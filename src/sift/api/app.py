"""FastAPI application for multi-tenant querying, ingestion submission, and the demo UI."""

import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4

import httpx
import redis.asyncio as redis_asyncio
import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from sift import metrics
from sift.answer.citations import Citation
from sift.answer.generate import answer
from sift.answer.llm import LLM, LLMUnavailable, make_llm
from sift.answer.prompt import PROMPT_VERSION
from sift.answer.refusal import gate1
from sift.auth.api_keys import ApiKeyAuthenticator, current_tenant
from sift.cache import REDIS_DOWN, AnswerCache, QueryEmbeddingCache, SpendBudget, answer_key
from sift.common.logging import configure_logging
from sift.common.ratelimit.limiter import RateLimiter
from sift.common.ratelimit.redis_health import RedisHealth
from sift.config import Settings, get_settings
from sift.db import create_engine, set_tenant
from sift.hopper import HopperBusy, enqueue_step
from sift.ingest import pipeline
from sift.ingest.parsers import ParseError
from sift.retrieval.filters import Filters
from sift.retrieval.rerank import remote_reranker
from sift.retrieval.search import DEPTH, Strategy, search

STATIC = Path(__file__).parent / "static"
CSP_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
    "img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
CITATION_FAILURE_REASON = "Could not produce an answer whose citations check out."
_REQ_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

INSERT_QUERY_LOG = """
INSERT INTO query_logs (
    tenant_id, collection_id, request_id, strategy, cache_hit,
    refused, refusal_gate, top_rerank_score, timings_ms,
    tokens_in, tokens_out, cost_usd, prompt_version, model
) VALUES (
    :tenant_id, :collection_id, :request_id, :strategy, :cache_hit,
    :refused, :refusal_gate, :top_rerank_score, CAST(:timings_ms AS jsonb),
    :tokens_in, :tokens_out, :cost_usd, :prompt_version, :model
)
"""

logger = structlog.get_logger()

NonEmptyQuestion = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1000),
]
CollectionName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
]


class DemoQueryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: NonEmptyQuestion
    strategy: Strategy = "hybrid_rerank"
    filters: Filters | None = None
    k: int = Field(default=8, ge=1, le=20)


class QueryIn(DemoQueryIn):
    collection: CollectionName


class QueryOut(BaseModel):
    answer: str
    refused: bool
    refusal_gate: str | None
    refusal_reason: str | None
    citations: list[Citation]
    suggestions: list[Citation]
    strategy: str
    cache_hit: bool
    timings_ms: dict[str, float]
    cost_usd: float
    request_id: str
    prompt_version: str


class CollectionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: CollectionName


class CollectionOut(BaseModel):
    id: UUID
    name: str


class DocumentSubmittedOut(BaseModel):
    document_id: UUID
    version_id: UUID
    status: str


class DocumentVersionOut(BaseModel):
    id: UUID
    status: str
    chunk_count: int | None
    created_at: datetime


class DocumentOut(BaseModel):
    id: UUID
    title: str
    source_uri: str
    doc_version: str | None
    active_version_id: UUID | None
    versions: list[DocumentVersionOut]


async def _embed_question(
    http: httpx.AsyncClient,
    models_url: str,
    question: str,
) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    resp = await http.post(
        f"{models_url.rstrip('/')}/embed",
        json={"texts": [question], "kind": "query"},
    )
    resp.raise_for_status()
    data = resp.json()
    vec = [float(x) for x in data["embeddings"][0]]
    return vec, round((time.perf_counter() - t0) * 1000.0, 1)


async def _write_query_log(
    db_engine: AsyncEngine,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    request_id: str,
    strategy: str,
    cache_hit: bool,
    refused: bool,
    refusal_gate: str | None,
    top_rerank_score: float | None,
    timings_ms: dict[str, float],
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    prompt_version: str,
    model: str | None,
) -> None:
    try:
        async with db_engine.begin() as conn:
            await set_tenant(conn, tenant_id)
            await conn.execute(
                text(INSERT_QUERY_LOG),
                {
                    "tenant_id": tenant_id,
                    "collection_id": collection_id,
                    "request_id": request_id,
                    "strategy": strategy,
                    "cache_hit": cache_hit,
                    "refused": refused,
                    "refusal_gate": refusal_gate,
                    "top_rerank_score": top_rerank_score,
                    "timings_ms": json.dumps(timings_ms),
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "cost_usd": cost_usd,
                    "prompt_version": prompt_version,
                    "model": model,
                },
            )
    except DBAPIError as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc


async def _execute_query(
    request: Request,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    index_version: int,
    body: DemoQueryIn,
) -> QueryOut:
    t_total = time.perf_counter()
    settings: Settings = request.app.state.settings
    db_engine: AsyncEngine = request.app.state.engine
    http_client: httpx.AsyncClient = request.app.state.http
    llm_client: LLM = request.app.state.llm
    qemb_cache: QueryEmbeddingCache = request.app.state.query_embedding_cache
    ans_cache: AnswerCache = request.app.state.answer_cache
    budget: SpendBudget = request.app.state.budget
    req_id: str = request.state.request_id

    cache_model = settings.answer_model if settings.llm_provider == "gemini" else "stub"
    ans_key = answer_key(
        tenant_id=tenant_id,
        collection_id=collection_id,
        index_version=index_version,
        strategy=body.strategy,
        filters=body.filters,
        k=body.k,
        prompt_version=PROMPT_VERSION,
        model=cache_model,
        question=body.question,
    )

    if settings.answer_cache_enabled:
        cached = await ans_cache.get(ans_key)
        if cached is not None:
            total_ms = round((time.perf_counter() - t_total) * 1000.0, 1)
            hit_timings: dict[str, float] = {
                "embed": 0.0,
                "dense": 0.0,
                "lexical": 0.0,
                "fuse": 0.0,
                "rerank": 0.0,
                "llm": 0.0,
                "total": total_ms,
            }
            hit_out = QueryOut.model_validate(
                {
                    **cached,
                    "cache_hit": True,
                    "timings_ms": hit_timings,
                    "cost_usd": 0.0,
                    "request_id": req_id,
                }
            )
            await _write_query_log(
                db_engine,
                tenant_id=tenant_id,
                collection_id=collection_id,
                request_id=req_id,
                strategy=body.strategy,
                cache_hit=True,
                refused=hit_out.refused,
                refusal_gate=hit_out.refusal_gate,
                top_rerank_score=None,
                timings_ms=hit_timings,
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                prompt_version=hit_out.prompt_version,
                model=cache_model,
            )
            metrics.QUERY_SECONDS.labels(stage="total").observe(total_ms / 1000.0)
            metrics.QUERIES.labels(strategy=body.strategy).inc()
            logger.info(
                "query",
                request_id=req_id,
                strategy=body.strategy,
                cache_hit=True,
                refused=hit_out.refused,
                refusal_gate=hit_out.refusal_gate,
                timings_ms=hit_timings,
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
            )
            return hit_out

    try:
        if body.strategy == "lexical":
            query_vec: list[float] | None = None
            embed_ms = 0.0
        else:
            t_embed = time.perf_counter()
            cached_vec = await qemb_cache.get(body.question)
            if cached_vec is not None:
                query_vec = cached_vec
                embed_ms = round((time.perf_counter() - t_embed) * 1000.0, 1)
            else:
                query_vec, embed_ms = await _embed_question(
                    http_client, settings.models_url, body.question
                )
                await qemb_cache.put(body.question, query_vec)

        reranker = (
            remote_reranker(http_client, settings.models_url)
            if body.strategy == "hybrid_rerank"
            else None
        )
        search_res = await search(
            db_engine,
            tenant_id=tenant_id,
            collection_id=collection_id,
            strategy=body.strategy,
            question=body.question,
            query_vector=query_vec,
            filters=body.filters,
            k=body.k,
            depth=DEPTH,
            rerank_depth=settings.rerank_depth,
            reranker=reranker,
        )
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise HTTPException(status_code=503, detail="model service unavailable") from exc
    except DBAPIError as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc

    tau = settings.refusal_tau if body.strategy == "hybrid_rerank" else None
    best_score = float(search_res.hits[0].score) if search_res.hits else None
    will_call_llm = bool(search_res.hits) and (tau is None or not gate1(best_score, tau))
    if will_call_llm:
        retry = await budget.check(tenant_id)
        if retry is not None:
            metrics.BUDGET_REJECTIONS.inc()
            raise HTTPException(
                status_code=429,
                detail="budget_exceeded",
                headers={"Retry-After": str(retry)},
            )

    try:
        ans_res = await answer(
            body.question,
            search_res.hits,
            llm_client,
            tau=tau,
            context_k=body.k,
        )
    except LLMUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="answer model unavailable",
            headers={"Retry-After": "10"},
        ) from exc

    total_ms = round((time.perf_counter() - t_total) * 1000.0, 1)
    timings_ms: dict[str, float] = {
        "embed": embed_ms,
        "dense": search_res.timings_ms.get("dense", 0.0),
        "lexical": search_res.timings_ms.get("lexical", 0.0),
        "fuse": search_res.timings_ms.get("fuse", 0.0),
        "rerank": search_res.timings_ms.get("rerank", 0.0),
        "llm": ans_res.llm_ms,
        "total": total_ms,
    }
    cost_usd = round(
        (
            ans_res.tokens_in * settings.llm_price_in_per_mtok
            + ans_res.tokens_out * settings.llm_price_out_per_mtok
        )
        / 1_000_000.0,
        6,
    )
    await budget.add(tenant_id, cost_usd)

    top_rerank_score = best_score if body.strategy == "hybrid_rerank" else None

    await _write_query_log(
        db_engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        request_id=req_id,
        strategy=body.strategy,
        cache_hit=False,
        refused=ans_res.refused,
        refusal_gate=ans_res.refusal_gate,
        top_rerank_score=top_rerank_score,
        timings_ms=timings_ms,
        tokens_in=ans_res.tokens_in,
        tokens_out=ans_res.tokens_out,
        cost_usd=cost_usd,
        prompt_version=PROMPT_VERSION,
        model=ans_res.model,
    )

    if body.strategy != "lexical":
        metrics.QUERY_SECONDS.labels(stage="embed").observe(embed_ms / 1000.0)
    for stage_name, stage_ms in search_res.timings_ms.items():
        metrics.QUERY_SECONDS.labels(stage=stage_name).observe(stage_ms / 1000.0)
    if will_call_llm:
        metrics.QUERY_SECONDS.labels(stage="llm").observe(ans_res.llm_ms / 1000.0)
    metrics.QUERY_SECONDS.labels(stage="total").observe(total_ms / 1000.0)
    metrics.LLM_TOKENS.labels(direction="in").inc(ans_res.tokens_in)
    metrics.LLM_TOKENS.labels(direction="out").inc(ans_res.tokens_out)
    metrics.QUERY_COST.inc(cost_usd)
    if ans_res.refused and ans_res.refusal_gate is not None:
        metrics.REFUSALS.labels(gate=ans_res.refusal_gate).inc()
    metrics.QUERIES.labels(strategy=body.strategy).inc()

    logger.info(
        "query",
        request_id=req_id,
        strategy=body.strategy,
        cache_hit=False,
        refused=ans_res.refused,
        refusal_gate=ans_res.refusal_gate,
        refusal_reason=ans_res.refusal_reason,
        timings_ms=timings_ms,
        tokens_in=ans_res.tokens_in,
        tokens_out=ans_res.tokens_out,
        cost_usd=cost_usd,
    )

    client_reason = (
        CITATION_FAILURE_REASON if ans_res.refusal_gate == "citation" else ans_res.refusal_reason
    )
    out = QueryOut(
        answer=ans_res.answer,
        refused=ans_res.refused,
        refusal_gate=ans_res.refusal_gate,
        refusal_reason=client_reason,
        citations=ans_res.citations,
        suggestions=ans_res.suggestions,
        strategy=body.strategy,
        cache_hit=False,
        timings_ms=timings_ms,
        cost_usd=cost_usd,
        request_id=req_id,
        prompt_version=PROMPT_VERSION,
    )
    if settings.answer_cache_enabled and ans_res.refusal_gate != "citation":
        payload = out.model_dump(
            mode="json",
            exclude={"request_id", "timings_ms", "cache_hit"},
        )
        await ans_cache.put(ans_key, payload)
    return out


def create_app(
    *,
    settings: Settings | None = None,
    engine: AsyncEngine | None = None,
    http: httpx.AsyncClient | None = None,
    llm: LLM | None = None,
    redis: redis_asyncio.Redis | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings = settings if settings is not None else get_settings()
        configure_logging(active_settings.log_level)

        pepper_str = active_settings.api_key_pepper.get_secret_value()
        if not pepper_str:
            raise RuntimeError("api_key_pepper must not be empty")

        own_engine = engine is None
        own_http = http is None
        own_redis = redis is None
        active_engine = (
            engine
            if engine is not None
            else create_engine(active_settings, role=active_settings.api_db_role)
        )
        active_http = (
            http
            if http is not None
            else httpx.AsyncClient(timeout=active_settings.models_timeout_seconds)
        )
        active_llm = llm if llm is not None else make_llm(active_settings, active_http)
        active_redis: redis_asyncio.Redis = (
            redis if redis is not None else redis_asyncio.from_url(active_settings.redis_url)
        )
        health = RedisHealth(active_settings.redis_retry_seconds)
        embed_model_tag = f"{active_settings.embed_model}@{active_settings.embed_revision[:12]}"
        stop_metrics = metrics.serve(active_settings.metrics_port)

        demo_ids: tuple[UUID, UUID] | None = None
        if active_settings.demo_tenant is not None:
            async with active_engine.connect() as conn:
                row = (
                    await conn.execute(
                        text(
                            "SELECT c.id, c.tenant_id "
                            "FROM collections c JOIN tenants t ON t.id = c.tenant_id "
                            "WHERE t.name = :t AND c.name = :c"
                        ),
                        {
                            "t": active_settings.demo_tenant,
                            "c": active_settings.demo_collection,
                        },
                    )
                ).one_or_none()
            if row is not None:
                demo_ids = (row.tenant_id, row.id)
            else:
                logger.warning(
                    "demo_collection_not_found",
                    tenant=active_settings.demo_tenant,
                    collection=active_settings.demo_collection,
                )

        app.state.settings = active_settings
        app.state.engine = active_engine
        app.state.http = active_http
        app.state.llm = active_llm
        app.state.redis = active_redis
        app.state.redis_health = health
        app.state.limiter = RateLimiter(active_redis, namespace="sift", health=health)
        app.state.query_embedding_cache = QueryEmbeddingCache(
            active_redis,
            health,
            model=embed_model_tag,
            ttl_seconds=active_settings.query_embedding_ttl_seconds,
        )
        app.state.answer_cache = AnswerCache(
            active_redis,
            health,
            ttl_seconds=active_settings.answer_cache_ttl_seconds,
        )
        app.state.budget = SpendBudget(
            active_redis,
            health,
            daily_usd=active_settings.daily_budget_usd,
        )
        app.state.api_keys = ApiKeyAuthenticator(active_engine, pepper_str.encode())
        app.state.demo_ids = demo_ids
        try:
            yield
        finally:
            stop_metrics()
            if own_redis:
                await active_redis.aclose()
            if own_http:
                await active_http.aclose()
            if own_engine:
                await active_engine.dispose()

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def _security_and_request_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        structlog.contextvars.clear_contextvars()
        raw_id = request.headers.get("X-Request-ID", "")
        req_id = raw_id if _REQ_ID_RE.match(raw_id) else uuid4().hex
        request.state.request_id = req_id
        structlog.contextvars.bind_contextvars(request_id=req_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        response.headers["Content-Security-Policy"] = CSP_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, str]:
        db_engine: AsyncEngine = request.app.state.engine
        http_client: httpx.AsyncClient = request.app.state.http
        cfg: Settings = request.app.state.settings
        redis_client: Any = request.app.state.redis
        health: RedisHealth = request.app.state.redis_health

        try:
            async with db_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc

        try:
            resp = await http_client.get(f"{cfg.models_url.rstrip('/')}/healthz")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="model service unavailable") from exc

        try:
            await redis_client.ping()
        except REDIS_DOWN as exc:
            health.failed("readyz_ping", exc)
            return {"status": "ok", "redis": "down"}

        return {"status": "ok", "redis": "ok"}

    @app.get("/", response_model=None)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.post("/v1/query", response_model=QueryOut)
    async def query_endpoint(
        body: QueryIn,
        request: Request,
        tenant_id: Annotated[UUID, Depends(current_tenant)],
    ) -> QueryOut:
        cfg: Settings = request.app.state.settings
        limiter: RateLimiter = request.app.state.limiter
        decision = await limiter.take(
            f"query:{tenant_id}",
            capacity=cfg.query_burst,
            rate=cfg.query_rate_per_sec,
        )
        if not decision.allowed:
            raise HTTPException(
                status_code=429,
                detail="rate limit",
                headers={
                    "Retry-After": str(decision.retry_after_seconds),
                    "X-RateLimit-Limit": str(decision.limit),
                    "X-RateLimit-Remaining": str(decision.remaining),
                },
            )

        db_engine: AsyncEngine = request.app.state.engine
        try:
            async with db_engine.connect() as conn:
                row = (
                    await conn.execute(
                        text(
                            "SELECT id, index_version FROM collections "
                            "WHERE tenant_id = :t AND name = :c"
                        ),
                        {"t": tenant_id, "c": body.collection},
                    )
                ).one_or_none()
        except DBAPIError as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc

        if row is None:
            raise HTTPException(status_code=404, detail="collection not found")

        return await _execute_query(
            request,
            tenant_id=tenant_id,
            collection_id=UUID(str(row.id)),
            index_version=int(row.index_version),
            body=body,
        )

    @app.post("/demo/query", response_model=QueryOut)
    async def demo_query_endpoint(body: DemoQueryIn, request: Request) -> QueryOut:
        demo_ids: tuple[UUID, UUID] | None = request.app.state.demo_ids
        if demo_ids is None:
            raise HTTPException(status_code=404, detail="demo disabled")

        cfg: Settings = request.app.state.settings
        limiter: RateLimiter = request.app.state.limiter
        client_ip = request.client.host if request.client else "unknown"
        decision = await limiter.take(
            f"demo:{client_ip}",
            capacity=cfg.demo_rate_per_minute,
            rate=cfg.demo_rate_per_minute / 60.0,
        )
        if not decision.allowed:
            raise HTTPException(
                status_code=429,
                detail="demo rate limit",
                headers={
                    "Retry-After": str(decision.retry_after_seconds),
                    "X-RateLimit-Limit": str(decision.limit),
                    "X-RateLimit-Remaining": str(decision.remaining),
                },
            )

        tenant_id, collection_id = demo_ids
        db_engine: AsyncEngine = request.app.state.engine
        try:
            async with db_engine.connect() as conn:
                raw_ver = await conn.scalar(
                    text("SELECT index_version FROM collections WHERE id = :c"),
                    {"c": collection_id},
                )
        except DBAPIError as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc

        if raw_ver is None:
            raise HTTPException(status_code=404, detail="collection not found")

        return await _execute_query(
            request,
            tenant_id=tenant_id,
            collection_id=collection_id,
            index_version=int(raw_ver),
            body=body,
        )

    @app.post("/v1/collections", response_model=CollectionOut, status_code=201)
    async def create_collection_endpoint(
        body: CollectionIn,
        request: Request,
        tenant_id: Annotated[UUID, Depends(current_tenant)],
    ) -> CollectionOut:
        db_engine: AsyncEngine = request.app.state.engine
        try:
            async with db_engine.begin() as conn:
                row = (
                    await conn.execute(
                        text(
                            "INSERT INTO collections (tenant_id, name) VALUES (:t, :n) "
                            "ON CONFLICT (tenant_id, name) DO NOTHING RETURNING id, name"
                        ),
                        {"t": tenant_id, "n": body.name},
                    )
                ).one_or_none()
        except DBAPIError as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc

        if row is None:
            raise HTTPException(status_code=409, detail="collection already exists")
        return CollectionOut(id=row.id, name=row.name)

    @app.get("/v1/collections", response_model=list[CollectionOut])
    async def list_collections_endpoint(
        request: Request,
        tenant_id: Annotated[UUID, Depends(current_tenant)],
    ) -> list[CollectionOut]:
        db_engine: AsyncEngine = request.app.state.engine
        try:
            async with db_engine.connect() as conn:
                rows = (
                    await conn.execute(
                        text("SELECT id, name FROM collections WHERE tenant_id = :t ORDER BY name"),
                        {"t": tenant_id},
                    )
                ).all()
        except DBAPIError as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return [CollectionOut(id=r.id, name=r.name) for r in rows]

    @app.post(
        "/v1/collections/{collection_id}/documents",
        response_model=DocumentSubmittedOut,
        status_code=202,
    )
    async def upload_document_endpoint(
        collection_id: UUID,
        request: Request,
        tenant_id: Annotated[UUID, Depends(current_tenant)],
        filename: Annotated[str, Query(min_length=1)],
        source: Annotated[str, Query(min_length=1)],
        source_uri: str | None = None,
        doc_version: str | None = None,
        published_on: date | None = None,
    ) -> DocumentSubmittedOut:
        cfg: Settings = request.app.state.settings
        db_engine: AsyncEngine = request.app.state.engine
        http_client: httpx.AsyncClient = request.app.state.http

        data = await request.body()
        if len(data) > cfg.max_upload_bytes:
            raise HTTPException(status_code=413, detail="file exceeds max_upload_bytes")
        if not data:
            raise HTTPException(status_code=422, detail="empty file body")

        try:
            async with db_engine.connect() as conn:
                owned = await conn.scalar(
                    text("SELECT 1 FROM collections WHERE id = :c AND tenant_id = :t"),
                    {"c": collection_id, "t": tenant_id},
                )
        except DBAPIError as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc

        if not owned:
            raise HTTPException(status_code=404, detail="collection not found")

        try:
            submitted = await pipeline.submit(
                db_engine,
                tenant_id=tenant_id,
                collection_id=collection_id,
                filename=filename,
                data=data,
                source_uri=source_uri or filename,
                source=source,
                doc_version=doc_version,
                published_on=published_on,
            )
            async with db_engine.begin() as conn:
                await set_tenant(conn, tenant_id)
                doc_id = await conn.scalar(
                    text("SELECT document_id FROM document_versions WHERE id = :v"),
                    {"v": submitted.version_id},
                )
        except ParseError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except DBAPIError as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc

        if submitted.needs_parse:
            ver_str = str(submitted.version_id)
            try:
                await enqueue_step(
                    http_client,
                    "parse",
                    {"version_id": ver_str},
                    f"parse:{ver_str}",
                )
            except HopperBusy as exc:
                retry = int(exc.retry_after) if exc.retry_after is not None else 5
                raise HTTPException(
                    status_code=503,
                    detail="ingestion queue busy",
                    headers={"Retry-After": str(retry)},
                ) from exc

        return DocumentSubmittedOut(
            document_id=UUID(str(doc_id)),
            version_id=submitted.version_id,
            status=submitted.status,
        )

    @app.get("/v1/documents/{document_id}", response_model=DocumentOut)
    async def get_document_endpoint(
        document_id: UUID,
        request: Request,
        tenant_id: Annotated[UUID, Depends(current_tenant)],
    ) -> DocumentOut:
        db_engine: AsyncEngine = request.app.state.engine
        try:
            async with db_engine.begin() as conn:
                await set_tenant(conn, tenant_id)
                doc = (
                    await conn.execute(
                        text(
                            "SELECT id, title, source_uri, doc_version, active_version_id "
                            "FROM documents WHERE id = :id AND tenant_id = :t"
                        ),
                        {"id": document_id, "t": tenant_id},
                    )
                ).one_or_none()
                if doc is None:
                    raise HTTPException(status_code=404, detail="document not found")
                versions = (
                    await conn.execute(
                        text(
                            "SELECT id, status, chunk_count, created_at "
                            "FROM document_versions WHERE document_id = :d ORDER BY created_at"
                        ),
                        {"d": document_id},
                    )
                ).all()
        except DBAPIError as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc

        return DocumentOut(
            id=doc.id,
            title=doc.title,
            source_uri=doc.source_uri,
            doc_version=doc.doc_version,
            active_version_id=doc.active_version_id,
            versions=[
                DocumentVersionOut(
                    id=v.id,
                    status=v.status,
                    chunk_count=v.chunk_count,
                    created_at=v.created_at,
                )
                for v in versions
            ],
        )

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
