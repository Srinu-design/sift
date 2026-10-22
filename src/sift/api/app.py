"""FastAPI application for multi-tenant querying, ingestion submission, and the demo UI."""

import json
import re
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import httpx
import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.answer.citations import Citation
from sift.answer.generate import answer
from sift.answer.llm import LLM, LLMUnavailable, make_llm
from sift.answer.prompt import PROMPT_VERSION
from sift.auth.api_keys import ApiKeyAuthenticator, current_tenant
from sift.common.logging import configure_logging
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


async def _execute_query(
    request: Request,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    body: DemoQueryIn,
) -> QueryOut:
    settings: Settings = request.app.state.settings
    db_engine: AsyncEngine = request.app.state.engine
    http_client: httpx.AsyncClient = request.app.state.http
    llm_client: LLM = request.app.state.llm
    req_id: str = request.state.request_id

    try:
        if body.strategy == "lexical":
            query_vec: list[float] | None = None
            embed_ms = 0.0
        else:
            query_vec, embed_ms = await _embed_question(
                http_client, settings.models_url, body.question
            )

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

    timings_ms: dict[str, float] = {
        "embed": embed_ms,
        "dense": search_res.timings_ms.get("dense", 0.0),
        "lexical": search_res.timings_ms.get("lexical", 0.0),
        "fuse": search_res.timings_ms.get("fuse", 0.0),
        "rerank": search_res.timings_ms.get("rerank", 0.0),
        "llm": ans_res.llm_ms,
    }
    cost_usd = round(
        (
            ans_res.tokens_in * settings.llm_price_in_per_mtok
            + ans_res.tokens_out * settings.llm_price_out_per_mtok
        )
        / 1_000_000.0,
        6,
    )
    top_rerank_score = (
        float(search_res.hits[0].score)
        if body.strategy == "hybrid_rerank" and search_res.hits
        else None
    )

    try:
        async with db_engine.begin() as conn:
            await set_tenant(conn, tenant_id)
            await conn.execute(
                text(INSERT_QUERY_LOG),
                {
                    "tenant_id": tenant_id,
                    "collection_id": collection_id,
                    "request_id": req_id,
                    "strategy": body.strategy,
                    "cache_hit": False,
                    "refused": ans_res.refused,
                    "refusal_gate": ans_res.refusal_gate,
                    "top_rerank_score": top_rerank_score,
                    "timings_ms": json.dumps(timings_ms),
                    "tokens_in": ans_res.tokens_in,
                    "tokens_out": ans_res.tokens_out,
                    "cost_usd": cost_usd,
                    "prompt_version": PROMPT_VERSION,
                    "model": ans_res.model,
                },
            )
    except DBAPIError as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc

    logger.info(
        "query",
        request_id=req_id,
        strategy=body.strategy,
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
    return QueryOut(
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


def create_app(
    *,
    settings: Settings | None = None,
    engine: AsyncEngine | None = None,
    http: httpx.AsyncClient | None = None,
    llm: LLM | None = None,
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
        app.state.api_keys = ApiKeyAuthenticator(active_engine, pepper_str.encode())
        app.state.demo_ids = demo_ids
        # shortcut: per process and in memory; the Redis rate limiter replaces it in week 6.
        demo_requests: dict[str, deque[float]] = defaultdict(deque)
        app.state.demo_requests = demo_requests
        try:
            yield
        finally:
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

        return {"status": "ok"}

    @app.get("/", response_model=None)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.post("/v1/query", response_model=QueryOut)
    async def query_endpoint(
        body: QueryIn,
        request: Request,
        tenant_id: Annotated[UUID, Depends(current_tenant)],
    ) -> QueryOut:
        db_engine: AsyncEngine = request.app.state.engine
        try:
            async with db_engine.connect() as conn:
                collection_id = await conn.scalar(
                    text("SELECT id FROM collections WHERE tenant_id = :t AND name = :c"),
                    {"t": tenant_id, "c": body.collection},
                )
        except DBAPIError as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc

        if collection_id is None:
            raise HTTPException(status_code=404, detail="collection not found")

        return await _execute_query(
            request,
            tenant_id=tenant_id,
            collection_id=UUID(str(collection_id)),
            body=body,
        )

    @app.post("/demo/query", response_model=QueryOut)
    async def demo_query_endpoint(body: DemoQueryIn, request: Request) -> QueryOut:
        demo_ids: tuple[UUID, UUID] | None = request.app.state.demo_ids
        if demo_ids is None:
            raise HTTPException(status_code=404, detail="demo disabled")

        cfg: Settings = request.app.state.settings
        # shortcut: per process and in memory; the Redis rate limiter replaces it in week 6.
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = 60.0
        history: deque[float] = request.app.state.demo_requests[client_ip]
        while history and (now - history[0]) >= window:
            history.popleft()
        if len(history) >= cfg.demo_rate_per_minute:
            retry_after = max(1, int(window - (now - history[0])) + 1)
            raise HTTPException(
                status_code=429,
                detail="demo rate limit",
                headers={"Retry-After": str(retry_after)},
            )
        history.append(now)

        tenant_id, collection_id = demo_ids
        return await _execute_query(
            request,
            tenant_id=tenant_id,
            collection_id=collection_id,
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
