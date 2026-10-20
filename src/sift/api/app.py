"""FastAPI application for grounded querying, health checks, and the demo UI."""

import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import httpx
import structlog
from fastapi import FastAPI, HTTPException, Request, Response
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
from sift.common.logging import configure_logging
from sift.config import Settings, get_settings
from sift.db import create_engine
from sift.ingest import pipeline
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


class QueryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=1000),
    ]
    strategy: Strategy = "hybrid_rerank"
    filters: Filters | None = None
    k: int = Field(default=8, ge=1, le=20)


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


def create_app(
    *,
    engine: AsyncEngine | None = None,
    http: httpx.AsyncClient | None = None,
    llm: LLM | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        configure_logging(settings.log_level)

        own_engine = engine is None
        own_http = http is None
        active_engine = engine if engine is not None else create_engine(settings)
        active_http = (
            http if http is not None else httpx.AsyncClient(timeout=settings.models_timeout_seconds)
        )
        active_llm = llm if llm is not None else make_llm(settings, active_http)

        # shortcut: one dev tenant, no auth; API keys and tenancy replace this in week 5.
        tenant_id, collection_id = await pipeline.ensure_collection(
            active_engine, settings.dev_tenant, settings.dev_collection
        )

        app.state.settings = settings
        app.state.engine = active_engine
        app.state.http = active_http
        app.state.llm = active_llm
        app.state.tenant_id = tenant_id
        app.state.collection_id = collection_id
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
        raw_id = request.headers.get("X-Request-ID", "")
        req_id = raw_id if _REQ_ID_RE.match(raw_id) else uuid4().hex
        request.state.request_id = req_id

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
        settings: Settings = request.app.state.settings

        try:
            async with db_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc

        try:
            resp = await http_client.get(f"{settings.models_url.rstrip('/')}/healthz")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="model service unavailable") from exc

        return {"status": "ok"}

    @app.get("/", response_model=None)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.post("/v1/query", response_model=QueryOut)
    async def query_endpoint(body: QueryIn, request: Request) -> QueryOut:
        settings: Settings = request.app.state.settings
        db_engine: AsyncEngine = request.app.state.engine
        http_client: httpx.AsyncClient = request.app.state.http
        llm_client: LLM = request.app.state.llm
        tenant_id: UUID = request.app.state.tenant_id
        collection_id: UUID = request.app.state.collection_id
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
            CITATION_FAILURE_REASON
            if ans_res.refusal_gate == "citation"
            else ans_res.refusal_reason
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

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
