"""sift-ingest: the internal endpoints Hopper's http jobs call. Never routed publicly.

Every call must carry a valid Hopper-Signature made with Sift's tenant secret. Answers tell
Hopper what to do next: 2xx done, 422 permanent (straight to the DLQ: the file cannot be
parsed), 503 retry later with backoff (sift-models, Hopper or Postgres is unavailable).
"""

import base64
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import httpx
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.exc import DBAPIError

from sift.common.logging import configure_logging
from sift.config import get_settings
from sift.db import create_engine
from sift.hopper import HopperBusy, enqueue_step
from sift.ingest import pipeline, signature
from sift.ingest.parsers import ParseError

log = structlog.get_logger()


class VersionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version_id: UUID


class BatchIn(VersionIn):
    batch: int = Field(ge=0)


class Unauthorized(Exception):
    pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    secret = settings.hopper_signing_secret.get_secret_value()
    if not secret or not settings.hopper_api_key.get_secret_value():
        raise RuntimeError(
            "HOPPER_SIGNING_SECRET and HOPPER_API_KEY must be set (make hopper-tenant)"
        )
    app.state.secret = base64.urlsafe_b64decode(secret)
    app.state.engine = create_engine(settings)
    app.state.http = httpx.AsyncClient(timeout=settings.models_timeout_seconds)
    yield
    await app.state.http.aclose()
    await app.state.engine.dispose()


app = FastAPI(title="sift-ingest", lifespan=lifespan, openapi_url=None)


@app.exception_handler(ParseError)
async def _parse_error(_: Request, exc: ParseError) -> JSONResponse:
    log.warning("parse_failed", error=str(exc))
    return JSONResponse({"error": "unparseable", "detail": str(exc)[:500]}, status_code=422)


@app.exception_handler(LookupError)
async def _missing(_: Request, exc: LookupError) -> JSONResponse:
    return JSONResponse({"error": "not_found", "detail": str(exc)}, status_code=422)


@app.exception_handler(ValidationError)
async def _invalid(_: Request, exc: ValidationError) -> JSONResponse:
    return JSONResponse({"error": "invalid_body", "detail": exc.errors()[:3]}, status_code=422)


@app.exception_handler(Unauthorized)
async def _unauthorized(_: Request, exc: Unauthorized) -> JSONResponse:
    return JSONResponse({"error": "bad_signature"}, status_code=401)


async def _unavailable(_: Request, exc: Exception) -> JSONResponse:
    log.warning("step_retryable", error=f"{type(exc).__name__}: {exc}"[:500])
    retry_after = getattr(exc, "retry_after", None) or 5
    return JSONResponse(
        {"error": "unavailable"}, status_code=503, headers={"Retry-After": str(int(retry_after))}
    )


for _exc in (
    pipeline.NotReady,
    HopperBusy,
    httpx.TransportError,
    httpx.HTTPStatusError,
    DBAPIError,
):
    app.add_exception_handler(_exc, _unavailable)


async def verified(request: Request) -> bytes:
    """The raw body, after checking Hopper's signature over exactly these bytes."""
    body = await request.body()
    ok = signature.verify(
        request.app.state.secret,
        request.headers.get("hopper-signature"),
        body,
        now=time.time(),
        tolerance=get_settings().signature_tolerance_seconds,
    )
    if not ok:
        raise Unauthorized()
    # One document can be followed across retries: every log line carries the job and attempt.
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        hopper_job_id=request.headers.get("hopper-job-id"),
        hopper_attempt=request.headers.get("hopper-attempt"),
        path=request.url.path,
    )
    return body


async def embed_remote(request: Request, texts: list[str]) -> list[list[float]]:
    response = await request.app.state.http.post(
        f"{get_settings().models_url}/embed", json={"texts": texts, "kind": "passage"}
    )
    response.raise_for_status()
    vectors: list[list[float]] = response.json()["embeddings"]
    return vectors


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/internal/parse")
async def parse(request: Request) -> dict[str, int]:
    body = VersionIn.model_validate_json(await verified(request))
    v = body.version_id
    structlog.contextvars.bind_contextvars(version_id=str(v))
    n = await pipeline.parse_version(request.app.state.engine, v)
    http = request.app.state.http
    for b in range(n):
        await enqueue_step(http, "embed", {"version_id": str(v), "batch": b}, f"embed:{v}:{b}")
    if n == 0:  # no chunks (or already done): activation is next, and it is idempotent
        await enqueue_step(http, "activate", {"version_id": str(v)}, f"activate:{v}")
    log.info("parsed", batches=n)
    return {"batches": n}


@app.post("/internal/embed")
async def embed(request: Request) -> dict[str, bool]:
    body = BatchIn.model_validate_json(await verified(request))
    v = body.version_id
    structlog.contextvars.bind_contextvars(version_id=str(v), batch=body.batch)
    done = await pipeline.embed_batch(
        request.app.state.engine, v, body.batch, lambda texts: embed_remote(request, texts)
    )
    if done:  # several last batches may all see zero left: the key makes it one job
        await enqueue_step(
            request.app.state.http, "activate", {"version_id": str(v)}, f"activate:{v}"
        )
    log.info("embedded", version_complete=done)
    return {"version_complete": done}


@app.post("/internal/activate")
async def activate(request: Request) -> dict[str, str]:
    body = VersionIn.model_validate_json(await verified(request))
    structlog.contextvars.bind_contextvars(version_id=str(body.version_id))
    outcome = await pipeline.activate(request.app.state.engine, body.version_id)
    log.info("activate", outcome=outcome)
    return {"outcome": outcome}
