"""What the integration tests share: docs pages and their ingestion, API keys, and the API app
with sift-models and Hopper mocked."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal
from uuid import UUID

import httpx
import redis.asyncio as redis_asyncio
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.answer.llm import LLM, LLMResult, LLMUnavailable, StubLLM
from sift.api.app import create_app
from sift.auth import keys
from sift.config import Settings
from sift.ingest import pipeline
from tests.conftest import FakeEmbedder

PEPPER = b"test-pepper"


def page(sections: int = 3, paragraphs: int = 4, extra: str = "") -> bytes:
    """A docs page: one h2 section per id, each with a few paragraphs."""
    body = "".join(
        f'<div class="sect2" id="SEC-{s}"><h3>Section {s}</h3>'
        + "".join(
            f"<p>Paragraph {p} of section {s} explains option_{s}_{p} in some detail. "
            + "Vacuum and analyze keep statistics fresh. " * 6
            + "</p>"
            for p in range(paragraphs)
        )
        + "</div>"
        for s in range(sections)
    )
    root = (
        f'<div id="docContent"><div class="sect1" id="P"><h2>A Page</h2>{extra}{body}</div></div>'
    )
    html = f"<html><body>{root}</body></html>"
    return html.encode()


async def setup(engine: AsyncEngine) -> tuple[UUID, UUID]:
    return await pipeline.ensure_collection(engine, "acme", "docs")


async def submit(engine: AsyncEngine, ids: tuple[UUID, UUID], data: bytes) -> pipeline.Submitted:
    return await pipeline.submit(
        engine,
        tenant_id=ids[0],
        collection_id=ids[1],
        filename="page.html",
        data=data,
        source_uri="https://docs.test/page.html",
        source="test",
        doc_version="17",
    )


async def ingest(engine: AsyncEngine, version_id: UUID, embed: FakeEmbedder) -> str:
    for b in range(await pipeline.parse_version(engine, version_id)):
        await pipeline.embed_batch(engine, version_id, b, embed)
    return await pipeline.activate(engine, version_id)


async def scalar(engine: AsyncEngine, sql: str, **params: object) -> object:
    async with engine.connect() as conn:
        return await conn.scalar(text(sql), params)


async def add_doc(
    engine: AsyncEngine,
    embedder: FakeEmbedder,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    body_text: str,
    title: str = "PostgreSQL Manual",
    heading: str = "Server Configuration",
    source_uri: str = "https://docs.test/config.html",
    source: str = "pgdocs",
    doc_version: str | None = "17",
    published_on: date | None = None,
) -> UUID:
    """A one-section page, ingested and live. Returns its document id."""
    html = (
        '<html><body><div id="docContent"><div class="sect1" id="P">'
        f"<h2>{title}</h2>"
        f'<div class="sect2" id="SEC-1"><h3>{heading}</h3><p>{body_text}</p></div>'
        "</div></div></body></html>"
    ).encode()
    sub = await pipeline.submit(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        filename="page.html",
        data=html,
        source_uri=source_uri,
        source=source,
        doc_version=doc_version,
        published_on=published_on,
    )
    await ingest(engine, sub.version_id, embedder)
    return sub.document_id


async def make_key(engine: AsyncEngine, tenant_id: UUID, *, revoked: bool = False) -> keys.NewKey:
    nk = keys.generate(PEPPER)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO api_keys (tenant_id, prefix, key_hash, name, revoked_at) "
                "VALUES (:t, :p, :h, 'test', CASE WHEN :rev THEN now() ELSE NULL END)"
            ),
            {"t": tenant_id, "p": nk.prefix, "h": nk.key_hash, "rev": revoked},
        )
    return nk


async def seed(
    engine: AsyncEngine, embedder: FakeEmbedder, tenant: str = "pgdocs", collection: str = "pgdocs"
) -> tuple[UUID, UUID, str]:
    """A collection with page() live and an API key: (tenant id, collection id, key)."""
    ids = await pipeline.ensure_collection(engine, tenant, collection)
    await ingest(engine, (await submit(engine, ids, page())).version_id, embedder)
    return *ids, (await make_key(engine, ids[0])).plaintext


class CountingLLM:
    model: str = "counting-stub"

    def __init__(self, fail: bool = False, error: type[LLMUnavailable] = LLMUnavailable) -> None:
        self.calls = 0
        self._fail = fail
        self._error = error
        self._stub = StubLLM()

    async def generate(
        self,
        system: str,
        turns: list[tuple[Literal["user", "model"], str]],
    ) -> LLMResult:
        self.calls += 1
        if self._fail:
            raise self._error("simulated rate limit")
        return await self._stub.generate(system, turns)


def models_http(
    embedder: FakeEmbedder,
    *,
    embed_calls: list[int] | None = None,
    top_rerank_score: float = 0.9,
    fail_status: int | None = None,
    hopper_requests: list[httpx.Request] | None = None,
) -> httpx.AsyncClient:
    """sift-models (embed, rerank) and Hopper's job API, answered in process."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if fail_status is not None:
            return httpx.Response(fail_status, json={"detail": "boom"})
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/embed":
            texts: list[str] = json.loads(request.content)["texts"]
            if embed_calls is not None:
                embed_calls.append(len(texts))
            vecs = await embedder(texts)
            return httpx.Response(200, json={"model": "fake-embed", "embeddings": vecs})
        if request.url.path == "/rerank":
            passages: list[str] = json.loads(request.content)["passages"]
            scores = [max(0.0, round(top_rerank_score - i * 0.05, 4)) for i in range(len(passages))]
            return httpx.Response(200, json={"model": "fake-rerank", "scores": scores})
        if request.url.path == "/v1/jobs":
            if hopper_requests is not None:
                hopper_requests.append(request)
            return httpx.Response(201, json={"id": "job-1", "status": "queued"})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def settings(engine: AsyncEngine, **overrides: object) -> Settings:
    return Settings.model_validate(
        {
            "database_url": engine.url.render_as_string(hide_password=False),
            "api_key_pepper": SecretStr(PEPPER.decode()),
            "api_db_role": "sift_app",
            "hopper_api_key": SecretStr("hopper-test-key"),
            "demo_tenant": "pgdocs",
            "demo_collection": "pgdocs",
            "metrics_port": 0,
            **overrides,
        }
    )


@asynccontextmanager
async def api(
    engine: AsyncEngine,
    embedder: FakeEmbedder,
    *,
    app_engine: AsyncEngine | None = None,
    models: httpx.AsyncClient | None = None,
    llm: LLM | None = None,
    redis: redis_asyncio.Redis | None = None,
    **overrides: object,
) -> AsyncIterator[httpx.AsyncClient]:
    """A client of the API app, started on the test database (queried through app_engine if
    given) with settings(engine, **overrides) and models_http(embedder) unless models is given."""
    async with models or models_http(embedder) as http:
        app = create_app(
            settings=settings(engine, **overrides),
            engine=app_engine or engine,
            http=http,
            llm=llm or StubLLM(),
            redis=redis,
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            yield client
