"""Integration tests for the FastAPI query endpoint, refusal gates, and health probes."""

import json
from typing import Literal

import httpx
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.answer.llm import LLMQuotaExceeded, LLMResult, LLMUnavailable, StubLLM
from sift.answer.prompt import PROMPT_VERSION
from sift.api.app import create_app
from sift.auth import keys
from sift.config import Settings
from sift.ingest import pipeline
from tests.conftest import FakeEmbedder
from tests.integration.test_pipeline import ingest, page

PEPPER = b"test-pepper"


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


def _test_settings(engine: AsyncEngine, **overrides: object) -> Settings:
    url = engine.url.render_as_string(hide_password=False)
    base: dict[str, object] = {
        "database_url": url,
        "api_key_pepper": SecretStr("test-pepper"),
        "api_db_role": "sift_app",
        "demo_tenant": "pgdocs",
        "demo_collection": "pgdocs",
    }
    base.update(overrides)
    return Settings.model_validate(base)


async def _seed_pgdocs(engine: AsyncEngine, embedder: FakeEmbedder) -> str:
    tenant_id, collection_id = await pipeline.ensure_collection(engine, "pgdocs", "pgdocs")
    sub = await pipeline.submit(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        filename="page.html",
        data=page(),
        source_uri="https://docs.test/page.html",
        source="pgdocs",
        doc_version="17",
    )
    await ingest(engine, sub.version_id, embedder)
    nk = keys.generate(PEPPER)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO api_keys (tenant_id, prefix, key_hash, name) "
                "VALUES (:t, :p, :h, 'test')"
            ),
            {"t": tenant_id, "p": nk.prefix, "h": nk.key_hash},
        )
    return nk.plaintext


def _make_models_http(
    embedder: FakeEmbedder,
    *,
    embed_calls: list[int] | None = None,
    top_rerank_score: float = 0.9,
    fail_status: int | None = None,
) -> httpx.AsyncClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        if fail_status is not None:
            return httpx.Response(fail_status, json={"detail": "boom"})
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/embed":
            body = json.loads(request.content.decode())
            texts: list[str] = body["texts"]
            if embed_calls is not None:
                embed_calls.append(len(texts))
            vecs = await embedder(texts)
            return httpx.Response(200, json={"model": "fake-embed", "embeddings": vecs})
        if request.url.path == "/rerank":
            body = json.loads(request.content.decode())
            passages: list[str] = body["passages"]
            scores = [
                max(0.0, round(top_rerank_score - idx * 0.05, 4)) for idx in range(len(passages))
            ]
            return httpx.Response(200, json={"model": "fake-rerank", "scores": scores})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_hybrid_rerank_query_and_log_row(engine: AsyncEngine, embedder: FakeEmbedder) -> None:
    api_key = await _seed_pgdocs(engine, embedder)
    async with _make_models_http(embedder) as models_http:
        app = create_app(
            settings=_test_settings(engine),
            engine=engine,
            http=models_http,
            llm=StubLLM(),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            r_no_key = await client.post(
                "/v1/query",
                json={"collection": "pgdocs", "question": "Paragraph 0 of section 0"},
            )
            assert r_no_key.status_code == 401

            resp = await client.post(
                "/v1/query",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "X-Request-ID": "req_test_01",
                },
                json={"collection": "pgdocs", "question": "Paragraph 0 of section 0"},
            )

    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == "req_test_01"
    data = resp.json()
    assert not data["refused"]
    assert data["refusal_gate"] is None
    assert data["request_id"] == "req_test_01"
    assert data["prompt_version"] == PROMPT_VERSION
    assert set(data["timings_ms"]) == {
        "embed",
        "dense",
        "lexical",
        "fuse",
        "rerank",
        "llm",
        "total",
    }
    assert len(data["citations"]) >= 1
    for cit in data["citations"]:
        assert cit["url"].startswith("https://docs.test/page.html#SEC-")

    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT strategy, prompt_version, refused, cache_hit, top_rerank_score "
                    "FROM query_logs WHERE request_id = :r"
                ),
                {"r": "req_test_01"},
            )
        ).one()
    assert row.strategy == "hybrid_rerank"
    assert row.prompt_version == PROMPT_VERSION
    assert row.refused is False
    assert row.cache_hit is False
    assert row.top_rerank_score is not None


async def test_lexical_strategy_skips_embed(engine: AsyncEngine, embedder: FakeEmbedder) -> None:
    api_key = await _seed_pgdocs(engine, embedder)
    embed_calls: list[int] = []
    async with _make_models_http(embedder, embed_calls=embed_calls) as models_http:
        app = create_app(
            settings=_test_settings(engine),
            engine=engine,
            http=models_http,
            llm=StubLLM(),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            resp = await client.post(
                "/v1/query",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "collection": "pgdocs",
                    "question": "Paragraph 0 section",
                    "strategy": "lexical",
                },
            )

    assert resp.status_code == 200
    assert embed_calls == []
    assert resp.json()["timings_ms"]["embed"] == 0.0


async def test_unknown_keys_rejected_with_422(engine: AsyncEngine, embedder: FakeEmbedder) -> None:
    api_key = await _seed_pgdocs(engine, embedder)
    headers = {"Authorization": f"Bearer {api_key}"}
    async with _make_models_http(embedder) as models_http:
        app = create_app(
            settings=_test_settings(engine),
            engine=engine,
            http=models_http,
            llm=StubLLM(),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            r_top = await client.post(
                "/v1/query",
                headers=headers,
                json={"collection": "pgdocs", "question": "Paragraph 0", "bogus": 123},
            )
            r_flt = await client.post(
                "/v1/query",
                headers=headers,
                json={
                    "collection": "pgdocs",
                    "question": "Paragraph 0",
                    "filters": {"unknown_filter": "17"},
                },
            )

    assert r_top.status_code == 422
    assert r_flt.status_code == 422


async def test_low_rerank_score_refuses_at_retrieval_gate_without_calling_llm(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    api_key = await _seed_pgdocs(engine, embedder)
    counting_llm = CountingLLM()
    async with _make_models_http(embedder, top_rerank_score=0.01) as models_http:
        app = create_app(
            settings=_test_settings(engine, refusal_tau=0.19),
            engine=engine,
            http=models_http,
            llm=counting_llm,
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            resp = await client.post(
                "/v1/query",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"collection": "pgdocs", "question": "Paragraph 0 of section 0"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["refused"] is True
    assert data["refusal_gate"] == "retrieval"
    assert counting_llm.calls == 0


async def test_llm_quota_exceeded_says_so(engine: AsyncEngine, embedder: FakeEmbedder) -> None:
    api_key = await _seed_pgdocs(engine, embedder)
    async with _make_models_http(embedder) as models_http:
        app = create_app(
            settings=_test_settings(engine),
            engine=engine,
            http=models_http,
            llm=CountingLLM(fail=True, error=LLMQuotaExceeded),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            r = await client.post(
                "/v1/query",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"collection": "pgdocs", "question": "Paragraph 0 of section 0"},
            )

    assert r.status_code == 503
    assert r.headers.get("Retry-After") == "60"
    assert "free answer-model quota" in r.json()["detail"]


async def test_llm_and_models_unavailable_return_503(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    api_key = await _seed_pgdocs(engine, embedder)
    headers = {"Authorization": f"Bearer {api_key}"}
    async with _make_models_http(embedder) as models_http:
        app = create_app(
            settings=_test_settings(engine),
            engine=engine,
            http=models_http,
            llm=CountingLLM(fail=True),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            r_llm = await client.post(
                "/v1/query",
                headers=headers,
                json={"collection": "pgdocs", "question": "Paragraph 0 of section 0"},
            )

    assert r_llm.status_code == 503
    assert r_llm.headers.get("Retry-After") == "10"
    assert r_llm.json() == {"detail": "answer model unavailable"}

    async with _make_models_http(embedder, fail_status=500) as failing_models_http:
        app2 = create_app(
            settings=_test_settings(engine),
            engine=engine,
            http=failing_models_http,
            llm=StubLLM(),
        )
        async with (
            app2.router.lifespan_context(app2),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app2), base_url="http://test"
            ) as client,
        ):
            r_mod = await client.post(
                "/v1/query",
                headers=headers,
                json={"collection": "pgdocs", "question": "Paragraph 0 of section 0"},
            )

    assert r_mod.status_code == 503
    assert r_mod.json() == {"detail": "model service unavailable"}


async def test_index_csp_and_health_endpoints(engine: AsyncEngine, embedder: FakeEmbedder) -> None:
    await _seed_pgdocs(engine, embedder)
    async with _make_models_http(embedder) as models_http:
        app = create_app(
            settings=_test_settings(engine),
            engine=engine,
            http=models_http,
            llm=StubLLM(),
        )
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            r_index = await client.get("/")
            r_health = await client.get("/healthz")
            r_ready = await client.get("/readyz")

    assert r_index.status_code == 200
    assert "text/html" in r_index.headers["content-type"]
    assert "default-src 'none'" in r_index.headers["Content-Security-Policy"]
    assert r_index.headers["X-Content-Type-Options"] == "nosniff"
    assert r_index.headers["Referrer-Policy"] == "no-referrer"
    assert r_health.status_code == 200
    assert r_health.json() == {"status": "ok"}
    assert r_ready.status_code == 200
    assert r_ready.json() == {"status": "ok", "redis": "ok"}
