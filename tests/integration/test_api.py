"""Integration tests for the FastAPI query endpoint, refusal gates, and health probes."""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.answer.llm import LLMQuotaExceeded
from sift.answer.prompt import PROMPT_VERSION
from tests.conftest import FakeEmbedder
from tests.integration.helpers import CountingLLM, api, models_http, seed


async def test_hybrid_rerank_query_and_log_row(engine: AsyncEngine, embedder: FakeEmbedder) -> None:
    _, _, api_key = await seed(engine, embedder)
    async with api(engine, embedder) as client:
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
    _, _, api_key = await seed(engine, embedder)
    embed_calls: list[int] = []
    models = models_http(embedder, embed_calls=embed_calls)
    async with api(engine, embedder, models=models) as client:
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
    _, _, api_key = await seed(engine, embedder)
    headers = {"Authorization": f"Bearer {api_key}"}
    async with api(engine, embedder) as client:
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
    _, _, api_key = await seed(engine, embedder)
    counting_llm = CountingLLM()
    models = models_http(embedder, top_rerank_score=0.01)
    async with api(engine, embedder, models=models, llm=counting_llm, refusal_tau=0.19) as client:
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
    _, _, api_key = await seed(engine, embedder)
    llm = CountingLLM(fail=True, error=LLMQuotaExceeded)
    async with api(engine, embedder, llm=llm) as client:
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
    _, _, api_key = await seed(engine, embedder)
    headers = {"Authorization": f"Bearer {api_key}"}
    async with api(engine, embedder, llm=CountingLLM(fail=True)) as client:
        r_llm = await client.post(
            "/v1/query",
            headers=headers,
            json={"collection": "pgdocs", "question": "Paragraph 0 of section 0"},
        )

    assert r_llm.status_code == 503
    assert r_llm.headers.get("Retry-After") == "10"
    assert r_llm.json() == {"detail": "answer model unavailable"}

    async with api(engine, embedder, models=models_http(embedder, fail_status=500)) as client:
        r_mod = await client.post(
            "/v1/query",
            headers=headers,
            json={"collection": "pgdocs", "question": "Paragraph 0 of section 0"},
        )

    assert r_mod.status_code == 503
    assert r_mod.json() == {"detail": "model service unavailable"}


async def test_index_csp_and_health_endpoints(engine: AsyncEngine, embedder: FakeEmbedder) -> None:
    await seed(engine, embedder)
    async with api(engine, embedder) as client:
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
