"""Integration tests for Redis caches, tenant isolation, spend budgets, and rate limits."""

import redis.asyncio as redis_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from tests.conftest import FakeEmbedder
from tests.integration.helpers import CountingLLM, api, models_http, seed


async def test_answer_cache_hit_and_query_logs(
    engine: AsyncEngine,
    embedder: FakeEmbedder,
    redis_client: redis_asyncio.Redis,
) -> None:
    tenant_id, _, api_key = await seed(engine, embedder)
    llm = CountingLLM()
    async with api(engine, embedder, llm=llm, redis=redis_client) as client:
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"collection": "pgdocs", "question": "What is MVCC?"}
        r1 = await client.post("/v1/query", json=payload, headers=headers)
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["cache_hit"] is False
        assert "total" in d1["timings_ms"]

        r2 = await client.post("/v1/query", json=payload, headers=headers)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["cache_hit"] is True
        assert d2["cost_usd"] == 0.0
        assert d2["request_id"] != d1["request_id"]
        assert d2["timings_ms"]["embed"] == 0.0
        assert d2["timings_ms"]["llm"] == 0.0
        assert "total" in d2["timings_ms"]

    assert llm.calls == 1
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT cache_hit, tokens_in, tokens_out, cost_usd "
                    "FROM query_logs WHERE tenant_id = :t"
                ),
                {"t": tenant_id},
            )
        ).all()
    assert len(rows) == 2
    assert sorted(bool(r.cache_hit) for r in rows) == [False, True]
    hit_row = next(r for r in rows if r.cache_hit)
    assert hit_row.tokens_in == 0
    assert hit_row.tokens_out == 0
    assert float(hit_row.cost_usd) == 0.0


async def test_tenant_isolation_in_answer_cache(
    engine: AsyncEngine,
    embedder: FakeEmbedder,
    redis_client: redis_asyncio.Redis,
) -> None:
    _, _, key_a = await seed(engine, embedder, "tenant_a", "docs")
    _, _, key_b = await seed(engine, embedder, "tenant_b", "docs")
    llm = CountingLLM()
    async with api(engine, embedder, llm=llm, redis=redis_client, demo_tenant=None) as client:
        payload = {"collection": "docs", "question": "What is MVCC?"}
        ra = await client.post(
            "/v1/query",
            json=payload,
            headers={"Authorization": f"Bearer {key_a}"},
        )
        assert ra.status_code == 200
        assert ra.json()["cache_hit"] is False
        assert llm.calls == 1

        rb = await client.post(
            "/v1/query",
            json=payload,
            headers={"Authorization": f"Bearer {key_b}"},
        )
        assert rb.status_code == 200
        assert rb.json()["cache_hit"] is False
        assert llm.calls == 2


async def test_activation_bumps_index_version_and_invalidates_cache(
    engine: AsyncEngine,
    embedder: FakeEmbedder,
    redis_client: redis_asyncio.Redis,
) -> None:
    _, collection_id, api_key = await seed(engine, embedder)
    llm = CountingLLM()
    async with api(engine, embedder, llm=llm, redis=redis_client) as client:
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"collection": "pgdocs", "question": "What is MVCC?"}
        r1 = await client.post("/v1/query", json=payload, headers=headers)
        assert r1.status_code == 200
        assert r1.json()["cache_hit"] is False

        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE collections SET index_version = index_version + 1 WHERE id = :c"),
                {"c": collection_id},
            )

        r2 = await client.post("/v1/query", json=payload, headers=headers)
        assert r2.status_code == 200
        assert r2.json()["cache_hit"] is False
        assert llm.calls == 2


async def test_daily_budget_blocks_new_llm_calls_but_allows_cache_hits(
    engine: AsyncEngine,
    embedder: FakeEmbedder,
    redis_client: redis_asyncio.Redis,
) -> None:
    _, _, api_key = await seed(engine, embedder)
    llm = CountingLLM()
    async with api(
        engine,
        embedder,
        llm=llm,
        redis=redis_client,
        daily_budget_usd=0.01,
        llm_price_in_per_mtok=100_000.0,
        llm_price_out_per_mtok=100_000.0,
    ) as client:
        headers = {"Authorization": f"Bearer {api_key}"}
        r1 = await client.post(
            "/v1/query",
            json={"collection": "pgdocs", "question": "What is MVCC?"},
            headers=headers,
        )
        assert r1.status_code == 200
        assert r1.json()["cost_usd"] > 0.01

        r2 = await client.post(
            "/v1/query",
            json={"collection": "pgdocs", "question": "How does concurrency work?"},
            headers=headers,
        )
        assert r2.status_code == 429
        assert r2.json() == {"detail": "budget_exceeded"}
        assert int(r2.headers["Retry-After"]) >= 1

        r3 = await client.post(
            "/v1/query",
            json={"collection": "pgdocs", "question": "What is MVCC?"},
            headers=headers,
        )
        assert r3.status_code == 200
        assert r3.json()["cache_hit"] is True


async def test_query_rate_limit(
    engine: AsyncEngine,
    embedder: FakeEmbedder,
    redis_client: redis_asyncio.Redis,
) -> None:
    _, _, api_key = await seed(engine, embedder)
    async with api(
        engine,
        embedder,
        redis=redis_client,
        query_burst=2,
        query_rate_per_sec=0.001,
    ) as client:
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"collection": "pgdocs", "question": "What is MVCC?"}
        r1 = await client.post("/v1/query", json=payload, headers=headers)
        r2 = await client.post("/v1/query", json=payload, headers=headers)
        r3 = await client.post("/v1/query", json=payload, headers=headers)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 429
        assert r3.json() == {"detail": "rate limit"}
        assert int(r3.headers["Retry-After"]) >= 1
        assert "X-RateLimit-Limit" in r3.headers
        assert "X-RateLimit-Remaining" in r3.headers


async def test_redis_down_fails_open(
    engine: AsyncEngine,
    embedder: FakeEmbedder,
) -> None:
    _, _, api_key = await seed(engine, embedder)
    dead_redis: redis_asyncio.Redis = redis_asyncio.from_url("redis://127.0.0.1:1/0")
    try:
        async with api(engine, embedder, redis=dead_redis) as client:
            resp = await client.post(
                "/v1/query",
                json={"collection": "pgdocs", "question": "What is MVCC?"},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            assert resp.status_code == 200
            assert resp.json()["cache_hit"] is False

            ready = await client.get("/readyz")
            assert ready.status_code == 200
            assert ready.json() == {"status": "ok", "redis": "down"}
    finally:
        await dead_redis.aclose()


async def test_query_embedding_cache_skips_second_embed_call(
    engine: AsyncEngine,
    embedder: FakeEmbedder,
    redis_client: redis_asyncio.Redis,
) -> None:
    _, _, api_key = await seed(engine, embedder)
    embed_calls: list[int] = []
    models = models_http(embedder, embed_calls=embed_calls)
    async with api(
        engine,
        embedder,
        models=models,
        redis=redis_client,
        answer_cache_enabled=False,
    ) as client:
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"collection": "pgdocs", "question": "What is MVCC?"}
        r1 = await client.post("/v1/query", json=payload, headers=headers)
        r2 = await client.post("/v1/query", json=payload, headers=headers)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert len(embed_calls) == 1
