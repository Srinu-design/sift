"""Integration tests for hybrid_rerank search strategy."""

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.ingest import pipeline
from sift.retrieval.search import search
from tests.conftest import FakeEmbedder
from tests.integration.helpers import add_doc


async def test_hybrid_rerank_orders_by_reranker_and_records_timing(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    tenant_id, collection_id = await pipeline.ensure_collection(engine, "acme", "docs")
    for idx in range(1, 5):
        await add_doc(
            engine,
            embedder,
            tenant_id=tenant_id,
            collection_id=collection_id,
            title=f"Manual {idx}",
            heading=f"Section {idx}",
            body_text=f"PostgreSQL vacuum_cost_limit configuration parameter details {idx}.",
            source_uri=f"https://docs.test/doc-{idx}.html",
        )

    [q_vec] = await embedder(["vacuum_cost_limit configuration"])

    async def fake_reranker(question: str, passages: list[str]) -> list[float]:
        # Assign higher score to higher section number so order is deterministic: 4, 3, 2, 1
        scores: list[float] = []
        for p in passages:
            for idx in range(1, 5):
                if f"Section {idx}" in p:
                    scores.append(idx * 0.2)
                    break
            else:
                scores.append(0.0)
        return scores

    res = await search(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        strategy="hybrid_rerank",
        question="vacuum_cost_limit configuration",
        query_vector=q_vec,
        k=2,
        reranker=fake_reranker,
    )
    assert len(res.hits) == 2
    assert [h.source_uri for h in res.hits] == [
        "https://docs.test/doc-4.html",
        "https://docs.test/doc-3.html",
    ]
    assert [h.score for h in res.hits] == [pytest.approx(0.8), pytest.approx(0.6)]
    assert set(res.timings_ms) == {"dense", "lexical", "fuse", "rerank"}

    with pytest.raises(ValueError, match="reranker"):
        await search(
            engine,
            tenant_id=tenant_id,
            collection_id=collection_id,
            strategy="hybrid_rerank",
            question="vacuum_cost_limit configuration",
            query_vector=q_vec,
            k=2,
            reranker=None,
        )
