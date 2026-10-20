"""Integration tests for hybrid_rerank search strategy."""

from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.ingest import pipeline
from sift.retrieval.search import search
from tests.conftest import FakeEmbedder
from tests.integration.test_pipeline import ingest


async def _ingest_doc(
    engine: AsyncEngine,
    embedder: FakeEmbedder,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    idx: int,
    text_body: str,
) -> None:
    html = (
        '<html><body><div id="docContent"><div class="sect1" id="P">'
        f"<h2>Manual {idx}</h2>"
        f'<div class="sect2" id="SEC-{idx}"><h3>Section {idx}</h3><p>{text_body}</p></div>'
        "</div></div></body></html>"
    ).encode()
    sub = await pipeline.submit(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        filename=f"doc-{idx}.html",
        data=html,
        source_uri=f"https://docs.test/doc-{idx}.html",
        source="pgdocs",
        doc_version="17",
    )
    await ingest(engine, sub.version_id, embedder)


async def test_hybrid_rerank_orders_by_reranker_and_records_timing(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    tenant_id, collection_id = await pipeline.ensure_collection(engine, "acme", "docs")
    for idx in range(1, 5):
        await _ingest_doc(
            engine,
            embedder,
            tenant_id=tenant_id,
            collection_id=collection_id,
            idx=idx,
            text_body=f"PostgreSQL vacuum_cost_limit configuration parameter details {idx}.",
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
