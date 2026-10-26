"""Integration tests for lexical, dense, and hybrid retrieval with metadata filters."""

from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.ingest import pipeline
from sift.retrieval.dense import dense
from sift.retrieval.filters import Filters
from sift.retrieval.lexical import lexical
from sift.retrieval.search import STRATEGIES, search
from tests.conftest import FakeEmbedder
from tests.integration.helpers import add_doc


async def fake_reranker(question: str, passages: list[str]) -> list[float]:
    return [1.0 / (i + 1) for i in range(len(passages))]  # keeps the fused order


async def test_lexical_identifier_punctuation_stopwords_and_modes(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    tenant_id, collection_id = await pipeline.ensure_collection(engine, "acme", "docs")
    await add_doc(
        engine,
        embedder,
        tenant_id=tenant_id,
        collection_id=collection_id,
        body_text="Raise max_connections in postgresql.conf when connection slots fill up.",
        source_uri="https://docs.test/connections.html",
    )
    await add_doc(
        engine,
        embedder,
        tenant_id=tenant_id,
        collection_id=collection_id,
        body_text="Tune work_mem and enable_seqscan for the query planner.",
        source_uri="https://docs.test/planner.html",
    )
    await add_doc(
        engine,
        embedder,
        tenant_id=tenant_id,
        collection_id=collection_id,
        body_text="Both max_connections and work_mem affect total backend memory usage.",
        source_uri="https://docs.test/memory.html",
    )

    hits = await lexical(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        question="How do I raise max_connections?",
    )
    assert any("max_connections" in h.content for h in hits)

    punct_hits = await lexical(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        question="What's the planner's default?!",
    )
    assert len(punct_hits) >= 1

    stop_hits = await lexical(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        question="what is the of and",
    )
    assert stop_hits == []

    or_hits = await lexical(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        question="max_connections work_mem",
        mode="or",
    )
    and_hits = await lexical(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        question="max_connections work_mem",
        mode="and",
    )
    or_uris = {h.source_uri for h in or_hits}
    and_uris = {h.source_uri for h in and_hits}
    assert and_uris < or_uris
    assert "https://docs.test/connections.html" in or_uris
    assert "https://docs.test/connections.html" not in and_uris
    assert and_uris == {"https://docs.test/memory.html"}


async def test_filters_across_all_strategies(engine: AsyncEngine, embedder: FakeEmbedder) -> None:
    tenant_id, collection_id = await pipeline.ensure_collection(engine, "acme", "docs")
    await add_doc(
        engine,
        embedder,
        tenant_id=tenant_id,
        collection_id=collection_id,
        body_text="Configure wal_level for replication and archiving in version 16.",
        source_uri="https://docs.test/v16.html",
        source="pgdocs",
        doc_version="16",
        published_on=date(2023, 9, 14),
    )
    await add_doc(
        engine,
        embedder,
        tenant_id=tenant_id,
        collection_id=collection_id,
        body_text="Configure wal_level for logical replication in version 17.",
        source_uri="https://docs.test/v17.html",
        source="pgdocs",
        doc_version="17",
        published_on=date(2024, 9, 26),
    )
    await add_doc(
        engine,
        embedder,
        tenant_id=tenant_id,
        collection_id=collection_id,
        body_text="Configure wal_level notes on the community wiki for version 18.",
        source_uri="https://docs.test/v18.html",
        source="wiki",
        doc_version="18",
        published_on=date(2025, 9, 25),
    )

    q_vec = (await embedder(["wal_level replication"]))[0]
    cases = [
        (Filters(version="16"), {"https://docs.test/v16.html"}),
        (Filters(sources=("wiki",)), {"https://docs.test/v18.html"}),
        (
            Filters(
                published_after=date(2024, 1, 1),
                published_before=date(2024, 12, 31),
            ),
            {"https://docs.test/v17.html"},
        ),
    ]
    for strategy in STRATEGIES:
        for flt, expected_uris in cases:
            res = await search(
                engine,
                tenant_id=tenant_id,
                collection_id=collection_id,
                strategy=strategy,
                question="wal_level replication",
                query_vector=q_vec,
                filters=flt,
                reranker=fake_reranker,
            )
            assert {h.source_uri for h in res.hits} == expected_uris


async def test_tenant_isolation_all_strategies(engine: AsyncEngine, embedder: FakeEmbedder) -> None:
    t1, c1 = await pipeline.ensure_collection(engine, "tenant-one", "docs")
    t2, c2 = await pipeline.ensure_collection(engine, "tenant-two", "docs")
    for tid, cid in ((t1, c1), (t2, c2)):
        await add_doc(
            engine,
            embedder,
            tenant_id=tid,
            collection_id=cid,
            body_text="Identical checkpoint_completion_target documentation text.",
            source_uri="https://docs.test/checkpoint.html",
        )

    q_vec = (await embedder(["checkpoint_completion_target"]))[0]
    t2_dense = await dense(engine, tenant_id=t2, collection_id=c2, query_vector=q_vec)
    t2_ids = {h.chunk_id for h in t2_dense}
    assert len(t2_ids) == 1

    for strategy in STRATEGIES:
        res = await search(
            engine,
            tenant_id=t1,
            collection_id=c1,
            strategy=strategy,
            question="checkpoint_completion_target",
            query_vector=q_vec,
            reranker=fake_reranker,
        )
        t1_ids = {h.chunk_id for h in res.hits}
        assert len(t1_ids) == 1
        assert t1_ids.isdisjoint(t2_ids)


async def test_dense_exact_matches_hnsw_and_hybrid_fields(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    tenant_id, collection_id = await pipeline.ensure_collection(engine, "acme", "docs")
    for idx in range(4):
        await add_doc(
            engine,
            embedder,
            tenant_id=tenant_id,
            collection_id=collection_id,
            title=f"Title {idx}",
            heading=f"Heading {idx}",
            body_text=f"shared_buffers configuration example {idx} for PostgreSQL.",
            source_uri=f"https://docs.test/doc-{idx}.html",
            source="pgdocs",
            doc_version="17",
        )

    q_vec = (await embedder(["shared_buffers configuration"]))[0]
    hnsw_hits = await dense(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        query_vector=q_vec,
        k=50,
        scan="hnsw",
    )
    exact_hits = await dense(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        query_vector=q_vec,
        k=50,
        scan="exact",
    )
    assert {h.chunk_id for h in hnsw_hits} == {h.chunk_id for h in exact_hits}

    res = await search(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        strategy="hybrid",
        question="shared_buffers configuration",
        query_vector=q_vec,
        k=2,
    )
    assert len(res.hits) == 2
    assert res.hits[0].score >= res.hits[1].score > 0.0
    assert set(res.timings_ms) == {"dense", "lexical", "fuse"}
    for h in res.hits:
        assert h.chunk_id > 0
        assert h.heading.startswith(f"{h.title} > Heading ")
        assert h.anchor == "SEC-1"
        assert len(h.anchor_path) >= 1
        assert h.doc_version == "17"
        assert h.source == "pgdocs"
        assert h.source_uri.startswith("https://docs.test/doc-")
        assert h.title.startswith("Title ")
        assert "shared_buffers" in h.content


async def test_vector_strategies_need_a_vector(engine: AsyncEngine) -> None:
    tenant_id, collection_id = await pipeline.ensure_collection(engine, "acme", "docs")
    for strategy in ("naive", "hybrid"):
        with pytest.raises(ValueError, match="query_vector"):
            await search(
                engine,
                tenant_id=tenant_id,
                collection_id=collection_id,
                strategy=strategy,
                question="anything",
                query_vector=None,
            )
