"""Ingestion steps against a real pgvector Postgres: idempotency, caching and atomic swaps."""

import asyncio
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.ingest import parsers, pipeline
from sift.retrieval.dense import dense
from tests.conftest import FakeEmbedder
from tests.integration.helpers import ingest, page, scalar, setup, submit


async def rows(engine: AsyncEngine, sql: str) -> list[tuple[object, ...]]:
    async with engine.connect() as conn:
        return [tuple(r) for r in await conn.execute(text(sql))]


async def test_an_unchanged_file_creates_no_version_no_job_and_no_embedding(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    ids = await setup(engine)
    first = await submit(engine, ids, page())
    assert (first.status, first.needs_parse) == ("new", True)
    assert await ingest(engine, first.version_id, embedder) == "activated"
    calls = len(embedder.texts)

    again = await submit(engine, ids, page())

    assert (again.version_id, again.status, again.needs_parse) == (first.version_id, "ready", False)
    assert len(embedder.texts) == calls  # nothing was embedded a second time
    assert await scalar(engine, "SELECT count(*) FROM document_versions") == 1


async def test_a_changed_file_only_pays_for_the_chunks_that_changed(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    ids = await setup(engine)
    v1 = await submit(engine, ids, page(sections=4))
    await ingest(engine, v1.version_id, embedder)
    first_cost = len(embedder.texts)

    # One more section at the end: every earlier chunk's text is unchanged.
    v2 = await submit(engine, ids, page(sections=5))
    await ingest(engine, v2.version_id, embedder)

    new_texts = len(embedder.texts) - first_cost
    total_v2 = await scalar(
        engine, "SELECT count(*) FROM chunks WHERE version_id = :v", v=v2.version_id
    )
    assert 0 < new_texts < total_v2  # the rest came from embedding_cache


async def test_queries_see_the_old_version_until_activation_then_only_the_new(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    ids = await setup(engine)
    v1 = await submit(engine, ids, page())
    await ingest(engine, v1.version_id, embedder)
    v2 = await submit(engine, ids, page(extra="<p>Brand new paragraph about freezing.</p>"))
    for b in range(await pipeline.parse_version(engine, v2.version_id)):
        await pipeline.embed_batch(engine, v2.version_id, b, embedder)

    async def live_versions() -> set[UUID]:
        [vector] = await embedder(["anything"])
        hits = await dense(
            engine, tenant_id=ids[0], collection_id=ids[1], query_vector=vector, k=100
        )
        ids_ = {h.chunk_id for h in hits}
        async with engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT DISTINCT version_id FROM chunks WHERE id = ANY(:ids)"),
                {"ids": list(ids_)},
            )
            return {r[0] for r in rows}

    index_before = await scalar(engine, "SELECT index_version FROM collections")
    assert await live_versions() == {v1.version_id}  # v2 is fully embedded but not active

    assert await pipeline.activate(engine, v2.version_id) == "activated"

    assert await live_versions() == {v2.version_id}
    statuses = dict(await rows(engine, "SELECT id, status FROM document_versions"))
    assert statuses == {v1.version_id: "superseded", v2.version_id: "ready"}
    assert await scalar(engine, "SELECT index_version FROM collections") == index_before + 1
    v1_chunks = await scalar(
        engine, "SELECT count(*) FROM chunks WHERE version_id = :v", v=v1.version_id
    )
    assert v1_chunks == 0  # a superseded version's chunks are deleted


async def test_an_older_version_finishing_late_never_replaces_a_newer_one(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    ids = await setup(engine)
    old = await submit(engine, ids, page(sections=2))
    new = await submit(engine, ids, page(sections=3))
    for v in (old, new):
        for b in range(await pipeline.parse_version(engine, v.version_id)):
            await pipeline.embed_batch(engine, v.version_id, b, embedder)

    assert await pipeline.activate(engine, new.version_id) == "activated"
    assert await pipeline.activate(engine, old.version_id) == "superseded"
    active = await scalar(engine, "SELECT active_version_id FROM documents")
    assert active == new.version_id
    old_chunks = await scalar(
        engine, "SELECT count(*) FROM chunks WHERE version_id = :v", v=old.version_id
    )
    assert old_chunks == 0  # the late version never goes live, and its chunks are deleted


async def test_every_step_repeats_safely(engine: AsyncEngine, embedder: FakeEmbedder) -> None:
    """At-least-once delivery: each step may run twice, even at the same time."""
    ids = await setup(engine)
    v = (await submit(engine, ids, page(sections=70, paragraphs=1))).version_id

    batches = await asyncio.gather(
        pipeline.parse_version(engine, v), pipeline.parse_version(engine, v)
    )
    assert batches[0] == batches[1] > 1
    await asyncio.gather(*(pipeline.embed_batch(engine, v, 0, embedder) for _ in range(2)))
    for b in range(batches[0]):
        await pipeline.embed_batch(engine, v, b, embedder)
    paid = len(embedder.texts)
    for b in range(batches[0]):  # a replayed batch finds its vectors and calls nothing
        assert await pipeline.embed_batch(engine, v, b, embedder)
    assert len(embedder.texts) == paid

    assert [await pipeline.activate(engine, v) for _ in range(2)] == ["activated", "already"]
    [(count, distinct, n)] = await rows(
        engine,
        "SELECT count(*), count(DISTINCT ordinal), (SELECT chunk_count FROM document_versions)"
        " FROM chunks WHERE embedding IS NOT NULL AND is_active",
    )
    assert count == distinct == n


async def test_activation_waits_for_every_embedding(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    ids = await setup(engine)
    v = (await submit(engine, ids, page(sections=70, paragraphs=1))).version_id
    assert await pipeline.parse_version(engine, v) > 1
    await pipeline.embed_batch(engine, v, 0, embedder)
    with pytest.raises(pipeline.NotReady):
        await pipeline.activate(engine, v)


async def test_a_job_for_a_deleted_version_is_permanent(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    missing = UUID(int=1)
    for step in (
        pipeline.parse_version(engine, missing),
        pipeline.embed_batch(engine, missing, 0, embedder),
        pipeline.activate(engine, missing),
    ):
        with pytest.raises(LookupError):
            await step


async def test_a_parse_failure_marks_the_version_and_a_retry_after_the_fix_succeeds(
    engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a DLQ replay does after a parser fix: the same job, run again."""
    ids = await setup(engine)
    v = (await submit(engine, ids, page())).version_id

    def broken(parser: str, data: bytes) -> parsers.ParsedDoc:
        raise parsers.ParseError("parser bug")

    monkeypatch.setattr(parsers, "parse", broken)
    with pytest.raises(parsers.ParseError):
        await pipeline.parse_version(engine, v)
    assert await scalar(engine, "SELECT status FROM document_versions") == "failed"
    assert await scalar(engine, "SELECT count(*) FROM chunks") == 0

    monkeypatch.undo()  # the fix ships
    assert await pipeline.parse_version(engine, v) >= 1
    assert await scalar(engine, "SELECT status FROM document_versions") == "embedding"
