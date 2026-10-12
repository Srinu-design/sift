"""Ingestion through the real stack: Hopper's workers calling sift-ingest and sift-models.

    make up && make hopper-tenant && make test-e2e

Each test uses its own tenant and unique text, so the embedding cache cannot hide the work
and the dev data is untouched. Both tests clean up after themselves.
"""

import asyncio
import hashlib
import os
import subprocess
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from sift.config import get_settings
from sift.hopper import enqueue_step
from sift.ingest import pipeline

ROOT = Path(__file__).resolve().parent.parent.parent
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.environ.get("SIFT_E2E") != "1", reason="needs the stack: make test-e2e"),
]


def hopper_db_url() -> str:
    return os.environ["DATABASE_URL"].rsplit("/", 1)[0] + "/hopper"


@pytest.fixture
async def live() -> AsyncIterator[tuple[AsyncEngine, UUID, UUID]]:
    """The dev database (sift-ingest's), the real file store, and a throwaway tenant."""
    if os.environ.get("HOPPER_API_KEY", "").startswith("hop_test"):
        pytest.skip("run through make test-e2e, which exports .env")
    os.environ["FILE_STORE"] = str(ROOT / "data/files")  # bind-mounted into sift-ingest
    get_settings.cache_clear()
    engine = create_async_engine(os.environ["DATABASE_URL"])
    tenant = f"e2e-{uuid.uuid4().hex[:8]}"
    tenant_id, collection_id = await pipeline.ensure_collection(engine, tenant, "docs")
    try:
        yield engine, tenant_id, collection_id
    finally:
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM documents WHERE tenant_id = :t"), {"t": tenant_id})
            await conn.execute(
                text("DELETE FROM collections WHERE tenant_id = :t"), {"t": tenant_id}
            )
            await conn.execute(text("DELETE FROM tenants WHERE id = :t"), {"t": tenant_id})
        await engine.dispose()
        get_settings.cache_clear()


def unique_page(n_sections: int) -> bytes:
    """~300 tokens a section, like the real corpus, so one 64-chunk batch takes seconds to
    embed and a kill lands inside a job, not between two."""
    nonce = uuid.uuid4().hex  # new text every run: every chunk misses the embedding cache
    sections = "".join(
        f'<div class="sect2" id="S{i}"><h3>Topic {i}</h3><p>Note {nonce} {i}: '
        + " ".join(
            f"Setting tune_{i}_{j} controls how often checkpoint {i} flushes {j} dirty buffers."
            for j in range(22)
        )
        + "</p></div>"
        for i in range(n_sections)
    )
    page = f'<div id="docContent"><div class="sect1" id="P"><h2>Page</h2>{sections}</div></div>'
    return f"<html><body>{page}</body></html>".encode()


async def submit_all(
    engine: AsyncEngine, tenant_id: UUID, collection_id: UUID, pages: list[bytes]
) -> list[UUID]:
    ids = []
    async with httpx.AsyncClient(timeout=30) as http:
        for n, data in enumerate(pages):
            s = await pipeline.submit(
                engine,
                tenant_id=tenant_id,
                collection_id=collection_id,
                filename=f"page-{n}.html",
                data=data,
                source_uri=f"https://e2e.test/page-{n}.html",
                source="e2e",
            )
            await enqueue_step(
                http, "parse", {"version_id": str(s.version_id)}, f"parse:{s.version_id}"
            )
            ids.append(s.version_id)
    return ids


async def wait_for(check: Callable[[], Awaitable[bool]], timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await check():
            return
        await asyncio.sleep(1)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


async def test_killing_a_worker_mid_embed_loses_and_duplicates_nothing(
    live: tuple[AsyncEngine, UUID, UUID],
) -> None:
    engine, tenant_id, collection_id = live
    versions = await submit_all(
        engine, tenant_id, collection_id, [unique_page(130) for _ in range(4)]
    )
    hopper = create_async_engine(hopper_db_url())

    async def embedding_started() -> bool:
        async with engine.connect() as conn:
            n = await conn.scalar(
                text("SELECT count(embedding) FROM chunks WHERE collection_id = :c"),
                {"c": collection_id},
            )
        return bool(n)

    async def running_embed_worker() -> str | None:
        async with hopper.connect() as conn:
            return await conn.scalar(
                text(
                    "SELECT lease_owner FROM jobs WHERE status = 'running' AND queue = 'sift-embed'"
                    " AND payload->'body'->>'version_id' = ANY(:v) LIMIT 1"
                ),
                {"v": [str(v) for v in versions]},
            )

    await wait_for(embedding_started, 120, "the first embeddings")
    owner = None
    deadline = time.monotonic() + 60
    while owner is None and time.monotonic() < deadline:
        owner = await running_embed_worker()
        await asyncio.sleep(0.2)
    assert owner, "no embed job was running to kill"
    container = owner.split("-")[0]  # worker ids start with the container's hostname
    await asyncio.to_thread(subprocess.run, ["docker", "kill", container], check=True)
    # docker kill counts as a manual stop, so no restart policy brings it back: start it again
    # as a supervisor would. Its job is not on it any more: the lease must expire first.
    await asyncio.sleep(2)
    await asyncio.to_thread(subprocess.run, ["docker", "start", container], check=True)

    async def all_ready() -> bool:
        async with engine.connect() as conn:
            left = await conn.scalar(
                text(
                    "SELECT count(*) FROM document_versions"
                    " WHERE id = ANY(:v) AND status <> 'ready'"
                ),
                {"v": versions},
            )
        return left == 0

    # The killed job's lease (30 s) expires, the reaper requeues it, the other worker or the
    # restarted one runs it again.
    await wait_for(all_ready, 300, "every version to be ready")

    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    """
                    SELECT v.chunk_count, count(c.id) AS n, count(DISTINCT c.ordinal) AS distinct_n,
                           max(c.ordinal) AS top, count(c.embedding) AS embedded,
                           bool_and(c.is_active) AS active
                    FROM document_versions v JOIN chunks c ON c.version_id = v.id
                    WHERE v.id = ANY(:v) GROUP BY v.id, v.chunk_count
                    """
                ),
                {"v": versions},
            )
        ).all()
    assert len(rows) == len(versions)
    for r in rows:
        assert (
            r.n == r.distinct_n == r.chunk_count == r.top + 1 == r.embedded
        )  # nothing lost or doubled
        assert r.active
    async with hopper.connect() as conn:
        reruns = await conn.scalar(
            text(
                "SELECT count(*) FROM jobs WHERE queue = 'sift-embed' AND attempts > 1"
                " AND payload->'body'->>'version_id' = ANY(:v)"
            ),
            {"v": [str(v) for v in versions]},
        )
    await hopper.dispose()
    assert reruns >= 1, "the kill did not interrupt a job, so this run proved nothing"


async def test_a_dead_parse_job_replays_once_the_cause_is_fixed(
    live: tuple[AsyncEngine, UUID, UUID],
) -> None:
    """The raw file is missing when parse runs: 422, straight to Hopper's DLQ, version failed.
    Put the file back, replay the job through Hopper's DLQ API, and it completes."""
    engine, tenant_id, collection_id = live
    data = unique_page(5)
    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 5))
    s = await pipeline.submit(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        filename="lost.html",
        data=data,
        source_uri="https://e2e.test/lost.html",
        source="e2e",
    )
    v = s.version_id
    stored = pipeline.file_path(get_settings().file_store, tenant_id, hashlib.sha256(data).digest())
    hidden = stored.with_suffix(".hidden")
    stored.rename(hidden)  # the cause: the file store lost the upload
    settings = get_settings()
    api = {"Authorization": f"Bearer {settings.hopper_api_key.get_secret_value()}"}

    async def status() -> str | None:
        async with engine.connect() as conn:
            return await conn.scalar(
                text("SELECT status FROM document_versions WHERE id = :v"), {"v": v}
            )

    async with httpx.AsyncClient(base_url=settings.hopper_url, headers=api, timeout=30) as hopper:
        await enqueue_step(hopper, "parse", {"version_id": str(v)}, f"parse:{v}")

        async def dead_job() -> str | None:
            page = (await hopper.get("/v1/dlq", params={"since": since, "limit": 200})).json()
            return next(
                (j["id"] for j in page["jobs"] if j["payload"]["body"].get("version_id") == str(v)),
                None,
            )

        async def is_dead() -> bool:
            return await dead_job() is not None

        await wait_for(is_dead, 60, "the parse job to reach the DLQ")
        job_id = await dead_job()
        assert await status() == "failed"

        hidden.rename(stored)  # the fix
        replay = await hopper.post("/v1/dlq/replay", json={"ids": [job_id], "spread_seconds": 0})
        assert replay.json()["replayed"] == 1

        async def ready() -> bool:
            return await status() == "ready"

        await wait_for(ready, 120, "the replayed version to be ready")
        job = (await hopper.get(f"/v1/jobs/{job_id}")).json()
    assert (job["status"], job["replay_count"]) == ("succeeded", 1)
