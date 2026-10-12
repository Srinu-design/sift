"""Ingestion steps: submit, parse, embed a batch, activate. Each one is idempotent.

Hopper runs them as jobs (sift.ingest.service) and the local script runs them in a loop
(sift.ingest.corpus --via local). A crash, retry or DLQ replay at any point repeats nothing
that already happened: chunks insert ON CONFLICT DO NOTHING, vectors come from the cache
when they exist, and activation is a guarded UPDATE.
"""

import asyncio
import hashlib
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from sift.config import get_settings
from sift.ingest import chunker, parsers

EMBED_BATCH = 64
Embed = Callable[[list[str]], Awaitable[list[list[float]]]]


class NotReady(Exception):
    """A step ran before the one it depends on finished. Retryable (HTTP 503)."""


def embed_model_key() -> str:
    """Model name plus commit: a new model revision must never reuse cached vectors."""
    s = get_settings()
    return f"{s.embed_model}@{s.embed_revision[:12]}"


def file_path(store: str, tenant_id: UUID, content_hash: bytes) -> Path:
    # Tenant first, so one tenant's files can never be reached through another's path.
    return Path(store) / str(tenant_id) / content_hash.hex()


async def ensure_collection(engine: AsyncEngine, tenant: str, collection: str) -> tuple[UUID, UUID]:
    """(tenant id, collection id), created if missing. Dev and eval tenants only, until the
    admin API arrives with tenancy (week 5)."""
    async with engine.begin() as conn:
        tenant_id = await conn.scalar(
            text(
                "INSERT INTO tenants (name) VALUES (:n) ON CONFLICT (name)"
                " DO UPDATE SET name = EXCLUDED.name RETURNING id"
            ),
            {"n": tenant},
        )
        collection_id = await conn.scalar(
            text(
                "INSERT INTO collections (tenant_id, name) VALUES (:t, :n)"
                " ON CONFLICT (tenant_id, name) DO UPDATE SET name = EXCLUDED.name RETURNING id"
            ),
            {"t": tenant_id, "n": collection},
        )
    assert tenant_id is not None and collection_id is not None
    return tenant_id, collection_id


@dataclass(frozen=True, slots=True)
class Submitted:
    version_id: UUID
    # new | pending | parsing | embedding: needs (or has) a parse job.
    # ready: unchanged, nothing to do. failed: replay the parse job from Hopper's DLQ.
    # superseded: this exact file was active before and has since been replaced.
    status: str

    @property
    def needs_parse(self) -> bool:
        return self.status in ("new", "pending", "parsing", "embedding")


UPSERT_DOCUMENT = """
INSERT INTO documents (tenant_id, collection_id, source_uri, title, source, doc_version,
                       published_on)
VALUES (:tenant_id, :collection_id, :source_uri, :title, :source, :doc_version, :published_on)
ON CONFLICT (collection_id, source_uri, doc_version)
DO UPDATE SET source = EXCLUDED.source, published_on = EXCLUDED.published_on
RETURNING id
"""

INSERT_VERSION = """
INSERT INTO document_versions (document_id, content_hash, parser, chunker)
VALUES (:document_id, :content_hash, :parser, :chunker)
ON CONFLICT (document_id, content_hash, parser, chunker) DO NOTHING
RETURNING id
"""


async def submit(
    engine: AsyncEngine,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    filename: str,
    data: bytes,
    source_uri: str,
    source: str,
    doc_version: str | None = None,
    published_on: date | None = None,
) -> Submitted:
    """Store the file under its SHA-256 and record a version. An unchanged file finds its
    existing version, so the caller enqueues nothing and no embedding is paid for twice."""
    parser = parsers.parser_for(filename)
    content_hash = hashlib.sha256(data).digest()
    path = file_path(get_settings().file_store, tenant_id, content_hash)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic: a reader never sees half a file
    async with engine.begin() as conn:
        document_id = await conn.scalar(
            text(UPSERT_DOCUMENT),
            {
                "tenant_id": tenant_id,
                "collection_id": collection_id,
                "source_uri": source_uri,
                "title": Path(filename).stem,  # replaced by the parsed title
                "source": source,
                "doc_version": doc_version,
                "published_on": published_on,
            },
        )
        version_id = await conn.scalar(
            text(INSERT_VERSION),
            {
                "document_id": document_id,
                "content_hash": content_hash,
                "parser": parser,
                "chunker": chunker.NAME,
            },
        )
        if version_id is not None:
            return Submitted(version_id, "new")
        row = (
            await conn.execute(
                text(
                    "SELECT id, status FROM document_versions"
                    " WHERE document_id = :d AND content_hash = :h AND parser = :p AND chunker = :c"
                ),
                {"d": document_id, "h": content_hash, "p": parser, "c": chunker.NAME},
            )
        ).one()
    # shortcut: re-uploading a superseded file does not re-activate it; add that (a fresh
    # job key per activation) when the 24 h retention of superseded chunks lands in week 6.
    return Submitted(row.id, row.status)


async def _version(conn: AsyncConnection, version_id: UUID, lock: bool = False) -> Any:
    row = (
        await conn.execute(
            text(
                """
                SELECT v.id, v.status, v.parser, v.content_hash, v.chunk_count, v.created_at,
                       d.id AS document_id, d.tenant_id, d.collection_id, d.source,
                       d.doc_version, d.published_on, d.active_version_id
                FROM document_versions v JOIN documents d ON d.id = v.document_id
                WHERE v.id = :v
                """
                + (" FOR UPDATE OF d" if lock else "")
            ),
            {"v": version_id},
        )
    ).first()
    if row is None:
        raise LookupError(f"no version {version_id}")
    return row


def batches(chunk_count: int) -> int:
    return -(-chunk_count // EMBED_BATCH)


INSERT_CHUNK = """
INSERT INTO chunks (tenant_id, collection_id, version_id, ordinal, heading, anchor, anchor_path,
                    page_start, page_end, content, token_count, content_hash,
                    source, doc_version, published_on)
VALUES (:tenant_id, :collection_id, :version_id, :ordinal, :heading, :anchor, :anchor_path,
        :page_start, :page_end, :content, :token_count, :content_hash,
        :source, :doc_version, :published_on)
ON CONFLICT (version_id, ordinal) DO NOTHING
"""


async def parse_version(engine: AsyncEngine, version_id: UUID) -> int:
    """Parse and chunk the file; return how many embed batches the version has.

    Raises parsers.ParseError (permanent) after marking the version failed. A version that
    is already parsed returns its batch count again, so a retry re-enqueues the same
    (idempotent) embed jobs; a version that is ready or superseded returns 0.
    """
    async with engine.begin() as conn:
        v = await _version(conn, version_id)
        if v.status in ("ready", "superseded"):
            return 0
        if v.status == "embedding":
            return batches(v.chunk_count)
        await conn.execute(
            text("UPDATE document_versions SET status = 'parsing', error = NULL WHERE id = :v"),
            {"v": version_id},
        )
    path = file_path(get_settings().file_store, v.tenant_id, v.content_hash)
    try:
        data = await asyncio.to_thread(path.read_bytes)
        doc = await asyncio.to_thread(parsers.parse, v.parser, data)
        chunks = await asyncio.to_thread(chunker.chunk, doc.blocks)
    except (parsers.ParseError, FileNotFoundError) as exc:
        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE document_versions SET status = 'failed', error = :e WHERE id = :v"),
                {"v": version_id, "e": str(exc)[:1000]},
            )
        raise parsers.ParseError(str(exc)) from exc

    rows = [
        {
            "tenant_id": v.tenant_id,
            "collection_id": v.collection_id,
            "version_id": version_id,
            "ordinal": c.ordinal,
            "heading": c.heading,
            "anchor": c.anchor,
            "anchor_path": list(c.anchor_path),
            "page_start": c.page_start,
            "page_end": c.page_end,
            "content": c.content,
            "token_count": c.token_count,
            "content_hash": c.content_hash,
            "source": v.source,
            "doc_version": v.doc_version,
            "published_on": v.published_on,
        }
        for c in chunks
    ]
    # One transaction: the chunks and the 'embedding' status land together, before any
    # embed job exists, so activation (guarded on 'embedding') can never run too early.
    async with engine.begin() as conn:
        if rows:
            await conn.execute(text(INSERT_CHUNK), rows)
        await conn.execute(
            text(
                "UPDATE document_versions SET status = 'embedding', chunk_count = :n"
                " WHERE id = :v AND status = 'parsing'"
            ),
            {"v": version_id, "n": len(chunks)},
        )
        if doc.title:
            await conn.execute(
                text("UPDATE documents SET title = :t WHERE id = :d"),
                {"t": doc.title, "d": v.document_id},
            )
    return batches(len(chunks))


async def embed_batch(engine: AsyncEngine, version_id: UUID, batch: int, embed: Embed) -> bool:
    """Fill this batch's missing vectors, from the cache first and the model for the rest.
    True when the whole version has no missing vector left (time to activate)."""
    async with engine.connect() as conn:
        if (
            await conn.scalar(
                text("SELECT 1 FROM document_versions WHERE id = :v"), {"v": version_id}
            )
            is None
        ):
            # Without this, a stale job for a deleted version finds "no vector missing" and
            # would report the version complete.
            raise LookupError(f"no version {version_id}")
    model = embed_model_key()
    params = {
        "v": version_id,
        "lo": batch * EMBED_BATCH,
        "hi": (batch + 1) * EMBED_BATCH,
        "m": model,
    }
    in_batch = "c.version_id = :v AND c.ordinal >= :lo AND c.ordinal < :hi AND c.embedding IS NULL"
    async with engine.begin() as conn:
        # Cache hits are copied inside Postgres; the vectors never travel to Python and back.
        await conn.execute(
            text(
                f"""
                UPDATE chunks c SET embedding = ec.embedding, embed_model = ec.model
                FROM embedding_cache ec
                WHERE {in_batch} AND ec.model = :m AND ec.content_hash = c.content_hash
                """
            ),
            params,
        )
        misses = (
            await conn.execute(
                text(f"SELECT c.heading, c.content, c.content_hash FROM chunks c WHERE {in_batch}"),
                params,
            )
        ).all()
    if misses:
        unique = {m.content_hash: chunker.embed_text(m.heading, m.content) for m in misses}
        vectors = await embed(list(unique.values()))
        from sift.db import vector_literal

        cache_rows = [
            {"m": model, "h": h, "e": vector_literal(vec)}
            for h, vec in zip(unique, vectors, strict=True)
        ]
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO embedding_cache (model, content_hash, embedding)"
                    " VALUES (:m, :h, CAST(:e AS vector)) ON CONFLICT DO NOTHING"
                ),
                cache_rows,
            )
            await conn.execute(
                text(
                    f"""
                    UPDATE chunks c SET embedding = ec.embedding, embed_model = ec.model
                    FROM embedding_cache ec
                    WHERE {in_batch} AND ec.model = :m AND ec.content_hash = c.content_hash
                    """
                ),
                params,
            )
    async with engine.connect() as conn:
        left = await conn.scalar(
            text(
                "SELECT EXISTS (SELECT 1 FROM chunks WHERE version_id = :v AND embedding IS NULL)"
            ),
            {"v": version_id},
        )
    return not left


async def activate(engine: AsyncEngine, version_id: UUID) -> str:
    """Make this version the one queries see, in one transaction. Returns what happened:
    'activated', 'already' (a repeat), or 'superseded' (a newer version is already live)."""
    async with engine.begin() as conn:
        v = await _version(conn, version_id, lock=True)  # serialises activations per document
        if v.status != "embedding":
            return "already"
        missing = await conn.scalar(
            text("SELECT count(*) FROM chunks WHERE version_id = :v AND embedding IS NULL"),
            {"v": version_id},
        )
        if missing:
            raise NotReady(f"{missing} chunks still have no embedding")
        old = v.active_version_id
        if old is not None:
            old_created = await conn.scalar(
                text("SELECT created_at FROM document_versions WHERE id = :o"), {"o": old}
            )
            if old_created is not None and old_created > v.created_at:
                # An older upload finished after a newer one: never roll the document back.
                await conn.execute(
                    text("UPDATE document_versions SET status = 'superseded' WHERE id = :v"),
                    {"v": version_id},
                )
                return "superseded"
        await conn.execute(
            text(
                "UPDATE document_versions SET status = 'ready'"
                " WHERE id = :v AND status = 'embedding'"
            ),
            {"v": version_id},
        )
        await conn.execute(
            text(
                "UPDATE chunks SET is_active = (version_id = :v)"
                " WHERE version_id = :v OR version_id = :o"
            ),
            {"v": version_id, "o": old},
        )
        if old is not None:
            await conn.execute(
                text("UPDATE document_versions SET status = 'superseded' WHERE id = :o"),
                {"o": old},
            )
        await conn.execute(
            text("UPDATE documents SET active_version_id = :v WHERE id = :d"),
            {"v": version_id, "d": v.document_id},
        )
        # Every cached answer keyed on the old index_version becomes unreachable (week 6).
        await conn.execute(
            text("UPDATE collections SET index_version = index_version + 1 WHERE id = :c"),
            {"c": v.collection_id},
        )
    return "activated"
