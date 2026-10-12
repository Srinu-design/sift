"""Dense search: cosine distance over the HNSW index, under the tenant and version filter.

pgvector applies WHERE filters after the index scan, so with the default ef_search of 40 a
filter that keeps 1 row in 3 (one docs version) returns short lists. Iterative scans
(pgvector 0.8.0+) keep scanning until enough rows pass, up to hnsw.max_scan_tuples.
"""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.db import vector_literal

# The index scan touches chunks alone; only the top k rows are joined for their document.
DENSE = """
SELECT top.*, d.source_uri
FROM (
  SELECT c.id, c.version_id, c.embedding <=> CAST(:q AS vector) AS distance,
         c.heading, c.anchor, c.anchor_path, c.page_start, c.page_end, c.doc_version
  FROM chunks c
  WHERE c.tenant_id = :tenant AND c.collection_id = :collection AND c.is_active
    AND (CAST(:version AS text) IS NULL OR c.doc_version = :version)
  ORDER BY distance
  LIMIT :k
) top
JOIN document_versions v ON v.id = top.version_id
JOIN documents d ON d.id = v.document_id
"""


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: int
    score: float  # higher is better: 1 - cosine distance
    heading: str
    anchor: str | None
    anchor_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    doc_version: str | None
    source_uri: str


async def dense(
    engine: AsyncEngine,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    query_vector: list[float],
    version: str | None = None,
    k: int = 50,
) -> list[Hit]:
    async with engine.begin() as conn:
        await conn.execute(text("SET LOCAL hnsw.ef_search = 100"))
        await conn.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        rows = (
            await conn.execute(
                text(DENSE),
                {
                    "q": vector_literal(query_vector),
                    "tenant": tenant_id,
                    "collection": collection_id,
                    "version": version,
                    "k": k,
                },
            )
        ).all()
    # relaxed_order can return rows slightly out of order: re-sort before ranking.
    rows = sorted(rows, key=lambda r: r.distance)
    return [
        Hit(
            r.id,
            1.0 - r.distance,
            r.heading,
            r.anchor,
            tuple(r.anchor_path),
            r.page_start,
            r.page_end,
            r.doc_version,
            r.source_uri,
        )
        for r in rows
    ]
