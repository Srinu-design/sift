"""Dense search: cosine distance over the HNSW index, under tenant and metadata filters.

pgvector applies WHERE filters after the index scan, so with the default ef_search of 40 a
filter that keeps 1 row in 3 (one docs version) returns short lists. Iterative scans
(pgvector 0.8.0+) keep scanning until enough rows pass, up to hnsw.max_scan_tuples.
"""

from typing import Literal
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.db import set_tenant, vector_literal
from sift.retrieval.filters import FILTER_SQL, Filters
from sift.retrieval.hit import Hit

# The index scan touches chunks alone; only the top k rows are joined for their document.
# The tenant scope (as set_tenant) and both HNSW settings in one round trip, transaction-local
# like SET LOCAL.
HNSW_SETUP = (
    "SELECT set_config('app.tenant_id', :t, true),"
    " set_config('hnsw.ef_search', '100', true),"
    " set_config('hnsw.iterative_scan', 'relaxed_order', true)"
)

DENSE = f"""
SELECT top.*, d.source_uri, d.title
FROM (
  SELECT c.id, c.version_id, c.embedding <=> CAST(:q AS vector) AS distance,
         c.heading, c.anchor, c.anchor_path, c.page_start, c.page_end,
         c.doc_version, c.source, c.content
  FROM chunks c
  WHERE c.tenant_id = :tenant AND c.collection_id = :collection AND c.is_active
    {FILTER_SQL}
  ORDER BY distance
  LIMIT :k
) top
JOIN document_versions v ON v.id = top.version_id
JOIN documents d ON d.id = v.document_id
"""


async def dense(
    engine: AsyncEngine,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    query_vector: list[float],
    filters: Filters | None = None,
    k: int = 50,
    scan: Literal["auto", "hnsw", "exact"] = "auto",
) -> list[Hit]:
    """`auto` lets the planner choose (on a small table it scans exactly, ignoring HNSW);
    `hnsw` forces the index and `exact` forbids it, for the exact-versus-HNSW check."""
    params: dict[str, object] = {
        "q": vector_literal(query_vector),
        "tenant": tenant_id,
        "collection": collection_id,
        "k": k,
        **(filters or Filters()).params(),
    }
    async with engine.begin() as conn:
        if scan == "exact":
            await set_tenant(conn, tenant_id)
            await conn.execute(text("SET LOCAL enable_indexscan = off"))
        else:
            await conn.execute(text(HNSW_SETUP), {"t": str(tenant_id)})
        if scan == "hnsw":
            await conn.execute(text("SET LOCAL enable_seqscan = off"))
            await conn.execute(text("SET LOCAL enable_sort = off"))
        rows = (await conn.execute(text(DENSE), params)).all()
    # relaxed_order can return rows slightly out of order: re-sort before ranking.
    rows = sorted(rows, key=lambda r: r.distance)
    return [Hit.from_row(r, 1.0 - r.distance) for r in rows]
