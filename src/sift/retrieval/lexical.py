"""Lexical search: PostgreSQL full-text search with ts_rank_cd over weighted tsvectors."""

from typing import Literal
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.retrieval.filters import FILTER_SQL, Filters
from sift.retrieval.hit import Hit

LEXICAL = f"""
WITH q AS (
  SELECT to_tsquery('simple', string_agg(quote_literal(lexeme), :sep)) AS q
  FROM unnest(to_tsvector('english', :question))
)
SELECT top.*, d.source_uri, d.title
FROM (
  SELECT c.id, c.version_id, ts_rank_cd(c.fts, q.q) AS score,
         c.heading, c.anchor, c.anchor_path, c.page_start, c.page_end,
         c.doc_version, c.source, c.content
  FROM chunks c, q
  WHERE c.tenant_id = :tenant AND c.collection_id = :collection AND c.is_active
    {FILTER_SQL}
    AND c.fts @@ q.q
  ORDER BY score DESC, c.id
  LIMIT :k
) top
JOIN document_versions v ON v.id = top.version_id
JOIN documents d ON d.id = v.document_id
"""


async def lexical(
    engine: AsyncEngine,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    question: str,
    filters: Filters | None = None,
    k: int = 50,
    mode: Literal["or", "and"] = "or",
) -> list[Hit]:
    sep = " | " if mode == "or" else " & "
    params: dict[str, object] = {
        "question": question,
        "sep": sep,
        "tenant": tenant_id,
        "collection": collection_id,
        "k": k,
        **(filters or Filters()).params(),
    }
    async with engine.connect() as conn:
        rows = (await conn.execute(text(LEXICAL), params)).all()
    rows = sorted(rows, key=lambda r: (-r.score, r.id))
    return [
        Hit(
            r.id,
            float(r.score),
            r.heading,
            r.anchor,
            tuple(r.anchor_path),
            r.page_start,
            r.page_end,
            r.doc_version,
            r.source,
            r.source_uri,
            r.title,
            r.content,
        )
        for r in rows
    ]
