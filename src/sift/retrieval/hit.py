"""Retrieval hit returned by dense, lexical, and hybrid search."""

from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: int
    score: float  # higher is better: 1 - cosine distance, ts_rank_cd, or RRF score
    heading: str
    anchor: str | None
    anchor_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    doc_version: str | None
    source: str
    source_uri: str
    title: str
    content: str

    @classmethod
    def from_row(cls, r: Any, score: float) -> Self:
        """A dense or lexical result row (both select the same columns)."""
        return cls(
            r.id,
            score,
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
