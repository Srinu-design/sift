"""Retrieval hit returned by dense, lexical, and hybrid search."""

from dataclasses import dataclass


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
