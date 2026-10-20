"""Unified retrieval entrypoint supporting naive, lexical, and hybrid strategies."""

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncEngine

from sift.retrieval.dense import dense
from sift.retrieval.filters import Filters
from sift.retrieval.fuse import rrf
from sift.retrieval.hit import Hit
from sift.retrieval.lexical import lexical

Strategy = Literal["naive", "lexical", "hybrid"]  # hybrid_rerank arrives next week
STRATEGIES: tuple[Strategy, ...] = ("naive", "lexical", "hybrid")
DEPTH = 50  # what each retriever returns before fusion


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float]  # per stage: "dense", "lexical", "fuse" (only the stages that ran)


async def search(
    engine: AsyncEngine,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    strategy: Strategy,
    question: str,
    query_vector: list[float] | None,
    filters: Filters | None = None,
    k: int = 8,
    depth: int = DEPTH,
    lexical_mode: Literal["or", "and"] = "or",
) -> SearchResult:
    async def _dense(vec: list[float], limit: int) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        hits = await dense(
            engine,
            tenant_id=tenant_id,
            collection_id=collection_id,
            query_vector=vec,
            filters=filters,
            k=limit,
        )
        return hits, round((time.perf_counter() - t0) * 1000.0, 1)

    async def _lexical(limit: int) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        hits = await lexical(
            engine,
            tenant_id=tenant_id,
            collection_id=collection_id,
            question=question,
            filters=filters,
            k=limit,
            mode=lexical_mode,
        )
        return hits, round((time.perf_counter() - t0) * 1000.0, 1)

    if strategy == "lexical":
        hits, lex_ms = await _lexical(k)
        return SearchResult(hits=hits, timings_ms={"lexical": lex_ms})

    if query_vector is None:
        raise ValueError(f"query_vector is required for strategy {strategy!r}")

    if strategy == "naive":
        hits, dense_ms = await _dense(query_vector, k)
        return SearchResult(hits=hits, timings_ms={"dense": dense_ms})

    (dense_hits, dense_ms), (lex_hits, lex_ms) = await asyncio.gather(
        _dense(query_vector, depth),
        _lexical(depth),
    )
    t0 = time.perf_counter()
    by_id: dict[int, Hit] = {h.chunk_id: h for h in lex_hits}
    by_id.update({h.chunk_id: h for h in dense_hits})
    fused = rrf([[h.chunk_id for h in dense_hits], [h.chunk_id for h in lex_hits]])
    hits = [replace(by_id[cid], score=score) for cid, score in fused[:k]]
    fuse_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return SearchResult(
        hits=hits,
        timings_ms={"dense": dense_ms, "lexical": lex_ms, "fuse": fuse_ms},
    )
