"""Retrieval metrics over ranked chunks, against gold locations labelled by section or page.

Gold is never a chunk id, so re-chunking does not invalidate the golden set. A retrieved
chunk is a gold hit when it is the same document (and so the same version) and either the
gold section is on its anchor path (the section itself or one it contains) or its page range
covers the gold page. A gold location with neither section nor page matches the whole doc.
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Gold:
    doc: str
    section: str | None = None
    page: int | None = None


@dataclass(frozen=True, slots=True)
class Retrieved:
    doc: str
    anchor_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None


def is_hit(chunk: Retrieved, gold: Gold) -> bool:
    if chunk.doc != gold.doc:
        return False
    if gold.section is not None:
        return gold.section in chunk.anchor_path
    if gold.page is not None:
        return (
            chunk.page_start is not None
            and chunk.page_end is not None
            and chunk.page_start <= gold.page <= chunk.page_end
        )
    return True


def first_hit_rank(ranked: Sequence[Retrieved], golds: Sequence[Gold]) -> int | None:
    """1-based rank of the first chunk that hits any gold location; None if none does."""
    for rank, chunk in enumerate(ranked, start=1):
        if any(is_hit(chunk, g) for g in golds):
            return rank
    return None


def hit_at_k(ranked: Sequence[Retrieved], golds: Sequence[Gold], k: int) -> float:
    rank = first_hit_rank(ranked[:k], golds)
    return 1.0 if rank is not None else 0.0


def recall_at_k(ranked: Sequence[Retrieved], golds: Sequence[Gold], k: int) -> float:
    """Share of this question's gold locations found in the top k."""
    top = ranked[:k]
    found = sum(any(is_hit(c, g) for c in top) for g in golds)
    return found / len(golds)


def reciprocal_rank(ranked: Sequence[Retrieved], golds: Sequence[Gold], k: int = 10) -> float:
    rank = first_hit_rank(ranked[:k], golds)
    return 1.0 / rank if rank is not None else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = mean,
    resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile interval. With ~100 questions one question moves a mean by ~1 point,
    so a number without its interval says less than it seems to."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    return stats[int(0.025 * resamples)], stats[int(0.975 * resamples) - 1]
