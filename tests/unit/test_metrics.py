"""Retrieval metrics on hand-made rankings, where the right answer is obvious by eye."""

import math

from eval.metrics import (
    Gold,
    Retrieved,
    bootstrap_ci,
    first_hit_rank,
    hit_at_k,
    is_hit,
    recall_at_k,
    reciprocal_rank,
)

GIN = Gold("pg17/indexes-types", "INDEXES-TYPES-GIN")
BTREE = Gold("pg17/indexes-types", "INDEXES-TYPES-BTREE")


def chunk(doc: str, *path: str) -> Retrieved:
    return Retrieved(doc, tuple(path))


def test_a_chunk_in_the_gold_section_or_a_subsection_hits() -> None:
    assert is_hit(chunk("pg17/indexes-types", "INDEXES-TYPES", "INDEXES-TYPES-GIN"), GIN)
    page = Gold("pg17/indexes-types", "INDEXES-TYPES")  # gold is the parent section
    assert is_hit(chunk("pg17/indexes-types", "INDEXES-TYPES", "INDEXES-TYPES-GIN"), page)


def test_the_parent_section_does_not_hit_a_subsection_gold() -> None:
    assert not is_hit(chunk("pg17/indexes-types", "INDEXES-TYPES"), GIN)


def test_the_same_section_in_another_version_is_a_miss() -> None:
    assert not is_hit(chunk("pg16/indexes-types", "INDEXES-TYPES", "INDEXES-TYPES-GIN"), GIN)


def test_page_gold_needs_the_page_inside_the_chunk_range() -> None:
    gold = Gold("rbi/circular-12", page=3)
    assert is_hit(Retrieved("rbi/circular-12", (), 2, 3), gold)
    assert not is_hit(Retrieved("rbi/circular-12", (), 4, 5), gold)
    assert not is_hit(Retrieved("rbi/circular-12"), gold)  # no page info: not a hit


def test_a_doc_level_gold_matches_any_chunk_of_the_doc() -> None:
    assert is_hit(chunk("pg17/sql-vacuum", "SQL-VACUUM"), Gold("pg17/sql-vacuum"))


def test_hit_rank_mrr_at_k() -> None:
    miss = chunk("pg17/mvcc", "MVCC")
    hit = chunk("pg17/indexes-types", "INDEXES-TYPES", "INDEXES-TYPES-GIN")
    ranked = [miss, miss, hit, miss]
    assert first_hit_rank(ranked, [GIN]) == 3
    assert hit_at_k(ranked, [GIN], 2) == 0.0
    assert hit_at_k(ranked, [GIN], 3) == 1.0
    assert reciprocal_rank(ranked, [GIN], k=10) == 1 / 3
    assert reciprocal_rank(ranked, [GIN], k=2) == 0.0  # outside the cut-off counts as zero


def test_recall_counts_gold_locations_not_chunks() -> None:
    gin = chunk("pg17/indexes-types", "INDEXES-TYPES", "INDEXES-TYPES-GIN")
    ranked = [gin, gin, gin]  # three hits on the same gold location
    assert recall_at_k(ranked, [GIN, BTREE], 3) == 0.5
    btree = chunk("pg17/indexes-types", "INDEXES-TYPES", "INDEXES-TYPES-BTREE")
    assert recall_at_k([gin, btree], [GIN, BTREE], 2) == 1.0
    assert recall_at_k([gin, btree], [GIN, BTREE], 1) == 0.5


def test_empty_ranking() -> None:
    assert hit_at_k([], [GIN], 8) == 0.0
    assert recall_at_k([], [GIN], 8) == 0.0
    assert first_hit_rank([], [GIN]) is None


def test_bootstrap_interval_brackets_the_mean() -> None:
    values = [1.0] * 70 + [0.0] * 30
    low, high = bootstrap_ci(values)
    assert low < 0.7 < high
    assert low > 0.55 and high < 0.85
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)
    assert all(math.isnan(x) for x in bootstrap_ci([]))
