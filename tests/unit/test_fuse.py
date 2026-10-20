"""Unit tests for reciprocal-rank fusion."""

import pytest

from sift.retrieval.fuse import rrf


def test_empty_input() -> None:
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_single_list_keeps_order() -> None:
    fused = rrf([[10, 20, 30]])
    assert [cid for cid, _ in fused] == [10, 20, 30]


def test_disjoint_lists_interleave_and_tie_break() -> None:
    fused = rrf([[1, 2], [3, 4]])
    assert [cid for cid, _ in fused] == [1, 3, 2, 4]


def test_shared_chunk_beats_single_retriever_and_keeps_singles() -> None:
    fused = rrf([[1, 2], [3, 2]])
    assert [cid for cid, _ in fused] == [2, 1, 3]


def test_repeated_id_in_one_list_counts_once() -> None:
    fused = rrf([[7, 7, 8]], k=60)
    assert fused == [
        (7, pytest.approx(1.0 / 61.0)),
        (8, pytest.approx(1.0 / 63.0)),
    ]


def test_exact_scores() -> None:
    fused = rrf([[1, 2], [2, 3]], k=60)
    assert fused == [
        (2, pytest.approx(1.0 / 62.0 + 1.0 / 61.0)),
        (1, pytest.approx(1.0 / 61.0)),
        (3, pytest.approx(1.0 / 62.0)),
    ]
