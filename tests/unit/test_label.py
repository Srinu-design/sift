"""Unit tests for reproducible sampling and resume filtering in eval.label."""

from typing import Any

from eval.label import select_sample


def _make_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(1, 21):
        rows.append(
            {
                "id": f"q-{i:02d}",
                "type": "factual",
                "answerable": True,
                "refused": i % 5 == 0,
                "answer": f"Answer {i}",
            }
        )
    return rows


def test_sample_is_reproducible_and_skips_labeled_ids() -> None:
    rows = _make_rows()
    reversed_rows = list(reversed(rows))

    sample_a = select_sample(rows, n=6, seed=20261023)
    sample_b = select_sample(reversed_rows, n=6, seed=20261023)

    ids_a = [r["id"] for r in sample_a]
    ids_b = [r["id"] for r in sample_b]
    assert len(ids_a) == 6
    assert ids_a == ids_b
    assert all(not r["refused"] for r in sample_a)

    already_labeled = {ids_a[0], ids_a[3]}
    remaining = select_sample(
        rows,
        n=6,
        seed=20261023,
        labeled_ids=already_labeled,
    )
    assert [r["id"] for r in remaining] == [qid for qid in ids_a if qid not in already_labeled]
