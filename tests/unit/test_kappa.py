"""Unit tests for unweighted and quadratic-weighted Cohen's kappa."""

import pytest

from eval.kappa import CORRECTNESS_LABELS, FAITHFUL_LABELS, cohen_kappa


def test_perfect_agreement_returns_one() -> None:
    labels = ["yes", "partly", "no", "yes", "no"]
    assert cohen_kappa(labels, labels, FAITHFUL_LABELS, weights="none") == pytest.approx(1.0)
    assert cohen_kappa(labels, labels, FAITHFUL_LABELS, weights="quadratic") == pytest.approx(1.0)


def test_hand_computed_two_class_example() -> None:
    a = ["y", "y", "n", "n", "y", "n", "y", "y", "n", "y"]
    b = ["y", "n", "n", "n", "y", "n", "y", "y", "y", "y"]
    # Observed agreement 8/10. Each rater says y 6 times in 10, so expected agreement is
    # 0.6 * 0.6 + 0.4 * 0.4 = 0.52, and kappa = (0.80 - 0.52) / (1 - 0.52) = 7/12.
    kappa = cohen_kappa(a, b, ["y", "n"], weights="none")
    assert kappa == pytest.approx(7.0 / 12.0)


def test_quadratic_weighting_penalises_far_disagreement_more_than_near() -> None:
    a = ["1", "1", "0.5", "0.5", "0", "0"]
    b_near = ["0.5", "1", "1", "0.5", "0", "0"]
    b_far = ["0", "1", "0.5", "0.5", "1", "0"]

    kappa_near = cohen_kappa(a, b_near, CORRECTNESS_LABELS, weights="quadratic")
    kappa_far = cohen_kappa(a, b_far, CORRECTNESS_LABELS, weights="quadratic")
    assert kappa_near > kappa_far


def test_zero_expected_disagreement_returns_one() -> None:
    a = ["yes", "yes", "yes", "yes"]
    b = ["yes", "yes", "yes", "yes"]
    assert cohen_kappa(a, b, FAITHFUL_LABELS, weights="none") == pytest.approx(1.0)
    assert cohen_kappa(a, b, FAITHFUL_LABELS, weights="quadratic") == pytest.approx(1.0)
