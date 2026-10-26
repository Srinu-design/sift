"""Unit tests for the CI retrieval regression gate."""

from typing import Any

from eval.gate import check


def _metric(val: float) -> dict[str, Any]:
    return {"mean": val, "ci95": [val - 0.02, val + 0.02]}


def _make_report(
    *,
    strategy: str = "hybrid_rerank",
    golden: str = "v1",
    split: str = "dev",
    hit5: float = 0.85,
    hit8: float = 0.90,
    recall10: float = 0.88,
    mrr10: float = 0.72,
    type_recall10: float = 0.85,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "golden": golden,
        "split": split,
        "overall": {
            "n": 50,
            "hit@5": _metric(hit5),
            "hit@8": _metric(hit8),
            "recall@10": _metric(recall10),
            "mrr@10": _metric(mrr10),
        },
        "by_type": {
            "factual": {
                "n": 25,
                "hit@5": _metric(hit5),
                "hit@8": _metric(hit8),
                "recall@10": _metric(type_recall10),
                "mrr@10": _metric(mrr10),
            }
        },
    }


def test_unchanged_report_passes() -> None:
    base_entry = _make_report()
    baseline = {"hybrid_rerank": base_entry}
    findings = check(_make_report(), baseline)
    assert not any(f.failed for f in findings)
    assert {f.metric: f.status for f in findings} == {
        "hit@5": "info",
        "hit@8": "pass",
        "recall@10": "pass",
        "mrr@10": "info",
    }


def test_overall_hit8_drop_003_on_hybrid_rerank_fails() -> None:
    baseline = {"hybrid_rerank": _make_report(hit8=0.90)}
    current = _make_report(hit8=0.87)
    findings = check(current, baseline)
    failed = [f for f in findings if f.failed]
    assert len(failed) == 1
    assert failed[0].metric == "hit@8"
    assert failed[0].status == "fail"


def test_overall_hit8_drop_001_on_hybrid_rerank_passes() -> None:
    baseline = {"hybrid_rerank": _make_report(hit8=0.90)}
    current = _make_report(hit8=0.89)
    findings = check(current, baseline)
    assert not any(f.failed for f in findings)


def test_per_type_recall10_drop_006_fails() -> None:
    baseline = {"hybrid_rerank": _make_report(type_recall10=0.85)}
    current = _make_report(type_recall10=0.79)
    findings = check(current, baseline)
    failed = [f for f in findings if f.failed]
    assert len(failed) == 1
    assert failed[0].metric == "factual:recall@10"
    assert failed[0].status == "fail"


def test_drop_on_naive_is_reported_not_failed() -> None:
    baseline = {"naive": _make_report(strategy="naive", hit8=0.80, recall10=0.80)}
    current = _make_report(strategy="naive", hit8=0.60, recall10=0.60, type_recall10=0.50)
    findings = check(current, baseline)
    assert len(findings) == 4
    assert not any(f.failed for f in findings)
    assert all(f.status == "info" for f in findings)


def test_golden_mismatch_fails() -> None:
    baseline = {"hybrid_rerank": _make_report(golden="v1")}
    current = _make_report(golden="v2")
    findings = check(current, baseline)
    assert any(f.failed for f in findings)
    assert findings[0].message is not None
    assert "mismatch" in findings[0].message


def test_missing_baseline_entry_is_not_a_failure() -> None:
    baseline = {"naive": _make_report(strategy="naive")}
    current = _make_report(strategy="hybrid_rerank")
    findings = check(current, baseline)
    assert not any(f.failed for f in findings)
    assert all(f.status == "no baseline" for f in findings)
