"""Compute Cohen's kappa (unweighted and quadratic-weighted) between human and judge labels.

python -m eval.kappa --labels eval/labels/human-v1.jsonl --judged JUDGED_JSONL
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from eval import HERE, read_jsonl, save

FAITHFUL_LABELS: tuple[str, ...] = ("yes", "partly", "no")
CORRECTNESS_LABELS: tuple[str, ...] = ("1", "0.5", "0")
KAPPA_WARNING_THRESHOLD = 0.6


def bucket_faithfulness(score: float) -> str:
    if score >= 1.0:
        return "yes"
    if score <= 0.0:
        return "no"
    return "partly"


def bucket_correctness(score: float) -> str:
    nearest = min((1.0, 0.5, 0.0), key=lambda target: abs(target - score))
    if nearest == 1.0:
        return "1"
    if nearest == 0.5:
        return "0.5"
    return "0"


def cohen_kappa(
    a: Sequence[str],
    b: Sequence[str],
    labels: Sequence[str],
    weights: Literal["none", "quadratic"] = "none",
) -> float:
    if len(a) != len(b):
        raise ValueError("sequences a and b must have the same length")
    n = len(a)
    if n == 0:
        raise ValueError("sequences must not be empty")

    k = len(labels)
    if k <= 1:
        return 1.0
    index_of = {label: idx for idx, label in enumerate(labels)}

    observed = [[0] * k for _ in range(k)]
    row_totals = [0] * k
    col_totals = [0] * k
    for label_a, label_b in zip(a, b, strict=True):
        i = index_of[label_a]
        j = index_of[label_b]
        observed[i][j] += 1
        row_totals[i] += 1
        col_totals[j] += 1

    obs_disagreement = 0.0
    exp_disagreement = 0.0
    denom = float((k - 1) ** 2)
    for i in range(k):
        for j in range(k):
            if i == j:
                w = 0.0
            elif weights == "none":
                w = 1.0
            else:
                w = ((i - j) ** 2) / denom
            obs_disagreement += w * (observed[i][j] / n)
            exp_prob = (row_totals[i] / n) * (col_totals[j] / n)
            exp_disagreement += w * exp_prob

    if exp_disagreement == 0.0:
        return 1.0
    return 1.0 - (obs_disagreement / exp_disagreement)


def confusion_matrix(
    a: Sequence[str],
    b: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        row_label: {col_label: 0 for col_label in labels} for row_label in labels
    }
    for la, lb in zip(a, b, strict=True):
        matrix[la][lb] += 1
    return matrix


def evaluate_dimension(
    human_labels: Sequence[str],
    judge_labels: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    n = len(human_labels)
    if n == 0:
        return {
            "n": 0,
            "raw_agreement": 0.0,
            "kappa": 0.0,
            "quadratic_kappa": 0.0,
            "confusion": confusion_matrix([], [], labels),
        }
    agree = sum(1 for h, j in zip(human_labels, judge_labels, strict=True) if h == j)
    raw = agree / n
    kappa = cohen_kappa(human_labels, judge_labels, labels, weights="none")
    q_kappa = cohen_kappa(human_labels, judge_labels, labels, weights="quadratic")
    return {
        "n": n,
        "raw_agreement": round(raw, 4),
        "kappa": round(kappa, 4),
        "quadratic_kappa": round(q_kappa, 4),
        "confusion": confusion_matrix(human_labels, judge_labels, labels),
    }


def compare_files(labels_path: Path, judged_path: Path) -> dict[str, Any]:
    human_by_id = {str(r["id"]): r for r in read_jsonl(labels_path)}
    judged_by_id = {str(r["id"]): r for r in read_jsonl(judged_path)}
    shared_ids = sorted(set(human_by_id) & set(judged_by_id))

    h_faith: list[str] = []
    j_faith: list[str] = []
    h_corr: list[str] = []
    j_corr: list[str] = []

    for qid in shared_ids:
        h_row = human_by_id[qid]
        j_row = judged_by_id[qid]
        if h_row.get("faithful") is not None and j_row.get("faithfulness") is not None:
            h_faith.append(bucket_faithfulness(float(h_row["faithful"])))
            j_faith.append(bucket_faithfulness(float(j_row["faithfulness"])))
        if h_row.get("correctness") is not None and j_row.get("correctness") is not None:
            h_corr.append(bucket_correctness(float(h_row["correctness"])))
            j_corr.append(bucket_correctness(float(j_row["correctness"])))

    return {
        "labels_file": str(labels_path),
        "judged_file": str(judged_path),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "shared_ids": len(shared_ids),
        "faithfulness": evaluate_dimension(h_faith, j_faith, FAITHFUL_LABELS),
        "correctness": evaluate_dimension(h_corr, j_corr, CORRECTNESS_LABELS),
    }


def _format_dimension(
    name: str,
    stats: dict[str, Any],
    labels: Sequence[str],
) -> list[str]:
    lines = [
        f"{name}: n={stats['n']} | raw={stats['raw_agreement']:.3f} | "
        f"kappa={stats['kappa']:.3f} | quadratic_kappa={stats['quadratic_kappa']:.3f}",
        f"{'human \\ judge':<14}" + "".join(f"{col:>8}" for col in labels),
    ]
    conf: dict[str, dict[str, int]] = stats["confusion"]
    for row_label in labels:
        row_cells = "".join(f"{conf[row_label][col]:>8}" for col in labels)
        lines.append(f"{row_label:<14}{row_cells}")
    if stats["n"] > 0 and float(stats["kappa"]) < KAPPA_WARNING_THRESHOLD:
        lines.append(
            f"WARNING ({name}): kappa {stats['kappa']:.3f} < {KAPPA_WARNING_THRESHOLD} - "
            "fix the judge prompt before trusting its numbers"
        )
    return lines


def format_report(report: dict[str, Any]) -> str:
    lines: list[str] = [
        f"kappa agreement | shared_ids={report['shared_ids']}",
        *_format_dimension("faithfulness", report["faithfulness"], FAITHFUL_LABELS),
        "",
        *_format_dimension("correctness", report["correctness"], CORRECTNESS_LABELS),
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval.kappa",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--labels", type=Path, default=HERE / "labels/human-v1.jsonl")
    p.add_argument("--judged", type=Path, required=True)
    args = p.parse_args()

    report = compare_files(args.labels, args.judged)
    print(format_report(report))

    out_path = save(report, "kappa")
    print(f"kappa written: {out_path.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
