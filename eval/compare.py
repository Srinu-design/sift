"""Paired bootstrap comparison of two eval.run result files.

python -m eval.compare eval/results/A.json eval/results/B.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval import metric_names
from eval.metrics import bootstrap_ci, mean


def compare(a: dict[str, Any], b: dict[str, Any]) -> str:
    if a.get("split") != b.get("split") or a.get("golden") != b.get("golden"):
        raise SystemExit(
            f"split/golden mismatch: {a.get('split')}/{a.get('golden')} "
            f"vs {b.get('split')}/{b.get('golden')}"
        )

    a_rows = {q["id"]: q for q in a.get("per_question", [])}
    b_rows = {q["id"]: q for q in b.get("per_question", [])}
    if set(a_rows) != set(b_rows):
        raise SystemExit("question id sets differ between result files")

    ids = list(a_rows)
    lines = [
        f"A = {a['strategy']} | B = {b['strategy']} | golden {a['golden']} | "
        f"split {a['split']} | n = {len(ids)}",
        f"{'metric':<10} | {'A':>7} | {'B':>7} | {'B-A':>8} | {'95% CI':>18} | {'real':<4}",
    ]
    for name in metric_names():
        a_vals = [float(a_rows[qid]["scores"][name]) for qid in ids]
        b_vals = [float(b_rows[qid]["scores"][name]) for qid in ids]
        diffs = [bv - av for av, bv in zip(a_vals, b_vals, strict=True)]
        low, high = bootstrap_ci(diffs)
        real = "yes" if (low > 0.0 or high < 0.0) else "no"
        ci = f"[{low:+.3f}, {high:+.3f}]"
        lines.append(
            f"{name:<10} | {mean(a_vals):>7.3f} | {mean(b_vals):>7.3f} | "
            f"{mean(diffs):>+8.3f} | {ci:>18} | {real:<4}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval.compare",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    args = p.parse_args()
    a_report = json.loads(args.a.read_text())
    b_report = json.loads(args.b.read_text())
    print(compare(a_report, b_report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
