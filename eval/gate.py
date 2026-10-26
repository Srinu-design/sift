"""CI retrieval regression gate comparing eval.run outputs against a pinned baseline.

python -m eval.gate eval/results/*-dev-v1.json --baseline eval/baselines/retrieval.json
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from eval import HERE

REPORT_METRICS: tuple[str, ...] = ("hit@5", "hit@8", "recall@10", "mrr@10")
GATED_METRICS: tuple[str, ...] = ("hit@8", "recall@10")
GATED_STRATEGY = "hybrid_rerank"
OVERALL_MAX_DROP = 0.02
TYPE_MAX_DROP = 0.05


@dataclass(frozen=True, slots=True)
class Finding:
    strategy: str
    metric: str
    baseline: float | None
    now: float | None
    change: float | None
    status: Literal["pass", "fail", "info", "no baseline"]
    failed: bool
    message: str | None = None


def check(results: dict[str, Any], baseline: dict[str, Any]) -> list[Finding]:
    strategy = str(results["strategy"])
    base_entry = baseline.get(strategy)
    if not isinstance(base_entry, dict):
        return [
            Finding(
                strategy=strategy,
                metric=m,
                baseline=None,
                now=float(results["overall"][m]["mean"]),
                change=None,
                status="no baseline",
                failed=False,
            )
            for m in REPORT_METRICS
        ]

    if results.get("golden") != base_entry.get("golden") or results.get("split") != base_entry.get(
        "split"
    ):
        msg = (
            f"golden/split mismatch for {strategy}: "
            f"results has {results.get('golden')}/{results.get('split')}, "
            f"baseline has {base_entry.get('golden')}/{base_entry.get('split')}"
        )
        return [
            Finding(
                strategy=strategy,
                metric="golden/split",
                baseline=None,
                now=None,
                change=None,
                status="fail",
                failed=True,
                message=msg,
            )
        ]

    findings: list[Finding] = []
    is_gated = strategy == GATED_STRATEGY

    for m in REPORT_METRICS:
        base_val = float(base_entry["overall"][m]["mean"])
        now_val = float(results["overall"][m]["mean"])
        change = round(now_val - base_val, 4)
        drop = round(base_val - now_val, 6)

        if is_gated and m in GATED_METRICS:
            failed = drop > OVERALL_MAX_DROP
            status: Literal["pass", "fail", "info", "no baseline"] = "fail" if failed else "pass"
        else:  # reported, never gated: a drop here is not a pass
            failed = False
            status = "info"

        findings.append(
            Finding(
                strategy=strategy,
                metric=m,
                baseline=base_val,
                now=now_val,
                change=change,
                status=status,
                failed=failed,
            )
        )

    if is_gated:
        base_types = base_entry.get("by_type") or {}
        res_types = results.get("by_type") or {}
        for qtype in sorted(res_types):
            if qtype not in base_types:
                continue
            for m in GATED_METRICS:
                base_val = float(base_types[qtype][m]["mean"])
                now_val = float(res_types[qtype][m]["mean"])
                change = round(now_val - base_val, 4)
                drop = round(base_val - now_val, 6)
                if drop > TYPE_MAX_DROP:
                    findings.append(
                        Finding(
                            strategy=strategy,
                            metric=f"{qtype}:{m}",
                            baseline=base_val,
                            now=now_val,
                            change=change,
                            status="fail",
                            failed=True,
                        )
                    )

    return findings


def render_markdown(findings: Sequence[Finding]) -> str:
    lines = [
        "| strategy | metric | baseline | now | change | status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for f in findings:
        b_str = f"{f.baseline:.3f}" if f.baseline is not None else "-"
        n_str = f"{f.now:.3f}" if f.now is not None else "-"
        c_str = f"{f.change:+.3f}" if f.change is not None else "-"
        status_str = f"{f.status} ({f.message})" if f.message else f.status
        lines.append(f"| {f.strategy} | {f.metric} | {b_str} | {n_str} | {c_str} | {status_str} |")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python -m eval.gate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument(
        "--baseline",
        type=Path,
        default=HERE / "baselines/retrieval.json",
    )
    args = p.parse_args()

    baseline: dict[str, Any] = (
        json.loads(args.baseline.read_text()) if args.baseline.exists() else {}
    )
    all_findings: list[Finding] = []
    for path in args.results:
        report = json.loads(path.read_text())
        all_findings.extend(check(report, baseline))

    md = render_markdown(all_findings)
    print(md)

    summary_env = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_env:
        summary_path = Path(summary_env)
        with summary_path.open("a", encoding="utf-8") as fp:
            fp.write(md + "\n")

    return 1 if any(f.failed for f in all_findings) else 0


if __name__ == "__main__":
    sys.exit(main())
