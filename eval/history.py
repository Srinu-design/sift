"""Every dated eval run, one row per run, for the demo page's "Eval history" tab.

python -m eval.history   (or make eval-history) rewrites src/sift/api/static/eval-history.json
from eval/results/: retrieval runs (hit@k, recall@10, MRR@10) and answer runs (refusals,
citations). Overlap, tau, kappa and judge files are other analyses and are left out.
"""

import json
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).parent / "results"
OUT = Path(__file__).parents[1] / "src" / "sift" / "api" / "static" / "eval-history.json"
RETRIEVAL_METRICS = ("hit@5", "hit@8", "recall@10", "mrr@10")
ANSWER_METRICS = ("refusal_recall", "refusal_precision", "citation_accuracy")


def row(result: dict[str, Any]) -> dict[str, Any] | None:
    overall = result.get("overall")
    if isinstance(overall, dict) and isinstance(overall.get("mrr@10"), dict):
        kind, metrics = "retrieval", {m: overall[m]["mean"] for m in RETRIEVAL_METRICS}
    elif "citation_accuracy" in result:
        kind, metrics = "answers", {m: result[m] for m in ANSWER_METRICS}
    else:
        return None
    return {
        "run_at": result["run_at"],
        "commit": result.get("commit"),  # None for runs before eval.run recorded it
        "kind": kind,
        "strategy": result.get("strategy"),
        "split": result.get("split"),
        "golden": result.get("golden"),
        "metrics": metrics,
    }


def history(results: Path = RESULTS) -> list[dict[str, Any]]:
    rows = (row(json.loads(p.read_text())) for p in sorted(results.glob("*.json")))
    return sorted((r for r in rows if r is not None), key=lambda r: r["run_at"])


def main() -> None:
    runs = history()
    OUT.write_text(json.dumps({"runs": runs}, indent=1) + "\n")
    print(f"wrote {OUT} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
