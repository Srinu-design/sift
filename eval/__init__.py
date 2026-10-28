"""Helpers the eval scripts share. Standard library only, so importing them stays cheap."""

import json
import math
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
KS = (5, 8, 10, 20)


def metric_names() -> list[str]:
    return [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS] + ["mrr@10"]


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, math.ceil(q * len(ordered)) - 1)
    return round(ordered[idx], 1)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def save(report: dict[str, Any], name: str) -> Path:
    """Write the report to eval/results/<UTC timestamp>-<name>.json and return the path."""
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = results / f"{stamp}-{name}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    return out
