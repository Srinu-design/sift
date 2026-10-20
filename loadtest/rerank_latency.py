"""Benchmark CPU cross-encoder rerank latency through sift-models for 50 and 30 pairs.

python loadtest/rerank_latency.py --url http://127.0.0.1:8100 --runs 30
"""

import argparse
import asyncio
import json
import math
import os
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import text

from sift.config import get_settings
from sift.db import create_engine

HERE = Path(__file__).resolve().parent
QUESTION = "How does PostgreSQL configure autovacuum vacuum scale factor on large tables?"
PAIR_COUNTS = (50, 30)
WARMUP_CALLS = 3


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, math.ceil(q * len(ordered)) - 1)
    return round(ordered[idx], 2)


async def fetch_passages(limit: int = 50) -> list[str]:
    engine = create_engine(get_settings())
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT heading, content FROM chunks WHERE is_active"
                        " ORDER BY random() LIMIT :n"
                    ),
                    {"n": limit},
                )
            ).all()
    finally:
        await engine.dispose()
    return [f"{r.heading}\n{r.content}" for r in rows]


async def benchmark(url: str, runs: int) -> dict[str, Any]:
    passages = await fetch_passages(max(PAIR_COUNTS))
    if not passages:
        raise SystemExit("no active chunks found in database")

    endpoint = f"{url.rstrip('/')}/rerank"
    settings = get_settings()
    model_name = settings.rerank_model
    by_size: dict[str, Any] = {}

    async with httpx.AsyncClient(timeout=settings.models_timeout_seconds) as http:
        for size in PAIR_COUNTS:
            batch = passages[:size]
            for _ in range(WARMUP_CALLS):
                resp = await http.post(endpoint, json={"query": QUESTION, "passages": batch})
                resp.raise_for_status()
                model_name = resp.json().get("model", model_name)

            timings: list[float] = []
            for _ in range(runs):
                t0 = time.perf_counter()
                resp = await http.post(endpoint, json={"query": QUESTION, "passages": batch})
                resp.raise_for_status()
                timings.append((time.perf_counter() - t0) * 1000.0)

            mean_ms = round(sum(timings) / len(timings), 2)
            p50_ms = percentile(timings, 0.50)
            p95_ms = percentile(timings, 0.95)
            by_size[str(size)] = {
                "pairs": len(batch),
                "runs": runs,
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
                "mean_ms": mean_ms,
            }
            print(
                f"pairs={len(batch):>2} | n={runs:>3} | "
                f"p50={p50_ms:>7.2f} ms | p95={p95_ms:>7.2f} ms | mean={mean_ms:>7.2f} ms"
            )

    return {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "rerank_model": model_name,
        "runs": runs,
        "results": by_size,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python loadtest/rerank_latency.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--runs", type=int, default=30)
    args = p.parse_args()

    report = asyncio.run(benchmark(args.url, args.runs))
    date_stamp = datetime.now(UTC).strftime("%Y%m%d")
    out_dir = HERE / "results" / f"{date_stamp}-rerank"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
