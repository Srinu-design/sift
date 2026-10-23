"""Summarise a query load test (loadtest/query_bench.sh): k6's latency and error rate per run,
and per-stage p50/p95 from query_logs for the same time window.

    uv run python loadtest/query_report.py loadtest/results/<date>-query
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import text

from sift.config import get_settings
from sift.db import create_engine

STAGES = ("embed", "dense", "lexical", "rerank", "llm")
WINDOW = """
SELECT count(*) AS n, avg(cache_hit::int) AS hit_rate,
  {stages}
FROM query_logs WHERE created_at BETWEEN :start AND :end
"""


async def main(out: Path) -> None:
    pct = ", ".join(
        f"percentile_cont({q}) WITHIN GROUP (ORDER BY (timings_ms->>'{s}')::float)"
        f" FILTER (WHERE NOT cache_hit) AS {s}_p{int(q * 100)}"
        for s in STAGES
        for q in (0.5, 0.95)
    )
    engine = create_engine(get_settings())
    lines = [
        "| cache | req/s | requests | errors | p50 ms | p95 ms | cache hits "
        + "".join(f"| {s} p50/p95 " for s in STAGES)
        + "|",
        "|---" * (7 + len(STAGES)) + "|",
    ]
    for run in map(json.loads, (out / "runs.jsonl").read_text().splitlines()):
        k6 = json.loads(
            (out / f"k6-cache-{str(run['cache']).lower()}-{run['rate_per_min']}.json").read_text()
        )
        dur = k6["metrics"]["http_req_duration"]["values"]
        failed = k6["metrics"]["http_req_failed"]["values"]["rate"]
        async with engine.connect() as conn:
            window = {k: datetime.fromisoformat(run[k]) for k in ("start", "end")}
            row = (await conn.execute(text(WINDOW.format(stages=pct)), window)).mappings().one()
        stage_cells = "".join(
            f"| {row[f'{s}_p50'] or 0:.0f}/{row[f'{s}_p95'] or 0:.0f} " for s in STAGES
        )
        lines.append(
            f"| {'on' if run['cache'] else 'off'} | {run['rate_per_min'] / 60:g} | {row['n']} "
            f"| {failed:.1%} "
            f"| {dur['med']:.0f} | {dur['p(95)']:.0f} | {float(row['hit_rate'] or 0):.0%} "
            f"{stage_cells}|"
        )
    await engine.dispose()
    table = "\n".join(lines)
    (out / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
