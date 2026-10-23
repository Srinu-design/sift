"""Writes deploy/grafana/dashboards/sift.json from the panel list below (edit here, rerun).

uv run python deploy/grafana/build_dashboard.py
"""

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}  # Hopper's provisioned data source
PANELS = [  # (title, unit, [(promql, legend)])
    (
        "p95 latency per stage",
        "s",
        [
            (
                "histogram_quantile(0.95,"
                " sum by (le, stage) (rate(sift_query_seconds_bucket[5m])))",
                "{{stage}}",
            )
        ],
    ),
    (
        "p50 latency per stage",
        "s",
        [
            (
                "histogram_quantile(0.5, sum by (le, stage) (rate(sift_query_seconds_bucket[5m])))",
                "{{stage}}",
            )
        ],
    ),
    (
        "Queries per second",
        "reqps",
        [("sum by (strategy) (rate(sift_queries_total[1m]))", "{{strategy}}")],
    ),
    (
        "Cost per query (USD)",
        "currencyUSD",
        [("sum(rate(sift_query_cost_usd_total[5m])) / sum(rate(sift_queries_total[5m]))", "cost")],
    ),
    (
        "Cache hit rate",
        "percentunit",
        [
            (
                "sum by (layer) (rate(sift_cache_hits_total[5m]))"
                " / sum by (layer) (rate(sift_cache_lookups_total[5m]))",
                "{{layer}}",
            )
        ],
    ),
    ("Refusals by gate", "reqps", [("sum by (gate) (rate(sift_refusals_total[5m]))", "{{gate}}")]),
    (
        "Errors and rejections",
        "reqps",
        [
            (
                'sum by (status) (rate(sift_http_responses_total{status=~"4xx|5xx"}[5m]))',
                "{{status}}",
            ),
            ("sum(rate(sift_budget_rejections_total[5m]))", "budget 429"),
        ],
    ),
    (
        "LLM tokens per second",
        "short",
        [("sum by (direction) (rate(sift_llm_tokens_total[5m]))", "{{direction}}")],
    ),
    (
        "Ingestion backlog (Hopper queues)",
        "short",
        [
            (
                'sum by (queue) (hopper_queue_depth{state="ready", queue=~"default|sift-embed"})',
                "{{queue}} ready",
            )
        ],
    ),
    (
        "Chunks ingested per second",
        "short",
        [("sum(rate(sift_ingest_chunks_total[5m]))", "chunks")],
    ),
]


def panel(i: int, title: str, unit: str, targets: list[tuple[str, str]]) -> dict[str, object]:
    return {
        "id": i + 1,
        "type": "timeseries",
        "title": title,
        "datasource": DS,
        "gridPos": {"h": 8, "w": 12, "x": 12 * (i % 2), "y": 8 * (i // 2)},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "targets": [
            {"refId": chr(65 + j), "datasource": DS, "expr": e, "legendFormat": legend}
            for j, (e, legend) in enumerate(targets)
        ],
    }


dashboard = {
    "uid": "sift",
    "title": "Sift",
    "schemaVersion": 39,
    "time": {"from": "now-1h", "to": "now"},
    "refresh": "10s",
    "tags": ["sift"],
    "panels": [panel(i, *p) for i, p in enumerate(PANELS)],
}
out = Path(__file__).parent / "dashboards" / "sift.json"
out.write_text(json.dumps(dashboard, indent=1) + "\n")
print(f"wrote {out} ({len(PANELS)} panels)")
