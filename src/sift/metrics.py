"""Prometheus metrics. Each process serves its own on an internal port (METRICS_PORT, 9100).

No tenant label: unbounded label values blow up Prometheus memory, and per-tenant cost comes from
SQL on query_logs. Label values are fixed sets (stages, gates, cache layers, directions).
"""

from collections.abc import Callable

from prometheus_client import Counter, Histogram, start_http_server

# Stage latencies run from ~1 ms (fuse) to tens of seconds (a free-tier LLM call): buckets
# cover both, since histogram_quantile interpolates inside a bucket.
QUERY_SECONDS = Histogram(
    "sift_query_seconds",
    "Time per query stage: embed, dense, lexical, fuse, rerank, llm, and total.",
    ["stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 40),
)
LLM_TOKENS = Counter("sift_llm_tokens_total", "LLM tokens, by direction (in, out).", ["direction"])
QUERY_COST = Counter("sift_query_cost_usd_total", "LLM spend in USD, from the price table.")
CACHE_LOOKUPS = Counter(
    "sift_cache_lookups_total", "Cache lookups by layer (query_embedding, answer).", ["layer"]
)
CACHE_HITS = Counter(
    "sift_cache_hits_total", "Cache hits by layer (query_embedding, answer).", ["layer"]
)
REFUSALS = Counter("sift_refusals_total", "Refusals by gate: retrieval, model, citation.", ["gate"])
QUERIES = Counter("sift_queries_total", "Answered queries by strategy.", ["strategy"])
BUDGET_REJECTIONS = Counter("sift_budget_rejections_total", "Queries refused (429) over budget.")
INGEST_CHUNKS = Counter("sift_ingest_chunks_total", "Chunks written by parse steps.")
RATELIMIT_FALLBACK = Counter(
    "sift_ratelimit_fallback_total",
    "Rate-limit decisions made by an in-process bucket because Redis was unavailable.",
)


def serve(port: int) -> Callable[[], None]:
    """Serve this process's metrics on 0.0.0.0:port from a daemon thread (copied from Hopper).

    Returns a function that stops the server. Port 0 serves nothing (tests).
    """
    if port == 0:
        return lambda: None
    server, thread = start_http_server(port)

    def stop() -> None:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    return stop
