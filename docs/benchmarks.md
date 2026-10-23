# Benchmarks

Everything here was measured on one development laptop, not the EC2 host: Intel Core i7-13700H
(20 logical CPUs), 14 GB RAM, the whole Compose stack (Hopper + Sift + Postgres + Redis +
Prometheus + Grafana) on the same machine. CPU only, no GPU. Week 7 repeats the rerank latency on
the EC2 host.

## Query load (2026-10-23, `make load-query`)

k6 at a constant arrival rate on `POST /v1/query` (`loadtest/k6/query.js`), 2 minutes per row,
strategy `hybrid_rerank`, rerank depth 30, 8 sources per answer. **The LLM is the stub** (valid
cited JSON after a fixed 800 ms), so the numbers measure Sift's own pipeline, not Gemini.
Question mix: each request picks one of the 86 dev golden questions uniformly at random (repeats
allowed). sift-models is restarted and the answer cache emptied before every row
(`loadtest/query_bench.sh`); raw k6 summaries and the table: `loadtest/results/20261023-query/`.

| cache | req/s | requests | errors | p50 ms | p95 ms | cache hits | embed p50/p95 | dense p50/p95 | lexical p50/p95 | rerank p50/p95 | llm p50/p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 0.5 | 59 | 0.0% | 2007 | 2090 | 0% | 12/19 | 8/13 | 11/23 | 1186/1238 | 801/801 |
| off | 1 | 119 | 0.0% | 9153 | 18373 | 0% | 490/9300 | 12/22 | 18/49 | 6486/11168 | 801/802 |
| off | 2 | 134 | 27.5% | 44131 | 60001 | 0% | 795/29950 | 13/19 | 18/58 | 28696/55010 | 801/802 |
| off | 5 | 120 | 74.7% | 60000 | 60001 | 0% | 1/30899 | 14/22 | 18/38 | 48141/77693 | 801/802 |
| on | 0.5 | 59 | 0.0% | 1798 | 1882 | 31% | 13/19 | 9/11 | 12/26 | 994/1046 | 801/802 |
| on | 1 | 120 | 0.0% | 1095 | 2556 | 48% | 38/834 | 6/11 | 10/23 | 1015/1081 | 801/802 |
| on | 2 | 236 | 5.0% | 9880 | 58050 | 44% | 7893/30309 | 11/21 | 18/61 | 15100/47310 | 801/802 |
| on | 5 | 304 | 40.2% | 32479 | 60001 | 62% | 3975/40029 | 14/21 | 21/37 | 39605/69761 | 801/802 |

Stage columns come from `query_logs` (uncached queries only); "errors" are k6's 60 s timeouts.

**Highest rate with p95 under 3 s:** 0.5 requests/s without the answer cache (p95 2.1 s);
1 request/s with it (p95 2.6 s, 48% hits under this mix).

**What saturates first: the cross-encoder on sift-models.** One rerank of 30 pairs takes about
1.0–1.2 s of the whole CPU, and sift-models runs one model call at a time (ADR-0006), so the
ceiling is a little under 1 uncached query per second. Past it, the rerank queue grows without
bound (rerank p50 6.5 s at 1 req/s, 29 s at 2), and query embedding waits in the same queue (embed
p95 9.3 s at 1 req/s against 19 ms at 0.5). Dense and lexical search stay under ~60 ms at every
rate: Postgres is nowhere near its limit. The cache lifts the ceiling by the hit rate: at 1 req/s
only about half the requests reach the reranker.

**Levers, in order of effect** (ADR-0003): a smaller rerank depth (top 20, measured against
quality first), ONNX/int8 export of the cross-encoder, shorter passages (256 tokens), a second
sift-models replica with its own CPU, and a higher cache hit rate on real traffic.

The first run of this test was discarded: k6 abandons a request after 60 s, but the server
keeps working on it, so the backlog from the saturated 2 req/s row spilled into every later row
(even "cache on, 1 req/s" showed 100% errors). Restarting sift-models before every row fixed it.

## Cost per query

The answer model is `gemini-3.5-flash-lite` on the free tier: $0 today (`llm_price_*` in
`config.py` are 0). Measured tokens, dev split, prompt answer-v2: 259,473 in and 15,184 out over
79 model calls for 86 questions (7 refused by gate 1 without a call, 7 needed the one retry).

| per 1,000 queries | tokens in | tokens out |
|---|---|---|
| no cache | ~3.0 M | ~0.18 M |
| 48% answer-cache hits (the mix above) | ~1.6 M | ~0.09 M |

Cost per 1,000 queries = tokens in x input price + tokens out x output price; with the paid
price table filled in (and dated) in `config.py`, `query_logs.cost_usd` and the Grafana panel
report it per query.

## Rerank latency (2026-10-20, `make rerank-latency`)

| pairs | p50 | p95 |
|---|---|---|
| 50 | 1,769 ms | 1,977 ms |
| 30 | 1,188 ms | 1,269 ms |

## Ingestion

- PostgreSQL docs, 182 HTML pages, through Hopper: 81–88 s with 1–4 embed workers (bound by
  sift-models), `loadtest/results/2026-10-12-ingest/`. In CI (GitHub runner, in-process): 158 s.
- SEBI circulars, 24 PDFs, in-process with Docling: 906 s and ~5 GB RAM (layout and table models
  on CPU), which is why Docling is not in the 4 GB server image.
