# Sift: design

Question answering over a tenant's documents that cites the section (or PDF page) behind every
claim, refuses when the evidence is missing, and proves every retrieval change with a measured
eval. Every number below links to how it was measured.

## Components

```
                     ┌──────────── Hopper (Project 1): api, workers, scheduler ────────────┐
upload ─► sift-api ─►│ parse:<v> ─► embed:<v>:<n> (queue sift-embed) ─► activate:<v>        │
                     └──────┬───────────────────────────────────────────────────────────────┘
                            │ signed http jobs (HMAC, internal-host allowlist)
                            ▼
                       sift-ingest ──► sift-models /embed            Postgres 16 + pgvector
                            │                                        (chunks, embeddings,
query ─► Caddy /sift/ ─► sift-api ──► sift-models /embed, /rerank     full-text index, RLS)
                            │  ├─► Postgres: dense (HNSW) ‖ lexical (tsvector) ─► RRF
                            │  ├─► Redis: answer cache, query-embedding cache, rate limits, budget
                            │  └─► Gemini (answer, JSON with [S#] citations) ─► citation checks
                            ▼
                 cited answer, or a refusal with "you may want to read"
```

| Process | Role |
|---|---|
| sift-api | query API, uploads, collections, demo page; runs as the non-owner role `sift_app` |
| sift-models | bge-small embeddings and the MiniLM cross-encoder, loaded once (one model call at a time) |
| sift-ingest | internal endpoints that Hopper's jobs call: parse, embed, activate |
| Hopper | retries, dead-letter queue and replay for ingestion (ADR-0006) |
| Postgres | one store for chunks, vectors, full-text and filters (ADR-0001) |
| Redis | caches, token buckets, daily spend (every use fails open) |

## The query path

1. **Auth and tenancy:** the tenant comes from the API key (`sift_live_…`, HMAC with a pepper),
   never the request; every transaction is scoped with `set_tenant` and row-level security is
   the second layer (ADR-0012).
2. **Answer cache:** key = tenant, collection, index_version, strategy, k, filters, prompt
   version, model, normalised question (ADR-0011).
3. **Retrieval** (`hybrid_rerank`): dense top 50 (HNSW, iterative scan under filters) and lexical
   top 50 (OR of the question's lexemes, `ts_rank_cd`) run concurrently, fused with RRF k = 60
   (ADR-0002), the fused top 30 rescored by the cross-encoder (ADR-0003), top 8 to the model.
   Metadata filters (version, dates, sources) apply inside both retrievers.
4. **Gate 1:** best rerank score below τ = 0.19 refuses without an LLM call (ADR-0008).
5. **Answer:** numbered, fenced sources; JSON with `[S#]` markers and short quotes; code checks
   every id, every quote (fuzzy, 90%) and that every sentence cites; one retry, then refusal.
   The model may refuse (gate 2). Sources are data: prompt v2 resists planted instructions.
6. **Log:** a `query_logs` row (timings per stage, tokens, cost, gate, cache hit) and Prometheus
   metrics; the daily budget is charged.

## The ingest path

Upload → file stored under its SHA-256, document upserted, version inserted (an unchanged file
creates nothing) → Hopper `parse` job (HTML by DOM sections; PDF by Docling with page numbers;
chunks of ~400 tokens, 15% overlap, never across a section or page, ≤ 512 tokens, ADR-0004) →
`embed` jobs per 64 chunks on their own queue (embedding cache first) → `activate`, which swaps
the document's live version atomically and bumps the collection's index_version.

## Evaluation

- Golden set v1: 123 PostgreSQL questions (17% unanswerable), labelled by section, frozen
  dev/test split; v2-sebi: 18 questions with page labels and prompt-injection cases (ADR-0007).
- Retrieval metrics (hit@k, recall@k, MRR@10, bootstrap CIs, paired comparisons) gate every PR
  (ADR-0009); answers, refusals, citations and the LLM judge run nightly (ADR-0010).
- Results: test split `hybrid_rerank` hit@5 0.968, refusal recall 1.00, 83% of answers cite a
  gold section (docs/eval-report.md); load: 0.5 req/s under 3 s p95 uncached, 1 req/s with the
  cache (docs/benchmarks.md).

## Decisions

| ADR | Decision |
|---|---|
| [0001](adr/0001-postgres-for-search.md) | Postgres (pgvector + full-text) instead of a vector database |
| [0002](adr/0002-rrf-fusion.md) | RRF with k = 60 instead of weighted score fusion |
| [0003](adr/0003-cross-encoder-rerank.md) | MiniLM cross-encoder on the fused top 30, from measured latency and quality |
| [0004](adr/0004-chunking.md) | ~400-token structure-aware chunks with 15% overlap, from the ablation |
| [0005](adr/0005-embedding-model.md) | Local bge-small-en-v1.5 rather than an API embedding model |
| [0006](adr/0006-ingestion-on-hopper.md) | Ingestion as Hopper http jobs; internal-host allowlist |
| [0007](adr/0007-gold-labels.md) | Gold labels at section or page level, never chunk ids |
| [0008](adr/0008-two-gate-refusal.md) | Two-gate refusal with τ tuned on dev |
| [0009](adr/0009-eval-in-ci.md) | Retrieval eval on every PR; generation eval nightly |
| [0010](adr/0010-llm-judge.md) | LLM judge, validated against human labels (correctness not yet trusted) |
| [0011](adr/0011-answer-cache.md) | Answer cache keyed by index_version; no semantic cache |
| [0012](adr/0012-tenant-isolation.md) | Filters plus row-level security on a shared index |

## What breaks first at 10x

| Load | First limit | Evidence | Lever |
|---|---|---|---|
| more queries | cross-encoder CPU (~1 s per query, one call at a time) | docs/benchmarks.md | rerank depth 20, ONNX/int8, more sift-models replicas, cache hit rate |
| more documents | embedding throughput at ingest (model-bound) | ADR-0006 | batch across documents, replicas behind the `sift-embed` queue |
| more tenants / chunks | shared HNSW index (one tenant's vectors slow another's) | ADR-0012 | list-partition chunks by tenant past ~1M chunks |
| answer cost | LLM tokens (~3,000 in per answer) | docs/benchmarks.md | cache, gate-1 refusals, fewer sources, a smaller model |

## Known limits

- The judge's correctness score is not validated (kappa 0.489 < 0.6, ADR-0010); faithfulness is.
- Golden questions were drafted by a model and only partly verified by a person.
- On a 2-vCPU host the reranker allows well under one uncached query per second.
- PDF ingestion (Docling) needs ~5 GB of RAM and is not in the production image.
- Out of scope: agents, multi-turn chat, fine-tuning, reading images or tables as pictures.
