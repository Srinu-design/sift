# ADR-0005: Embedding model: local bge-small-en-v1.5, not an API model

Status: Accepted · Date: 2026-10-23

## Context

Every chunk is embedded once and every query once. The embedder decides dense recall, the
vector size in Postgres, what an eval run costs, and whether CI can rebuild the index offline.

## Decision

`BAAI/bge-small-en-v1.5`, pinned to commit `5c38ec7c…` (MIT), 384 dimensions, cosine distance,
its retrieval instruction prefixed to queries only, run on CPU in sift-models
(`src/sift/models_svc/encoder.py`) and in-process for evals. Chunks are capped at its 512-token
limit (ADR-0004).

## Measured

- Dense-only (naive) on dev: hit@8 0.930, MRR@10 0.747; on test: hit@8 0.935, MRR 0.681. With
  hybrid search and the reranker on top: test hit@5 0.968.
- Whole corpus (1,903 chunks, 1,007 unique texts) through Hopper in ~85 s on the laptop
  (bounded by the model server, ADR-0006); in CI 158 s in-process.
- An embedding cache keyed by model and SHA-256 of the text (Postgres, ADR-0011) makes
  re-ingesting unchanged text free; a query-embedding cache in Redis does the same for queries.

## Alternatives considered

- **An API embedding model** (OpenAI, Gemini, Voyage). Possibly better recall, but every
  re-ingest and every eval run in CI costs money and network, the vector size changes the schema,
  and a model the provider retires changes the numbers. Not run: it is the one ablation of the
  build guide that needs a paid key.
- **bge-base / bge-large.** Better on benchmarks, 768/1024 dimensions and 3-10x slower on CPU,
  next to a cross-encoder that is already the bottleneck (docs/benchmarks.md).
- **Unpinned model versions.** A silent upstream update would change every eval number; the
  revision is part of every results file and of the CI index cache key.

## Consequences

- The 512-token limit rules out larger chunks (the guide's 800-token ablation cannot exist).
- Embedding is ~1% of query latency; the reranker is ~99% of retrieval time.

## When to revisit

- Dense recall becomes the limiting factor in the error analysis (today: 1 embedding miss of 17
  test problems, docs/eval-report.md), or a GPU makes a larger model cheap.
