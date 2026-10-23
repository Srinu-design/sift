# ADR-0011: Answer cache keyed by index_version; no semantic cache

Status: Accepted · Date: 2026-10-23

## Context

An answer costs an embedding, two retrievals, a cross-encoder pass (~1.1 s on CPU, the slowest
stage, ADR-0003) and an LLM call. Users repeat questions. A cached answer must never be another
tenant's, never be stale after a document changes, and never be the answer to a different
question that merely looks similar.

## Decision

Three layers (`src/sift/cache.py`):

| Layer | Key | Store | Invalidation |
|---|---|---|---|
| Chunk embeddings | model + SHA-256 of text | Postgres `embedding_cache` | none: content-addressed |
| Query embeddings | model@revision + SHA-256 of the normalised query | Redis, 1 day | none: content-addressed |
| Answers | **tenant**, collection, index_version, strategy, k, filters, prompt version, model, SHA-256 of the normalised query | Redis, 24 h | activation bumps `collections.index_version`, so old keys are never read again |

- The answer key starts with the tenant id, and every part that changes the answer is in it.
  Normalisation is deliberately small: lower case, collapsed whitespace, trailing punctuation.
- Nothing is deleted on activation: old keys simply stop being asked for and expire.
- An answer refused for failed citations (a transient model failure) is not cached.
- Cache hits and gate-1 refusals make no LLM call, so they cost nothing and are never blocked by
  the tenant's daily budget; a cache hit still writes its `query_logs` row (`cache_hit = true`).
- Every Redis use fails open: with Redis down Sift answers uncached (tested).

## Alternatives considered

- **A semantic cache** (reuse the answer of a question whose embedding is close enough).
  Rejected: "How do I take an incremental backup in PostgreSQL 16?" and "... in 17?" are near
  neighbours with opposite answers (16 has no incremental backup). The golden set is full of
  such pairs; a similarity threshold would serve wrong-version answers that look right.
- **Invalidate by deleting keys on activation.** Needs a key index per collection and a race
  between delete and a concurrent write; bumping a version in the key has neither.
- **Cache in the API process.** Lost on restart and not shared between replicas.

## Consequences

- Measured: a repeated query 3.3 s -> 0.4 ms (live, laptop). Hit rate under the load-test mix
  (uniform over the 86 dev questions, repeats allowed) is in `docs/benchmarks.md`.
- Tested: tenant B never hits tenant A's cached answer; an index_version bump makes the old answer
  unreachable; Redis down still answers (`tests/integration/test_ops.py`).
- Any new input that changes an answer (a new setting, a reranker swap) must join the key, or
  the cache serves the old answer for up to 24 h. Prompt and model are in it; tau, rerank depth
  and context size beyond k are not, and a change to them should bump `PROMPT_VERSION`.

## When to revisit

- Hit rate on real traffic stays near zero: drop the answer cache and keep the embedding cache.
- A tenant needs answers older than 24 h to change on a schedule other than activation.
