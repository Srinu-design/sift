# Resume bullets (every number links to its measurement)

- Built **Sift**, a multi-tenant RAG service (FastAPI, Postgres 16 + pgvector, Redis) that answers
  from 24 regulatory PDFs and the PostgreSQL 16–18 docs with a citation for every claim, refusing
  **all 6** unanswerable questions on a held-out test split while answering 29 of 31 answerable
  ones ([eval report](eval-report.md)).
- Raised held-out **hit@5 from 0.839 to 0.968** over dense-only retrieval with hybrid search
  (full-text + vector, reciprocal-rank fusion) and a cross-encoder reranker, choosing each step
  with paired-bootstrap tests on a 123-question golden set ([ADR-0002](adr/0002-rrf-fusion.md),
  [ADR-0003](adr/0003-cross-encoder-rerank.md)).
- Made every pull request prove its retrieval quality: a CI gate rebuilds the index from a pinned
  corpus, runs four strategies and **blocked a change that raised MRR but lost 2.8 points of hit@8**
  ([CI gate](ci-gate-demo.md), [ADR-0009](adr/0009-eval-in-ci.md)).
- Isolated tenants with API keys and Postgres row-level security under a non-owner role; tests
  show one tenant asking another's text verbatim gets **zero** of its chunks under every strategy
  ([ADR-0012](adr/0012-tenant-isolation.md)); found and fixed a prompt injection that made the model
  refuse on a planted instruction ([ADR-0008](adr/0008-two-gate-refusal.md)).
- Ran ingestion as jobs on my own queue (Hopper: retries, dead-letter replay, worker-kill tests) and
  deployed both services to one EC2 host; a retrieval smoke test **caught a release that passed
  health checks and rolled it back in 72 s** ([deploy](deploy.md)).
- Load-tested with k6 and a stub LLM: **1 req/s under 3 s p95 with a tenant-scoped answer cache**
  (48% hits), 0.5 req/s without; identified the CPU cross-encoder as the first bottleneck and
  measured the alternatives ([benchmarks](benchmarks.md)).
