# Sift

Question answering over your documents that cites the section behind every claim, refuses when
the evidence is missing, and proves every retrieval change with a measured eval. Ingestion
runs as jobs on [Hopper](../hopper), my own job queue.

> **Status: weeks 1–4 of 7.** Corpus, golden set, schema, ingestion (in-process and through
> Hopper), four retrieval strategies with metadata filters, cited answers with two refusal
> gates, the query API and a demo page are done. Tenancy, the judge, the CI eval gate, caching
> and deployment come next. Nothing
> below is claimed beyond what a test or a measured number shows.

## Retrieval: four strategies on the dev split

Golden set **v1**, dev split, 71 answerable questions (15 unanswerable not scored for retrieval),
95% bootstrap intervals. `make eval` reproduces it; the committed copy is
[`eval/baselines/retrieval.json`](eval/baselines/retrieval.json).

| Strategy | hit@5 | hit@8 | recall@10 | MRR@10 |
|---|---|---|---|---|
| naive (dense, bge-small) | 0.887 [0.82, 0.96] | 0.930 [0.87, 0.99] | 0.958 [0.91, 0.99] | 0.747 [0.66, 0.83] |
| lexical (OR full-text) | 0.817 [0.72, 0.90] | 0.873 [0.79, 0.94] | 0.852 [0.77, 0.93] | 0.607 [0.51, 0.69] |
| hybrid (RRF, k = 60) | 0.887 [0.82, 0.96] | 0.958 [0.90, 1.00] | 0.951 [0.89, 0.99] | 0.738 [0.65, 0.82] |
| **hybrid_rerank** (cross-encoder, top 30) | 0.887 [0.80, 0.96] | **0.986** [0.96, 1.00] | 0.979 [0.94, 1.00] | **0.841** [0.76, 0.91] |

| Question type | n | naive hit@5 | hybrid hit@5 | naive MRR@10 | hybrid MRR@10 |
|---|---|---|---|---|---|
| factoid | 31 | 0.935 | 0.871 | 0.844 | 0.757 |
| procedural | 15 | 0.800 | 0.800 | 0.572 | 0.649 |
| version-specific | 10 | 0.700 | 1.000 | 0.514 | 0.712 |
| paraphrase | 8 | 1.000 | 0.875 | 0.771 | 0.719 |
| multi-section | 7 | 1.000 | 1.000 | 1.000 | 0.905 |

**Reranking is the first real gain:** hybrid_rerank (top 50) minus naive, paired bootstrap,
MRR@10 +0.086 [+0.018, +0.163] and hit@8 +0.056 [+0.014, +0.113]. Reranking the top 30 instead of
50 keeps the quality at 60% of the latency (p95 1.27 s against 1.98 s on the dev laptop's CPU):
ADR-[0003](docs/adr/0003-cross-encoder-rerank.md).

**Hybrid alone beats lexical** (MRR@10 +0.131 [+0.062, +0.203]) **but ties naive**
(hit@8 +0.028 [−0.028, +0.085], MRR@10 −0.009 [−0.093, +0.078]). It fixes exact identifiers
(`IS JSON`, `JSON_EXISTS`, `ON_ERROR`: version-specific hit@5 0.70 → 1.00) and demotes chunks
that dense ranks first but lexical misses (paraphrases). Equal-weight RRF lets two medium ranks
beat one first place, which is what the reranker fixes. Full analysis, and AND versus
OR lexical queries: ADR-[0002](docs/adr/0002-rrf-fusion.md). **The golden set is drafted, not
yet human-verified** ([changelog](eval/golden/CHANGELOG.md)).

## Answers: cited, or refused (dev split)

`make eval-answers`, 86 dev questions, `hybrid_rerank`, gemini-3.5-flash-lite, prompt `answer-v1`:

| | answered | refused |
|---|---|---|
| answerable (71) | 68 | 3 (all gate 1) |
| unanswerable (15) | 2 | 13 (gate 1: 4, gate 2: 9) |

- **Refusal recall 0.87, wrong refusals 4.2%.** Gate 1 (rerank score below τ = 0.19) is mostly
  a cost saver: questions about a feature missing from the asked version score 0.95+. The model
  (gate 2) catches 9 of the 11 it lets through and never refused an answerable question.
  ADR-[0008](docs/adr/0008-two-gate-refusal.md).
- The 2 answered unanswerable questions are faithful partial answers, not inventions.
- **Citation accuracy 0.75** (cited sources overlapping a gold section); 88% of answers cite at
  least one gold section. Of the 24 off-gold citations, 13 are on the gold page (another version's
  copy or a sibling section): the judge in week 5 decides whether they support the claim.
- 3 of 79 model calls needed the one retry; none ended as a citation failure. LLM p50 2.0 s,
  p95 11.6 s on the free tier. About 3,000 tokens in per question.

## What works now

| Feature | Proof |
|---|---|
| Corpus snapshot: PostgreSQL docs 16.15 / 17.11 / 18.6, 182 pages, SHA-256 manifest | [`corpus/`](corpus/README.md) |
| 123-question golden set, 17% unanswerable, frozen dev/test split, every anchor and quote checked against the corpus | `make golden-check`, [`eval/golden_check.py`](eval/golden_check.py) |
| HTML parser that keeps section anchors and heading paths; PDF parser (Docling) that keeps pages | [`test_html_parser.py`](tests/unit/test_html_parser.py), [`test_pdf_parser.py`](tests/unit/test_pdf_parser.py) |
| Structure-aware chunking: ~400 tokens, 15% overlap, code and tables never split when they fit, 512-token guard | [`test_chunker.py`](tests/unit/test_chunker.py) |
| Retrieval metrics (hit@k, recall@k, MRR@10, bootstrap CIs), unit-tested on toy rankings | [`test_metrics.py`](tests/unit/test_metrics.py) |
| Naive, lexical and hybrid search behind one `search()`, same hit shape; dense and lexical run concurrently, timed per stage | [`test_retrieval.py`](tests/integration/test_retrieval.py) |
| Lexical OR query from the question's lexemes: safe with apostrophes, punctuation, all-stopword questions and identifiers like `max_connections` | same |
| RRF: one list, disjoint lists, ties, a chunk found by one retriever only | [`test_fuse.py`](tests/unit/test_fuse.py) |
| Filters (version, date range, sources; unknown keys rejected) inside both retrievers, so a filtered top k is never short | [`test_filters.py`](tests/unit/test_filters.py), `test_retrieval.py` |
| Filtered HNSW keeps recall: forced index versus exact scan on 86 version-filtered dev questions, overlap at 50 mean **0.999**, min 0.92, no short lists | `make eval-overlap` |
| A second tenant with identical documents never appears in the first tenant's results, for every strategy | `test_retrieval.py` |
| Cross-encoder rerank of the fused top 30 on sift-models, scores mapped to 0–1 | [`test_rerank.py`](tests/unit/test_rerank.py), `make rerank-latency` |
| Answers as validated JSON from numbered sources; malformed JSON or a failed citation check gets one retry, then a refusal | [`test_answer.py`](tests/unit/test_answer.py) |
| Citation checker in code: unknown source ids, quotes not in their chunk (whitespace and small punctuation changes tolerated), sentences without a citation | [`test_citations.py`](tests/unit/test_citations.py) |
| Gate 1 refusal: best rerank score below τ = 0.19 refuses without calling the LLM; τ swept on dev (wrongly refuses 3 of 71 answerable, 4.2%) | `make eval-tau` |
| Gemini client over REST (key in a header, never the URL; 429/5xx/timeouts are retryable errors) and a stub LLM for tests and load tests | [`test_llm.py`](tests/unit/test_llm.py) |
| `POST /v1/query`: strategy, filters (unknown keys 422), cited answer or refusal with "you may want to read", per-stage timings, a `query_logs` row per query; 503 with no internals when a dependency is down | [`test_api.py`](tests/integration/test_api.py) |
| Demo page at `/`: strategy and version pickers, clickable citations; no inline script, strict CSP, model output rendered as text only | `make up`, then http://127.0.0.1:8200 |
| Ingestion as Hopper jobs: the whole corpus in **~85 s**, every job on its first attempt; 1, 2 and 4 embed workers measured (bound by the model server) | [benchmark](loadtest/results/2026-10-12-ingest/summary.md), ADR-[0006](docs/adr/0006-ingestion-on-hopper.md) |
| Re-uploading an unchanged file: zero jobs, zero embedding calls | [`test_pipeline.py`](tests/integration/test_pipeline.py) |
| Embedding cache: identical text across versions embedded once (1,903 chunks, 1,007 model calls) | same |
| A worker `kill`ed mid-embed: every chunk present once, every embedding there | [`test_hopper_ingest.py`](tests/e2e/test_hopper_ingest.py) |
| A dead parse job replays through Hopper's DLQ after the cause is fixed | same |
| New version swaps in atomically; an old version finishing late never wins | [`test_pipeline.py`](tests/integration/test_pipeline.py) |
| sift-ingest accepts only calls signed by Hopper with Sift's secret | [`test_ingest_service.py`](tests/integration/test_ingest_service.py) |

## Quickstart

Hopper must be cloned next to this repo (`../hopper`, branch `sift-ingest-support`).

```bash
make up              # Hopper + Sift in one Compose project (first start downloads the models)
make hopper-tenant   # Sift's tenant in Hopper: API key, signing secret, internal-host allowlist
make corpus          # fetch the raw docs listed in corpus/manifest.jsonl (once)
make ingest          # the whole corpus through Hopper; make ingest-sample for 10 pages
make eval            # all four strategies on the dev split
make eval-tau        # sweep the gate-1 refusal threshold on dev
make rerank-latency  # cross-encoder p50/p95 at 50 and 30 pairs
make eval-answers    # cited answers and refusals with the real model (GEMINI_API_KEY in .env)
make eval-compare A=eval/results/<a>.json B=eval/results/<b>.json   # paired bootstrap
make eval-overlap    # filtered HNSW versus exact search
make test            # unit + integration (needs Postgres from make up)
make test-e2e        # kill test and DLQ replay against the running stack
```

`make ingest-local` runs the same steps in one process, without Hopper (the week-1 path).

## How ingestion works

```
CLI (later: POST /v1/collections/{id}/documents)
  └─ store file under its SHA-256, upsert document, insert version (unchanged file: stop here)
     └─ Hopper job  parse:<version>        queue default     → sift-ingest /internal/parse
          └─ Hopper jobs embed:<version>:<n>  queue sift-embed  → sift-ingest /internal/embed → sift-models /embed
               └─ Hopper job activate:<version> queue default   → sift-ingest /internal/activate
```

Processes: `sift-models` (embedding model and cross-encoder, loaded once), `sift-ingest`
(internal only, signed calls), `sift-migrate`, plus Hopper's api, workers, scheduler and two
workers for the `sift-embed` queue. One Postgres (`pgvector/pgvector:0.8.6-pg16`) holds Hopper's
database and Sift's.

## Decisions, and where this departs from the build guide

- **Postgres 16, not 18:** same major as Hopper, as the guide's delivery section says (its stack
  table says pg18). ADR-[0001](docs/adr/0001-postgres-for-search.md).
- **A separate Hopper queue for embedding**, after the first run retry-stormed. ADR-0006.
- **`chunks.anchor_path`** (not in the guide's schema): every section id from the page root to
  the chunk, so the eval can tell "the gold section or one of its subsections".
- **`parser` in a version's unique key**, so a parser fix re-parses (the guide keys on chunker only).
- **Hopper changes** (branch `sift-ingest-support`): per-tenant internal-host allowlist (admin
  only, link-local still refused) and a per-job `timeout` on the http task.
- **On 1,903 chunks Postgres does not use the HNSW index:** it scans and sorts, which is
  exact, so today's naive numbers are exact-search numbers. The overlap check forces the index
  to prove it keeps recall under filters once the table is big enough for the planner to use it.
- **Gate 1 alone refuses only 4 of 15 unanswerable dev questions.** Most ask about a feature in
  a version that lacks it ("JSON_TABLE in 16"); the reranker scores the neighbouring sections
  above 0.95. Gate 2 (the model reading the sources) has to catch those; it is measured with the
  answer eval next.
- **The query API has no auth yet** (one dev tenant, port published on localhost only): API
  keys and row-level security arrive in week 5.
- **No upload HTTP API yet:** the tenant must come from an API key, which arrives with tenancy
  in week 5. Until then a CLI submits files.
- **The `pdf` extra (Docling) is not in the Docker image yet:** about 400 MB more. PDFs in the
  container get a clear 422 until week 5 (tenant B) adds it.

## Layout

```
corpus/      fetch.py, manifest.jsonl, README (raw/ is git-ignored)
eval/        golden/v1.jsonl + CHANGELOG, golden_check.py, metrics.py, run.py, compare.py,
             overlap.py, tau.py, answers.py, baselines/, results/
src/sift/    config, db, hopper (client), ingest/ (parsers, chunker, pipeline, service, corpus CLI),
             models_svc/ (encoder, /embed, /rerank), retrieval/ (dense, lexical, fuse,
             filters, rerank, search), answer/ (prompt, llm, citations, refusal, generate),
             api/ (query API, demo page), common/ (from Hopper)
migrations/  0001: every table and index
tests/       unit/, integration/ (real pgvector Postgres), e2e/ (the running stack)
docs/        ai-usage.md, adr/
```

Links: [AI usage](docs/ai-usage.md) · [ADRs](docs/adr/) · [corpus](corpus/README.md) · [golden set](eval/golden/CHANGELOG.md)
