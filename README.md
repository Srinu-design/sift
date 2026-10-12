# Sift

Question answering over your documents that cites the section behind every claim, refuses when
the evidence is missing, and proves every retrieval change with a measured eval. Ingestion
runs as jobs on [Hopper](../hopper), my own job queue.

> **Status: weeks 1–2 of 7.** Corpus, golden set, schema, ingestion (in-process and through
> Hopper) and the naive retrieval baseline are done. Hybrid search, reranking, answers with
> citations, refusals, tenancy, the CI eval gate and deployment come next. Nothing below is
> claimed beyond what a test or a measured number shows.

## Baseline: dense retrieval only (`naive`)

Golden set **v1**, dev split, 71 answerable questions (15 unanswerable not scored for retrieval),
95% bootstrap intervals. `make eval` reproduces it; the committed copy is
[`eval/baselines/retrieval.json`](eval/baselines/retrieval.json).

| Strategy | hit@5 | hit@8 | recall@10 | MRR@10 |
|---|---|---|---|---|
| naive (bge-small, HNSW) | 0.887 [0.82, 0.96] | 0.930 [0.87, 0.99] | 0.958 [0.91, 0.99] | 0.747 [0.66, 0.83] |

| Question type | n | hit@5 | MRR@10 |
|---|---|---|---|
| factoid | 31 | 0.935 | 0.844 |
| procedural | 15 | 0.800 | 0.572 |
| version-specific | 10 | 0.700 | 0.514 |
| paraphrase | 8 | 1.000 | 0.771 |
| multi-section | 7 | 1.000 | 1.000 |

Most misses land on the right page but the wrong section, or are exact identifiers
(`ON_ERROR`, `JSON_EXISTS`, `IS JSON`) that dense search places near, not on: the case for
hybrid search in week 3. **The golden set is drafted, not yet human-verified**
([changelog](eval/golden/CHANGELOG.md)).

## What works now

| Feature | Proof |
|---|---|
| Corpus snapshot: PostgreSQL docs 16.15 / 17.11 / 18.6, 182 pages, SHA-256 manifest | [`corpus/`](corpus/README.md) |
| 123-question golden set, 17% unanswerable, frozen dev/test split, every anchor and quote checked against the corpus | `make golden-check`, [`eval/golden_check.py`](eval/golden_check.py) |
| HTML parser that keeps section anchors and heading paths; PDF parser (Docling) that keeps pages | [`test_html_parser.py`](tests/unit/test_html_parser.py), [`test_pdf_parser.py`](tests/unit/test_pdf_parser.py) |
| Structure-aware chunking: ~400 tokens, 15% overlap, code and tables never split when they fit, 512-token guard | [`test_chunker.py`](tests/unit/test_chunker.py) |
| Retrieval metrics (hit@k, recall@k, MRR@10, bootstrap CIs), unit-tested on toy rankings | [`test_metrics.py`](tests/unit/test_metrics.py) |
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
make eval            # naive retrieval on the dev split
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
- **No upload HTTP API yet:** the tenant must come from an API key, which arrives with tenancy
  in week 5. Until then a CLI submits files.
- **The `pdf` extra (Docling) is not in the Docker image yet:** about 400 MB more. PDFs in the
  container get a clear 422 until week 5 (tenant B) adds it.

## Layout

```
corpus/      fetch.py, manifest.jsonl, README (raw/ is git-ignored)
eval/        golden/v1.jsonl + CHANGELOG, golden_check.py, metrics.py, run.py, baselines/, results/
src/sift/    config, db, hopper (client), ingest/ (parsers, chunker, pipeline, service, corpus CLI),
             models_svc/ (encoder, /embed, /rerank), retrieval/ (dense), common/ (from Hopper)
migrations/  0001: every table and index
tests/       unit/, integration/ (real pgvector Postgres), e2e/ (the running stack)
docs/        ai-usage.md, adr/
```

Links: [AI usage](docs/ai-usage.md) · [ADRs](docs/adr/) · [corpus](corpus/README.md) · [golden set](eval/golden/CHANGELOG.md)
