# Sift

Question answering over your documents that cites the section behind every claim, refuses when
the evidence is missing, and proves every retrieval change with a measured eval. Ingestion
runs as jobs on [Hopper](https://github.com/Srinu-design/hopper), my own job queue.

**Live demo:** http://3.104.222.203/sift/ (PostgreSQL 16–18 docs; one shared rate limit).
Design: [docs/design.md](docs/design.md) · Eval: [docs/eval-report.md](docs/eval-report.md) ·
Load: [docs/benchmarks.md](docs/benchmarks.md) · Deploy: [docs/deploy.md](docs/deploy.md) ·
12 [ADRs](docs/adr/). Nothing below is claimed beyond what a test or a measured number shows.

## Architecture

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

Each process and decision: [docs/design.md](docs/design.md#components).

## Results on the held-out test split (run once, at the end)

37 PostgreSQL questions never used for any choice (31 answerable, 6 unanswerable), plus 6 SEBI:

| | result |
|---|---|
| retrieval, `hybrid_rerank` | **hit@5 0.968** [0.90, 1.00], MRR@10 0.744 (dense-only: hit@5 0.839, MRR 0.681) |
| refusals | **6 of 6 unanswerable refused**; 2 of 31 answerable wrongly refused (6.5%) |
| citations | 83% of answers cite a gold section; every citation checked in code |
| faithfulness (LLM judge) | 1.000; the judge caught 5 of 5 planted false claims |
| correctness (LLM judge) | 0.897, **not yet trusted** (kappa 0.489 against 40 hand labels, ADR-0010) |
| tenant B (SEBI PDFs) | 5 of 5 answered with a page citation, 1 of 1 refused, prompt injection resisted |
| load (stub LLM, laptop CPU) | 0.5 req/s under 3 s p95 uncached, 1 req/s with the answer cache; the reranker saturates first |
| CI | the retrieval gate blocked a deliberately bad PR ([docs/ci-gate-demo.md](docs/ci-gate-demo.md)) |
| deploy | on EC2 next to Hopper; a broken release caught by the retrieval smoke test and rolled back in 72 s |

Error analysis of every test miss: the reranker demoted the gold section 8 times (mostly
version-specific questions), the answer used a weaker source 4 times, 1 embedding miss, 1 gate-1
threshold, 1 chunk boundary, 1 too-narrow gold label ([eval report](docs/eval-report.md)).

## Retrieval: four strategies on the dev split (how each choice was made)

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

`make eval-answers`, 86 dev questions, `hybrid_rerank`, gemini-3.5-flash-lite, prompt `answer-v2`:

| | answered | refused |
|---|---|---|
| answerable (71) | 67 | 4 (gate 1: 3, gate 2: 1) |
| unanswerable (15) | 2 | 13 (gate 1: 4, gate 2: 9) |

- **Refusal recall 0.87, wrong refusals 5.6%.** Gate 1 (rerank score below τ = 0.19) is mostly
  a cost saver: questions about a feature missing from the asked version score 0.95+, and the
  model (gate 2) catches 9 of the 11 it lets through. Its one refusal of an answerable question
  is a retrieval miss (the gold section was not among the sources). ADR-[0008](docs/adr/0008-two-gate-refusal.md).
- The 2 answered unanswerable questions are faithful partial answers that say what is missing.
- **94% of answers cite at least one gold section**; citation accuracy (cited sources overlapping
  a gold section) is 0.71, because many citations are the same section in another version or a
  sibling section: the judge decides whether they support the claim.
- 7 of 79 model calls needed the one retry; none ended as a citation failure. LLM p50 1.8 s,
  p95 10.5 s on the free tier. About 3,300 tokens in per question that reached the model.
- **Prompt injection:** prompt v1 obeyed one of three instructions planted in a tenant-B document
  (it refused a question it could answer). Prompt v2 fences the sources and restates the rule
  after them; all three injected questions now pass (ADR-0008).

## Generation quality: the LLM judge (dev split)

`make eval-judge`: every answered dev question judged by `gemini-3.1-flash-lite` at temperature
0 (`eval/judge.py`, rubric judge-v2), against the sources the answer model saw:

| | n | faithfulness | answer correctness |
|---|---|---|---|
| overall | 69 | 1.000 (every claim supported) | 0.895 [0.84, 0.94] |
| factoid | 30 | 1.000 | 0.967 |
| procedural | 14 | 1.000 | 0.786 |

- **Faithfulness is 1.000 because the answers stay inside their sources**: a false sentence
  planted in each of 5 answers was flagged 5 out of 5 times.
- **Correctness is not validated yet.** Against 40 hand labels (`make label`, `make kappa`) the
  judge agrees on 34 of 40, but Cohen's kappa is 0.489, under the 0.6 bar: 35 of the 40 labels
  are "fully correct", so chance agreement is high and kappa stays low. The rubric was also
  clarified after the first round of labels (grade against what the question asks), so the
  number is measured on the data that shaped it. ADR-[0010](docs/adr/0010-llm-judge.md) has the
  history and the next step: a fresh, stratified set of labels.
- The judge is not the stronger model the guide asks for (free-tier quotas; see the ADR).

## Ablations (dev, one change at a time)

[docs/eval-report.md](docs/eval-report.md): 400-token chunks with 15% overlap stay (500 tokens and
no overlap lose MRR significantly; 200 doubles the chunks for no significant gain). The version
filter is the biggest single factor (without it MRR falls 0.84 -> 0.53). A bigger reranker,
bge-reranker-v2-m3, lifts hit@5 from 0.887 to 0.986 but takes 34 s per query on CPU.

## Load, cost and monitoring

`make load-query` (k6, stub LLM, laptop CPU; full table in [docs/benchmarks.md](docs/benchmarks.md)):
**0.5 requests/s with p95 under 3 s without the answer cache, 1 request/s with it** (48% hits
under a uniform mix of the dev questions). The cross-encoder saturates first: one rerank takes
about a second of the whole CPU and sift-models runs one model call at a time. Postgres search
stays under ~60 ms at every rate. About 3,000 tokens in and 180 out per query
(~3.0 M / 0.18 M per 1,000 queries; free tier today, so $0).

Prometheus scrapes sift-api and sift-ingest next to Hopper; Grafana has a **Sift** dashboard
(p50/p95 per stage, cost per query, cache hit rate, refusals by gate, errors and budget 429s,
tokens, Hopper's ingestion backlog) next to Hopper's (`deploy/grafana/`,
`docker/hopper.override.yaml`; Hopper's files are mounted unchanged).

## Tenant B: SEBI circulars (PDF, page citations)

24 SEBI circulars from July–October 2026 (`corpus/fetch_sebi.py`, SHA-256 manifest), parsed by
Docling with page numbers, plus the prompt-injection fixture. Golden set `v2-sebi`: 18 questions
with page-level gold ([changelog](eval/golden/CHANGELOG.md)). Dev split, `hybrid_rerank`:

- Retrieval: hit@5 1.00 on 10 answerable questions; the corpus is small (266 chunks), so this
  says the pipeline works on PDFs, not that the ranking is good.
- Answers: 10 of 10 answerable answered, every one citing a gold page; both unanswerable refused,
  including a PostgreSQL question that tenant `pgdocs` can answer (the answer step never sees
  another tenant's chunks); injected questions pass.
- Ingestion: 15 minutes and about 5 GB of RAM for the 24 PDFs on the laptop CPU (Docling layout and
  table models), which is why the PDF parser is not in the 4 GB server image yet.
- RBI was the first choice; its file server answers scripted downloads with a CAPTCHA page, so
  tenant B uses SEBI, which serves the PDFs.

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
| API keys (`sift_live_...`, HMAC with a pepper, one 401 for every bad key); tenant always from the key | [`test_api_keys.py`](tests/unit/test_api_keys.py) |
| Row-level security as a second layer: the API runs as non-owner `sift_app`; tenant B sees 0 of A's chunks even with `WHERE tenant_id = A`; an unset tenant sees nothing | [`test_tenancy.py`](tests/integration/test_tenancy.py), ADR-[0012](docs/adr/0012-tenant-isolation.md) |
| Tenant B asks A's text word for word under all four strategies: none of A's chunks; A's document id is a 404 for B | same |
| Collections and uploads over the API (`POST /v1/collections/{id}/documents`, raw body, refused with 413 while streaming past 20 MB) ingested through Hopper; in production, which runs no ingestion, uploads answer 503 | same |
| Ingestion as Hopper jobs: the whole corpus in **~85 s**, every job on its first attempt; 1, 2 and 4 embed workers measured (bound by the model server) | [benchmark](loadtest/results/2026-10-12-ingest/summary.md), ADR-[0006](docs/adr/0006-ingestion-on-hopper.md) |
| Re-uploading an unchanged file: zero jobs, zero embedding calls | [`test_pipeline.py`](tests/integration/test_pipeline.py) |
| Embedding cache: identical text across versions embedded once (1,903 chunks, 1,007 model calls) | same |
| A worker `kill`ed mid-embed: every chunk present once, every embedding there | [`test_hopper_ingest.py`](tests/e2e/test_hopper_ingest.py) |
| A dead parse job replays through Hopper's DLQ after the cause is fixed | same |
| New version swaps in atomically and the old version's chunks are deleted (their vectors stay in `embedding_cache`); an old version finishing late never wins | [`test_pipeline.py`](tests/integration/test_pipeline.py) |
| sift-ingest accepts only calls signed by Hopper with Sift's secret | [`test_ingest_service.py`](tests/integration/test_ingest_service.py) |

## Quickstart

Hopper must be cloned next to this repo (`../hopper`, branch `main`).

```bash
make env             # .env from .env.example (and Hopper's); then set API_KEY_PEPPER in .env,
                     # for example to the output of: openssl rand -hex 32 (sift-api refuses an empty one)
make up              # Hopper + Sift in one Compose project (first start downloads the models)
make hopper-tenant   # Sift's tenant in Hopper: API key, signing secret, internal-host allowlist
make up              # again, so sift-ingest restarts with the key and secret now in .env
# the evaluated corpus snapshot, as CI gets it (make corpus re-crawls postgresql.org instead)
gh release download corpus-v1 --repo Srinu-design/sift --pattern corpus-raw-v1.tar.gz
tar -xzf corpus-raw-v1.tar.gz -C corpus/
uv run python corpus/verify.py corpus/manifest.jsonl   # every file matches its SHA-256
make ingest          # the whole corpus through Hopper (creates tenant pgdocs); make ingest-sample for 10 pages
make api-key TENANT=pgdocs   # a Sift API key for the tenant, printed once
make eval            # all four strategies on the dev split
make eval-tau        # sweep the gate-1 refusal threshold on dev
make rerank-latency  # cross-encoder p50/p95 at 50 and 30 pairs
make eval-answers    # cited answers and refusals with the real model (GEMINI_API_KEY in .env)
make load-query      # k6 load test with the stub LLM, cache off and on
make ingest-sebi     # tenant B: SEBI circular PDFs and the injection fixture, in-process (needs the pdf extra)
make eval-compare A=eval/results/<a>.json B=eval/results/<b>.json   # paired bootstrap
make eval-overlap    # filtered HNSW versus exact search
make test            # unit + integration (needs Postgres from make up)
make test-e2e        # kill test and DLQ replay against the running stack
```

`make ingest-local` runs the same steps in one process, without Hopper (the week-1 path).
`make corpus` builds a new snapshot: it re-crawls postgresql.org and overwrites
`corpus/manifest.jsonl`, so only run it on purpose ([corpus/README.md](corpus/README.md)).

## How ingestion works

```
CLI, or POST /v1/collections/{id}/documents
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
- **Hopper changes** (Srinu-design/hopper#14, #15): per-tenant internal-host allowlist (admin
  only, link-local still refused), a per-job `timeout` on the http task, the pgvector Postgres
  image (the same 16.15 build) and a `/sift/` route in Caddy.
- **On 1,903 chunks Postgres does not use the HNSW index:** it scans and sorts, which is
  exact, so today's naive numbers are exact-search numbers. The overlap check forces the index
  to prove it keeps recall under filters once the table is big enough for the planner to use it.
- **Gate 1 alone refuses only 4 of 15 unanswerable dev questions.** Most ask about a feature in
  a version that lacks it ("JSON_TABLE in 16"); the reranker scores the neighbouring sections
  above 0.95. Gate 2 (the model reading the sources) catches 9 of the 11 it lets through.
- **Ingestion, evals and the admin CLI run as the tables' owner**, which bypasses row-level
  security by design: they are internal paths (Hopper's signed jobs, a shell). Only the API runs
  as `sift_app` (ADR-0012).
- **The `pdf` extra (Docling) is not in the Docker image:** it needs ~5 GB of RAM to parse the
  SEBI PDFs, more than the 3.8 GB host; tenant B is ingested in-process (`make ingest-sebi`) and
  restored to production from a dump. PDFs uploaded to the container get a clear 422.
- **Judge and answer models are free-tier Gemini** (`gemini-3.5-flash-lite` answers,
  `gemini-3.1-flash-lite` judges), not the stronger judge the guide asks for (ADR-0010).

## Layout

```
corpus/      fetch.py, fetch_sebi.py, verify.py, manifests (*manifest.jsonl), fixtures/,
             README (raw/ is git-ignored)
eval/        golden/ (v1.jsonl, v2-sebi.jsonl, CHANGELOG), golden_check.py, metrics.py, run.py,
             compare.py, overlap.py, tau.py, answers.py, gate.py (the CI gate), judge.py,
             label.py, kappa.py, history.py, ablations.sh, baselines/, labels/, results/
src/sift/    config, db, hopper (client), ingest/ (parsers, chunker, pipeline, service, corpus CLI),
             models_svc/ (encoder, /embed, /rerank), retrieval/ (dense, lexical, fuse,
             filters, rerank, search), answer/ (prompt, llm, citations, refusal, generate),
             api/ (query API, demo page), auth/ (API keys), cache.py (Redis caches, budget),
             metrics.py, smoke.py (the deploy smoke test),
             admin.py (tenants, keys, collections), common/ (from Hopper)
migrations/  0001: every table and index; 0002: API keys, sift_app role, row-level security
scripts/     hopper_tenant.py (make hopper-tenant)
loadtest/    k6/ (query load), query_bench.sh, ingest_bench.py, rerank_latency.py, results/
deploy/      deploy.sh (release, smoke test, rollback), prometheus/, grafana/ (Sift dashboard)
docker/      Dockerfile, compose.yaml (dev, with Hopper), compose.prod.yaml, hopper.override.yaml
tests/       unit/, integration/ (real pgvector Postgres), e2e/ (the running stack)
docs/        design.md, eval-report.md, benchmarks.md, deploy.md, ci-gate-demo.md, ai-usage.md,
             resume-bullets.md, adr/
```

Links: [AI usage](docs/ai-usage.md) · [ADRs](docs/adr/) · [corpus](corpus/README.md) · [golden set](eval/golden/CHANGELOG.md)
