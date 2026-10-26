# Every target is in the .PHONY list below.
# Host tools read settings from the environment only (like Hopper): export .env to them.
-include .env
export

COMPOSE := docker compose -f docker/compose.yaml --env-file .env
EVAL := uv run --extra models python -m eval.run

.PHONY: env up down logs corpus migrate hopper-tenant api-key ingest-local ingest-sample ingest ingest-sebi ingest-bench eval \
	eval-baseline eval-compare eval-overlap eval-tau eval-answers eval-judge label kappa load-query ablations rerank-latency eval-history golden-check golden-review test test-e2e lint fmt

env:
	@test -f .env || cp .env.example .env
	@test -f ../hopper/.env || cp ../hopper/.env.example ../hopper/.env

# Sift and Hopper together. sift-ingest refuses to start until make hopper-tenant has run.
up: env
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

# Snapshot the PostgreSQL docs (corpus/fetch.py). Only to build a new snapshot: evals use
# the files listed in corpus/manifest.jsonl and never re-crawl.
corpus:
	uv run python corpus/fetch.py

migrate:
	uv run python -m sift.db
	uv run alembic upgrade head

# Sift's tenant in Hopper: platform admin, tenant, internal-host allowlist, API key.
hopper-tenant: env
	uv run python scripts/hopper_tenant.py

# Week 1 path: parse, chunk and embed in this process (no Hopper).
ingest-local:
	uv run --extra models python -m sift.ingest.corpus --via local

# Through Hopper (make up and make hopper-tenant first).
ingest-sample:
	uv run python -m sift.ingest.corpus --via hopper --versions 17 --limit 10 --wait

ingest:
	uv run python -m sift.ingest.corpus --via hopper --wait

# Tenant B: SEBI circulars as PDFs, in-process (Docling is not in the Docker image yet), and
# the prompt-injection fixture that v2-sebi's "injected" questions ask about.
ingest-sebi:
	uv run --extra models --extra pdf python -m sift.ingest.corpus --via local \
		--manifest corpus/sebi_manifest.jsonl --tenant sebi --collection circulars
	uv run --extra models --extra pdf python -m sift.ingest.corpus --via local \
		--manifest corpus/fixture_manifest.jsonl --tenant sebi --collection circulars

STRATEGIES := naive lexical hybrid hybrid_rerank

# A new API key for a tenant, printed once: make api-key TENANT=pgdocs
api-key:
	uv run python -m sift.admin create-key $(TENANT)

# Retrieval eval of every strategy on the dev split. Results land in eval/results/.
eval:
	for s in $(STRATEGIES); do $(EVAL) --strategy $$s --split dev || exit 1; done

# Deliberate: rewrites eval/baselines/retrieval.json, which a reviewer sees in the diff.
eval-baseline:
	for s in $(STRATEGIES); do $(EVAL) --strategy $$s --split dev --write-baseline || exit 1; done

# Paired bootstrap of two results files: make eval-compare A=eval/results/x.json B=eval/results/y.json
eval-compare:
	uv run python -m eval.compare $(A) $(B)

# Gate-1 threshold: the largest tau that wrongly refuses at most 5% of answerable dev questions.
eval-tau:
	uv run --extra models python -m eval.tau --split dev

# Cross-encoder latency at 50 and 30 pairs through sift-models (make up first).
rerank-latency:
	uv run python loadtest/rerank_latency.py

# Answers with the real model on dev (resumable; LLM_PROVIDER and GEMINI_API_KEY from .env).
eval-answers:
	uv run --extra models python -m eval.answers --split dev

# The LLM judge on the dev answers (faithfulness, correctness), then your hand labels and kappa.
eval-judge:
	uv run python -m eval.judge eval/results/answers-dev-v1-hybrid_rerank-answer-v2.jsonl \
		--golden eval/golden/v1.jsonl

label:
	uv run python -m eval.label eval/results/answers-dev-v1-hybrid_rerank-answer-v2.jsonl \
		--golden eval/golden/v1.jsonl

kappa:
	uv run python -m eval.kappa --labels eval/labels/human-v1.jsonl \
		--judged eval/results/answers-dev-v1-hybrid_rerank-answer-v2-judge-v2.jsonl

# Query load test with the stub LLM, cache off and on (16 minutes). Needs make up.
load-query:
	loadtest/query_bench.sh

# Retrieval ablations on dev, one change at a time (about an hour on a laptop CPU).
ablations:
	eval/ablations.sh

# Filtered dev questions: HNSW (iterative scan) against exact search, overlap at 50.
eval-overlap:
	uv run --extra models python -m eval.overlap --split dev

# Whole-corpus ingestion with 1, 2 and 4 embed workers (empties the local embedding cache).
ingest-bench:
	uv run python loadtest/ingest_bench.py

# Every gold location must exist in the corpus snapshot.
golden-check:
	uv run python -m eval.golden_check

# The demo page's Eval history tab: every dated run in eval/results/. Rerun after an eval.
eval-history:
	uv run python -m eval.history

# A checklist with a link to every gold section, for verifying the set by hand.
golden-review:
	uv run python -m eval.golden_check --review > eval/golden/review-v1.md
	@echo "wrote eval/golden/review-v1.md"

test:
	uv run pytest -q

# Needs make up, make hopper-tenant; kills Hopper workers mid-ingest.
test-e2e:
	SIFT_E2E=1 uv run pytest -q tests/e2e

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

fmt:
	uv run ruff format .
	uv run ruff check --fix .
