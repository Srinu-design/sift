# Targets: env, up, down, logs, corpus, migrate, hopper-tenant, ingest-local, ingest-sample,
# ingest, ingest-bench, eval, eval-baseline, golden-check, golden-review, test, test-e2e, lint, fmt.
# Host tools read settings from the environment only (like Hopper): export .env to them.
-include .env
export

COMPOSE := docker compose -f docker/compose.yaml --env-file .env
EVAL := uv run --extra models python -m eval.run

.PHONY: env up down logs corpus migrate hopper-tenant ingest-local ingest-sample ingest ingest-bench eval \
	eval-baseline golden-check golden-review test test-e2e lint fmt

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

# Retrieval eval on the dev split. Results land in eval/results/.
eval:
	$(EVAL) --strategy naive --split dev

# Deliberate: rewrites eval/baselines/retrieval.json, which a reviewer sees in the diff.
eval-baseline:
	$(EVAL) --strategy naive --split dev --write-baseline

# Whole-corpus ingestion with 1, 2 and 4 embed workers (empties the local embedding cache).
ingest-bench:
	uv run python loadtest/ingest_bench.py

# Every gold location must exist in the corpus snapshot.
golden-check:
	uv run python -m eval.golden_check

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
