#!/usr/bin/env bash
# Retrieval ablations on the dev split, one change at a time (build guide, week 6), all against
# hybrid_rerank on the default index unless the change needs its own index. Index variants are
# ingested in-process into their own collections (pgdocs-<variant>) of tenant pgdocs.
#   make ablations        (an hour or so on a laptop CPU; needs Postgres from make up)
set -euo pipefail
cd "$(dirname "$0")/.."
eval_run() { uv run --extra models python -m eval.run --strategy hybrid_rerank --split dev "$@"; }
ingest() {  # $1: collection, then env assignments for the chunker
  local collection=$1; shift
  env "$@" uv run --extra models python -m sift.ingest.corpus --via local --collection "$collection"
}
variant() {  # $1: collection, rest: chunker env
  local collection=$1; shift
  ingest "$collection" "$@"
  env "$@" eval_run --collection "$collection" | tee -a eval/results/ablations.log
}
export -f eval_run ingest
: > eval/results/ablations.log
variant pgdocs-c200 CHUNK_TARGET_TOKENS=200 CHUNK_OVERLAP_TOKENS=30
# 800 (the guide's third size) cannot exist: bge-small reads 512 tokens and the chunker refuses
# longer chunks rather than have them silently truncated. 500 is the largest that fits.
variant pgdocs-c500 CHUNK_TARGET_TOKENS=500 CHUNK_OVERLAP_TOKENS=75
variant pgdocs-o0 CHUNK_OVERLAP_TOKENS=0
variant pgdocs-nohead CHUNK_EMBED_HEADING=0
eval_run --no-filters | tee -a eval/results/ablations.log
RERANK_MODEL=BAAI/bge-reranker-v2-m3 RERANK_REVISION=953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
  eval_run | tee -a eval/results/ablations.log
