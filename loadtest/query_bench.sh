#!/usr/bin/env bash
# Query load test (build guide, week 6): k6 at 0.5, 1, 2 and 5 requests/s for 2 minutes each,
# answer cache off and on, stub LLM (800 ms). Results: loadtest/results/<date>-query/.
# sift-models is restarted before every run: k6 abandons a request after 60 s but the server
# keeps working on it, so without a restart a saturated run's backlog spills into the next.
#   make load-query        (needs make up and SIFT_API_KEY in .env)
set -euo pipefail
cd "$(dirname "$0")/.."
out="loadtest/results/$(date -u +%Y%m%d)-query"
mkdir -p "$out"
compose=(docker compose -f docker/compose.yaml -f loadtest/compose.bench.yaml --env-file .env)
key=$(grep '^SIFT_API_KEY=' .env | cut -d= -f2-)
for cache in false true; do
  ANSWER_CACHE_ENABLED=$cache "${compose[@]}" up -d --no-deps sift-api >/dev/null
  until curl -sf http://127.0.0.1:8200/readyz >/dev/null; do sleep 1; done
  for rate in 30 60 120 300; do   # per minute
    docker compose -f docker/compose.yaml --env-file .env exec -T redis redis-cli -n 1 FLUSHDB >/dev/null
    docker compose -f docker/compose.yaml --env-file .env restart sift-models >/dev/null
    until curl -sf http://127.0.0.1:8200/readyz | grep -q '"status":"ok"'; do sleep 2; done
    sleep 20   # the models load on start; let the first call not pay for it
    start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    docker run --rm --network host --user "$(id -u):$(id -g)" -v "$PWD/loadtest:/work" -w /work \
      -e BASE=http://127.0.0.1:8200 -e KEY="$key" -e RATE=$rate -e DURATION="${DURATION:-2m}" \
      -e SUMMARY="results/$(basename "$out")/k6-cache-$cache-$rate.json" grafana/k6:2.3.0 \
      run --quiet k6/query.js || true   # a failed threshold is a result, not an error
    end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "{\"cache\": $cache, \"rate_per_min\": $rate, \"start\": \"$start\", \"end\": \"$end\"}" >> "$out/runs.jsonl"
  done
done
docker compose -f docker/compose.yaml --env-file .env up -d --no-deps sift-api >/dev/null  # back to normal
uv run python loadtest/query_report.py "$out"
