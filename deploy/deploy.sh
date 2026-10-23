#!/usr/bin/env bash
# Deploy one release of Sift on the production host, with a retrieval smoke test and automatic
# rollback (the shape of Hopper's deploy.sh, smaller: Sift has one API replica and no Caddy of
# its own; Hopper's Caddy routes /sift/ to sift-api).
#
#   deploy.sh <git ref>    build sift:<sha> from /opt/sift/src at that ref and deploy it
#   deploy.sh --rollback   go back to the previous good release (no migrations)
#   deploy.sh --status     print the live and previous releases
#
# A release: 1. build the image from the checked-out ref (compose file and code always match)
# 2. migrate with the new image (migrations are backward compatible) 3. replace sift-models and
# sift-api and wait until healthy 4. smoke test: python -m sift.smoke (one question, gold section
# in the top 5, no LLM call). If 3 or 4 fail, the previous release comes back the same way and is
# smoke tested too. Exit codes: 0 deployed, 1 failed and rolled back, 2 bad usage or failed before
# touching the live release, 3 the rollback failed as well.
set -euo pipefail
ROOT="${SIFT_ROOT:-/opt/sift}"
SRC="$ROOT/src"
WAIT="${HEALTH_TIMEOUT:-300}"   # sift-models loads two models on start
LOG="$ROOT/deploy-history.log"

log() { echo "$(date -u +%H:%M:%S) deploy: $*" | tee -a "$LOG" >&2; }
compose() { SIFT_IMAGE="sift:$1" docker compose -p sift -f "$SRC/docker/compose.prod.yaml" --env-file "$ROOT/.env" "${@:2}"; }

healthy() {  # $1 container; waits for Docker's health status
  local deadline=$((SECONDS + WAIT))
  while ((SECONDS < deadline)); do
    [[ "$(docker inspect -f '{{.State.Health.Status}}' "$1" 2>/dev/null)" == healthy ]] && return 0
    sleep 3
  done
  return 1
}

start() {  # $1 sha: replace the services and smoke test them
  compose "$1" up -d sift-models sift-api || return 1
  healthy sift-sift-models-1 || { log "sift-models not healthy"; return 1; }
  healthy sift-sift-api-1 || { log "sift-api not healthy"; return 1; }
  compose "$1" exec -T sift-api python -m sift.smoke 2>&1 | tee -a "$LOG" >&2
  return "${PIPESTATUS[0]}"
}

cleanup() {  # keep the live and previous images only: each is ~2 GB on a 19 GB disk
  local keep=("$(cat "$ROOT/current" 2>/dev/null)" "$(cat "$ROOT/previous" 2>/dev/null)")
  docker image ls sift --format '{{.Tag}}' | grep -vxF -e "${keep[0]:-none}" -e "${keep[1]:-none}" \
    | xargs -r -I{} docker image rm -f "sift:{}" >/dev/null 2>&1 || true
  docker builder prune -f >/dev/null 2>&1 || true   # dangling build cache only
}

rollback() {  # $1: the release to go back to (default: the recorded previous release)
  local prev=${1:-$(cat "$ROOT/previous" 2>/dev/null)}
  [[ -n "$prev" ]] || { log "no previous release to roll back to"; return 3; }
  log "rolling back to $prev"
  if start "$prev"; then
    echo "$prev" > "$ROOT/current"
    log "rolled back to $prev"
    cleanup
    return 1
  fi
  log "ROLLBACK FAILED: $prev did not pass the smoke test either"
  return 3
}

case "${1:-}" in
  --status)
    echo "current: $(cat "$ROOT/current" 2>/dev/null || echo none)"
    echo "previous: $(cat "$ROOT/previous" 2>/dev/null || echo none)"
    exit 0 ;;
  --rollback) rollback; exit $? ;;
  ""|-*) echo "usage: deploy.sh <git ref> | --rollback | --status" >&2; exit 2 ;;
esac

git -C "$SRC" fetch -q origin || { log "git fetch failed"; exit 2; }
sha=$(git -C "$SRC" rev-parse --verify -q "$1^{commit}" || git -C "$SRC" rev-parse --verify -q "origin/$1^{commit}") \
  || { log "unknown ref $1"; exit 2; }
log "release $sha ($1)"
git -C "$SRC" checkout -q --detach "$sha"
docker build -q -t "sift:$sha" -f "$SRC/docker/Dockerfile" "$SRC" >/dev/null || { log "build failed"; exit 2; }
compose "$sha" run --rm sift-migrate >/dev/null 2>&1 || { log "migrations failed"; exit 2; }
live=$(cat "$ROOT/current" 2>/dev/null || true)
if start "$sha"; then
  [[ -n "$live" && "$live" != "$sha" ]] && echo "$live" > "$ROOT/previous"
  echo "$sha" > "$ROOT/current"
  log "deployed $sha"
  cleanup
  exit 0
fi
log "release $sha failed"
[[ -n "$live" && "$live" != "$sha" ]] || { log "nothing to roll back to"; exit 1; }
rollback "$live"
