#!/usr/bin/env bash
# Deploy one release of Sift on the production host, with a retrieval smoke test and automatic
# rollback (the shape of Hopper's deploy.sh, smaller: Sift has one API replica and no Caddy of
# its own; Hopper's Caddy routes /sift/ to sift-api).
#
#   deploy.sh <git ref>    build sift:<sha> from /opt/sift/src at that ref and deploy it
#   deploy.sh --rollback   go back to the previous good release (no migrations)
#   deploy.sh --status     print the live and previous releases
#   deploy.sh --from-ssh   forced command of the CI deploy key (docs/deploy.md): the one word the
#                          key sent (a commit sha, --rollback or --status) arrives in
#                          SSH_ORIGINAL_COMMAND, and nothing else can run
#
# A release: 1. build the image from the checked-out ref (compose file and code always match)
# 2. migrate with the new image (migrations are backward compatible) 3. replace sift-models and
# sift-api and wait until healthy 4. smoke test: python -m sift.smoke (one question, gold section
# in the top 5, no LLM call). If 3 or 4 fail, the previous release comes back the same way and is
# smoke tested too. Exit codes: 0 deployed (or, for --rollback, rolled back), 1 failed and rolled
# back, 2 bad usage or failed before touching the live release, 3 the rollback failed as well.
set -euo pipefail
ROOT="${SIFT_ROOT:-/opt/sift}"
SRC="$ROOT/src"
WAIT="${HEALTH_TIMEOUT:-300}"   # sift-models loads two models on start
LOG="$ROOT/deploy-history.log"

log() { echo "$(date -u +%H:%M:%S) deploy: $*" | tee -a "$LOG" >&2; }
compose() { SIFT_IMAGE="sift:$1" SIFT_ROOT="$ROOT" docker compose -p sift -f "$SRC/docker/compose.prod.yaml" --env-file "$ROOT/.env" "${@:2}"; }

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
  # shellcheck disable=SC2012  # log names are a time stamp, a process id and a word
  ls -1t "$ROOT/logs" 2>/dev/null | tail -n +101 | xargs -r -I{} rm -f "$ROOT/logs/{}" || true
}

rollback() {  # $1: the release to go back to (default: the recorded previous release)
  local prev=${1:-$(cat "$ROOT/previous" 2>/dev/null)}
  [[ -n "$prev" ]] || { log "no previous release to roll back to"; return 3; }
  log "rolling back to $prev"
  # Its own compose file and scripts too, not the failed release's.
  git -C "$SRC" checkout -q --detach "$prev" || { log "cannot check out $prev"; return 3; }
  if start "$prev"; then
    echo "$prev" > "$ROOT/current"
    log "rolled back to $prev"
    cleanup
    return 0
  fi
  log "ROLLBACK FAILED: $prev did not pass the smoke test either"
  return 3
}

# The CI key's forced command (Hopper's deploy.sh does the same). What the key sent must be one
# word: a full commit sha, --rollback or --status, never a command. The deploy runs detached,
# writing to a log this session follows, so a dropped SSH connection cannot kill it halfway.
from_ssh() {
  local word="${SSH_ORIGINAL_COMMAND:-}" logfile pid code=0
  [[ "$word" == --status || "$word" == --rollback || "$word" =~ ^[0-9a-f]{40}$ ]] \
    || { echo "expected a commit sha, --rollback or --status, got: $(printf '%q' "$word")" >&2; exit 2; }
  [[ "$word" == --status ]] && exec "$BASH" "$0" --status
  mkdir -p "$ROOT/logs" || exit 2
  logfile="$ROOT/logs/$(date -u +%Y%m%dT%H%M%SZ)-$$-${word#--}.log"
  : >"$logfile"  # before tail opens it
  # Without SSH_ORIGINAL_COMMAND, the child can never take this path again.
  env -u SSH_ORIGINAL_COMMAND setsid --wait "$BASH" "$0" "$word" </dev/null >>"$logfile" 2>&1 &
  pid=$!
  tail -n +1 -f --pid="$pid" "$logfile" >&2 || true
  wait "$pid" || code=$?
  echo "this log is kept on the host: $logfile" >&2
  exit "$code"
}

case "${1:-}" in
  --from-ssh) from_ssh ;;
  --status)
    echo "current: $(cat "$ROOT/current" 2>/dev/null || echo none)"
    echo "previous: $(cat "$ROOT/previous" 2>/dev/null || echo none)"
    exit 0 ;;
esac
# CI and a person deploying by hand must never overlap.
exec 9>"$ROOT/.deploy.lock"
flock -n 9 || { log "another deploy is running"; exit 2; }

case "${1:-}" in
  --rollback) rollback; exit $? ;;
  ""|-*) echo "usage: deploy.sh <git ref> | --rollback | --status | --from-ssh" >&2; exit 2 ;;
esac

git -C "$SRC" fetch -q origin || { log "git fetch failed"; exit 2; }
# origin/<ref> first: the clone's own branches are never updated, so a bare "main" would
# resolve to whatever main was when the host cloned (a stale build once filled the disk).
sha=$(git -C "$SRC" rev-parse --verify -q "origin/$1^{commit}" || git -C "$SRC" rev-parse --verify -q "$1^{commit}") \
  || { log "unknown ref $1"; exit 2; }
log "release $sha ($1)"
git -C "$SRC" checkout -q --detach "$sha"
docker build -q -t "sift:$sha" -f "$SRC/docker/Dockerfile" "$SRC" >/dev/null || { log "build failed"; exit 2; }
out=$(compose "$sha" run --rm sift-migrate 2>&1) || { log "migrations failed"; echo "$out" >&2; exit 2; }
live=$(cat "$ROOT/current" 2>/dev/null || true)
if start "$sha"; then
  [[ -n "$live" && "$live" != "$sha" ]] && echo "$live" > "$ROOT/previous"
  echo "$sha" > "$ROOT/current"
  log "deployed $sha"
  cleanup
  exit 0
fi
log "release $sha failed"
# A failed redeploy of the live release goes back one further.
target=$live
[[ "$live" == "$sha" ]] && target=$(cat "$ROOT/previous" 2>/dev/null || true)
[[ -n "$target" && "$target" != "$sha" ]] || { log "nothing to roll back to"; exit 1; }
rollback "$target" && exit 1
exit 3
