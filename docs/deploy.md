# Deploying Sift

Sift runs on the same EC2 host as Hopper (2 vCPU, 3.8 GB RAM, ap-southeast-2), as its own Compose
project (`sift`) on Hopper's Docker network, reached through Hopper's Caddy at
**`http://<host>/sift/`**. Hopper's side (one PR, Srinu-design/hopper#15): the Postgres image
became `pgvector/pgvector:0.8.6-pg16-trixie` (the same Postgres build as `postgres:16.15`, plus the
extension) and Caddy routes `/sift/` to `sift-api`.

| Process | Memory limit | Notes |
|---|---|---|
| sift-models | 1.5 GB | bge-small + MiniLM cross-encoder, CPU torch, loads once |
| sift-api | 400 MB | query API and demo page |
| sift-migrate | (one-off) | creates the `sift` database and runs Alembic |

Not deployed: sift-ingest (the index is restored from a dump, below; PDF parsing needs ~5 GB of
RAM), so uploads answer 503 "document uploads are unavailable" in production. Hopper's production
Prometheus scrapes sift-api, and its Grafana has the Sift dashboard (Srinu-design/hopper
commit 2e7cf53).

## One-time setup on the host

```bash
sudo mkdir -p /opt/sift && sudo chown ubuntu: /opt/sift
git clone https://github.com/Srinu-design/sift /opt/sift/src
# /opt/sift/.env (chmod 600): DATABASE_URL to Hopper's Postgres (database "sift"),
# API_KEY_PEPPER (random, generated on the host), LLM_PROVIDER, GEMINI_API_KEY
```

The search index is not rebuilt on the host (embedding 1,900+ chunks on 2 vCPUs, and Docling for
the PDFs, would take long and need more RAM than the host has). It is restored from a data-only
dump of the two tenants (`pgdocs`, `sebi`) taken from the evaluated local database:
`gunzip -c index.sql.gz | docker exec -i hopper-postgres-1 psql -U hopper -d sift`.

## A release

`/opt/sift/src/deploy/deploy.sh <git ref>` (exit codes and steps in the script's header):

1. build `sift:<sha>` from the checked-out ref (the compose file and the code always match);
2. run the migrations with it (backward compatible, so the live release keeps working);
3. replace sift-models and sift-api and wait until Docker reports them healthy;
4. **retrieval smoke test**, `python -m sift.smoke` in the new sift-api: one known question
   ("Can CREATE INDEX CONCURRENTLY run inside a transaction block?", version 17) must find the
   gold section in the top 5 through the live embedder, index and reranker. No LLM call, so a
   deploy never depends on the answer model's quota.

If 3 or 4 fail, the previous release comes back the same way and is smoke tested too.
`deploy.sh --rollback` and `deploy.sh --status` do what they say. Every step is logged with a
time stamp to `/opt/sift/deploy-history.log`.

## Deploying from CI

`.github/workflows/deploy.yml` deploys every push to main once `ci` (lint, tests, the retrieval
gate) is green, and on demand any sha or `--rollback` (Actions > deploy > Run workflow). It runs
`ssh -- sift <sha>` with a key that can do one thing: its line in the host's
`~/.ssh/authorized_keys` is

```
restrict,command="/opt/sift/src/deploy/deploy.sh --from-ssh" ssh-ed25519 ... sift-github-deploy
```

so whatever the key sends arrives as `SSH_ORIGINAL_COMMAND`, and `deploy.sh --from-ssh` accepts
only a full commit sha, `--rollback` or `--status` (`tests/unit/test_deploy_script.py`). The
deploy runs detached with its log in `/opt/sift/logs/`, so a dropped connection cannot stop it
halfway, and a lock keeps CI and a hand deploy from overlapping.

One-time setup (the private key goes from the owner's machine to GitHub, never through tooling):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/sift-deploy -N "" -C sift-github-deploy
# on the host: append the restrict,command=... line above with the .pub key
gh secret set EC2_SSH_KEY -R Srinu-design/sift < ~/.ssh/sift-deploy
ssh-keygen -F 3.104.222.203 | grep -v '^#' | gh secret set EC2_KNOWN_HOSTS -R Srinu-design/sift
gh variable set EC2_HOST -R Srinu-design/sift -b 3.104.222.203
gh variable set PUBLIC_URL -R Srinu-design/sift -b http://3.104.222.203/sift/
```

## Record (2026-10-23, from `/opt/sift/deploy-history.log`)

**Hopper side:** Srinu-design/hopper#15 merged, Hopper's own pipeline deployed it (CI, then
`deploy.sh` over SSH, smoke test, public URL check: all green). Postgres came back on
`pgvector/pgvector:0.8.6-pg16-trixie`, healthy, Hopper's `/readyz` 200 throughout.

**First release:** database created and migrated, index restored (2 tenants, 207 documents,
2,173 active chunks, none without an embedding), then:

```
12:13:39 deploy: release 8bfeffcc… (week7-final)
smoke pass: top 5 anchors ['SQL-CREATEINDEX-CONCURRENTLY', …], rerank 3232 ms
12:14:32 deploy: deployed 8bfeffcc…
```

`http://3.104.222.203/sift/` serves the page; a demo query answered in 3.8 s (rerank ~3.2–3.5 s
on 2 vCPUs, against ~1.1 s on the laptop). Memory: sift-models 476 MB, sift-api 78 MB; 1.6 GB of
the host's 3.8 GB still available.

**Incident: the second release filled the disk.** The image copied the whole Python environment
(~2 GB, with Sift installed into it) as one layer, so every release, even a one-line change,
added ~2 GB, and Docker keeps each image both compressed and unpacked. The next build stopped
with "no space left on device" (19 GB disk, 100% for a moment). What held: the build fails before
the live release is touched, so Sift kept serving and Hopper's Postgres stayed healthy (checked
at once; build cache and half-built images pruned back to 6 GB free). The fix (commit 8cbfa91):
dependencies in their own layer, which changes only with `uv.lock`, and Sift's code (324 kB) on
`PYTHONPATH`; `deploy.sh` now keeps only the live and previous images. The first build of the new
layout still needed ~6 GB, so Sift (not Hopper) was taken down for 3.5 minutes and its old image
removed. Lesson: measure an image's footprint per release, not its size.

**Broken release, rolled back.** A branch with a one-character bug (the reranker sorting
ascending: it builds, starts and passes health checks, but puts the worst passages first):

```
12:29:24 deploy: release 68d84966… (demo-broken-release)
smoke FAIL: top 5 anchors ['INDEXES-INDEX-ONLY-SCANS', 'id-1.9.3.69.8', 'id-1.9.3.178.8', …]
12:30:08 deploy: release 68d84966… failed
12:30:08 deploy: rolling back to 8cbfa919…
smoke pass: top 5 anchors ['SQL-CREATEINDEX-CONCURRENTLY', …]
12:30:36 deploy: rolled back to 8cbfa919…
```

72 seconds from release to rolled back; the public page kept answering, and the build reused the
cached dependency layer (disk unchanged). Health checks alone would have shipped this bug: only a
check of *what retrieval returns* catches it.

**A second disk scare, same day:** `deploy.sh main` resolved the host clone's *local* `main`
(still the commit from clone time, with the old 2 GB image layout) before `origin/main`, and its
build filled the disk again. The live release was untouched (exit 2: the build fails before any
swap); space was freed at once. `deploy.sh` now resolves `origin/<ref>` first.

## Turning on real answers

Done on 2026-10-25: the host answers with Gemini. The key is entered by the owner, never by
tooling:

```bash
ssh -i ~/.ssh/hopper-login.pem ubuntu@3.104.222.203
sed -i 's/^LLM_PROVIDER=stub/LLM_PROVIDER=gemini/' /opt/sift/.env
read -rsp 'Gemini API key: ' k && echo "GEMINI_API_KEY=$k" >> /opt/sift/.env && unset k
cd /opt/sift && SIFT_IMAGE=sift:$(cat current) docker compose -p sift -f src/docker/compose.prod.yaml --env-file .env up -d --force-recreate sift-api
```
