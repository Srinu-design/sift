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
RAM), and Prometheus scraping of Sift (Hopper's production Prometheus config is Hopper's).

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

## Record

Filled in from the host's deploy log after the first release and the broken-release drill.
