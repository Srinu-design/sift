# ADR-0006: Ingestion as Hopper http jobs; internal-host allowlist; a queue for embedding

Status: Accepted · Date: 2026-10-12

## Context

Ingesting a file is three steps of very different cost: parse and chunk (CPU, seconds), embed
(the model server, seconds per 64 chunks), activate (one transaction). Any step can crash or
time out, and a retry must never duplicate chunks or pay for an embedding twice. Hopper
(Project 1) already provides at-least-once delivery, retries with backoff, a dead-letter queue
and replay.

## Decision

1. **Every step is a Hopper `http` job** that calls an idempotent sift-ingest endpoint:
   `parse` (key `parse:<version>`), `embed` per 64-chunk batch (`embed:<version>:<batch>`),
   `activate` (`activate:<version>`). Hopper's idempotency keys make a repeated enqueue the
   same job, so several last-batch finishers create one activation.
2. **Idempotent steps:** chunks insert `ON CONFLICT (version_id, ordinal) DO NOTHING`, and the
   `embedding` status is set in the same transaction, before any embed job exists. Embeddings
   come from a content-addressed cache first. Activation is a guarded update under the
   document's row lock, and an older version that finishes late never replaces a newer one.
3. **Status codes tell Hopper what to do:** 422 for a file that cannot be parsed (permanent:
   straight to the DLQ; replay after the fix), 503 with `Retry-After` when sift-models, Hopper or
   Postgres is unavailable, 401 for a missing or bad `Hopper-Signature`. sift-ingest verifies
   the HMAC with Sift's tenant secret and rejects timestamps more than 5 minutes off.
4. **Hopper change: a per-tenant internal-host allowlist.** Hopper's SSRF guard refuses private
   addresses, and sift-ingest is one. A platform admin may now allow named hosts for one
   tenant. Those calls use a separate client whose guard still refuses link-local (the cloud
   metadata endpoint) and multicast, and which never shares pooled connections with other
   tenants' jobs. Host names only: IP literals and `localhost` are rejected.
5. **Hopper change: a per-job `timeout`** (up to 120 s) on the http task, because parsing a
   large page or PDF takes longer than Hopper's 10 s default. A total-time bound keeps every
   other job's behaviour exactly as before.
6. **Embedding has its own queue, `sift-embed`,** drained by Hopper workers sized to the model
   server (2 replicas × 1 slot). Parse and activate stay on `default`.

## Why the separate queue (measured)

The first whole-corpus run put embed jobs on `default`, where 3 workers × 20 slots ran up to
60 at once. sift-models runs one model call at a time, so calls queued behind each other past
sift-ingest's 60 s timeout; each 503 was retried, and the abandoned calls kept the model busy,
so the queue only grew. Jobs reached attempt 7 of 8. With the dedicated queue the same corpus
(182 pages, 1,903 chunks, 1,007 unique embeddings) ingested in **80.9 to 87.6 s with 1 to 4 embed
workers, every job on its first attempt** (`loadtest/results/2026-10-12-ingest/summary.md`). The
backlog now waits in Hopper's queue (Postgres), not in open HTTP connections.

## Alternatives considered

- **FastAPI background tasks.** Lost on a crash or deploy; no retries, no DLQ.
- **Celery or RQ.** A third queue system beside Hopper and Redis, for what Hopper already does.
- **Calling sift-models directly from Hopper.** Hopper would need to know about embeddings and
  the cache; sift-ingest keeps that logic in Sift.
- **A bigger timeout on the shared queue.** Hides the storm instead of removing it: 60 waiting
  calls still hold 60 connections and the model still does the abandoned work.
- **Limiting concurrency inside sift-ingest** (a semaphore). Works per process, but the backlog
  would sit in HTTP calls that Hopper times out; a queue is the right place for it.

## Consequences

- Proven by tests: an unchanged file creates no version, job or embedding call; a worker killed
  mid-embed (`make test-e2e`) ends with every chunk present once and embedded; a dead parse job
  replays after the cause is fixed; queries see only the old version until activation.
- Ingest throughput is bounded by sift-models by design, and measured: 87.5 s with 1 embed
  worker, 84.8 s with 2, 81.4 s with 4 (`loadtest/results/2026-10-12-ingest/`). Scale it with
  more sift-models replicas and more `sift-embed` slots together.
- Sift depends on two small Hopper changes (migration 0006, the admin route, the http task).

## When to revisit

- Embedding becomes the bottleneck of a real workload: batch across documents, or add model
  replicas behind the same queue.
- PDFs near the 120 s parse limit: split them into page-range jobs (50 pages each).
