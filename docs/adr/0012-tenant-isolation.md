# ADR-0012: Tenant isolation with filters plus row-level security, on a shared index

Status: Accepted · Date: 2026-10-23

## Context

Several tenants share one Postgres, one `chunks` table and one HNSW index. A tenant must never
retrieve, read or be cited another tenant's text, and an isolation bug is the worst bug Sift
can have: it leaks a customer's documents. Filters in application code are one forgotten
`WHERE` away from a leak.

## Decision

1. **The tenant comes from the API key, never from the request** (Hopper's rule). Keys are
   `sift_live_<prefix>_<secret>`; only HMAC-SHA256(pepper, secret) is stored
   (`src/sift/auth/keys.py`, copied from Hopper). One 401 for missing, malformed, unknown,
   revoked and wrong keys.
2. **Every query filters on tenant_id and collection_id** (both retrievers, before fusion).
3. **Row-level security as the second layer** on `chunks`, `documents`, `query_logs`, and on
   `document_versions` through its document (migration 0002). Policies compare `tenant_id` with
   `NULLIF(current_setting('app.tenant_id', true), '')::uuid`, set per transaction by
   `sift.db.set_tenant`. An unset tenant sees zero rows; it does not error.
4. **The API runs as `sift_app`, a non-owner role** (`SET ROLE` on every pooled connection),
   because RLS never applies to a table's owner. Ingestion, migrations, evals and the admin CLI
   stay the owner: internal paths, reached only through Hopper's signed jobs or a shell.
5. **Another tenant's document id is a 404, not a 403**: under RLS the row simply does not
   exist for the caller, so the API cannot even confirm it exists.

## Proof

`tests/integration/test_tenancy.py`, against Postgres, as `sift_app`:

- Tenant B, scoped to itself, counts 0 of tenant A's chunks, documents and versions even with
  `WHERE tenant_id = A`; with no tenant set every count is 0; writing a `query_logs` row for A
  while scoped to B is rejected by the policy's `WITH CHECK`.
- B asks a question copied word for word from A's page, under all four strategies: no citation
  or suggestion of A's comes back, and `search()` as B returns none of A's chunk ids.
- B requests A's document id: 404. Uploading to A's collection id with B's key: 404.

## Alternatives considered

- **Filters only.** One missed `WHERE` leaks. RLS makes that bug return nothing instead.
- **A schema or database per tenant.** Strong isolation, but migrations, the HNSW index and the
  connection pool multiply per tenant, for two tenants.
- **Partitioning chunks by tenant (list partitions, an HNSW index each).** pgvector's README
  suggests it for isolation at scale: one tenant's vectors then cannot slow another's searches.
  Worth it past roughly a million chunks or when one tenant dwarfs the others; today there are
  1,903.
- **FORCE ROW LEVEL SECURITY with the owner as the app role.** Possible, but ingestion would
  then need the tenant set in every step; the owner-for-internal-paths split is simpler.

## Consequences

- `SET ROLE` can be undone by `RESET ROLE` from inside the session, so RLS is a second layer
  against bugs, not against SQL injection. Production still works this way: the API connects as
  the tables' owner and runs `SET ROLE sift_app`, a `NOLOGIN` role (migration 0002); it has no
  login role of its own.
- The demo page queries one public tenant without a key, rate-limited per client IP by the same
  Redis token bucket as the API (`RateLimiter`). In production every request reaches sift-api
  from Caddy's address, so the demo's bucket is one bucket shared by every visitor
  (`docker/compose.prod.yaml`).
- Key lookups are cached for 60 s per API process: a revoked key keeps working on a replica for
  up to a minute (as in Hopper).

## When to revisit

- A tenant past ~1M chunks, or one tenant's queries slowing another's: list-partition chunks.
- Any path that must run as the owner on behalf of a tenant (none today).
