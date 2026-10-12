# ADR-0001: Postgres (pgvector + full-text search) instead of a dedicated vector database

Status: Accepted · Date: 2026-10-12

## Context

Sift stores chunks, their embeddings and a full-text index, and every query filters by tenant,
collection, docs version, date and source. Ingestion swaps a document's chunks from one version
to the next and must never show a half-swapped document. Hopper already runs one Postgres; the
target host is a single small EC2 instance (about 4 GB of RAM). The corpus is small: 1,903
chunks today, tens of thousands at most for this project.

## Decision

One Postgres holds everything Sift searches: the `pgvector/pgvector:0.8.6-pg16` image (Hopper's
major version, so Hopper's schema and data are unchanged), with a `sift` database beside
Hopper's. Chunks carry their embedding (`vector(384)`, HNSW index, cosine), a generated
`tsvector` (GIN index; the heading weighted A, the content B) and copies of the filter columns,
so a filtered search reads one table. Retrieval SQL is hand-written, like Hopper's queue SQL.

Filtered vector search uses iterative index scans (`hnsw.iterative_scan = relaxed_order`,
pgvector 0.8.0+): with a plain HNSW scan, filters apply after the index returns its
`ef_search` candidates, so a filter that keeps one docs version in three returns short lists.

## Alternatives considered

- **Qdrant, Weaviate or Elasticsearch/OpenSearch.** Stronger at scale (quantisation, sharding,
  BM25 built in), but a second store that has to be kept in sync with Postgres, which still
  holds documents and versions: activation could no longer be one transaction, and tenant
  filters would live in two places. One more service to run on a 4 GB host.
- **pgvector with IVFFlat.** Smaller and faster to build, but recall depends on the list count
  and on rebuilding as data changes; HNSW needs no training step.
- **The `postgres:16` image plus a pgvector build step.** Possible, but the official pgvector
  image pins both versions in one tag.

## Consequences

- Activation (old chunks off, new on, `index_version` bumped) is one transaction, and tests
  prove queries see the old version until the commit and only the new one after.
- Tenant isolation is SQL: every query filters on `tenant_id`, and row-level security as a
  non-owner role is the second layer (week 5).
- One HNSW graph is shared by every tenant, so one tenant's vectors affect another's recall
  and speed. Fine at this size; see "revisit".
- The guide's tech-stack table names the `-pg18` tag; its delivery section says to keep
  Hopper's major version. The second wins: changing majors would break Hopper's existing data
  volume and its CI pin. Upgrading both to 18 is a separate, deliberate migration.

## When to revisit

- More than about 1 million chunks, or HNSW no longer fitting in memory on the host.
- Filtered queries whose exact-versus-HNSW overlap at 50 drops below 0.9 (the test arrives in
  week 3), or many tenants of very different sizes: then list-partition chunks by tenant, as
  pgvector's README suggests, before reaching for another store.
- Needing BM25 itself rather than `ts_rank_cd` (stretch: ParadeDB's `pg_search`).
