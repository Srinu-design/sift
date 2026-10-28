"""initial schema: tenants, collections, documents, versions, chunks, caches, query logs

Revision ID: 0001
Revises:
Create Date: 2026-10-12

Hand-written SQL, like Hopper's, so every line can be read and defended. Chunks carry their
own filter columns (source, doc_version, published_on), so the hot query path reads one table.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    # Hopper's tenants live in Hopper's database, and a foreign key cannot cross databases,
    # so Sift keeps its own. API keys arrive with tenancy (week 5).
    """
    CREATE TABLE tenants (
      id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      name        text NOT NULL UNIQUE,
      created_at  timestamptz NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE collections (
      id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      tenant_id      uuid NOT NULL REFERENCES tenants(id),
      name           text NOT NULL,
      index_version  integer NOT NULL DEFAULT 1,  -- bumped on every activation; in cache keys
      UNIQUE (tenant_id, name)
    )
    """,
    # NULLS NOT DISTINCT: circulars have no doc_version, and plain UNIQUE treats every NULL
    # as different, so the same circular could be inserted twice.
    """
    CREATE TABLE documents (
      id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      tenant_id          uuid NOT NULL REFERENCES tenants(id),
      collection_id      uuid NOT NULL REFERENCES collections(id),
      source_uri         text NOT NULL,   -- original URL or file name, for citation links
      title              text NOT NULL,
      source             text NOT NULL,   -- chapter name, or 'sebi' / 'rbi'
      doc_version        text,            -- '16', '17', '18' for the Postgres docs
      published_on       date,            -- circular date
      active_version_id  uuid,
      UNIQUE NULLS NOT DISTINCT (collection_id, source_uri, doc_version)
    )
    """,
    """
    CREATE TABLE document_versions (
      id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      document_id   uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      content_hash  bytea NOT NULL,       -- SHA-256 of the raw file
      parser        text NOT NULL,        -- e.g. 'html-v1', 'docling-v2'
      chunker       text NOT NULL,        -- e.g. 'headings-400-15'
      status        text NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'parsing', 'embedding',
                                      'ready', 'superseded', 'failed')),
      chunk_count   integer,
      error         text,                 -- why parsing failed, when status = 'failed'
      created_at    timestamptz NOT NULL DEFAULT now(),
      UNIQUE (document_id, content_hash, parser, chunker)  -- a parser fix makes a new version
    )
    """,
    # anchor_path: section ids from the page root down to this chunk's section. The eval
    # counts a chunk as a gold hit when the gold section is anywhere on it, which is how
    # "the gold section or one of its subsections" is decided without re-parsing pages.
    """
    CREATE TABLE chunks (
      id             bigserial PRIMARY KEY,
      tenant_id      uuid NOT NULL,
      collection_id  uuid NOT NULL,
      version_id     uuid NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
      ordinal        integer NOT NULL,
      heading        text NOT NULL,       -- 'Indexes > Index Types > GIN'
      anchor         text,                -- HTML section id for citation links
      anchor_path    text[] NOT NULL DEFAULT '{}',
      page_start     integer,             -- PDF pages
      page_end       integer,
      content        text NOT NULL,
      token_count    integer NOT NULL,
      content_hash   bytea NOT NULL,      -- SHA-256 of the embedded text (heading + content)
      source         text NOT NULL,       -- filter columns copied from documents
      doc_version    text,
      published_on   date,
      is_active      boolean NOT NULL DEFAULT false,
      fts            tsvector GENERATED ALWAYS AS (
                       setweight(to_tsvector('english', heading), 'A') ||
                       setweight(to_tsvector('english', content), 'B')) STORED,
      embedding      vector(384),         -- NULL until its embed job runs
      embed_model    text,
      UNIQUE (version_id, ordinal)
    )
    """,
    # Keyed by text, never returned by any API: sharing it across tenants leaks nothing.
    """
    CREATE TABLE embedding_cache (
      model         text NOT NULL,
      content_hash  bytea NOT NULL,
      embedding     vector(384) NOT NULL,
      PRIMARY KEY (model, content_hash)
    )
    """,
    """
    CREATE TABLE query_logs (
      id                bigserial PRIMARY KEY,
      tenant_id         uuid NOT NULL,
      collection_id     uuid NOT NULL,
      request_id        text NOT NULL,
      strategy          text NOT NULL,     -- naive | lexical | hybrid | hybrid_rerank
      cache_hit         boolean NOT NULL,
      refused           boolean NOT NULL,
      refusal_gate      text,              -- 'retrieval' | 'model' | 'citation'
      top_rerank_score  real,
      timings_ms        jsonb NOT NULL,
      tokens_in         integer,
      tokens_out        integer,
      cost_usd          numeric(10,6),
      prompt_version    text NOT NULL,
      model             text,
      created_at        timestamptz NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX chunks_fts_idx ON chunks USING gin (fts)",
    # m 16, ef_construction 64 (the defaults). Built on an empty table here; a later
    # migration that rebuilds it must use CONCURRENTLY inside autocommit_block().
    "CREATE INDEX chunks_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops)",
    """
    CREATE INDEX chunks_filter_idx ON chunks (tenant_id, collection_id, doc_version, published_on)
      WHERE is_active
    """,
    "CREATE INDEX query_logs_tenant_idx ON query_logs (tenant_id, created_at DESC)",
]


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for table in (
        "query_logs",
        "embedding_cache",
        "chunks",
        "document_versions",
        "documents",
        "collections",
        "tenants",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table}")
