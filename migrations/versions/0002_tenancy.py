"""Multi-tenant API keys, non-owner application role, and Row-Level Security policies.

The API runs as the non-owner role `sift_app` so PostgreSQL enforces Row-Level Security
on tenant-owned tables (`chunks`, `documents`, `document_versions`, `query_logs`).
Internal trusted processes (migrations, ingestion workers, evaluations, admin CLI)
continue running as the table owner and bypass RLS intentionally.
"""

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

STATEMENTS: list[str] = [
    """
    CREATE TABLE api_keys (
        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id uuid NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
        prefix text NOT NULL UNIQUE,
        key_hash bytea NOT NULL,
        name text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        last_used_at timestamptz,
        revoked_at timestamptz
    )
    """,
    # Roles are cluster-wide, so test databases (and Hopper's) share it: create it only once.
    """
    DO $$ BEGIN
      IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'sift_app') THEN
        CREATE ROLE sift_app NOLOGIN;
      END IF;
    END $$
    """,
    "GRANT sift_app TO CURRENT_USER",
    "GRANT USAGE ON SCHEMA public TO sift_app",
    (
        "GRANT SELECT ON tenants, collections, api_keys, chunks, documents, "
        "document_versions, query_logs TO sift_app"
    ),
    "GRANT INSERT ON collections, documents, document_versions, query_logs TO sift_app",
    "GRANT UPDATE ON documents, document_versions TO sift_app",
    "GRANT UPDATE (last_used_at) ON api_keys TO sift_app",
    "GRANT USAGE, SELECT ON SEQUENCE query_logs_id_seq TO sift_app",
    "ALTER TABLE chunks ENABLE ROW LEVEL SECURITY",
    """
    CREATE POLICY tenant_isolation ON chunks
        USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
        WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    """,
    "ALTER TABLE documents ENABLE ROW LEVEL SECURITY",
    """
    CREATE POLICY tenant_isolation ON documents
        USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
        WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    """,
    "ALTER TABLE query_logs ENABLE ROW LEVEL SECURITY",
    """
    CREATE POLICY tenant_isolation ON query_logs
        USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
        WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    """,
    "ALTER TABLE document_versions ENABLE ROW LEVEL SECURITY",
    """
    CREATE POLICY tenant_isolation ON document_versions
        USING (EXISTS (SELECT 1 FROM documents d WHERE d.id = document_id))
        WITH CHECK (EXISTS (SELECT 1 FROM documents d WHERE d.id = document_id))
    """,
]

DOWNGRADE_STATEMENTS: list[str] = [
    "DROP POLICY IF EXISTS tenant_isolation ON document_versions",
    "DROP POLICY IF EXISTS tenant_isolation ON query_logs",
    "DROP POLICY IF EXISTS tenant_isolation ON documents",
    "DROP POLICY IF EXISTS tenant_isolation ON chunks",
    "ALTER TABLE document_versions DISABLE ROW LEVEL SECURITY",
    "ALTER TABLE query_logs DISABLE ROW LEVEL SECURITY",
    "ALTER TABLE documents DISABLE ROW LEVEL SECURITY",
    "ALTER TABLE chunks DISABLE ROW LEVEL SECURITY",
    "REVOKE ALL ON ALL TABLES IN SCHEMA public FROM sift_app",
    "REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM sift_app",
    "REVOKE USAGE ON SCHEMA public FROM sift_app",
    "DROP TABLE IF EXISTS api_keys",
]


def upgrade() -> None:
    for stmt in STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    # Leave the cluster-wide sift_app role in place because other databases may use it.
    for stmt in DOWNGRADE_STATEMENTS:
        op.execute(stmt)
