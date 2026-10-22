import asyncio
import re
import sys
from typing import Any
from uuid import UUID

from sqlalchemy import event, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from sift.config import Settings, get_settings


def create_engine(settings: Settings, *, role: str | None = None) -> AsyncEngine:
    """With `role`, every new connection runs SET ROLE, so row-level security applies to it
    (it never applies to the tables' owner). The API runs as sift_app; ingest and evals don't."""
    engine = create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
    )
    if role is not None:
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", role):
            raise ValueError(f"unexpected role name {role!r}")

        @event.listens_for(engine.sync_engine, "connect")
        def _set_role(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute(f"SET ROLE {role}")
            cursor.close()

    return engine


async def set_tenant(conn: AsyncConnection, tenant_id: UUID) -> None:
    """Scope this transaction to one tenant for row-level security (transaction-local, so a
    pooled connection never carries it into the next request)."""
    await conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": str(tenant_id)})


def vector_literal(values: list[float]) -> str:
    """pgvector's text form; bound as a string and CAST(:v AS vector) in SQL."""
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


async def ensure_database(url: str) -> bool:
    """Create the database in `url` if it is missing. Sift shares Hopper's Postgres, and an
    init script would only run on a brand-new volume. True if it was created."""
    target = make_url(url)
    name = target.database
    assert name and name.replace("_", "").isalnum(), f"unexpected database name {name!r}"
    admin = create_async_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}
            )
            if exists:
                return False
            await conn.execute(text(f'CREATE DATABASE "{name}"'))
            return True
    finally:
        await admin.dispose()


if __name__ == "__main__":  # python -m sift.db: create the database, before alembic upgrade
    created = asyncio.run(ensure_database(get_settings().database_url))
    print("created database" if created else "database exists", file=sys.stderr)
