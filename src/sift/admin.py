"""Administrative CLI for managing tenants, collections, and API keys.

python -m sift.admin create-tenant acme --collection docs
python -m sift.admin create-key acme --name ci
python -m sift.admin revoke-key abcd1234
python -m sift.admin create-collection acme rfcs
python -m sift.admin list-keys acme
"""

import argparse
import asyncio
import sys
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from sift.auth import keys
from sift.config import get_settings
from sift.db import create_engine


async def _get_tenant_id(conn: AsyncConnection, name: str) -> UUID:
    tenant_id = await conn.scalar(
        text("SELECT id FROM tenants WHERE name = :name"),
        {"name": name},
    )
    if tenant_id is None:
        raise SystemExit(f"tenant {name!r} not found")
    return UUID(str(tenant_id))


async def _cmd_create_tenant(
    engine: AsyncEngine,
    pepper: bytes,
    name: str,
    collection: str,
) -> None:
    new_key = keys.generate(pepper)
    async with engine.begin() as conn:
        tenant_id = await conn.scalar(
            text(
                "INSERT INTO tenants (name) VALUES (:name)"
                " ON CONFLICT (name) DO NOTHING RETURNING id"
            ),
            {"name": name},
        )
        if tenant_id is None:
            raise SystemExit(f"tenant {name!r} already exists")
        collection_id = await conn.scalar(
            text("INSERT INTO collections (tenant_id, name) VALUES (:t, :c) RETURNING id"),
            {"t": tenant_id, "c": collection},
        )
        await conn.execute(
            text(
                "INSERT INTO api_keys (tenant_id, prefix, key_hash, name) "
                "VALUES (:t, :p, :h, 'initial')"
            ),
            {"t": tenant_id, "p": new_key.prefix, "h": new_key.key_hash},
        )

    print(f"tenant_id={tenant_id} collection_id={collection_id}", file=sys.stderr)
    print("Shown once; store it now.", file=sys.stderr)
    print(new_key.plaintext)


async def _cmd_create_key(
    engine: AsyncEngine,
    pepper: bytes,
    tenant_name: str,
    key_name: str,
) -> None:
    new_key = keys.generate(pepper)
    async with engine.begin() as conn:
        tenant_id = await _get_tenant_id(conn, tenant_name)
        await conn.execute(
            text(
                "INSERT INTO api_keys (tenant_id, prefix, key_hash, name) VALUES (:t, :p, :h, :n)"
            ),
            {"t": tenant_id, "p": new_key.prefix, "h": new_key.key_hash, "n": key_name},
        )

    print(f"tenant_id={tenant_id} prefix={new_key.prefix} name={key_name}", file=sys.stderr)
    print("Shown once; store it now.", file=sys.stderr)
    print(new_key.plaintext)


async def _cmd_revoke_key(engine: AsyncEngine, prefix: str) -> None:
    async with engine.begin() as conn:
        key_id = await conn.scalar(
            text(
                "UPDATE api_keys SET revoked_at = COALESCE(revoked_at, now()) "
                "WHERE prefix = :p RETURNING id"
            ),
            {"p": prefix},
        )
        if key_id is None:
            raise SystemExit(f"key with prefix {prefix!r} not found")
    print(f"revoked prefix={prefix}", file=sys.stderr)


async def _cmd_create_collection(
    engine: AsyncEngine,
    tenant_name: str,
    collection_name: str,
) -> None:
    async with engine.begin() as conn:
        tenant_id = await _get_tenant_id(conn, tenant_name)
        collection_id = await conn.scalar(
            text(
                "INSERT INTO collections (tenant_id, name) VALUES (:t, :c) "
                "ON CONFLICT (tenant_id, name) DO NOTHING RETURNING id"
            ),
            {"t": tenant_id, "c": collection_name},
        )
        if collection_id is None:
            raise SystemExit(
                f"collection {collection_name!r} already exists for tenant {tenant_name!r}"
            )
    print(collection_id)


async def _cmd_list_keys(engine: AsyncEngine, tenant_name: str) -> None:
    async with engine.connect() as conn:
        tenant_id = await _get_tenant_id(conn, tenant_name)
        rows = (
            await conn.execute(
                text(
                    "SELECT prefix, name, created_at, last_used_at, revoked_at "
                    "FROM api_keys WHERE tenant_id = :t ORDER BY created_at"
                ),
                {"t": tenant_id},
            )
        ).all()

    for r in rows:
        created = r.created_at.isoformat() if r.created_at else "-"
        last_used = r.last_used_at.isoformat() if r.last_used_at else "-"
        revoked = r.revoked_at.isoformat() if r.revoked_at else "-"
        print(f"{r.prefix}\t{r.name}\t{created}\t{last_used}\t{revoked}")


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    pepper_str = settings.api_key_pepper.get_secret_value()
    if not pepper_str:
        raise SystemExit("api_key_pepper must not be empty")
    pepper = pepper_str.encode()

    engine = create_engine(settings)
    try:
        if args.command == "create-tenant":
            await _cmd_create_tenant(engine, pepper, args.name, args.collection)
        elif args.command == "create-key":
            await _cmd_create_key(engine, pepper, args.tenant_name, args.name)
        elif args.command == "revoke-key":
            await _cmd_revoke_key(engine, args.prefix)
        elif args.command == "create-collection":
            await _cmd_create_collection(engine, args.tenant_name, args.collection_name)
        elif args.command == "list-keys":
            await _cmd_list_keys(engine, args.tenant_name)
    finally:
        await engine.dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sift.admin",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tenant = sub.add_parser("create-tenant")
    p_tenant.add_argument("name")
    p_tenant.add_argument("--collection", default="default")

    p_key = sub.add_parser("create-key")
    p_key.add_argument("tenant_name")
    p_key.add_argument("--name", default="default")

    p_revoke = sub.add_parser("revoke-key")
    p_revoke.add_argument("prefix")

    p_coll = sub.add_parser("create-collection")
    p_coll.add_argument("tenant_name")
    p_coll.add_argument("collection_name")

    p_list = sub.add_parser("list-keys")
    p_list.add_argument("tenant_name")

    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
