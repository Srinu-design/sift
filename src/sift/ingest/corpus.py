"""Ingest the snapshotted corpus (corpus/manifest.jsonl).

    python -m sift.ingest.corpus --via local     # week 1: every step in this process
    python -m sift.ingest.corpus --via hopper    # week 2: parse/embed/activate as Hopper jobs
    python -m sift.ingest.corpus --via hopper --versions 17 --limit 10 --wait

--wait polls until every submitted version is ready (or failed) and prints the time taken.
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import text

from sift.config import get_settings
from sift.db import create_engine
from sift.hopper import HopperBusy, enqueue_step
from sift.ingest import pipeline

TENANT = COLLECTION = "pgdocs"


def manifest(path: Path, versions: set[str] | None, limit: int | None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if versions is None or r["version"] in versions]
    return rows[:limit] if limit else rows


async def run(args: argparse.Namespace) -> int:
    engine = create_engine(get_settings())
    root = args.manifest.parent
    rows = manifest(
        args.manifest, set(args.versions.split(",")) if args.versions else None, args.limit
    )
    tenant_id, collection_id = await pipeline.ensure_collection(
        engine, args.tenant, args.collection
    )
    statuses: Counter[str] = Counter()
    submitted: list[UUID] = []
    started = time.monotonic()
    embed: pipeline.Embed | None = None
    if args.via == "local":
        from sift.models_svc.encoder import encoder

        def embed(texts: list[str]) -> Any:
            return asyncio.to_thread(encoder().embed, texts, "passage")

    async with httpx.AsyncClient(timeout=30) as http:
        for row in rows:
            s = await pipeline.submit(
                engine,
                tenant_id=tenant_id,
                collection_id=collection_id,
                filename=Path(row["path"]).name,
                data=(root / row["path"]).read_bytes(),
                source_uri=row["url"],
                source=row["source"],
                doc_version=row["version"],
            )
            statuses[s.status] += 1
            if not s.needs_parse:
                continue
            submitted.append(s.version_id)
            v = s.version_id
            if embed is not None:  # in-process, the same steps the jobs run
                for b in range(await pipeline.parse_version(engine, v)):
                    await pipeline.embed_batch(engine, v, b, embed)
                await pipeline.activate(engine, v)
            else:
                while True:
                    try:
                        await enqueue_step(http, "parse", {"version_id": str(v)}, f"parse:{v}")
                        break
                    except HopperBusy as exc:  # rate limited: wait as told, then go on
                        await asyncio.sleep(exc.retry_after or 1)
    print(f"submitted {len(rows)} files: {dict(statuses)}", file=sys.stderr)

    if args.wait and submitted:
        while True:
            async with engine.connect() as conn:
                result = await conn.execute(
                    text(
                        "SELECT status, count(*) FROM document_versions"
                        " WHERE id = ANY(:ids) GROUP BY status"
                    ),
                    {"ids": submitted},
                )
                counts: dict[str, int] = {r[0]: r[1] for r in result}
            done = counts.get("ready", 0) + counts.get("failed", 0) + counts.get("superseded", 0)
            print(f"  {time.monotonic() - started:6.1f}s {counts}", file=sys.stderr)
            if done == len(submitted):
                break
            await asyncio.sleep(2)
    print(f"took {time.monotonic() - started:.1f}s", file=sys.stderr)
    await engine.dispose()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python -m sift.ingest.corpus",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, default=Path("corpus/manifest.jsonl"))
    p.add_argument("--via", choices=["local", "hopper"], default="hopper")
    p.add_argument("--tenant", default=TENANT)
    p.add_argument("--collection", default=COLLECTION)
    p.add_argument("--versions", help="comma separated, e.g. 17 or 16,18")
    p.add_argument("--limit", type=int)
    p.add_argument("--wait", action="store_true")
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
