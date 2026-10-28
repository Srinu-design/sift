"""Whole-corpus ingestion through Hopper with 1, 2 and 4 embed workers.

    make ingest-bench                 (make up and make hopper-tenant first; ~15 minutes)
    python loadtest/ingest_bench.py --workers 1,2,4 --runs 2

Each run ingests every file in corpus/manifest.jsonl into a throwaway tenant, so the real
pgdocs data is untouched, and starts with an EMPTY embedding cache: otherwise the second run
would pay for no model calls and measure nothing. Emptying a cache loses nothing (live chunks
keep their vectors), but the script refuses any database that is not on this machine. The
embed workers go back to 2 at the end. Results land in loadtest/results/<date>-ingest/.
"""

import argparse
import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = [
    "docker",
    "compose",
    "-f",
    str(ROOT / "docker/compose.yaml"),
    "--env-file",
    str(ROOT / ".env"),
]
DEFAULT_EMBED_WORKERS = 2  # as in docker/compose.yaml


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout


def scale(n: int) -> None:
    run(
        *COMPOSE,
        "up",
        "-d",
        "--no-recreate",
        "--scale",
        f"hopper-embed-worker={n}",
        "hopper-embed-worker",
    )
    for _ in range(60):
        if len(run(*COMPOSE, "ps", "-q", "hopper-embed-worker").split()) == n:
            return
        time.sleep(1)
    raise SystemExit(f"could not scale the embed workers to {n}")


async def one_run(sift: AsyncEngine, hopper: AsyncEngine, workers: int) -> dict[str, Any]:
    tenant = f"bench-w{workers}-{uuid.uuid4().hex[:6]}"
    async with sift.begin() as conn:
        await conn.execute(text("TRUNCATE embedding_cache"))
    started_at = datetime.now(UTC)
    t0 = time.monotonic()
    done = await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            "-m",
            "sift.ingest.corpus",
            "--via",
            "hopper",
            "--tenant",
            tenant,
            "--collection",
            "docs",
            "--wait",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    seconds = round(time.monotonic() - t0, 1)
    submitted = re.search(r"submitted (\d+) files", done.stderr)
    async with sift.connect() as conn:
        tenant_id = await conn.scalar(text("SELECT id FROM tenants WHERE name = :n"), {"n": tenant})
        chunks, embedded, ready = (
            await conn.execute(
                text(
                    "SELECT count(*), count(embedding),"
                    " (SELECT count(*) FROM document_versions v JOIN documents d"
                    "  ON d.id = v.document_id WHERE d.tenant_id = :t AND v.status = 'ready')"
                    " FROM chunks WHERE tenant_id = :t"
                ),
                {"t": tenant_id},
            )
        ).one()
        model_texts = await conn.scalar(text("SELECT count(*) FROM embedding_cache"))
    async with hopper.connect() as conn:
        jobs = [
            dict(r._mapping)
            for r in await conn.execute(
                text(
                    "SELECT queue, count(*) AS jobs,"
                    " count(*) FILTER (WHERE attempts > 1) AS retried,"
                    " count(*) FILTER (WHERE status <> 'succeeded') AS not_succeeded"
                    " FROM jobs WHERE created_at >= :s"
                    " AND tenant_id = (SELECT id FROM tenants WHERE name = :h)"
                    " GROUP BY queue ORDER BY queue"
                ),
                {"s": started_at, "h": os.environ.get("SIFT_HOPPER_TENANT", "sift")},
            )
        ]
    async with sift.begin() as conn:  # leave nothing behind but the (refilled) cache
        for sql in (
            "DELETE FROM documents WHERE tenant_id = :t",
            "DELETE FROM collections WHERE tenant_id = :t",
            "DELETE FROM tenants WHERE id = :t",
        ):
            await conn.execute(text(sql), {"t": tenant_id})
    shutil.rmtree(
        ROOT / os.environ.get("FILE_STORE", "data/files") / str(tenant_id), ignore_errors=True
    )
    return {
        "embed_workers": workers,
        "seconds": seconds,
        "files": int(submitted.group(1)) if submitted else None,
        "versions_ready": ready,
        "chunks": chunks,
        "chunks_embedded": embedded,
        "texts_sent_to_model": model_texts,
        "chunks_per_second": round(chunks / seconds, 1),
        "hopper_jobs": jobs,
    }


def environment() -> dict[str, str]:
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    return {
        "host": f"{cpu}, {os.cpu_count()} threads",
        "os": f"{platform.system()} {platform.release()}",
        "docker": "Docker engine " + run("docker", "version", "-f", "{{.Server.Version}}").strip(),
        # No commit yet is a normal state for a new repo; record it rather than crash.
        "git": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
        or "no commit yet",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def summary(results: list[dict[str, Any]], env: dict[str, str]) -> str:
    lines = [
        "# Whole-corpus ingestion through Hopper",
        "",
        f"{env['host']}; {env['docker']}; one sift-models (one model call at a time),",
        "one sift-ingest, Hopper's 3 default workers x 20 slots for parse and activate,",
        "N embed workers x 1 slot on the sift-embed queue.",
        "The embedding cache is emptied before each run.",
        "",
        "| Embed workers | Run | Seconds | Chunks/s | Embedded | Retried jobs | Failed jobs |",
        "|---|---|---|---|---|---|---|",
    ]
    runs: dict[int, int] = {}
    for r in results:
        runs[r["embed_workers"]] = runs.get(r["embed_workers"], 0) + 1
        retried = sum(j["retried"] for j in r["hopper_jobs"])
        failed = sum(j["not_succeeded"] for j in r["hopper_jobs"])
        lines.append(
            f"| {r['embed_workers']} | {runs[r['embed_workers']]} | {r['seconds']} | "
            f"{r['chunks_per_second']} | {r['texts_sent_to_model']} | {retried} | {failed} |"
        )
    return "\n".join(lines) + "\n"


async def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--workers", default="1,2,4")
    p.add_argument("--runs", type=int, default=2)
    args = p.parse_args()
    url = make_url(os.environ["DATABASE_URL"])
    if url.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(f"refusing to empty the embedding cache on {url.host}: local stacks only")
    sift = create_async_engine(url)
    hopper = create_async_engine(url.set(database="hopper"))
    plan = [int(w) for _ in range(args.runs) for w in args.workers.split(",")]  # interleaved
    results = []
    try:
        for workers in plan:
            scale(workers)
            print(f"{workers} embed workers ...", file=sys.stderr, flush=True)
            results.append(await one_run(sift, hopper, workers))
            r = results[-1]
            print(
                f"  {r['seconds']}s, {r['chunks']} chunks, {r['texts_sent_to_model']} embedded",
                file=sys.stderr,
                flush=True,
            )
    finally:
        scale(DEFAULT_EMBED_WORKERS)
        await sift.dispose()
        await hopper.dispose()

    env = environment()
    out = ROOT / "loadtest/results" / f"{datetime.now(UTC):%Y-%m-%d}-ingest"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps({"environment": env, "runs": results}, indent=1) + "\n"
    )
    (out / "summary.md").write_text(summary(results, env))
    print(summary(results, env))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
