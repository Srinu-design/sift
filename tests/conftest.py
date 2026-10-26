import asyncio
import hashlib
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest
import redis.asyncio as redis_asyncio
from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from sift.answer.llm import LLMResult
from sift.retrieval.hit import Hit

# Tests default to the local Compose stack's Postgres; CI overrides DATABASE_URL. Each test
# gets its own database, so the dev data in "sift" is never touched.
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://hopper:hopper@127.0.0.1:5432/sift")
os.environ["FILE_STORE"] = tempfile.mkdtemp(prefix="sift-files-")
os.environ.setdefault("HOPPER_URL", "http://hopper.test")
os.environ.setdefault("MODELS_URL", "http://models.test")
os.environ.setdefault("HOPPER_API_KEY", "hop_test_key")
os.environ.setdefault("HOPPER_SIGNING_SECRET", "dGVzdC1zaWduaW5nLXNlY3JldC1ub3QtYS1zZWNyZXQ=")
# Redis db 15 is the tests' own; the redis_client fixture empties it around every test.
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/15")
os.environ.setdefault("METRICS_PORT", "0")

ROOT = Path(__file__).resolve().parent.parent


def _url(name: str) -> URL:
    return make_url(os.environ["DATABASE_URL"]).set(database=name)


async def _admin(sql: str) -> None:
    admin = create_async_engine(_url("postgres"), isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            await conn.execute(text(sql))
    finally:
        await admin.dispose()


@pytest.fixture(scope="session")
def migrated_template() -> Iterator[str]:
    """Migrate once with the real alembic CLI; each test clones this database."""
    name = f"sift_test_{uuid.uuid4().hex[:10]}_tmpl"
    asyncio.run(_admin(f'CREATE DATABASE "{name}"'))
    try:
        env = {**os.environ, "DATABASE_URL": _url(name).render_as_string(hide_password=False)}
        done = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        yield name
    finally:
        asyncio.run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


@pytest.fixture
async def engine(migrated_template: str) -> AsyncIterator[AsyncEngine]:
    name = f"sift_test_{uuid.uuid4().hex[:12]}"
    await _admin(f'CREATE DATABASE "{name}" TEMPLATE "{migrated_template}"')
    eng = create_async_engine(_url(name))
    try:
        yield eng
    finally:
        await eng.dispose()
        await _admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
async def redis_client() -> AsyncIterator["redis_asyncio.Redis"]:
    client = redis_asyncio.from_url(os.environ["REDIS_URL"])
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


class FakeEmbedder:
    """Deterministic 384-dim vectors from the text's hash; counts every text it is sent."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        out = []
        for t in texts:
            seed = hashlib.sha256(t.encode()).digest()
            raw = [(seed[i % 32] - 128) / 128 + i * 1e-4 for i in range(384)]
            norm = sum(x * x for x in raw) ** 0.5
            out.append([x / norm for x in raw])
        return out


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def make_hit(n: int = 1, **fields: Any) -> Hit:
    """Retrieval hit number n; keyword arguments replace any of its fields."""
    hit = Hit(
        chunk_id=n,
        score=0.9,
        heading=f"Heading {n}",
        anchor=f"SEC-{n}",
        anchor_path=(f"SEC-{n}",),
        page_start=None,
        page_end=None,
        doc_version="17",
        source="pgdocs",
        source_uri=f"https://docs.test/page-{n}.html",
        title=f"Doc {n}",
        content=f"Source {n} explains that shared_buffers controls memory caching.",
    )
    return replace(hit, **fields)


class ScriptedLLM:
    """Gives the replies in order; records the turns of every call."""

    model: str = "scripted-test"

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[tuple[Literal["user", "model"], str]]] = []

    async def generate(
        self,
        system: str,
        turns: list[tuple[Literal["user", "model"], str]],
    ) -> LLMResult:
        self.calls.append(list(turns))
        return LLMResult(text=self._replies.pop(0), tokens_in=100, tokens_out=25, model=self.model)
