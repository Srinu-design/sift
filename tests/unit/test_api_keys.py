"""Unit tests for ApiKeyAuthenticator caching, revocation, and secret verification."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncEngine

from sift.auth import keys
from sift.auth.api_keys import ApiKeyAuthenticator

PEPPER = b"unit-test-pepper"


@dataclass
class _FakeRow:
    id: UUID
    tenant_id: UUID
    key_hash: bytes
    revoked: bool


class _FakeResult:
    def __init__(self, row: _FakeRow | None) -> None:
        self._row = row

    def one_or_none(self) -> _FakeRow | None:
        return self._row


class _FakeConn:
    def __init__(self, engine: "_FakeEngine") -> None:
        self._engine = engine

    async def execute(self, statement: Any, params: dict[str, Any] | None = None) -> _FakeResult:
        sql = str(statement)
        if "SELECT" in sql:
            self._engine.select_calls += 1
            prefix = (params or {}).get("p", "")
            return _FakeResult(self._engine.rows.get(prefix))
        if "UPDATE" in sql:
            self._engine.update_calls += 1
        return _FakeResult(None)


class _FakeEngine:
    def __init__(self) -> None:
        self.rows: dict[str, _FakeRow] = {}
        self.select_calls = 0
        self.update_calls = 0

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[_FakeConn]:
        yield _FakeConn(self)

    @asynccontextmanager
    async def begin(self) -> AsyncIterator[_FakeConn]:
        yield _FakeConn(self)


async def test_valid_wrong_malformed_unknown_and_revoked_keys() -> None:
    fake_engine = _FakeEngine()
    now = 1000.0
    auth = ApiKeyAuthenticator(
        cast(AsyncEngine, fake_engine),
        PEPPER,
        cache_seconds=60.0,
        clock=lambda: now,
    )

    tenant_id = uuid4()
    active_key = keys.generate(PEPPER)
    fake_engine.rows[active_key.prefix] = _FakeRow(
        id=uuid4(),
        tenant_id=tenant_id,
        key_hash=active_key.key_hash,
        revoked=False,
    )

    revoked_key = keys.generate(PEPPER)
    fake_engine.rows[revoked_key.prefix] = _FakeRow(
        id=uuid4(),
        tenant_id=tenant_id,
        key_hash=revoked_key.key_hash,
        revoked=True,
    )

    # Valid key resolves tenant_id
    assert await auth.authenticate(active_key.plaintext) == tenant_id
    assert fake_engine.select_calls == 1
    assert fake_engine.update_calls == 1

    # Wrong secret with valid prefix fails
    tampered = active_key.plaintext[:-4] + "XXXX"
    assert await auth.authenticate(tampered) is None

    # Malformed keys fail without querying DB
    before_selects = fake_engine.select_calls
    assert await auth.authenticate("") is None
    assert await auth.authenticate("not-a-sift-key") is None
    assert fake_engine.select_calls == before_selects

    # Unknown prefix fails
    unknown_key = keys.generate(PEPPER)
    assert await auth.authenticate(unknown_key.plaintext) is None

    # Revoked key fails
    assert await auth.authenticate(revoked_key.plaintext) is None


async def test_cache_serves_second_call_and_evict_forces_lookup() -> None:
    fake_engine = _FakeEngine()
    now = 500.0
    auth = ApiKeyAuthenticator(
        cast(AsyncEngine, fake_engine),
        PEPPER,
        cache_seconds=60.0,
        clock=lambda: now,
    )

    tenant_id = uuid4()
    new_key = keys.generate(PEPPER)
    fake_engine.rows[new_key.prefix] = _FakeRow(
        id=uuid4(),
        tenant_id=tenant_id,
        key_hash=new_key.key_hash,
        revoked=False,
    )

    assert await auth.authenticate(new_key.plaintext) == tenant_id
    assert fake_engine.select_calls == 1
    assert fake_engine.update_calls == 1

    # Second call within TTL uses cache and skips last_used_at update
    now += 10.0
    assert await auth.authenticate(new_key.plaintext) == tenant_id
    assert fake_engine.select_calls == 1
    assert fake_engine.update_calls == 1

    # Evicting the prefix forces a fresh DB lookup on the next call
    auth.evict(new_key.prefix)
    assert await auth.authenticate(new_key.plaintext) == tenant_id
    assert fake_engine.select_calls == 2

    # Advancing clock past 60s triggers both cache refresh and last_used_at update
    now += 61.0
    assert await auth.authenticate(new_key.plaintext) == tenant_id
    assert fake_engine.select_calls == 3
    assert fake_engine.update_calls == 2
