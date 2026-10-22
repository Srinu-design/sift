"""API key verification, prefix caching, and FastAPI authentication dependencies."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

import structlog
from fastapi import HTTPException, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.auth import keys

logger = structlog.get_logger()

AUTH_ERROR_DETAIL = "a valid API key is required: Authorization: Bearer sift_live_..."
TOUCH_INTERVAL_SECONDS = 60.0

SELECT_KEY_BY_PREFIX = """
SELECT id, tenant_id, key_hash, revoked_at IS NOT NULL AS revoked
FROM api_keys
WHERE prefix = :p
"""


@dataclass(frozen=True, slots=True)
class _CachedKey:
    id: UUID
    tenant_id: UUID
    key_hash: bytes
    revoked: bool
    expires_at: float


class ApiKeyAuthenticator:
    def __init__(
        self,
        engine: AsyncEngine,
        pepper: bytes,
        *,
        cache_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engine = engine
        self._pepper = pepper
        self._cache_seconds = cache_seconds
        self._clock = clock
        self._cache: dict[str, _CachedKey] = {}
        self._last_touch: dict[str, float] = {}

    def evict(self, prefix: str) -> None:
        self._cache.pop(prefix, None)

    async def authenticate(self, presented: str) -> UUID | None:
        parsed = keys.parse(presented)
        if parsed is None:
            return None
        prefix, secret = parsed

        now = self._clock()
        entry = self._cache.get(prefix)
        if entry is None or now >= entry.expires_at:
            async with self._engine.connect() as conn:
                row = (await conn.execute(text(SELECT_KEY_BY_PREFIX), {"p": prefix})).one_or_none()
            if row is None:
                self._cache.pop(prefix, None)
                return None
            entry = _CachedKey(
                id=row.id,
                tenant_id=row.tenant_id,
                key_hash=bytes(row.key_hash),
                revoked=bool(row.revoked),
                expires_at=now + self._cache_seconds,
            )
            self._cache[prefix] = entry

        valid = keys.matches(self._pepper, secret, entry.key_hash)
        if not valid or entry.revoked:
            return None

        last_touch = self._last_touch.get(prefix)
        if last_touch is None or (now - last_touch) >= TOUCH_INTERVAL_SECONDS:
            try:
                async with self._engine.begin() as conn:
                    await conn.execute(
                        text("UPDATE api_keys SET last_used_at = now() WHERE id = :id"),
                        {"id": entry.id},
                    )
                self._last_touch[prefix] = now
            except Exception as exc:
                logger.warning(
                    "api_key_last_used_update_failed",
                    prefix=prefix,
                    error=str(exc),
                )

        return entry.tenant_id


def bearer_token(request: Request) -> str | None:
    header = request.headers.get("Authorization")
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    stripped = token.strip()
    if scheme.lower() != "bearer" or not stripped:
        return None
    return stripped


async def current_tenant(request: Request) -> UUID:
    token = bearer_token(request)
    if token is None:
        raise HTTPException(
            status_code=401,
            detail=AUTH_ERROR_DETAIL,
            headers={"WWW-Authenticate": "Bearer"},
        )
    authenticator: ApiKeyAuthenticator = request.app.state.api_keys
    tenant_id = await authenticator.authenticate(token)
    if tenant_id is None:
        raise HTTPException(
            status_code=401,
            detail=AUTH_ERROR_DETAIL,
            headers={"WWW-Authenticate": "Bearer"},
        )
    structlog.contextvars.bind_contextvars(tenant_id=str(tenant_id))
    return tenant_id
