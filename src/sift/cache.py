"""Redis-backed query embedding cache, exact answer cache, and per-tenant daily spend budget."""

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, time, timedelta
from typing import Any, Protocol
from uuid import UUID

import redis.asyncio as redis_asyncio

from sift.common.ratelimit.limiter import REDIS_DOWN
from sift.common.ratelimit.redis_health import RedisHealth
from sift.metrics import CACHE_HITS, CACHE_LOOKUPS
from sift.retrieval.filters import Filters

_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT = ".?!,;: "


class SupportsRedis(Protocol):
    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: str, *, ex: int | None = None) -> Any: ...
    async def incrbyfloat(self, key: str, amount: float) -> Any: ...
    async def expire(self, key: str, seconds: int) -> Any: ...


class SupportsHealth(Protocol):
    def usable(self) -> bool: ...
    def failed(self, what: str, exc: BaseException) -> None: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def normalize_query(question: str) -> str:
    collapsed = _WS_RE.sub(" ", question.lower()).strip()
    return collapsed.rstrip(_TRAILING_PUNCT)


class QueryEmbeddingCache:
    def __init__(
        self,
        redis: redis_asyncio.Redis | SupportsRedis,
        health: RedisHealth | SupportsHealth,
        *,
        model: str,
        ttl_seconds: int,
    ) -> None:
        self._redis: Any = redis
        self._health = health
        self._model = model
        self._ttl_seconds = ttl_seconds

    def _key(self, question: str) -> str:
        digest = hashlib.sha256(normalize_query(question).encode("utf-8")).hexdigest()
        return f"sift:qemb:{self._model}:{digest}"

    async def get(self, question: str) -> list[float] | None:
        CACHE_LOOKUPS.labels(layer="query_embedding").inc()
        if not self._health.usable():
            return None
        try:
            raw = await self._redis.get(self._key(question))
        except REDIS_DOWN as exc:
            self._health.failed("query_embedding_get", exc)
            return None
        if raw is None:
            return None
        CACHE_HITS.labels(layer="query_embedding").inc()
        items: list[Any] = json.loads(raw)
        return [float(x) for x in items]

    async def put(self, question: str, vector: list[float]) -> None:
        if not self._health.usable():
            return
        try:
            await self._redis.set(
                self._key(question),
                json.dumps(vector),
                ex=self._ttl_seconds,
            )
        except REDIS_DOWN as exc:
            self._health.failed("query_embedding_put", exc)


def answer_key(
    *,
    tenant_id: UUID,
    collection_id: UUID,
    index_version: int,
    strategy: str,
    filters: Filters | Mapping[str, Any] | None,
    k: int,
    prompt_version: str,
    model: str,
    question: str,
) -> str:
    if filters is None:
        filters_json = "{}"
    elif isinstance(filters, Filters):
        filters_json = json.dumps(filters.model_dump(mode="json"), sort_keys=True)
    else:
        filters_json = json.dumps(dict(filters), sort_keys=True)
    filters_hash = hashlib.sha256(filters_json.encode("utf-8")).hexdigest()
    question_hash = hashlib.sha256(normalize_query(question).encode("utf-8")).hexdigest()
    return (
        f"sift:ans:{tenant_id}:{collection_id}:{index_version}:"
        f"{strategy}:{k}:{filters_hash}:{prompt_version}:{model}:{question_hash}"
    )


class AnswerCache:
    def __init__(
        self,
        redis: redis_asyncio.Redis | SupportsRedis,
        health: RedisHealth | SupportsHealth,
        *,
        ttl_seconds: int,
    ) -> None:
        self._redis: Any = redis
        self._health = health
        self._ttl_seconds = ttl_seconds

    async def get(self, key: str) -> dict[str, Any] | None:
        CACHE_LOOKUPS.labels(layer="answer").inc()
        if not self._health.usable():
            return None
        try:
            raw = await self._redis.get(key)
        except REDIS_DOWN as exc:
            self._health.failed("answer_cache_get", exc)
            return None
        if raw is None:
            return None
        CACHE_HITS.labels(layer="answer").inc()
        loaded: dict[str, Any] = json.loads(raw)
        return loaded

    async def put(self, key: str, value: dict[str, Any]) -> None:
        if not self._health.usable():
            return
        try:
            await self._redis.set(key, json.dumps(value), ex=self._ttl_seconds)
        except REDIS_DOWN as exc:
            self._health.failed("answer_cache_put", exc)


class SpendBudget:
    def __init__(
        self,
        redis: redis_asyncio.Redis | SupportsRedis,
        health: RedisHealth | SupportsHealth,
        *,
        daily_usd: float | None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._redis: Any = redis
        self._health = health
        self._daily_usd = daily_usd
        self._clock = clock

    def _now_utc(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            return now.replace(tzinfo=UTC)
        return now.astimezone(UTC)

    def _key(self, tenant_id: UUID, now: datetime) -> str:
        return f"sift:spend:{tenant_id}:{now.strftime('%Y-%m-%d')}"

    async def check(self, tenant_id: UUID) -> int | None:
        if self._daily_usd is None or not self._health.usable():
            return None
        now = self._now_utc()
        try:
            raw = await self._redis.get(self._key(tenant_id, now))
        except REDIS_DOWN as exc:
            self._health.failed("spend_budget_check", exc)
            return None
        spent = float(raw) if raw is not None else 0.0
        if spent < self._daily_usd:
            return None
        next_midnight = datetime.combine(
            now.date() + timedelta(days=1),
            time.min,
            tzinfo=UTC,
        )
        remaining = int((next_midnight - now).total_seconds())
        return max(1, remaining)

    async def add(self, tenant_id: UUID, usd: float) -> None:
        if usd <= 0.0 or not self._health.usable():
            return
        now = self._now_utc()
        key = self._key(tenant_id, now)
        try:
            await self._redis.incrbyfloat(key, usd)
            await self._redis.expire(key, 2 * 86_400)
        except REDIS_DOWN as exc:
            self._health.failed("spend_budget_add", exc)
