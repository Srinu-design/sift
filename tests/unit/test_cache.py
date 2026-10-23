"""Unit tests for query normalization, cache key construction, and spend budget checks."""

from datetime import UTC, datetime
from uuid import uuid4

from sift.cache import (
    AnswerCache,
    QueryEmbeddingCache,
    SpendBudget,
    answer_key,
    normalize_query,
)
from sift.retrieval.filters import Filters


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int | None] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        self.store[key] = value
        self.ttls[key] = ex

    async def incrbyfloat(self, key: str, amount: float) -> float:
        updated = float(self.store.get(key, "0.0")) + amount
        self.store[key] = str(updated)
        return updated

    async def expire(self, key: str, seconds: int) -> None:
        self.ttls[key] = seconds


class FakeHealth:
    def usable(self) -> bool:
        return True

    def failed(self, what: str, exc: BaseException) -> None:
        pass


def test_normalize_query() -> None:
    assert normalize_query("  What IS  MVCC?? ") == "what is mvcc"
    assert normalize_query("How does VACUUM work?!,;: ") == "how does vacuum work"


def test_answer_key_tenant_prefix_and_sensitivity() -> None:
    tenant_id = uuid4()
    collection_id = uuid4()
    base = answer_key(
        tenant_id=tenant_id,
        collection_id=collection_id,
        index_version=1,
        strategy="hybrid_rerank",
        filters=Filters(version="17"),
        k=8,
        prompt_version="answer-v2",
        model="stub",
        question="What is MVCC?",
    )
    assert base.startswith(f"sift:ans:{tenant_id}:")

    assert (
        answer_key(
            tenant_id=uuid4(),
            collection_id=collection_id,
            index_version=1,
            strategy="hybrid_rerank",
            filters=Filters(version="17"),
            k=8,
            prompt_version="answer-v2",
            model="stub",
            question="What is MVCC?",
        )
        != base
    )
    assert (
        answer_key(
            tenant_id=tenant_id,
            collection_id=uuid4(),
            index_version=1,
            strategy="hybrid_rerank",
            filters=Filters(version="17"),
            k=8,
            prompt_version="answer-v2",
            model="stub",
            question="What is MVCC?",
        )
        != base
    )
    assert (
        answer_key(
            tenant_id=tenant_id,
            collection_id=collection_id,
            index_version=2,
            strategy="hybrid_rerank",
            filters=Filters(version="17"),
            k=8,
            prompt_version="answer-v2",
            model="stub",
            question="What is MVCC?",
        )
        != base
    )
    assert (
        answer_key(
            tenant_id=tenant_id,
            collection_id=collection_id,
            index_version=1,
            strategy="dense",
            filters=Filters(version="17"),
            k=8,
            prompt_version="answer-v2",
            model="stub",
            question="What is MVCC?",
        )
        != base
    )
    assert (
        answer_key(
            tenant_id=tenant_id,
            collection_id=collection_id,
            index_version=1,
            strategy="hybrid_rerank",
            filters=Filters(version="17"),
            k=4,
            prompt_version="answer-v2",
            model="stub",
            question="What is MVCC?",
        )
        != base
    )
    assert (
        answer_key(
            tenant_id=tenant_id,
            collection_id=collection_id,
            index_version=1,
            strategy="hybrid_rerank",
            filters=Filters(version="17"),
            k=8,
            prompt_version="answer-v3",
            model="stub",
            question="What is MVCC?",
        )
        != base
    )
    assert (
        answer_key(
            tenant_id=tenant_id,
            collection_id=collection_id,
            index_version=1,
            strategy="hybrid_rerank",
            filters=Filters(version="17"),
            k=8,
            prompt_version="answer-v2",
            model="gemini-2.5-flash",
            question="What is MVCC?",
        )
        != base
    )
    assert (
        answer_key(
            tenant_id=tenant_id,
            collection_id=collection_id,
            index_version=1,
            strategy="hybrid_rerank",
            filters=Filters(version="16"),
            k=8,
            prompt_version="answer-v2",
            model="stub",
            question="What is MVCC?",
        )
        != base
    )

    same_norm = answer_key(
        tenant_id=tenant_id,
        collection_id=collection_id,
        index_version=1,
        strategy="hybrid_rerank",
        filters=Filters(version="17"),
        k=8,
        prompt_version="answer-v2",
        model="stub",
        question="  what   IS mvcc??? ",
    )
    assert same_norm == base

    f1 = answer_key(
        tenant_id=tenant_id,
        collection_id=collection_id,
        index_version=1,
        strategy="hybrid_rerank",
        filters={"version": "17", "sources": ["pgdocs"]},
        k=8,
        prompt_version="answer-v2",
        model="stub",
        question="What is MVCC?",
    )
    f2 = answer_key(
        tenant_id=tenant_id,
        collection_id=collection_id,
        index_version=1,
        strategy="hybrid_rerank",
        filters={"sources": ["pgdocs"], "version": "17"},
        k=8,
        prompt_version="answer-v2",
        model="stub",
        question="What is MVCC?",
    )
    assert f1 == f2


async def test_caches_and_spend_budget() -> None:
    redis = FakeRedis()
    health = FakeHealth()
    tenant_id = uuid4()

    qemb = QueryEmbeddingCache(redis, health, model="bge@1234567890ab", ttl_seconds=86_400)
    assert await qemb.get("What is MVCC?") is None
    await qemb.put("What is MVCC?", [0.25, 0.75])
    assert await qemb.get("  what is mvcc?? ") == [0.25, 0.75]

    acache = AnswerCache(redis, health, ttl_seconds=86_400)
    assert await acache.get("sift:ans:test") is None
    await acache.put("sift:ans:test", {"answer": "cached"})
    assert await acache.get("sift:ans:test") == {"answer": "cached"}

    fixed_now = datetime(2026, 5, 1, 23, 59, 30, tzinfo=UTC)
    budget = SpendBudget(redis, health, daily_usd=1.0, clock=lambda: fixed_now)
    assert await budget.check(tenant_id) is None

    await budget.add(tenant_id, 0.4)
    assert await budget.check(tenant_id) is None

    await budget.add(tenant_id, 0.6)
    assert await budget.check(tenant_id) == 30

    uncapped = SpendBudget(redis, health, daily_usd=None, clock=lambda: fixed_now)
    assert await uncapped.check(tenant_id) is None
