"""Unit tests for cross-encoder reranking and remote reranker HTTP client."""

import json

import httpx
import pytest

from sift.retrieval.rerank import passage, remote_reranker, rerank
from tests.conftest import make_hit


async def test_rerank_orders_ties_replaces_score_and_trims() -> None:
    hits = [make_hit(n, score=n / 10, content=f"Content for chunk {n}.") for n in range(1, 5)]
    seen_passages: list[str] = []

    async def fake_reranker(question: str, passages: list[str]) -> list[float]:
        assert question == "test query"
        seen_passages.extend(passages)
        # chunk 2 and 3 tie at 0.8; chunk 4 gets 0.95; chunk 1 gets 0.1
        return [0.1, 0.8, 0.8, 0.95]

    out = await rerank("test query", hits, fake_reranker, k=3)
    assert seen_passages == [passage(h) for h in hits]
    assert seen_passages[0] == "Heading 1\nContent for chunk 1."
    assert [h.chunk_id for h in out] == [4, 2, 3]
    assert [h.score for h in out] == [
        pytest.approx(0.95),
        pytest.approx(0.8),
        pytest.approx(0.8),
    ]


async def test_rerank_empty_input_does_not_call_reranker() -> None:
    called = False

    async def fake_reranker(question: str, passages: list[str]) -> list[float]:
        nonlocal called
        called = True
        return []

    out = await rerank("test query", [], fake_reranker, k=5)
    assert out == []
    assert not called


async def test_remote_reranker_posts_query_and_passages() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "http://sift-models:8100/rerank"
        body = json.loads(request.content.decode())
        assert body == {"query": "autovacuum", "passages": ["H1\nC1", "H2\nC2"]}
        return httpx.Response(200, json={"model": "ce", "scores": [0.25, 0.75]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        scorer = remote_reranker(client, "http://sift-models:8100")
        scores = await scorer("autovacuum", ["H1\nC1", "H2\nC2"])

    assert scores == [pytest.approx(0.25), pytest.approx(0.75)]
