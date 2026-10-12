"""sift-models' HTTP layer with a fake encoder: validation and routing, no model download."""

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from sift.models_svc import app as models_app


class FakeEncoder:
    embed_model = "fake-embed"
    rerank_model = "fake-rerank"

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        self.calls.append(("embed", kind))
        return [[1.0, 0.0] for _ in texts]

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append(("rerank", query))
        return [0.5 for _ in passages]


@pytest.fixture
async def client(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[httpx.AsyncClient, FakeEncoder]]:
    fake = FakeEncoder()
    monkeypatch.setattr(models_app, "encoder", lambda: fake)
    transport = httpx.ASGITransport(app=models_app.app)  # no lifespan: no model load
    async with httpx.AsyncClient(transport=transport, base_url="http://sift-models") as c:
        yield c, fake


async def test_embed_and_rerank_pass_through(client: Any) -> None:
    c, fake = client
    embed = await c.post("/embed", json={"texts": ["a", "b"], "kind": "query"})
    rerank = await c.post("/rerank", json={"query": "q", "passages": ["p1", "p2", "p3"]})
    assert embed.json() == {"model": "fake-embed", "embeddings": [[1.0, 0.0], [1.0, 0.0]]}
    assert rerank.json() == {"model": "fake-rerank", "scores": [0.5, 0.5, 0.5]}
    assert fake.calls == [("embed", "query"), ("rerank", "q")]


@pytest.mark.parametrize(
    "path, body",
    [
        ("/embed", {"texts": [], "kind": "query"}),
        ("/embed", {"texts": ["x"] * 257, "kind": "passage"}),  # more than one request should carry
        ("/embed", {"texts": ["x"], "kind": "document"}),  # only query or passage
        ("/embed", {"texts": ["x"], "kind": "query", "extra": 1}),
        ("/rerank", {"query": "", "passages": ["p"]}),
        ("/rerank", {"query": "q", "passages": ["p"] * 201}),
    ],
)
async def test_bad_requests_are_422_and_never_reach_the_model(
    client: Any, path: str, body: dict[str, Any]
) -> None:
    c, fake = client
    assert (await c.post(path, json=body)).status_code == 422
    assert fake.calls == []
