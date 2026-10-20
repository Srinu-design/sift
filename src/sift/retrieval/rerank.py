"""Cross-encoder reranking over candidate retrieval hits."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace

import httpx

from sift.retrieval.hit import Hit

Reranker = Callable[[str, list[str]], Awaitable[list[float]]]


def passage(hit: Hit) -> str:
    return f"{hit.heading}\n{hit.content}"


async def rerank(
    question: str,
    hits: list[Hit],
    reranker: Reranker,
    k: int,
) -> list[Hit]:
    if not hits or k <= 0:
        return []
    scores = await reranker(question, [passage(h) for h in hits])
    rescored = [replace(h, score=float(s)) for h, s in zip(hits, scores, strict=True)]
    rescored.sort(key=lambda h: h.score, reverse=True)
    return rescored[:k]


def remote_reranker(http: httpx.AsyncClient, base_url: str) -> Reranker:
    url = f"{base_url.rstrip('/')}/rerank"

    async def _score(question: str, passages: list[str]) -> list[float]:
        resp = await http.post(url, json={"query": question, "passages": passages})
        resp.raise_for_status()
        data = resp.json()
        return [float(s) for s in data["scores"]]

    return _score


def local_reranker() -> Reranker:
    from sift.models_svc.encoder import encoder

    enc = encoder()

    async def _score(question: str, passages: list[str]) -> list[float]:
        return await asyncio.to_thread(enc.rerank, question, passages)

    return _score
