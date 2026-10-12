"""sift-models: /embed and /rerank, internal only. Models load once per process, which
matters on a 4 GB host, and there is one place to swap them."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from sift.common.logging import configure_logging
from sift.config import get_settings
from sift.models_svc.encoder import Kind, encoder


class EmbedIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[str] = Field(min_length=1, max_length=256)
    kind: Kind


class EmbedOut(BaseModel):
    model: str
    embeddings: list[list[float]]


class RerankIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    passages: list[str] = Field(min_length=1, max_length=200)


class RerankOut(BaseModel):
    model: str
    scores: list[float]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging(get_settings().log_level)
    await asyncio.to_thread(encoder)  # load both models before the first request
    yield


app = FastAPI(title="sift-models", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedOut)
async def embed(body: EmbedIn) -> EmbedOut:
    enc = encoder()
    vectors = await asyncio.to_thread(enc.embed, body.texts, body.kind)
    return EmbedOut(model=enc.embed_model, embeddings=vectors)


@app.post("/rerank", response_model=RerankOut)
async def rerank(body: RerankIn) -> RerankOut:
    enc = encoder()
    scores = await asyncio.to_thread(enc.rerank, body.query, body.passages)
    return RerankOut(model=enc.rerank_model, scores=scores)
