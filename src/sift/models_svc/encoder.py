"""The two local models, loaded once. sift-models serves them over HTTP; the eval and the
in-process ingest script call them directly, so queries and chunks are embedded by the very
same code."""

import math
import threading
from functools import lru_cache
from typing import Literal

from sift.config import get_settings

# bge-small-en-v1.5's retrieval instruction: prefixed to queries, never to passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
Kind = Literal["query", "passage"]


class Encoder:
    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        s = get_settings()
        self.embed_model = s.embed_model
        self.rerank_model = s.rerank_model
        self._embedder = SentenceTransformer(s.embed_model, revision=s.embed_revision, device="cpu")
        self._reranker = CrossEncoder(s.rerank_model, revision=s.rerank_revision, device="cpu")
        # One model call at a time: torch already spreads one call over every core, and two
        # at once only fight over them.
        self._lock = threading.Lock()

    def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        with self._lock:
            vectors = self._embedder.encode(
                texts,
                prompt=QUERY_INSTRUCTION if kind == "query" else None,
                normalize_embeddings=True,  # cosine distance in pgvector
                batch_size=32,
                convert_to_numpy=True,
            )
        return vectors.tolist()  # type: ignore[no-any-return]  # numpy: Python floats, fast

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Relevance in 0-1. The model returns raw logits; the sigmoid makes a threshold
        mean the same thing for every query."""
        with self._lock:
            logits = self._reranker.predict([(query, p) for p in passages], batch_size=32)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


@lru_cache
def encoder() -> Encoder:
    return Encoder()
