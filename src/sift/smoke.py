"""Retrieval smoke test for a deploy: python -m sift.smoke (inside the sift-api container).

Embeds one known question through sift-models, runs hybrid_rerank against the live index and
checks the gold section is in the top 5. No LLM call: a deploy must not depend on the answer
model's quota, and retrieval is what a bad release breaks. Exit 0 pass, 1 fail.
"""

import asyncio
import sys

import httpx

from sift.config import get_settings
from sift.db import create_engine
from sift.ingest import pipeline
from sift.retrieval.filters import Filters
from sift.retrieval.rerank import remote_reranker
from sift.retrieval.search import search

QUESTION = "Can CREATE INDEX CONCURRENTLY run inside a transaction block?"
GOLD_ANCHOR = "SQL-CREATEINDEX-CONCURRENTLY"


async def main() -> int:
    settings = get_settings()
    engine = create_engine(settings)
    try:
        tenant_id, collection_id = await pipeline.ensure_collection(engine, "pgdocs", "pgdocs")
        async with httpx.AsyncClient(timeout=settings.models_timeout_seconds) as http:
            r = await http.post(
                f"{settings.models_url}/embed", json={"texts": [QUESTION], "kind": "query"}
            )
            r.raise_for_status()
            result = await search(
                engine,
                tenant_id=tenant_id,
                collection_id=collection_id,
                strategy="hybrid_rerank",
                question=QUESTION,
                query_vector=r.json()["embeddings"][0],
                filters=Filters(version="17"),
                k=5,
                rerank_depth=settings.rerank_depth,
                reranker=remote_reranker(http, settings.models_url),
            )
    finally:
        await engine.dispose()
    anchors = [h.anchor for h in result.hits]
    ok = GOLD_ANCHOR in anchors
    print(f"smoke {'pass' if ok else 'FAIL'}: top 5 anchors {anchors}, {result.timings_ms}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
