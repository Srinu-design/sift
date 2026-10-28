"""sift-ingest's endpoints as Hopper calls them: signed, idempotent, and answering with the
status code that tells Hopper what to do next (2xx done, 422 DLQ, 503 retry)."""

import base64
import hashlib
import hmac
import json
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.config import get_settings
from sift.ingest import pipeline
from sift.ingest.service import app
from tests.conftest import FakeEmbedder
from tests.integration.helpers import page, scalar, setup, submit

SECRET = base64.urlsafe_b64decode(get_settings().hopper_signing_secret.get_secret_value())


class Outside:
    """Hopper and sift-models, as seen from sift-ingest."""

    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []
        self.models_up = True
        self.embed = FakeEmbedder()

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/jobs":
            job = json.loads(request.content)
            self.jobs.append({**job, "key": request.headers["idempotency-key"]})
            return httpx.Response(201, json={"id": "job"})
        if request.url.path == "/embed":
            if not self.models_up:
                raise httpx.ConnectError("sift-models is down")
            texts = json.loads(request.content)["texts"]
            return httpx.Response(200, json={"model": "m", "embeddings": await self.embed(texts)})
        return httpx.Response(404)


@pytest.fixture
async def call(engine: AsyncEngine) -> AsyncIterator[tuple[Any, Outside]]:
    outside = Outside()
    app.state.secret = SECRET
    app.state.engine = engine
    app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(outside.handle))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://sift-ingest"
    ) as client:

        async def post(
            path: str, body: dict[str, Any], secret: bytes = SECRET, t: float | None = None
        ) -> httpx.Response:
            raw = json.dumps(body, separators=(",", ":")).encode()
            ts = int(t or time.time())
            mac = hmac.new(secret, f"{ts}.".encode() + raw, hashlib.sha256).hexdigest()
            headers = {
                "Hopper-Signature": f"t={ts},v1={mac}",
                "Hopper-Job-Id": "j1",
                "Hopper-Attempt": "1",
            }
            return await client.post(path, content=raw, headers=headers)

        yield post, outside
    await app.state.http.aclose()


async def version(engine: AsyncEngine, data: bytes) -> UUID:
    return (await submit(engine, await setup(engine), data)).version_id


async def test_unsigned_wrongly_signed_or_stale_calls_are_401(
    call: Any, engine: AsyncEngine
) -> None:
    post, outside = call
    v = await version(engine, page())
    body = {"version_id": str(v)}
    assert (
        await post("/internal/parse", body, secret=b"another tenant's secret")
    ).status_code == 401
    assert (await post("/internal/parse", body, t=time.time() - 3600)).status_code == 401
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://x"
    ) as raw:
        assert (await raw.post("/internal/parse", json=body)).status_code == 401
    assert outside.jobs == []
    assert await scalar(engine, "SELECT status FROM document_versions") == "pending"


async def test_parse_enqueues_each_batch_once_however_often_it_runs(
    call: Any, engine: AsyncEngine
) -> None:
    post, outside = call
    v = await version(engine, page(sections=70, paragraphs=1))  # 70 chunks: two batches

    first = await post("/internal/parse", {"version_id": str(v)})
    again = await post("/internal/parse", {"version_id": str(v)})  # Hopper's retry

    assert first.status_code == again.status_code == 200
    keys = [j["key"] for j in outside.jobs]
    assert keys == [f"embed:{v}:0", f"embed:{v}:1"] * 2  # same keys: Hopper keeps one job each
    assert {j["queue"] for j in outside.jobs} == {"sift-embed"}
    assert all(j["payload"]["url"].endswith("/internal/embed") for j in outside.jobs)


async def test_the_last_batch_enqueues_activation_and_activation_goes_live(
    call: Any, engine: AsyncEngine
) -> None:
    post, outside = call
    v = await version(engine, page())
    await post("/internal/parse", {"version_id": str(v)})
    done = await post("/internal/embed", {"version_id": str(v), "batch": 0})
    assert done.json() == {"version_complete": True}
    assert outside.jobs[-1]["key"] == f"activate:{v}"
    assert (await post("/internal/activate", {"version_id": str(v)})).json() == {
        "outcome": "activated"
    }
    assert (await post("/internal/activate", {"version_id": str(v)})).json() == {
        "outcome": "already"
    }


async def test_an_unparseable_file_is_422_so_hopper_dead_letters_it(
    call: Any, engine: AsyncEngine
) -> None:
    post, _ = call
    ids = await setup(engine)
    s = await pipeline.submit(
        engine,
        tenant_id=ids[0],
        collection_id=ids[1],
        filename="broken.pdf",
        data=b"%PDF-1.4 not really",
        source_uri="broken.pdf",
        source="test",
    )
    pytest.importorskip("docling")
    resp = await post("/internal/parse", {"version_id": str(s.version_id)})
    assert resp.status_code == 422
    assert await scalar(engine, "SELECT status FROM document_versions") == "failed"


async def test_models_down_is_503_with_retry_after(call: Any, engine: AsyncEngine) -> None:
    post, outside = call
    v = await version(engine, page())
    await post("/internal/parse", {"version_id": str(v)})
    outside.models_up = False
    resp = await post("/internal/embed", {"version_id": str(v), "batch": 0})
    assert resp.status_code == 503 and resp.headers["retry-after"]
    assert await scalar(engine, "SELECT count(embedding) FROM chunks") == 0


async def test_a_bad_body_or_unknown_version_is_422(call: Any) -> None:
    post, _ = call
    assert (await post("/internal/parse", {"version_id": "not-a-uuid"})).status_code == 422
    assert (await post("/internal/parse", {"version_id": str(UUID(int=7))})).status_code == 422
