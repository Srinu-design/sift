"""Integration tests for Postgres RLS, multi-tenant API isolation, uploads, and demo limits."""

import json
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine

from sift.answer.llm import StubLLM
from sift.api.app import create_app
from sift.auth import keys
from sift.config import Settings
from sift.db import create_engine, set_tenant
from sift.ingest import pipeline
from sift.retrieval.search import STRATEGIES, search
from tests.conftest import FakeEmbedder
from tests.integration.test_pipeline import ingest, page

PEPPER = b"test-pepper"


def _settings(engine: AsyncEngine, **overrides: object) -> Settings:
    url = engine.url.render_as_string(hide_password=False)
    base: dict[str, object] = {
        "database_url": url,
        "api_key_pepper": SecretStr("test-pepper"),
        "api_db_role": "sift_app",
        "hopper_api_key": SecretStr("hopper-test-key"),
        "demo_tenant": "pgdocs",
        "demo_collection": "pgdocs",
        "demo_rate_per_minute": 10,
        "max_upload_bytes": 20_000_000,
    }
    base.update(overrides)
    return Settings.model_validate(base)


async def _create_api_key(
    engine: AsyncEngine,
    tenant_id: UUID,
    *,
    name: str = "test",
    revoked: bool = False,
) -> keys.NewKey:
    nk = keys.generate(PEPPER)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO api_keys (tenant_id, prefix, key_hash, name, revoked_at) "
                "VALUES (:t, :p, :h, :n, CASE WHEN :rev THEN now() ELSE NULL END)"
            ),
            {
                "t": tenant_id,
                "p": nk.prefix,
                "h": nk.key_hash,
                "n": name,
                "rev": revoked,
            },
        )
    return nk


async def _ingest_custom_doc(
    engine: AsyncEngine,
    embedder: FakeEmbedder,
    *,
    tenant_id: UUID,
    collection_id: UUID,
    title: str,
    body_text: str,
    source_uri: str,
) -> UUID:
    html = (
        '<html><body><div id="docContent"><div class="sect1" id="P">'
        f"<h2>{title}</h2>"
        f'<div class="sect2" id="SEC-1"><h3>Section 1</h3><p>{body_text}</p></div>'
        "</div></div></body></html>"
    ).encode()
    sub = await pipeline.submit(
        engine,
        tenant_id=tenant_id,
        collection_id=collection_id,
        filename="doc.html",
        data=html,
        source_uri=source_uri,
        source="pgdocs",
        doc_version="17",
    )
    await ingest(engine, sub.version_id, embedder)
    async with engine.connect() as conn:
        doc_id = await conn.scalar(
            text("SELECT document_id FROM document_versions WHERE id = :v"),
            {"v": sub.version_id},
        )
    return UUID(str(doc_id))


def _mock_http(
    embedder: FakeEmbedder,
    *,
    hopper_requests: list[httpx.Request] | None = None,
) -> httpx.AsyncClient:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/embed":
            body = json.loads(request.content.decode())
            vecs = await embedder(body["texts"])
            return httpx.Response(200, json={"model": "fake-embed", "embeddings": vecs})
        if request.url.path == "/rerank":
            body = json.loads(request.content.decode())
            passages = body["passages"]
            scores = [max(0.1, round(0.95 - i * 0.05, 4)) for i in range(len(passages))]
            return httpx.Response(200, json={"model": "fake-rerank", "scores": scores})
        if request.url.path == "/v1/jobs":
            if hopper_requests is not None:
                hopper_requests.append(request)
            return httpx.Response(201, json={"id": "job-1", "status": "queued"})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_rls_isolation_unset_tenant_and_with_check(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    t_a, c_a = await pipeline.ensure_collection(engine, "tenant-a", "docs")
    t_b, c_b = await pipeline.ensure_collection(engine, "tenant-b", "docs")
    doc_a = await _ingest_custom_doc(
        engine,
        embedder,
        tenant_id=t_a,
        collection_id=c_a,
        title="Alpha Manual",
        body_text="Alpha secret configuration parameter details.",
        source_uri="https://docs.test/alpha.html",
    )
    await _ingest_custom_doc(
        engine,
        embedder,
        tenant_id=t_b,
        collection_id=c_b,
        title="Beta Manual",
        body_text="Beta public configuration parameter details.",
        source_uri="https://docs.test/beta.html",
    )

    app_engine = create_engine(_settings(engine), role="sift_app")
    try:
        # With tenant B active, tenant A's chunks, documents, and versions are invisible
        async with app_engine.begin() as conn:
            await set_tenant(conn, t_b)
            chunks_a = await conn.scalar(
                text("SELECT count(*) FROM chunks WHERE tenant_id = :a"), {"a": t_a}
            )
            docs_a = await conn.scalar(
                text("SELECT count(*) FROM documents WHERE tenant_id = :a"), {"a": t_a}
            )
            vers_a = await conn.scalar(
                text("SELECT count(*) FROM document_versions WHERE document_id = :d"),
                {"d": doc_a},
            )
            chunks_b = await conn.scalar(text("SELECT count(*) FROM chunks"))
            assert chunks_a == 0
            assert docs_a == 0
            assert vers_a == 0
            assert chunks_b and chunks_b > 0

        # With no tenant set on the connection, all counts are 0 and nothing raises
        async with app_engine.begin() as conn:
            for table in ("chunks", "documents", "document_versions", "query_logs"):
                cnt = await conn.scalar(text(f"SELECT count(*) FROM {table}"))
                assert cnt == 0

        # WITH CHECK prevents tenant B from inserting a query_logs row for tenant A
        with pytest.raises(DBAPIError):
            async with app_engine.begin() as conn:
                await set_tenant(conn, t_b)
                await conn.execute(
                    text(
                        "INSERT INTO query_logs ("
                        "tenant_id, collection_id, request_id, strategy, cache_hit, "
                        "refused, timings_ms, prompt_version"
                        ") VALUES (:t, :c, 'req1', 'naive', false, false, '{}'::jsonb, 'v1')"
                    ),
                    {"t": t_a, "c": c_a},
                )
    finally:
        await app_engine.dispose()


async def test_cross_tenant_query_and_document_isolation(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    t_a, c_a = await pipeline.ensure_collection(engine, "tenant-a", "docs")
    t_b, c_b = await pipeline.ensure_collection(engine, "tenant-b", "docs")

    alpha_phrase = "Alpha exclusive super_secret_guc_setting parameter explanation."
    doc_a = await _ingest_custom_doc(
        engine,
        embedder,
        tenant_id=t_a,
        collection_id=c_a,
        title="Alpha Manual",
        body_text=alpha_phrase,
        source_uri="https://docs.test/alpha-only.html",
    )
    await _ingest_custom_doc(
        engine,
        embedder,
        tenant_id=t_b,
        collection_id=c_b,
        title="Beta Manual",
        body_text="Beta standard checkpoint_timeout parameter explanation.",
        source_uri="https://docs.test/beta-only.html",
    )

    key_a = await _create_api_key(engine, t_a)
    key_b = await _create_api_key(engine, t_b)

    async with engine.connect() as conn:
        a_chunk_ids = set(
            (await conn.execute(text("SELECT id FROM chunks WHERE tenant_id = :a"), {"a": t_a}))
            .scalars()
            .all()
        )
    assert len(a_chunk_ids) >= 1

    [q_vec] = await embedder([alpha_phrase])
    app_engine = create_engine(_settings(engine), role="sift_app")
    try:

        async def fake_reranker(question: str, passages: list[str]) -> list[float]:
            return [0.9 for _ in passages]

        for strat in STRATEGIES:
            res = await search(
                app_engine,
                tenant_id=t_b,
                collection_id=c_b,
                strategy=strat,
                question=alpha_phrase,
                query_vector=q_vec,
                reranker=fake_reranker if strat == "hybrid_rerank" else None,
            )
            b_hit_ids = {h.chunk_id for h in res.hits}
            assert b_hit_ids.isdisjoint(a_chunk_ids)

        async with _mock_http(embedder) as models_http:
            app = create_app(
                settings=_settings(engine),
                engine=app_engine,
                http=models_http,
                llm=StubLLM(),
            )
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client,
            ):
                for strat in STRATEGIES:
                    resp = await client.post(
                        "/v1/query",
                        headers={"Authorization": f"Bearer {key_b.plaintext}"},
                        json={
                            "collection": "docs",
                            "question": alpha_phrase,
                            "strategy": strat,
                        },
                    )
                    assert resp.status_code == 200
                    body = resp.json()
                    urls = [
                        c["url"] for c in (*body.get("citations", []), *body.get("suggestions", []))
                    ]
                    assert all("alpha-only.html" not in u for u in urls)

                # Tenant B requesting Tenant A's document gets 404, never 403
                r_b_doc = await client.get(
                    f"/v1/documents/{doc_a}",
                    headers={"Authorization": f"Bearer {key_b.plaintext}"},
                )
                assert r_b_doc.status_code == 404
                assert r_b_doc.json() == {"detail": "document not found"}

                # Tenant A requesting its own document gets 200
                r_a_doc = await client.get(
                    f"/v1/documents/{doc_a}",
                    headers={"Authorization": f"Bearer {key_a.plaintext}"},
                )
                assert r_a_doc.status_code == 200
                doc_data = r_a_doc.json()
                assert doc_data["id"] == str(doc_a)
                assert doc_data["title"] == "Alpha Manual"
                assert len(doc_data["versions"]) == 1
                assert doc_data["versions"][0]["status"] == "ready"
    finally:
        await app_engine.dispose()


async def test_auth_errors_collections_uploads_and_demo_rate_limit(
    engine: AsyncEngine, embedder: FakeEmbedder
) -> None:
    t_pg, c_pg = await pipeline.ensure_collection(engine, "pgdocs", "pgdocs")
    await _ingest_custom_doc(
        engine,
        embedder,
        tenant_id=t_pg,
        collection_id=c_pg,
        title="PG Docs",
        body_text="PostgreSQL shared_buffers configuration documentation.",
        source_uri="https://docs.test/pg.html",
    )

    t_a, c_a = await pipeline.ensure_collection(engine, "tenant-a", "docs")
    _, c_b = await pipeline.ensure_collection(engine, "tenant-b", "docs")

    key_a = await _create_api_key(engine, t_a)
    revoked_key = await _create_api_key(engine, t_a, revoked=True)
    unknown_key = keys.generate(PEPPER)
    wrong_secret = key_a.plaintext[:-4] + "ZZZZ"

    hopper_requests: list[httpx.Request] = []
    app_engine = create_engine(_settings(engine), role="sift_app")
    cfg = _settings(engine, max_upload_bytes=100, demo_rate_per_minute=2)

    try:
        async with _mock_http(embedder, hopper_requests=hopper_requests) as models_http:
            app = create_app(
                settings=cfg,
                engine=app_engine,
                http=models_http,
                llm=StubLLM(),
            )
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client,
            ):
                # Missing, garbage, unknown-prefix, wrong-secret, and revoked keys -> identical 401
                expected_401 = {
                    "detail": "a valid API key is required: Authorization: Bearer sift_live_..."
                }
                for hdrs in (
                    {},
                    {"Authorization": "Bearer garbage"},
                    {"Authorization": f"Bearer {unknown_key.plaintext}"},
                    {"Authorization": f"Bearer {wrong_secret}"},
                    {"Authorization": f"Bearer {revoked_key.plaintext}"},
                ):
                    r_unauth = await client.get("/v1/collections", headers=hdrs)
                    assert r_unauth.status_code == 401
                    assert r_unauth.headers.get("WWW-Authenticate") == "Bearer"
                    assert r_unauth.json() == expected_401

                auth_a = {"Authorization": f"Bearer {key_a.plaintext}"}

                # Unknown collection in /v1/query -> 404
                r_missing_col = await client.post(
                    "/v1/query",
                    headers=auth_a,
                    json={"collection": "nonexistent", "question": "test"},
                )
                assert r_missing_col.status_code == 404
                assert r_missing_col.json() == {"detail": "collection not found"}

                # Create collection once -> 201; twice -> 409; list -> includes new collection
                r_col1 = await client.post(
                    "/v1/collections",
                    headers=auth_a,
                    json={"name": "rfcs"},
                )
                assert r_col1.status_code == 201
                r_col2 = await client.post(
                    "/v1/collections",
                    headers=auth_a,
                    json={"name": "rfcs"},
                )
                assert r_col2.status_code == 409
                r_list = await client.get("/v1/collections", headers=auth_a)
                assert r_list.status_code == 200
                assert {c["name"] for c in r_list.json()} == {"docs", "rfcs"}

                # Upload small HTML within max_upload_bytes (Temporarily raise limit for valid page)
                app.state.settings = cfg.model_copy(update={"max_upload_bytes": 20_000_000})
                html_bytes = page(sections=1, paragraphs=1)
                r_up = await client.post(
                    f"/v1/collections/{c_a}/documents",
                    params={
                        "filename": "small.html",
                        "source": "pgdocs",
                        "source_uri": "https://docs.test/small.html",
                        "doc_version": "17",
                    },
                    headers=auth_a,
                    content=html_bytes,
                )
                assert r_up.status_code == 202
                up_body = r_up.json()
                ver_id = up_body["version_id"]
                assert len(hopper_requests) == 1
                assert hopper_requests[0].headers["Idempotency-Key"] == f"parse:{ver_id}"
                assert json.loads(hopper_requests[0].content.decode())["task"] == "http"

                # Re-uploading the same bytes returns 202
                r_up_again = await client.post(
                    f"/v1/collections/{c_a}/documents",
                    params={
                        "filename": "small.html",
                        "source": "pgdocs",
                        "source_uri": "https://docs.test/small.html",
                        "doc_version": "17",
                    },
                    headers=auth_a,
                    content=html_bytes,
                )
                assert r_up_again.status_code == 202

                # Body over max_upload_bytes (100 bytes) -> 413
                app.state.settings = cfg
                r_too_big = await client.post(
                    f"/v1/collections/{c_a}/documents",
                    params={"filename": "big.html", "source": "pgdocs"},
                    headers=auth_a,
                    content=b"x" * 101,
                )
                assert r_too_big.status_code == 413

                # Uploading to Tenant B's collection id as Tenant A -> 404
                r_wrong_tenant_col = await client.post(
                    f"/v1/collections/{c_b}/documents",
                    params={"filename": "tiny.html", "source": "pgdocs"},
                    headers=auth_a,
                    content=b"<html></html>",
                )
                assert r_wrong_tenant_col.status_code == 404

                # Demo endpoint works without key and enforces demo_rate_per_minute (2)
                for _ in range(2):
                    r_demo_ok = await client.post(
                        "/demo/query",
                        json={"question": "shared_buffers configuration"},
                    )
                    assert r_demo_ok.status_code == 200

                r_demo_limited = await client.post(
                    "/demo/query",
                    json={"question": "shared_buffers configuration"},
                )
                assert r_demo_limited.status_code == 429
                assert r_demo_limited.json() == {"detail": "demo rate limit"}
                assert "Retry-After" in r_demo_limited.headers
    finally:
        await app_engine.dispose()
