"""Unit tests for GeminiLLM and StubLLM."""

import json

import httpx
import pytest

from sift.answer.citations import check
from sift.answer.llm import GeminiLLM, LLMUnavailable, StubLLM
from sift.answer.prompt import RESPONSE_SCHEMA, SYSTEM_PROMPT, ModelAnswer, user_prompt
from sift.retrieval.hit import Hit


def _sample_hit() -> Hit:
    return Hit(
        chunk_id=1,
        score=0.92,
        heading="WAL > Asynchronous Commit",
        anchor="WAL-ASYNC-COMMIT",
        anchor_path=("WAL-ASYNC-COMMIT",),
        page_start=None,
        page_end=None,
        doc_version="17",
        source="pgdocs",
        source_uri="https://www.postgresql.org/docs/17/wal-async-commit.html",
        title="Asynchronous Commit",
        content=(
            "Asynchronous commit provides an option to let transactions complete faster. "
            "It risks losing a small amount of data on a crash."
        ),
    )


async def test_gemini_llm_request_and_token_accounting() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": '{"refused": true, "answer": "", "citations": []}'}]
                        }
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 120,
                    "candidatesTokenCount": 30,
                    "thoughtsTokenCount": 15,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        llm = GeminiLLM(
            client,
            api_key="secret-test-key",
            model="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com",
            timeout=10.0,
            thinking_budget=None,
        )
        res = await llm.generate(
            SYSTEM_PROMPT,
            [("user", "First question"), ("model", "{}"), ("user", "Retry")],
        )

    assert len(captured) == 1
    req = captured[0]
    assert str(req.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    )
    assert "secret-test-key" not in str(req.url)
    assert req.headers["x-goog-api-key"] == "secret-test-key"

    body = json.loads(req.content.decode())
    assert body["systemInstruction"] == {"parts": [{"text": SYSTEM_PROMPT}]}
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "First question"}]},
        {"role": "model", "parts": [{"text": "{}"}]},
        {"role": "user", "parts": [{"text": "Retry"}]},
    ]
    gen_cfg = body["generationConfig"]
    assert gen_cfg["temperature"] == 0
    assert gen_cfg["responseMimeType"] == "application/json"
    assert gen_cfg["responseSchema"] == RESPONSE_SCHEMA
    assert "thinkingConfig" not in gen_cfg

    assert res.tokens_in == 120
    assert res.tokens_out == 45
    assert res.model == "gemini-2.5-flash"


async def test_gemini_llm_thinking_config_included_when_set() -> None:
    captured_bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
                "usageMetadata": {"promptTokenCount": 10},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        llm = GeminiLLM(
            client,
            api_key="secret-test-key",
            model="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com",
            timeout=10.0,
            thinking_budget=256,
        )
        res = await llm.generate(SYSTEM_PROMPT, [("user", "Hello")])

    gen_cfg = captured_bodies[0]["generationConfig"]
    assert isinstance(gen_cfg, dict)
    assert gen_cfg["thinkingConfig"] == {"thinkingBudget": 256}
    assert res.tokens_in == 10
    assert res.tokens_out == 0


@pytest.mark.parametrize("status_code", [429, 503])
async def test_gemini_llm_rate_limit_and_5xx_raise_unavailable(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": "unavailable"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        llm = GeminiLLM(
            client,
            api_key="super-secret-key",
            model="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com",
            timeout=10.0,
        )
        with pytest.raises(LLMUnavailable) as exc_info:
            await llm.generate(SYSTEM_PROMPT, [("user", "Hello")])
        assert "super-secret-key" not in str(exc_info.value)


async def test_gemini_llm_empty_candidates_raises_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        llm = GeminiLLM(
            client,
            api_key="super-secret-key",
            model="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com",
            timeout=10.0,
        )
        with pytest.raises(LLMUnavailable):
            await llm.generate(SYSTEM_PROMPT, [("user", "Hello")])


async def test_stub_llm_output_passes_check() -> None:
    hit = _sample_hit()
    stub = StubLLM(delay_ms=0)
    prompt = user_prompt("What is asynchronous commit?", [hit])
    res = await stub.generate(SYSTEM_PROMPT, [("user", prompt)])

    parsed = ModelAnswer.model_validate_json(res.text)
    assert not parsed.refused
    assert check(parsed, [hit]) == []
    assert res.tokens_in > 0
    assert res.tokens_out > 0

    empty_res = await stub.generate(SYSTEM_PROMPT, [("user", user_prompt("No sources?", []))])
    empty_parsed = ModelAnswer.model_validate_json(empty_res.text)
    assert empty_parsed.refused
