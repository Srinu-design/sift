"""LLM clients: Google Gemini over REST and a deterministic stub."""

import asyncio
import re
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

from sift.answer.prompt import RESPONSE_SCHEMA, ModelAnswer, ModelCitation
from sift.config import Settings

_FIRST_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    tokens_in: int
    tokens_out: int
    model: str


class LLMUnavailable(Exception):
    """Raised on LLM rate limits, timeouts, 5xx responses, or empty/blocked candidates."""


class LLM(Protocol):
    model: str

    async def generate(
        self,
        system: str,
        turns: list[tuple[Literal["user", "model"], str]],
    ) -> LLMResult: ...


class GeminiLLM:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float,
        thinking_budget: int | None = None,
    ) -> None:
        self._http = http
        self._api_key = api_key
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._thinking_budget = thinking_budget

    async def generate(
        self,
        system: str,
        turns: list[tuple[Literal["user", "model"], str]],
    ) -> LLMResult:
        url = f"{self._base_url}/v1beta/models/{self.model}:generateContent"
        gen_config: dict[str, object] = {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        }
        if self._thinking_budget is not None:
            gen_config["thinkingConfig"] = {"thinkingBudget": self._thinking_budget}

        payload: dict[str, object] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": role, "parts": [{"text": text}]} for role, text in turns],
            "generationConfig": gen_config,
        }

        try:
            resp = await self._http.post(
                url,
                headers={"x-goog-api-key": self._api_key},
                json=payload,
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            raise LLMUnavailable(f"Gemini transport error: {type(exc).__name__}") from exc

        if resp.status_code == 429 or resp.status_code >= 500:
            raise LLMUnavailable(f"Gemini HTTP {resp.status_code}")
        resp.raise_for_status()

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMUnavailable("Gemini returned no candidates")
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if not text.strip():
            raise LLMUnavailable("Gemini returned empty candidate text")

        usage = data.get("usageMetadata") or {}
        tokens_in = int(usage.get("promptTokenCount") or 0)
        tokens_out = int(usage.get("candidatesTokenCount") or 0) + int(
            usage.get("thoughtsTokenCount") or 0
        )
        return LLMResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=self.model,
        )


class StubLLM:
    model: str = "stub"

    def __init__(self, delay_ms: int = 0) -> None:
        self._delay_ms = delay_ms

    async def generate(
        self,
        system: str,
        turns: list[tuple[Literal["user", "model"], str]],
    ) -> LLMResult:
        if self._delay_ms > 0:
            await asyncio.sleep(self._delay_ms / 1000.0)

        prompt_text = system + "\n" + "\n".join(t for _, t in turns)
        last_user = turns[-1][1] if turns else ""
        s1_content = self._extract_s1(last_user)

        if not s1_content:
            reply = ModelAnswer(
                refused=True,
                answer="",
                citations=[],
                refusal_reason="No sources available to answer the question.",
            ).model_dump_json()
        else:
            first_sent = _FIRST_SENT_RE.split(s1_content.strip(), maxsplit=1)[0]
            words = first_sent.split()[:30]
            snippet = " ".join(words).rstrip(".!?,;:")
            quote = " ".join(s1_content.split()[:8])
            reply = ModelAnswer(
                refused=False,
                answer=f"{snippet} [S1].",
                citations=[ModelCitation(source="S1", quote=quote)],
                refusal_reason=None,
            ).model_dump_json()

        return LLMResult(
            text=reply,
            tokens_in=len(prompt_text) // 4,
            tokens_out=len(reply) // 4,
            model=self.model,
        )

    @staticmethod
    def _extract_s1(user_text: str) -> str:
        lines = user_text.splitlines()
        for idx, line in enumerate(lines):
            if line.startswith("[S1] ") and idx + 1 < len(lines):
                return lines[idx + 1].strip()
        return ""


def make_llm(settings: Settings, http: httpx.AsyncClient) -> LLM:
    if settings.llm_provider == "gemini":
        key = settings.gemini_api_key.get_secret_value()
        if not key:
            raise ValueError("gemini_api_key must not be empty when llm_provider is 'gemini'")
        return GeminiLLM(
            http,
            api_key=key,
            model=settings.answer_model,
            base_url=settings.gemini_url,
            timeout=settings.llm_timeout_seconds,
            thinking_budget=settings.llm_thinking_budget,
        )
    return StubLLM(delay_ms=settings.stub_delay_ms)
