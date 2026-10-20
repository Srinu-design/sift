"""Two-gate grounded answer pipeline with single-retry citation enforcement."""

import time
from dataclasses import dataclass
from typing import Literal

from pydantic import ValidationError

from sift.answer.citations import Citation, check, cited
from sift.answer.llm import LLM
from sift.answer.prompt import SYSTEM_PROMPT, ModelAnswer, user_prompt
from sift.answer.refusal import gate1, suggestions
from sift.retrieval.hit import Hit

RefusalGate = Literal["retrieval", "model", "citation"]


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    refused: bool
    refusal_gate: RefusalGate | None
    refusal_reason: str | None
    citations: list[Citation]
    suggestions: list[Citation]  # filled only on refusal
    attempts: int  # LLM calls made: 0, 1 or 2
    tokens_in: int
    tokens_out: int
    llm_ms: float
    model: str | None


async def answer(
    question: str,
    hits: list[Hit],
    llm: LLM,
    *,
    tau: float | None,
    context_k: int,
) -> AnswerResult:
    best_score = hits[0].score if hits else None
    if tau is not None and gate1(best_score, tau):
        return AnswerResult(
            answer="",
            refused=True,
            refusal_gate="retrieval",
            refusal_reason="Top retrieval score is below the confidence threshold.",
            citations=[],
            suggestions=suggestions(hits),
            attempts=0,
            tokens_in=0,
            tokens_out=0,
            llm_ms=0.0,
            model=None,
        )

    context_hits = hits[:context_k]
    turns: list[tuple[Literal["user", "model"], str]] = [
        ("user", user_prompt(question, context_hits))
    ]
    tokens_in = 0
    tokens_out = 0
    total_ms = 0.0
    last_errors: list[str] = []
    last_model: str | None = None

    for attempt in (1, 2):
        t0 = time.perf_counter()
        res = await llm.generate(SYSTEM_PROMPT, turns)
        total_ms += (time.perf_counter() - t0) * 1000.0
        tokens_in += res.tokens_in
        tokens_out += res.tokens_out
        last_model = res.model

        parsed: ModelAnswer | None = None
        try:
            parsed = ModelAnswer.model_validate_json(res.text)
        except ValidationError as exc:
            last_errors = [f"invalid JSON: {exc}"]
        else:
            last_errors = check(parsed, context_hits)

        if not last_errors and parsed is not None:
            if parsed.refused:
                return AnswerResult(
                    answer="",
                    refused=True,
                    refusal_gate="model",
                    refusal_reason=parsed.refusal_reason
                    or "The sources do not contain the answer.",
                    citations=[],
                    suggestions=suggestions(hits),
                    attempts=attempt,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    llm_ms=round(total_ms, 1),
                    model=last_model,
                )
            return AnswerResult(
                answer=parsed.answer.strip(),
                refused=False,
                refusal_gate=None,
                refusal_reason=None,
                citations=cited(parsed, context_hits),
                suggestions=[],
                attempts=attempt,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                llm_ms=round(total_ms, 1),
                model=last_model,
            )

        if attempt == 1:
            joined = "; ".join(last_errors)
            turns.append(("model", res.text))
            turns.append(
                (
                    "user",
                    f"Your response failed validation: {joined}. "
                    "Please fix these errors and reply with valid JSON only.",
                )
            )

    return AnswerResult(
        answer="",
        refused=True,
        refusal_gate="citation",
        refusal_reason="; ".join(last_errors),
        citations=[],
        suggestions=suggestions(hits),
        attempts=2,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        llm_ms=round(total_ms, 1),
        model=last_model,
    )
