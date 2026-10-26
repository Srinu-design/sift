"""Unit tests for the two-gate answer pipeline and retry behavior."""

from dataclasses import replace

from sift.answer.generate import answer
from sift.answer.prompt import ModelAnswer, ModelCitation
from tests.conftest import ScriptedLLM, make_hit


async def test_good_first_reply() -> None:
    hits = [make_hit(1), make_hit(2)]
    good = ModelAnswer(
        refused=False,
        answer="shared_buffers controls memory caching [S1].",
        citations=[
            ModelCitation(
                source="S1",
                quote="shared_buffers controls memory caching",
            )
        ],
    ).model_dump_json()
    llm = ScriptedLLM([good])

    res = await answer("What does shared_buffers do?", hits, llm, tau=0.2, context_k=8)
    assert not res.refused
    assert res.refusal_gate is None
    assert res.attempts == 1
    assert res.tokens_in == 100
    assert res.tokens_out == 25
    assert len(res.citations) == 1
    assert res.citations[0].n == 1
    assert res.suggestions == []


async def test_malformed_json_then_good_retries_with_error() -> None:
    hits = [make_hit(1)]
    good = ModelAnswer(
        refused=False,
        answer="shared_buffers controls memory caching [S1].",
        citations=[
            ModelCitation(
                source="S1",
                quote="shared_buffers controls memory caching",
            )
        ],
    ).model_dump_json()
    llm = ScriptedLLM(["{not valid json", good])

    res = await answer("What does shared_buffers do?", hits, llm, tau=0.2, context_k=8)
    assert not res.refused
    assert res.attempts == 2
    assert res.tokens_in == 200
    assert res.tokens_out == 50
    assert len(llm.calls) == 2
    second_call = llm.calls[1]
    assert len(second_call) == 3
    assert second_call[1] == ("model", "{not valid json")
    assert second_call[2][0] == "user"
    assert "invalid JSON" in second_call[2][1]


async def test_bad_twice_refuses_with_citation_gate() -> None:
    hits = [make_hit(1), make_hit(2)]
    bad_citation = ModelAnswer(
        refused=False,
        answer="Sentence without any source tag.",
        citations=[],
    ).model_dump_json()
    llm = ScriptedLLM([bad_citation, bad_citation])

    res = await answer("What does shared_buffers do?", hits, llm, tau=0.2, context_k=8)
    assert res.refused
    assert res.refusal_gate == "citation"
    assert res.refusal_reason is not None
    assert "missing a [S#] citation" in res.refusal_reason
    assert res.attempts == 2
    assert len(res.suggestions) == 2


async def test_model_refusal_returns_model_gate_and_suggestions() -> None:
    hits = [make_hit(1), make_hit(2)]
    refusal = ModelAnswer(
        refused=True,
        answer="",
        citations=[],
        refusal_reason="The sources do not mention Kubernetes operators.",
    ).model_dump_json()
    llm = ScriptedLLM([refusal])

    res = await answer("How to deploy on Kubernetes?", hits, llm, tau=0.2, context_k=8)
    assert res.refused
    assert res.refusal_gate == "model"
    assert res.refusal_reason == "The sources do not mention Kubernetes operators."
    assert res.attempts == 1
    assert len(res.suggestions) == 2


async def test_score_below_tau_triggers_retrieval_gate_without_llm_call() -> None:
    # First two hits share a URL so suggestions() must deduplicate by url.
    h1 = make_hit(1, score=0.15, anchor="SAME")
    h2 = replace(h1, chunk_id=2, score=0.12, content="Duplicate section url chunk.")
    h3 = make_hit(3, score=0.10, anchor="OTHER")
    llm = ScriptedLLM([])

    res = await answer("Unrelated question?", [h1, h2, h3], llm, tau=0.5, context_k=8)
    assert res.refused
    assert res.refusal_gate == "retrieval"
    assert res.attempts == 0
    assert len(llm.calls) == 0
    assert [s.url for s in res.suggestions] == [
        "https://docs.test/page-1.html#SAME",
        "https://docs.test/page-3.html#OTHER",
    ]


async def test_tau_none_skips_gate1_and_context_k_limits_sources() -> None:
    hits = [make_hit(i, score=-5.0) for i in range(1, 5)]
    good = ModelAnswer(
        refused=False,
        answer="shared_buffers controls memory caching [S1].",
        citations=[
            ModelCitation(
                source="S1",
                quote="shared_buffers controls memory caching",
            )
        ],
    ).model_dump_json()
    llm = ScriptedLLM([good])

    res = await answer("What does shared_buffers do?", hits, llm, tau=None, context_k=2)
    assert not res.refused
    assert res.attempts == 1
    sent_prompt = llm.calls[0][0][1]
    assert "[S1]" in sent_prompt
    assert "[S2]" in sent_prompt
    assert "[S3]" not in sent_prompt
