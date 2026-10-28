"""Unit tests for the LLM judge parser, faithfulness scoring, and retry handling."""

from typing import Any

import pytest

from eval.judge import JudgeReply, faithfulness_score, judge_rows
from tests.conftest import ScriptedLLM


def test_faithfulness_from_parsed_reply() -> None:
    three_claims = JudgeReply.model_validate_json(
        """
        {
          "claims": [
            {"claim": "c1", "supported": true, "source": "S1"},
            {"claim": "c2", "supported": false, "source": null},
            {"claim": "c3", "supported": true, "source": "S2"}
          ],
          "correctness": 1.0,
          "correctness_reason": "Matches all key facts."
        }
        """
    )
    assert faithfulness_score(three_claims.claims) == pytest.approx(0.6667)

    no_claims = JudgeReply.model_validate_json(
        '{"claims": [], "correctness": 0.5, "correctness_reason": "Partial."}'
    )
    assert faithfulness_score(no_claims.claims) == pytest.approx(1.0)


def test_correctness_rounding() -> None:
    r_half = JudgeReply.model_validate(
        {"claims": [], "correctness": 0.4, "correctness_reason": "Close to half."}
    )
    assert r_half.correctness == pytest.approx(0.5)

    r_one = JudgeReply.model_validate(
        {"claims": [], "correctness": 0.9, "correctness_reason": "Close to one."}
    )
    assert r_one.correctness == pytest.approx(1.0)

    r_null = JudgeReply.model_validate(
        {"claims": [], "correctness": None, "correctness_reason": "Unanswerable."}
    )
    assert r_null.correctness is None


async def test_judged_row_builder_skips_refused_and_retries_invalid_json() -> None:
    golden_by_id: dict[str, dict[str, Any]] = {
        "q1": {
            "id": "q1",
            "question": "How do I raise max_connections?",
            "answerable": True,
            "reference_answer": "Set max_connections in postgresql.conf and restart.",
        },
        "q2": {
            "id": "q2",
            "question": "Unanswered question?",
            "answerable": True,
            "reference_answer": "Some reference.",
        },
    }
    rows: list[dict[str, Any]] = [
        {
            "id": "q2",
            "type": "factual",
            "answerable": True,
            "refused": True,
            "refusal_gate": "retrieval",
            "answer": "",
            "citations": [],
            "sources": [],
        },
        {
            "id": "q1",
            "type": "factual",
            "answerable": True,
            "refused": False,
            "refusal_gate": None,
            "answer": "Set max_connections in postgresql.conf [S1].",
            "citations": [
                {"n": 1, "url": "https://docs.test/c#1", "section": "Conn", "version": "17"}
            ],
            "sources": [
                {
                    "n": 1,
                    "label": "PostgreSQL 17 - Connections",
                    "text": "Set max_connections in postgresql.conf and restart the server.",
                }
            ],
        },
    ]

    valid_reply = (
        '{"claims": [{"claim": "Set max_connections in postgresql.conf", '
        '"supported": true, "source": "S1"}], '
        '"correctness": 0.5, "correctness_reason": "Missed server restart."}'
    )
    llm = ScriptedLLM(["{malformed json", valid_reply])

    judged = await judge_rows(rows, golden_by_id, llm)
    assert len(judged) == 1
    assert judged[0]["id"] == "q1"
    assert judged[0]["faithfulness"] == pytest.approx(1.0)
    assert judged[0]["correctness"] == pytest.approx(0.5)
    assert len(llm.calls) == 2
    assert len(llm.calls[1]) == 3
    assert llm.calls[1][1] == ("model", "{malformed json")
