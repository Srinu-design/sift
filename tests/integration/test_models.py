"""The real pinned models. Skipped where the models extra is not installed, so CI (plain
`uv sync`) never runs them. Loading takes a few seconds."""

import math

import pytest

pytest.importorskip("sentence_transformers")

from sift.models_svc.encoder import QUERY_INSTRUCTION, encoder


def test_embeddings_are_384_dims_and_normalised() -> None:
    [v] = encoder().embed(["How do I build an index without blocking writes?"], "passage")
    assert len(v) == 384
    assert math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-4)  # cosine = dot product


def test_queries_get_the_retrieval_instruction_and_passages_do_not() -> None:
    text = "vacuum full locks"
    [query], [passage] = encoder().embed([text], "query"), encoder().embed([text], "passage")
    [manual] = encoder().embed([QUERY_INSTRUCTION + text], "passage")
    assert query != passage
    assert max(abs(a - b) for a, b in zip(query, manual, strict=True)) < 1e-5


def test_the_reranker_puts_the_relevant_passage_first_with_scores_in_0_1() -> None:
    scores = encoder().rerank(
        "How do I build an index without blocking writes?",
        [
            "VACUUM reclaims storage occupied by dead tuples.",
            "CREATE INDEX CONCURRENTLY builds the index without locking out writes.",
        ],
    )
    assert all(0.0 < s < 1.0 for s in scores)
    assert scores[1] > 0.9 > 0.1 > scores[0]
