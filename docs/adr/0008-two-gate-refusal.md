# ADR-0008: Two-gate refusal, with τ tuned on dev

Status: Accepted · Date: 2026-10-20

## Context

17% of the golden set is unanswerable: features asked about in a version that lacks them
("JSON_TABLE in 16"), and things the corpus does not cover (pgvector, pg_cron, Kubernetes).
The model already knows a lot about Postgres, so left alone it answers from memory: right
perhaps, but not supported by the sources, which is the failure Sift exists to prevent.
Refusing costs something too: a wrongly refused answerable question is a failed user.

## Decision

Two gates, cheapest first (`src/sift/answer/refusal.py`, `generate.py`):

1. **Gate 1, retrieval.** If the best cross-encoder score (sigmoid, 0–1) is below τ, refuse
   without calling the LLM. τ = **0.19**: the largest value that wrongly refuses at most 5% of
   answerable dev questions (`make eval-tau`). Applies to `hybrid_rerank` only, the one
   strategy whose scores are calibrated.
2. **Gate 2, model.** The prompt tells the model to set `refused: true` with a one-line reason
   when the numbered sources do not contain the answer, and to answer only the covered part
   (plus a "The sources do not ..." sentence) when they cover some of it.

A third path also ends in a refusal: an answer whose citations fail the code checks twice
(gate "citation"). Every refusal returns the two closest sections as "you may want to read".

## Measured (dev split, 86 questions, gemini-3.5-flash-lite, prompt answer-v1)

`make eval-answers`, `eval/results/20261020T111441Z-answers-dev-v1-hybrid_rerank.json`:

| | answered | refused |
|---|---|---|
| answerable (71) | 68 | 3 (gate 1: 3) |
| unanswerable (15) | 2 | 13 (gate 1: 4, gate 2: 9) |

- Refusal recall 0.867, precision 0.812, wrong-refusal rate 4.2%. No citation-gate refusals;
  3 questions needed the one retry.
- **Gate 1 alone catches 4 of 15.** The version-absent questions score 0.95–0.997: the
  neighbouring sections really are about the feature. Gate 2 catches 9 of the other 11, and
  never refused an answerable question.
- The 3 wrong refusals are all gate 1 and all phrased in user words (q058 "a lock identified
  by a number", q069, q076): the MS MARCO cross-encoder scores them low (ADR-0003).
- The 2 unanswerable questions answered are faithful partial answers, not inventions: q108 (old
  values from `UPDATE ... RETURNING` in 17) answers that RETURNING computes the new values;
  q120 (a Klingon text search configuration) explains how to create a configuration. Neither
  states what is missing, which prompt v1 asks for.
- Gate 1 skipped the LLM on 7 of 86 questions (8%).

Test-split numbers are published once, in week 7, as the build guide asks.

## Alternatives considered

- **Model-only refusal.** Every question pays for an LLM call, and the decision is hard to
  tune: there is no threshold to sweep.
- **Retrieval-only refusal (a higher τ).** τ = 0.85 refuses 10 of 15 unanswerable but 10% of
  answerable questions; τ = 0.977 reaches 13 of 15 at 30% wrong refusals, where the two
  gates together reach 13 of 15 at 4.2%
  (`eval/results/20261020T102625Z-tau-dev-v1.json`).
- **A separate "is this answerable?" classifier call.** A second LLM call per question for
  what gate 2 already decides while answering.

## Consequences

- A refusal costs an LLM call except below τ; gate 1 is a cost saver more than a safety net.
- Paraphrased questions are the ones at risk of a wrong refusal; a reranker that handles user
  wording (or a lower τ) is the lever.
- The tie-breaker for gate 2 is the prompt, so `prompt_version` is part of every log line and
  cache key.

## When to revisit

- Test-split wrong refusals above 5%, or a judge (week 5) finding unsupported claims in
  answered unanswerable questions.
- Another reranker (ADR-0003 ablation): τ must be swept again, since scores change.
