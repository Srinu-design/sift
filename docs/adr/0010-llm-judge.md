# ADR-0010: LLM judge, validated against human labels (not yet trusted for correctness)

Status: Accepted, with an open validation gap · Date: 2026-10-23

## Context

Faithfulness (is every claim supported by the sources?) and answer correctness cannot be computed
from strings: they need a reader. A model can read every answer on every run; a person cannot.
The build guide's rule: use a different, stronger model than the answerer at temperature 0 with
a rubric, hand-label 40 answers, and trust the judge only if Cohen's kappa against those labels is
at least 0.6.

## Decision

`eval/judge.py`: faithfulness by the RAGAS definition (atomic claims, each supported by the numbered
sources or not; supported / all), correctness 0 / 0.5 / 1, temperature 0, structured JSON, one
retry on invalid JSON, a question skipped (and retried on the next run) after three failed calls.
Labels: `eval/label.py`, a seeded sample of 40 answered dev questions; agreement: `eval/kappa.py`
(raw agreement, Cohen's kappa, quadratic-weighted kappa, confusion table).

**Judge model: `gemini-3.1-flash-lite`.** On the free tier (checked 2026-10-23) the pro models have
no quota, `gemini-3.5-flash` allows 20 requests a day, and Gemma 4 31B answered 500/503 for long
stretches. This is an older version of the answerer's own family, not the stronger model the
guide asks for.

## Measured (dev split, 69 answered questions, answer prompt v2)

| | judge-v1 rubric | judge-v2 rubric (current) |
|---|---|---|
| faithfulness | 1.000 (152/152 claims) | 1.000 |
| correctness (mean) | 0.851 | 0.895 [0.84, 0.94] |
| agreement with 40 labels, correctness | 32/40, kappa 0.584 (labels v1) | 34/40, **kappa 0.489** (labels v2) |
| agreement, faithfulness | 40/40, all one class | 40/40, all one class |

**How the labels and rubric evolved, in order:**
1. srinu labelled 40 answers ("correct vs the reference"). Kappa with judge-v1: 0.584.
2. The 8 disagreements went both ways and so did the labels themselves: some answers missing a
   detail of the reference got 1, others 0.5. The rubric was ambiguous.
3. One written rule was chosen: *grade against what the question asks; reference detail the
   question did not ask for is optional.* Claude proposed re-grades under that rule; srinu
   reviewed and approved them (12 labels changed 0.5 -> 1, each marked `suggested_by: claude`,
   `rubric: v2` in `eval/labels/human-v1.jsonl`).
4. The judge prompt got the same rule (judge-v2). Agreement 34/40, kappa 0.489.

**Why kappa is low although agreement is 85%:** 35 of the 40 labels are "fully correct", so
chance agreement is already high and kappa (which subtracts it) is small; one or two
disagreements move it a lot. Faithfulness is the extreme case: every label on both sides is
"faithful", so agreement is perfect and kappa says nothing.

**Faithfulness was checked another way:** a false sentence planted in each of 5 answers was
flagged in 5 of 5 (faithfulness 0.50-0.67 for those answers), so the judge is not waving claims
through.

## Consequences

- **Judge faithfulness: usable** (perfect agreement, plus the planted-claim check).
- **Judge correctness: not trusted yet** (kappa < 0.6). The README reports it with that caveat.
- Rubric v2 was written after seeing these 40 labels, so the 0.489 is measured on the data that
  shaped it and is, if anything, optimistic.

## Next (not done)

- Label a fresh, stratified sample (with deliberately half-right and wrong answers, for example
  from a weaker strategy or planted errors) so both classes are well represented, and compute
  kappa on it without touching the rubric.
- A stronger judge model once quota allows (paid tier), validated the same way.
