# ADR-0009: Retrieval eval on every PR; generation eval nightly

Status: Accepted · Date: 2026-10-23

## Context

Retrieval is deterministic, local and free to run; generation and its judge need a paid or
rate-limited API and are noisy. Both must catch regressions, at a cost CI can carry on every PR.

## Decision

- **`eval-retrieval` on every PR and push to main** (`.github/workflows/ci.yml`): restore the
  index from a cache keyed by the corpus manifest, ingest code, migrations and model pins
  (rebuilt on a miss from the `corpus-v1` release asset, never the live site); run all four
  strategies on dev; `eval/gate.py` fails the build if `hybrid_rerank` loses more than 2 points
  of hit@8 or recall@10 overall, or 5 on any question type; the table goes to the job summary.
- **`eval-nightly`** (`.github/workflows/eval-nightly.yml`), nightly, on demand, and on the
  `run-full-eval` label: answers with the real model and the judge, hard-capped at 120 answer and
  120 judge calls per run.
- Updating a baseline is a deliberate `make eval-baseline` commit a reviewer sees in the diff.

## Measured

- The gate reproduced every gated number exactly on GitHub's runner (index rebuilt there in
  158 s); a cached run takes about 4–6 minutes.
- It blocked a deliberately bad PR (rerank depth 10, which raised hit@5 and MRR but lost 2.8
  points of hit@8 and recall@10): docs/ci-gate-demo.md.

## Alternatives considered

- **Generation eval on every PR:** quota and cost per PR, and an LLM-judged number is noisier
  than the 2-point threshold.
- **No cached index (re-embed every run):** slower CI and network dependence on every PR.
- **Gate on MRR too:** MRR moves with small reorderings that do not change what the answer step
  sees; hit@8 and recall@10 are what the top-8 context depends on.

## Consequences

- A change that only hurts answers (prompt, model) is caught nightly, not on the PR.
- The gate covers the dev split only; the test split stays untouched until the final eval.
