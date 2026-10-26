# ADR-0003: Rerank the fused top 30 with ms-marco-MiniLM-L6-v2 on CPU

Status: Accepted · Date: 2026-10-20

## Context

Hybrid search (ADR-0002) ties dense search on dev: equal-weight RRF lets two medium ranks beat
one first place. A cross-encoder reads the question and each passage together, so it can
re-order the fused list on relevance instead of rank position. It costs one model pass per
(question, passage) pair, on the CPU of one small EC2 host that also runs everything else.

## Decision

`hybrid_rerank`: dense 50 + lexical 50 → RRF → the top **30** scored by
`cross-encoder/ms-marco-MiniLM-L6-v2` (pinned commit) on (question, heading + content) in one
batch on sift-models, sigmoid scores in 0–1 → top 8 to the answer step
(`src/sift/retrieval/rerank.py`, `rerank_depth` in `config.py`).

## Measured

Quality, dev split, 71 answerable questions (golden v1):

| Strategy | hit@5 | hit@8 | recall@10 | MRR@10 |
|---|---|---|---|---|
| naive | 0.887 | 0.930 | 0.958 | 0.747 |
| hybrid | 0.887 | 0.958 | 0.951 | 0.738 |
| hybrid_rerank, top 50 | 0.887 | 0.986 | 0.979 | 0.834 |
| **hybrid_rerank, top 30** | 0.887 | 0.986 | 0.979 | **0.841** |

Paired bootstrap, hybrid_rerank (top 50) minus naive: MRR@10 +0.086 [+0.018, +0.163],
hit@8 +0.056 [+0.014, +0.113]. Minus hybrid: MRR@10 +0.096 [+0.022, +0.168]. The first change
whose interval excludes zero. hit@5 does not move (0.887 for all three).

Latency through sift-models, 30 timed calls after 3 warm-ups, 50 random corpus chunks
(`make rerank-latency`, `loadtest/results/20261020-rerank/`), on the development laptop
(20 logical CPUs, no GPU):

| Pairs | p50 | p95 |
|---|---|---|
| 50 | 1,769 ms | 1,977 ms |
| 30 | 1,188 ms | 1,269 ms |

Retrieval before it (dense and lexical concurrently, RRF) takes about 10–20 ms at p95: the
reranker is ~99% of retrieval time. Top 30 keeps the quality and costs 60% of top 50.

## Alternatives considered

- **No reranker.** Ties naive (ADR-0002) and gives gate 1 no calibrated score to threshold.
- **BAAI/bge-reranker-v2-m3.** Stronger and multilingual, but about 568M parameters against
  MiniLM's 22M. Measured in the week 6 ablation (docs/eval-report.md): hit@5 0.887 -> 0.986
  (significant), MRR 0.841 -> 0.854, but reranking takes 34 s per query at p50 (77 s p95) on
  the laptop CPU against 1.1 s, and the model is too large next to the others on a 4 GB host.
- **Rerank in the API process.** Every API replica would load the model; sift-models loads it
  once for every caller.
- **An API reranker (Cohere, Voyage).** Paid per call, and a network hop on every query.

## Consequences

- About 1.2 s per query on the laptop. On the EC2 host's 2 vCPUs a rerank took ~3.2–3.5 s
  (docs/deploy.md), and it is the first bottleneck at 10x.
- The cross-encoder is trained on MS MARCO web queries. It scores questions in user wording low:
  of the three answerable dev questions that gate 1 refuses, two are paraphrases and the third
  ("a lock identified by a number" for advisory locks) avoids the docs' terms too (ADR-0008).

## When to revisit

- p95 on EC2 above ~2 s: export to ONNX (int8), cut passages to 256 tokens, or rerank the top 20,
  and publish each against this table. **This trigger is met** (~3.2–3.5 s on EC2,
  docs/deploy.md); the work is deferred because the live demo gets little traffic.
- hit@5 stays flat while MRR rises: the top 8 is already right and the order inside it is what
  improves, so check whether a smaller context_k loses anything.
