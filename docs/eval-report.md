# Eval report

Golden set v1 (PostgreSQL docs 16–18), **dev split**, 71 answerable questions, strategy
`hybrid_rerank` unless stated. 95% intervals are paired bootstraps of per-question differences
against the baseline (`make eval-compare`); "real" means the interval excludes zero. The test
split is used once, at the end (week 7).

## Ablations (2026-10-23, `make ablations`)

One change at a time against the baseline: 400-token chunks with 15% overlap, heading embedded
with each chunk, cross-encoder `ms-marco-MiniLM-L6-v2` on the fused top 30, version filter on.
Index variants were ingested into their own collections (`eval/ablations.sh`,
`eval/results/ablations.log`).

| change | chunks | hit@5 | hit@8 | recall@10 | MRR@10 | verdict |
|---|---|---|---|---|---|---|
| **baseline** | 1,903 | 0.887 | 0.986 | 0.979 | 0.841 | |
| chunks of 200 tokens | 3,737 | 0.944 (+5.6) | 0.986 | 0.979 | 0.802 (−3.8) | no real difference; twice the chunks |
| chunks of 500 tokens | 1,590 | 0.930 (+4.2) | 0.958 (−2.8) | 0.993 | 0.796 (**−4.5, real**) | worse |
| no overlap | 1,949 | 0.901 | 0.986 | 0.979 | 0.809 (**−3.2, real**) | worse |
| heading not embedded | 1,903 | 0.887 | 0.986 | 0.979 | 0.841 | no effect after reranking |
| no version filter | 1,903 | 0.859 | 0.859 (**−12.7, real**) | 0.847 (**−13.1, real**) | 0.531 (**−31.0, real**) | the filter is essential |
| bge-reranker-v2-m3 | 1,903 | 0.986 (**+9.9, real**) | 0.986 | 0.979 | 0.854 | better, but 34 s per query |
| OR vs AND lexical query | | see ADR-0002 | | | | OR kept |
| rerank top 10 instead of 30 | | 0.915 | 0.958 (−2.8) | 0.951 | 0.844 | blocked by the CI gate (docs/ci-gate-demo.md) |

What they say:

- **Chunk size: 400 stays** (ADR-0004). 200 lifts hit@5 but loses MRR and doubles the chunks (and
  the embedding and storage cost); 500 loses MRR significantly and drops a gold section out of the
  top 8. **The guide's 800 cannot exist here:** bge-small reads 512 tokens, and the chunker refuses
  longer chunks rather than have the model silently truncate them.
- **Overlap earns its 2.5% more chunks:** without it MRR drops 3.2 points (significant).
- **Embedding the heading has no measurable effect on the final ranking.** The dense embeddings do
  change (no chunk pair shares a vector), but the cross-encoder reads heading + content and the
  full-text index weights headings, so the reranked top 10 is the same. Kept: it costs nothing
  and helps the dense-only strategy.
- **The version filter is the biggest single factor:** without it the same page from 16, 17 and 18
  competes, and MRR falls from 0.84 to 0.53. Questions that name a version should get the filter
  (the guide's optional "turn 'in Postgres 16' into a filter" is the next step).
- **A bigger reranker buys real quality at an impossible price on CPU:** bge-reranker-v2-m3
  (568M parameters) puts a gold section in the top 5 for 70 of 71 questions (hit@5 0.986 against
  0.887), but reranking takes 34 s per query at p50 and 77 s at p95 on the laptop, against 1.1 s.
  Worth it only on a GPU, or distilled; ADR-0003.
- Not run: an API embedding model (paid) and context size k = 5 / 12 for the answer step (each
  run costs ~80 model calls on the free tier; hit@5 and hit@10 above bound what k changes).

## Error analysis (test split)

Written in week 7, after the one test-split run: every miss labelled with one cause (parse error,
chunk boundary, lexical mismatch, embedding miss, wrong filter, reranker demoted it, answer
ignored context, bad gold label).
