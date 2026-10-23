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

## Final results: the test split (2026-10-23, run once)

The test split (37 PostgreSQL questions, 6 SEBI) was not used for any choice above. Retrieval
(`eval/results/20261023T1112*-test-v1.json`), 31 answerable questions:

| strategy | hit@5 | hit@8 | recall@10 | MRR@10 |
|---|---|---|---|---|
| naive | 0.839 [0.71, 0.94] | 0.935 [0.84, 1.00] | 0.919 | 0.681 [0.56, 0.81] |
| lexical | 0.710 [0.55, 0.84] | 0.839 [0.71, 0.97] | 0.887 | 0.547 [0.41, 0.69] |
| hybrid | 0.839 [0.68, 0.97] | 0.935 [0.84, 1.00] | 0.919 | 0.775 [0.65, 0.91] |
| **hybrid_rerank** | **0.968** [0.90, 1.00] | **0.968** [0.90, 1.00] | 0.952 | 0.744 [0.62, 0.86] |

Answers (`hybrid_rerank`, gemini-3.5-flash-lite, prompt answer-v2), PostgreSQL test split:

| | answered | refused |
|---|---|---|
| answerable (31) | 29 | 2 (gate 1) |
| unanswerable (6) | 0 | 6 (gate 1: 0, gate 2: 6) |

Refusal recall 1.00, wrong refusals 6.5% (2 of 31), 83% of answers cite a gold section, citation
accuracy 0.65. Judge (gemini-3.1-flash-lite, judge-v2): faithfulness 1.000 on 29 answers,
correctness 0.897 [0.83, 0.97], **with the caveat of ADR-0010** (kappa 0.489 on dev labels).
SEBI test split: 5 of 5 answerable answered citing a gold page, 1 of 1 unanswerable refused, the
injected-instruction question passed.

Test against dev: hit@5 and hit@8 of `hybrid_rerank` hold (0.968 against 0.887 / 0.986), but its
MRR is lower (0.744 against 0.841), below plain hybrid on this split. The error analysis below
shows why: the reranker moves version-specific gold sections from rank 1 to ranks 2-5.

## Error analysis (test split)

Every test question with a problem anywhere in the pipeline, one cause each (`hybrid_rerank`;
"rank" is the first gold chunk's rank after reranking, hybrid's rank in brackets).

| id | type | what happened | cause |
|---|---|---|---|
| q074 | paraphrase | "treat 'colour' and 'color' as the same word": gold (synonym dictionary) not in the top 10 (hybrid 34); gate 1 refused | embedding miss (the question avoids every term the section uses) |
| q072 | paraphrase | serialization failures in user words: gold ranked 1, but the rerank score fell below τ; gate 1 refused | gate-1 threshold: the cross-encoder scores user wording low (ADR-0003) |
| q089, q091, q095, q097 | version-specific | gold section at rank 3-5 (hybrid: 1); a sibling section of the same page first | reranker demoted it |
| q011, q067, q085 | factoid, paraphrase, multi-section | gold at rank 2-4 (hybrid: 1); q085 answered only half (one of its two sections) | reranker demoted it |
| q054 | procedural | gold at rank 2; the answer used the neighbouring GIN-index page and missed the expression index | reranker demoted it |
| q101 | version-specific | gold at rank 2 in both hybrid and rerank: the sibling MERGE section that also mentions the clause ranks first | chunk boundary (two sections share the answer) |
| q027, q062, q086 | factoid, procedural, multi-section | gold in the top 2 of the sources, but the answer cited other sources and was half right | answer ignored context |
| q042 | factoid | gold cited, judged half right on how ANALYZE sizes its sample | answer ignored context |
| q046 | procedural | correct answer (REINDEX CONCURRENTLY) citing the REINDEX page, not the gold section | bad gold label (too narrow: two sections answer it) |

Counts: **reranker demoted it 8**, answer ignored context 4, embedding miss 1, gate-1 threshold 1,
chunk boundary 1, bad gold label 1. No parse errors and no wrong filters.

What it says to fix next:

1. **The reranker on version-specific questions** (half the test misses): MiniLM, trained on web
   queries, prefers a section that repeats the question's words over the one that answers it.
   bge-reranker-v2-m3 fixed exactly this on dev (hit@5 0.986) but costs 34 s per query on CPU;
   on a GPU host, or rerank only when the fused top 1 and 2 are close.
2. **Answers using the second-best source**: 4 answers had the gold section in context and
   answered from another. A prompt rule ("prefer the source that answers the exact question")
   is the cheap experiment, measured on dev first.
3. **User-worded paraphrases** fail at both gates (q072, q074): query rewriting (stretch goal) or a
   reranker that handles paraphrase.
