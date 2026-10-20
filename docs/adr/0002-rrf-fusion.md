# ADR-0002: Reciprocal-rank fusion with k = 60 instead of weighted score fusion

Status: Accepted · Date: 2026-10-20

## Context

Hybrid search runs dense (cosine over bge-small embeddings) and lexical (Postgres full-text,
`ts_rank_cd`) retrieval side by side and must merge two ranked lists of 50. Their scores are
on unrelated scales: cosine similarity sits in a narrow band near 0.6–0.9, while `ts_rank_cd`
is unbounded and depends on how many query words a chunk contains. Both retrievers run under the
same tenant and metadata filters, so fusion only ever merges filtered lists.

## Decision

Fuse with reciprocal-rank fusion (Cormack, Clarke and Büttcher, SIGIR 2009):
score(d) = Σ 1 / (60 + rank_r(d)) over the retrievers that returned d. Ranks only, no scores.
Ties keep first-seen order (dense list first), so a run is deterministic
(`src/sift/retrieval/fuse.py`, `tests/unit/test_fuse.py`).

The lexical side uses an **OR** of the question's lexemes, not `websearch_to_tsquery`'s AND.

## Measured (dev split, 71 answerable questions, golden v1)

| Strategy | hit@5 | hit@8 | recall@10 | MRR@10 |
|---|---|---|---|---|
| naive (dense) | 0.887 | 0.930 | 0.958 | 0.747 |
| lexical (OR) | 0.817 | 0.873 | 0.852 | 0.607 |
| lexical (AND) | 0.352 | 0.352 | 0.336 | 0.290 |
| hybrid (RRF, OR) | 0.887 | 0.958 | 0.951 | 0.738 |
| hybrid (RRF, AND) | 0.873 | 0.915 | 0.958 | 0.766 |

Paired bootstrap on per-question differences (`make eval-compare`):

- **Hybrid beats lexical** on every metric; the 95% interval excludes zero for hit@8, recall@k
  and MRR@10 (MRR +0.131 [+0.062, +0.203]).
- **Hybrid does not beat naive.** hit@8 +0.028 [−0.028, +0.085], MRR@10 −0.009
  [−0.093, +0.078]: no difference either way.

Why hybrid ties naive: it moved 31 of 71 questions, 17 up and 14 down. The gains are exact
identifiers that dense search places near but not on: 7 of the 10 version-specific questions
changed and 6 improved (`SERIALIZE`, `IS JSON`, `JSON_EXISTS`, `ON_ERROR`, `MAINTAIN`), and
version_specific hit@5 went from 0.70 to 1.00. The losses are chunks that dense ranks first but
lexical misses or ranks low: equal-weight RRF lets two medium ranks beat one first place. The
worst are paraphrases and questions in user words: q071 "back to exactly how it was at 3pm
yesterday" (lexical finds nothing; dense rank 3 → hybrid 19), q002 (18 → 40), q038 (1 → 6).
On this golden set dense is already strong (hit@8 0.93), so there is little for fusion to add
at k = 8. The case for hybrid is recall at depth for the reranker: the cross-encoder (ADR-0003)
re-orders the fused 50, and the version-specific gains are hits that dense alone pushes out of
the top 5.

## Alternatives considered

- **Weighted score fusion** (min-max normalise each list, α·dense + (1−α)·lexical). Needs α
  tuned on dev and re-tuned whenever a model or the tsvector weights change; min-max over 50
  rows is dominated by the top score. Not run here, because RRF plus a reranker is the standard
  design. Revisit if the reranker cannot recover the dense first places that RRF demotes.
- **Weighted RRF** (a larger weight for dense). Would likely recover the paraphrase losses; it
  adds the same tuning knob. A one-line change in `rrf()` if the rerank results ask for it.
- **AND lexical query.** Precise but misses most natural-language questions (lexical hit@8
  0.352). Inside hybrid it has the best MRR (0.766), because it adds few, precise candidates and
  demotes less. It is still worse than OR at hit@8 (0.915), and it returns nothing for wordy
  questions, so OR stays.
- **Dense only.** Equal on this golden set, but it misses the identifier questions that users of
  a docs site ask most.

## Consequences

- Filters run inside both retrievers, so a filtered top 8 is never short. Exact versus HNSW
  (forced with `enable_seqscan = off`), version-filtered, at 50: 86 dev questions, mean overlap
  0.999, minimum 0.92, no short lists (`make eval-overlap`).
- **On 1,903 chunks the planner does not use the HNSW index at all**: it scans and sorts, which
  is exact. The naive numbers above are exact-search numbers. HNSW starts to matter as the
  table grows, and the forced-index overlap above shows it keeps recall under filters when it does.
- Hybrid costs one extra query per request. Both run concurrently; `timings_ms` per stage is in
  every results file.

## When to revisit

- hybrid_rerank (week 4) does not beat naive on MRR@10: try weighted RRF before anything else.
- A corpus where lexical search is weak (scanned PDFs, other languages): measure per tenant.
