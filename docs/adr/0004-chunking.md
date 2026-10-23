# ADR-0004: Structure-aware chunking at ~400 tokens with 15% overlap, from the ablation

Status: Accepted · Date: 2026-10-23

## Context

Chunks are what retrieval ranks and what the answer cites. Too small and a chunk loses the context
that makes it relevant (and the index doubles); too large and one chunk mixes topics, ranks worse,
and costs more tokens per answer. The embedding model, bge-small-en-v1.5, reads at most 512
tokens; anything longer is silently truncated by the model.

## Decision

`src/sift/ingest/chunker.py`, chunker `headings-400-15`:

- Split on sections first (HTML h1–h3 with their anchors; PDF pages), never across one, so a
  citation's anchor or page is always exact.
- Inside a section, pack blocks up to ~400 tokens (heading included, counted with the embedding
  model's own tokenizer), with ~60 tokens (15%) of overlap; code blocks and tables are not split
  when they fit.
- Fail the ingest step if any chunk exceeds 512 tokens rather than let the model truncate it.
- Embed `heading + content`; the heading also gets full-text weight A.
- The chunker's name is part of every document version's unique key, so changing it re-chunks.

## Measured (dev split, hybrid_rerank, paired bootstrap; docs/eval-report.md)

| chunker | chunks | hit@5 | hit@8 | MRR@10 |
|---|---|---|---|---|
| 200 tokens, 15% | 3,737 | 0.944 | 0.986 | 0.802 (−3.8, n.s.) |
| **400 tokens, 15%** | **1,903** | **0.887** | **0.986** | **0.841** |
| 500 tokens, 15% | 1,590 | 0.930 | 0.958 | 0.796 (−4.5, significant) |
| 400 tokens, no overlap | 1,949 | 0.901 | 0.986 | 0.809 (−3.2, significant) |
| 400 tokens, heading not embedded | 1,903 | 0.887 | 0.986 | 0.841 (no change) |

## Alternatives considered

- **200 tokens:** higher hit@5 but lower MRR, not significant either way, at twice the chunks to
  embed, store and search. Revisit if hit@5 starts to matter more than MRR (smaller context k).
- **500 tokens** (the largest that fits under 512 with the heading): significantly worse MRR.
- **800 tokens** (the guide's third size): impossible with this embedder without truncation.
- **No overlap:** significantly worse MRR for 2.5% fewer chunks.
- **Fixed-size windows ignoring structure:** breaks the exact section/page citation, the product's
  core promise.

## Consequences

- 1,903 chunks for 182 pages; ~3,000 tokens of sources per answer at k = 8.
- The heading in the embedding does not change the reranked result; it is kept because it costs
  nothing and helps dense-only retrieval.

## When to revisit

- A different embedding model (longer context): rerun the sizes, including 800.
- A different reranker or context size k.
