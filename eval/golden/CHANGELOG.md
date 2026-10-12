# Golden set changelog

Every results file names the golden-set version it used (`golden: "v1"`). Questions are never
edited in place once a version is published; fixes go into the next version.

## v1 — 2026-10-12 (corpus snapshot of 2026-10-12, `corpus/manifest.jsonl`)

123 questions for tenant `pgdocs`:

| Type | n | dev | test |
|---|---|---|---|
| factoid | 44 | 31 | 13 |
| procedural | 21 | 15 | 6 |
| version_specific | 15 | 10 | 5 |
| paraphrase | 12 | 8 | 4 |
| multi_section | 10 | 7 | 3 |
| unanswerable | 21 (17%) | 15 | 6 |
| **total** | **123** | **86** | **37** |

- **Split:** 70/30 per question type, shuffled with seed `20261012`, frozen in the file. Tune
  thresholds, chunk size and k on `dev` only; publish `test` numbers.
- **Gold** is labelled by section anchor (`{"doc": "pg17/sql-createindex", "section":
  "SQL-CREATEINDEX-CONCURRENTLY"}`), never by chunk id, so re-chunking keeps the set valid.
- **Unanswerable:** 8 ask about a feature in a version that does not have it (incremental
  backup, JSON_TABLE, MERGE RETURNING, COPY ON_ERROR in 16; REJECT_LIMIT, RETURNING OLD in 17…)
  and carry an `absent` pattern that the checker proves appears nowhere in that version's
  chunks. 13 ask about things that do not exist or are out of scope (pgvector HNSW, pg_cron,
  BM25, PgBouncer, PL/pgSQL, a made-up function and setting).
- Each answerable question carries **`evidence`**: short quotes the checker finds inside the
  gold section, so the answer is grounded in the section it cites.

### Verification status: drafted, not yet verified

All 123 were **drafted by Claude** (`drafted_by`), and **`verified_by` is null on every one.**
The build guide asks for a person to verify each question against the source page, and to
hand-write at least half. That has not happened yet. Until it does, treat v1 as a draft.

What *has* been checked mechanically (`make golden-check`, needs `make corpus` first):

- every gold document exists and matches the question's version filter;
- every gold section anchor exists in that document;
- every evidence quote appears in the gold section (case and whitespace aside);
- every `absent` pattern is absent from that version's chunks;
- ids unique, splits valid, at least 15% unanswerable.

The first draft failed 10 of these checks (8 quotes worded differently from the docs, 2
`absent` patterns that matched unrelated text); all were fixed against the page text.

**Second pass (Claude, same day):** every answerable question's reference answer was read
against the text of its gold section. 12 were corrected: one was wrong for this corpus (q042
quoted ANALYZE's "300 × statistics target" sample size, which is in PostgreSQL's source code
but not in the docs), and eleven said more than their gold section supports (for example q099
named `pg_maintain`, which is not on that page) or had a weak evidence quote. A reference
answer should be checkable against its own gold sections, because faithfulness is graded
against retrieved text. Still no `verified_by`: this was a model checking a model.

**To verify by hand:** `make golden-review` writes `review-v1.md`, a checklist with a link to
every gold section on postgresql.org.

To verify a question: open its link in `review-v1.md`, check the reference answer, edit what is
wrong in `v1.jsonl`, and set `verified_by`. Rewriting questions in
your own words also covers "hand-write at least half". Run `make golden-check` after editing.

### Known limits of v1

- Many factoid questions share words with their section, which favours lexical search;
  paraphrase questions (12) are the counterweight. Check the per-type breakdown, not just the total.
- Naive dense retrieval already scores hit@8 0.93 on dev (`eval/baselines/retrieval.json`).
  hit@8 leaves little room to show later gains; MRR@10 (0.75) and hit@5 (0.89) have more.
  v2 should add harder questions: subsection-level gold, more paraphrase and multi-section.
- Tenant B (SEBI/RBI circulars, ~15 questions with page-level gold) arrives with tenancy in
  week 5, as v2.
