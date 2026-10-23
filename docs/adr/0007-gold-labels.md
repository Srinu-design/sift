# ADR-0007: Gold labels at section or page level, never chunk ids

Status: Accepted · Date: 2026-10-12 (recorded 2026-10-23)

## Context

The golden set must survive every change to parsing and chunking, because the eval exists to
compare those changes. A label that names chunk 4711 dies with the next re-chunk.

## Decision

Gold is a location in the source: `{"doc": "pg17/sql-createindex", "section":
"SQL-CREATEINDEX-CONCURRENTLY"}` for HTML (the section's anchor) and `{"doc": "sebi/<id>",
"page": 2}` for PDFs. A retrieved chunk hits a gold location when it is the same document (and
so the same version) and the gold section is on its anchor path (the section or one inside it),
or its page range covers the gold page (`eval/metrics.py`). Each answerable question also
carries `evidence` quotes that the checker proves are inside its gold section (`make
golden-check`), and unanswerable ones an `absent` pattern proven absent from that version.

## Consequences

- Five chunker variants (ADR-0004) were evaluated against the same golden set without relabelling.
- One gold section per question is sometimes too narrow: the test error analysis found one answer
  that was right but cited a second section that also answers it (q046), and citation accuracy
  counts the same section in another version as a miss. Labels can list several gold locations
  where that matters (multi-section questions do).
- Section granularity cannot tell a chunk at the start of a long section from one at its end.

## Alternatives considered

- **Chunk ids:** invalid after any re-chunk.
- **Answer strings only** (no location): cannot score retrieval or citations.
- **Character offsets:** precise, but break when the parser changes whitespace or markup.
