# AI usage

Only what actually happened is recorded.

## Weeks 1–2 (2026-10-12): built with Claude Code

Claude Code (Opus 5.5) wrote the first version of everything in this repository at that
point, and the Hopper changes on branch `sift-ingest-support`, from the build guide. It was
committed on 2026-10-12 (`defac85`, "Weeks 1-2").

**The guide says to write these yourself, because interviewers probe them:** the retrieval
SQL (`src/sift/retrieval/dense.py`), the metric code (`eval/metrics.py`), and later RRF, the
citation checker and the refusal logic. Claude wrote the first two. Before claiming them:
read them line by line, be able to rewrite them without looking, and change anything you
would have done differently.

**The golden set (`eval/golden/v1.jsonl`) was drafted by Claude** and is unverified
(`verified_by` is null on all 123). See `eval/golden/CHANGELOG.md`.

### Mistakes caught, and what caught them

| What went wrong | Caught by | Fix |
|---|---|---|
| The HTML parser joined inline markup with spaces ("TOAST ed", "( integer )") | Reading section text while writing golden questions | Spaces only at block boundaries; unit test `test_inline_markup_joins_without_spaces` |
| A parser fix would never re-parse a file: versions were unique on (document, hash, chunker), not the parser | Noticed while fixing the bug above | `parser` added to the unique key |
| One pinned model revision (the cross-encoder) was a wrong hash | Checking the hashes against the Hugging Face API before use | Real commit pinned in `config.py` |
| First Hopper ingest **retry-stormed**: 60 worker slots sent 60 embed calls at once into one model server; calls timed out, retries piled on | Running the whole corpus through Hopper (jobs at attempt 7 of 8) | Embedding on its own Hopper queue with workers sized to the model server; the backlog waits in Postgres. ADR-0006 |
| An embed job for a deleted version reported the version complete | Dead `activate` jobs after that storm | `embed_batch` raises for a missing version; `test_a_job_for_a_deleted_version_is_permanent` |
| The PDF parser turned every Docling exception into a permanent 422, including "a model failed to load" | Docling failing on its first run | Only a file Docling rejects is a 422; other errors are retried |
| torchvision came from PyPI (CUDA build) while torch came from the CPU index: Docling could not load | Docling's first run (`torchvision::nms does not exist`) | torchvision pinned to the CPU index too |
| The kill test killed a worker *between* jobs, so it proved nothing | The test's own guard (`assert reruns >= 1`) | Realistic chunk sizes, so a batch takes seconds |
| `docker kill` is a manual stop: the restart policy never brought the worker back, and repeated runs removed both embed workers | The next e2e runs timing out | The test restarts the container itself |
| 10 golden questions failed the checker (8 quotes worded differently from the docs, 2 `absent` patterns matching unrelated text) | `make golden-check` | Fixed against the page text |
| Golden set: q042's answer used a number from PostgreSQL's source code that the docs never state; 11 more answers claimed more than their gold section says | A second pass reading every answer against its section | Corrected; see `eval/golden/CHANGELOG.md` |
| The `pdf` extra's types broke `mypy` in a venv without it (as in CI) | Running lint and tests in a fresh venv like CI's | `docling_core` marked optional for mypy |

## After weeks 1–2

This log was not kept up after weeks 1–2. What git shows: RRF (`src/sift/retrieval/fuse.py`)
arrived in week 3, the citation checker and refusal (`src/sift/answer/citations.py`,
`refusal.py`) in week 4, and the rest through week 7. Who wrote those is not recorded here.
Two things are recorded in the data itself: v2-sebi's questions were drafted by Claude
(`drafted_by`, `verified_by` null on all 18), and 12 of the 40 hand labels in
`eval/labels/human-v1.jsonl` were suggested by Claude and reviewed by a person (`suggested_by`,
commit 3f4ed91).
