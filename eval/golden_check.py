"""Check a golden set against the corpus snapshot. Exits 1 on any problem.

    python -m eval.golden_check [eval/golden/v1.jsonl]
    python -m eval.golden_check --review [eval/golden/v1.jsonl] > review.md   (a checklist)

- Every gold document exists, matches the question's version filter, and has the section.
- Every evidence quote appears (case and whitespace aside) inside one of the gold sections.
- Every 'absent' term (for "wrong version" unanswerable questions) appears nowhere in that
  version's chunks: proof the documents really cannot answer it.
- At least 15% of questions are unanswerable; ids are unique; the split is dev or test.

Passing this is not verification: a person still checks each answer against the page and
records verified_by. It catches made-up anchors and quotes from the wrong section.
"""

import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from sift.ingest.chunker import chunk
from sift.ingest.parsers import Block, parse, parser_for

ROOT = Path(__file__).resolve().parent.parent
TYPES = {"factoid", "procedural", "version_specific", "multi_section", "paraphrase", "unanswerable"}
MIN_UNANSWERABLE = 0.15


def norm(s: str) -> str:
    return " ".join(s.lower().split())


@lru_cache
def manifest() -> dict[str, dict[str, Any]]:
    lines = (ROOT / "corpus/manifest.jsonl").read_text().splitlines()
    return {m["doc"]: m for m in map(json.loads, lines)}


@lru_cache
def blocks(doc: str) -> list[Block]:
    m = manifest()[doc]
    path = ROOT / "corpus" / m["path"]
    return parse(parser_for(path.name), path.read_bytes()).blocks


def section_text(doc: str, section: str) -> str | None:
    texts = [b.text for b in blocks(doc) if section in b.anchor_path]
    return norm(" ".join(texts)) if texts else None


@lru_cache
def version_chunks(version: str | None) -> list[str]:
    docs = [d for d, m in manifest().items() if version is None or m["version"] == version]
    return [f"{c.heading}\n{c.content}" for d in docs for c in chunk(blocks(d))]


def check(rows: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    ids = Counter(r["id"] for r in rows)
    problems += [f"{i}: duplicate id" for i, n in ids.items() if n > 1]
    for r in rows:
        qid = r["id"]
        if r["type"] not in TYPES:
            problems.append(f"{qid}: unknown type {r['type']!r}")
        if r.get("split") not in ("dev", "test"):
            problems.append(f"{qid}: split must be dev or test")
        version = (r.get("filters") or {}).get("version")
        if r["answerable"] == (r["type"] == "unanswerable"):
            problems.append(f"{qid}: answerable={r['answerable']} contradicts type {r['type']}")
        if r["answerable"] and not r["gold"]:
            problems.append(f"{qid}: answerable but no gold location")
        if not r["answerable"] and r["gold"]:
            problems.append(f"{qid}: unanswerable but has gold")
        texts = []
        for g in r["gold"]:
            if g["doc"] not in manifest():
                problems.append(f"{qid}: no document {g['doc']}")
                continue
            if version and manifest()[g["doc"]]["version"] != version:
                problems.append(f"{qid}: gold {g['doc']} is not version {version}")
            if g.get("section"):
                text = section_text(g["doc"], g["section"])
                if text is None:
                    problems.append(f"{qid}: {g['doc']} has no section {g['section']}")
                else:
                    texts.append(text)
            else:
                texts.append(norm(" ".join(b.text for b in blocks(g["doc"]))))
        for quote in r.get("evidence", []):
            if texts and not any(norm(quote) in t for t in texts):
                problems.append(f"{qid}: evidence not in gold sections: {quote!r}")
        if r.get("absent"):
            rx = re.compile(r["absent"], re.I)
            hits = [c.split("\n")[0] for c in version_chunks(version) if rx.search(c)]
            if hits:
                problems.append(f"{qid}: 'absent' {r['absent']!r} found under {hits[0]!r}")
    unanswerable = sum(not r["answerable"] for r in rows)
    if rows and unanswerable / len(rows) < MIN_UNANSWERABLE:
        problems.append(f"only {unanswerable}/{len(rows)} unanswerable (need >= 15%)")
    return problems


def review(rows: list[dict[str, Any]]) -> str:
    """A checklist for the person verifying: each question, its answer, and links to the gold
    sections on postgresql.org (the snapshot's own URLs, with the section anchor)."""
    out = [
        "# Golden set review",
        "",
        "Tick a question once its answer is checked against the page,",
        "then set `verified_by` on its line in the JSONL.",
        "",
    ]
    for r in rows:
        mark = "x" if r.get("verified_by") else " "
        version = (r.get("filters") or {}).get("version", "-")
        out.append(f"- [{mark}] **{r['id']}** ({r['type']}, v{version}) {r['question']}")
        out.append(f"  - Answer: {r['reference_answer']}")
        for g in r["gold"]:
            url = manifest()[g["doc"]]["url"] + (f"#{g['section']}" if g.get("section") else "")
            out.append(f"  - Source: [{g['doc']}#{g.get('section') or ''}]({url})")
        if r.get("absent"):
            out.append(
                f"  - Proof it is unanswerable: `{r['absent']}` appears nowhere in that version"
            )
    return "\n".join(out) + "\n"


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--review"]
    path = Path(args[0]) if args else ROOT / "eval/golden/v1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if "--review" in sys.argv:
        print(review(rows), end="")
        return 0
    problems = check(rows)
    for p in problems:
        print(p)
    types = Counter(r["type"] for r in rows)
    splits = Counter(r["split"] for r in rows)
    verified = sum(bool(r.get("verified_by")) for r in rows)
    print(
        f"{path.name}: {len(rows)} questions {dict(types)}; splits {dict(splits)}; "
        f"human-verified {verified}/{len(rows)}; {len(problems)} problems"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
