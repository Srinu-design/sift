"""Deterministic citation verification and source mapping."""

import difflib
import re

from pydantic import BaseModel

from sift.answer.prompt import ModelAnswer
from sift.retrieval.hit import Hit

QUOTE_MATCH_RATIO = 0.9

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(`])")
_TAG_RE = re.compile(r"\[S(\d+)\]")
_WS_RE = re.compile(r"\s+")
_SURROUNDING_QUOTES = "\"'`"


class Citation(BaseModel):
    n: int
    title: str
    version: str | None
    section: str
    url: str
    page: int | None


def _normalise(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def _quote_in_source(quote: str, hit: Hit) -> bool:
    norm_quote = _normalise(quote.strip().strip(_SURROUNDING_QUOTES))
    if not norm_quote:
        return False
    norm_source = _normalise(f"{hit.heading}\n{hit.content}")
    if norm_quote in norm_source:
        return True
    match = difflib.SequenceMatcher(
        None, norm_quote, norm_source, autojunk=False
    ).find_longest_match(0, len(norm_quote), 0, len(norm_source))
    return (match.size / len(norm_quote)) >= QUOTE_MATCH_RATIO


def check(answer: ModelAnswer, sources: list[Hit]) -> list[str]:
    if answer.refused:
        return []

    errors: list[str] = []
    text = answer.answer.strip()
    if not text:
        errors.append("answer is empty but refused is false")

    max_n = len(sources)
    for match in _TAG_RE.finditer(text):
        n = int(match.group(1))
        if n < 1 or n > max_n:
            errors.append(f"citation [S{n}] in answer is out of range 1..{max_n}")

    if text:
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        for sentence in sentences:
            if sentence.startswith("The sources do not"):
                continue
            if not _TAG_RE.search(sentence):
                errors.append(f"sentence is missing a [S#] citation: {sentence!r}")

    for cit in answer.citations:
        if not cit.source.startswith("S") or not cit.source[1:].isdigit():
            errors.append(f"invalid source identifier {cit.source!r}")
            continue
        n = int(cit.source[1:])
        if n < 1 or n > max_n:
            errors.append(f"citation source {cit.source} is out of range 1..{max_n}")
            continue
        if not _quote_in_source(cit.quote, sources[n - 1]):
            errors.append(f"quote for {cit.source} not found in source: {cit.quote!r}")

    return errors


def citation_of(n: int, hit: Hit) -> Citation:
    section = hit.heading.rsplit(" > ", 1)[-1]
    url = f"{hit.source_uri}#{hit.anchor}" if hit.anchor else hit.source_uri
    return Citation(
        n=n,
        title=hit.title,
        version=hit.doc_version,
        section=section,
        url=url,
        page=hit.page_start,
    )


def cited(answer: ModelAnswer, sources: list[Hit]) -> list[Citation]:
    if answer.refused:
        return []
    max_n = len(sources)
    used: set[int] = set()
    for match in _TAG_RE.finditer(answer.answer):
        n = int(match.group(1))
        if 1 <= n <= max_n:
            used.add(n)
    for cit in answer.citations:
        if cit.source.startswith("S") and cit.source[1:].isdigit():
            n = int(cit.source[1:])
            if 1 <= n <= max_n:
                used.add(n)
    return [citation_of(n, sources[n - 1]) for n in sorted(used)]
