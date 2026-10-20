"""Refusal gate 1 (retrieval confidence) and fallback reading suggestions."""

from sift.answer.citations import Citation, citation_of
from sift.retrieval.hit import Hit


def gate1(best_score: float | None, tau: float) -> bool:
    return best_score is None or best_score < tau


def suggestions(hits: list[Hit], n: int = 2) -> list[Citation]:
    out: list[Citation] = []
    seen_urls: set[str] = set()
    for idx, hit in enumerate(hits, start=1):
        cit = citation_of(idx, hit)
        if cit.url in seen_urls:
            continue
        seen_urls.add(cit.url)
        out.append(cit)
        if len(out) >= n:
            break
    return out
