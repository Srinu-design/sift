"""Snapshot the PostgreSQL docs corpus once, with a manifest. Evals never re-crawl.

    uv run python corpus/fetch.py            # writes corpus/raw/pg{16,17,18}/*.html
                                             # and corpus/manifest.jsonl

Chapters are walked through their table of contents, so a page added in one version is
picked up without editing this list. A page that does not exist in a version (404) is
skipped and reported, which is expected: pg_combinebackup only exists from 17 on.
"""

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

VERSIONS = ("16", "17", "18")
BASE = "https://www.postgresql.org/docs/{version}/"
# Chapter roots: the root page plus every page its table of contents links to.
CHAPTERS = ("indexes", "textsearch", "mvcc", "backup", "maintenance")
# Single pages: JSON, a subset of the SQL command reference, and the backup tools.
PAGES = (
    "datatype-json",
    "functions-json",
    "sql-analyze",
    "sql-cluster",
    "sql-copy",
    "sql-createindex",
    "sql-delete",
    "sql-dropindex",
    "sql-explain",
    "sql-insert",
    "sql-lock",
    "sql-merge",
    "sql-reindex",
    "sql-set-transaction",
    "sql-update",
    "sql-vacuum",
    "app-pgbasebackup",
    "app-pgcombinebackup",
    "app-pgdump",
    "app-pgrestore",
)
USER_AGENT = "sift-corpus-snapshot/1 (one-off snapshot for a RAG eval; github.com/Srinu-design)"
DELAY_SECONDS = 0.3

ROOT = Path(__file__).resolve().parent


def toc_pages(html: str) -> list[str]:
    """Pages a chapter's table of contents links to, in order, without fragments."""
    soup = BeautifulSoup(html, "lxml")
    seen: list[str] = []
    for a in soup.select("#docContent div.toc a[href]"):
        page = str(a["href"]).split("#")[0].removesuffix(".html")
        if page and page not in seen:
            seen.append(page)
    return seen


def main() -> int:
    raw = ROOT / "raw"
    manifest: list[dict[str, str]] = []
    missing: list[str] = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30) as client:

        def get(version: str, page: str) -> str | None:
            url = urljoin(BASE.format(version=version), f"{page}.html")
            time.sleep(DELAY_SECONDS)
            response = client.get(url)
            if response.status_code == 404:
                missing.append(f"pg{version}/{page}")
                return None
            response.raise_for_status()
            path = raw / f"pg{version}" / f"{page}.html"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            manifest.append(
                {
                    "doc": f"pg{version}/{page}",
                    "url": url,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(response.content).hexdigest(),
                    "version": version,
                    "source": chapter_of.get(page, page),
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )
            return response.text

        for version in VERSIONS:
            chapter_of: dict[str, str] = {}
            pages: list[str] = []
            for chapter in CHAPTERS:
                html = get(version, chapter)
                if html is None:
                    continue
                chapter_of[chapter] = chapter
                for page in toc_pages(html):
                    chapter_of[page] = chapter
                    pages.append(page)
            for page in PAGES:
                # "sql-*" pages are one chapter (SQL commands); the rest are their own source.
                chapter_of[page] = "sql-commands" if page.startswith("sql-") else page
                pages.append(page)
            for page in pages:
                get(version, page)
            print(f"pg{version}: {sum(m['version'] == version for m in manifest)} pages")

    manifest.sort(key=lambda m: m["doc"])
    with (ROOT / "manifest.jsonl").open("w") as f:
        for line in manifest:
            f.write(json.dumps(line) + "\n")
    print(f"{len(manifest)} files; not found (expected for new pages): {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
