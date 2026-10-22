"""Snapshot tenant B's corpus: the latest SEBI circulars, as PDFs.

    uv run python corpus/fetch_sebi.py           # writes corpus/raw/sebi/*.pdf and
                                                 # corpus/sebi_manifest.jsonl

The listing shows one row per circular (date, then a link to its page); each circular page
links its PDF under sebi_data/attachdocs. Run once: evals use the manifest, never the live
site. SEBI circulars are public regulatory documents; raw files stay git-ignored. (RBI's file
server answers scripts with a CAPTCHA page, so tenant B uses SEBI.)
"""

import hashlib
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

LISTING = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0"
PDF_LINK = re.compile(r"https://www\.sebi\.gov\.in/sebi_data/attachdocs/[^\"'\s]+\.pdf")
HERE = Path(__file__).resolve().parent


def parse_listing(html: str) -> list[dict[str, str]]:
    """Circulars in page order: title, page url and date."""
    rows: list[dict[str, str]] = []
    for tr in BeautifulSoup(html, "lxml").find_all("tr"):
        cells = tr.find_all("td")
        link = tr.find("a", href=re.compile(r"/legal/circulars/"))
        if len(cells) < 2 or link is None:
            continue
        rows.append(
            {
                "title": " ".join(link.get_text().split()),
                "page": link["href"],
                "published_on": datetime.strptime(cells[0].get_text(strip=True), "%b %d, %Y")
                .date()
                .isoformat(),
            }
        )
    return rows


def main() -> int:
    raw = HERE / "raw" / "sebi"
    raw.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, str]] = []
    with httpx.Client(timeout=60, headers={"User-Agent": "sift-corpus-snapshot"}) as http:
        for row in parse_listing(http.get(LISTING).raise_for_status().text):
            found = PDF_LINK.search(http.get(row["page"]).raise_for_status().text)
            if found is None:
                print(f"no PDF on {row['page']}", file=sys.stderr)
                continue
            url = found.group(0)
            stem = url.rsplit("/", 1)[1].removesuffix(".pdf")
            path = raw / f"{stem}.pdf"
            if not path.exists():
                data = http.get(url).raise_for_status().content
                if not data.startswith(b"%PDF"):  # a block page is not a document
                    print(f"not a PDF: {url}", file=sys.stderr)
                    continue
                path.write_bytes(data)
            time.sleep(0.5)  # be polite to the regulator's server
            out.append(
                {
                    "doc": f"sebi/{stem}",
                    "url": url,
                    "path": f"raw/sebi/{stem}.pdf",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source": "sebi",
                    "title": row["title"],
                    "page_url": row["page"],
                    "published_on": row["published_on"],
                    "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )
    (HERE / "sebi_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{len(out)} circulars -> corpus/sebi_manifest.jsonl", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
