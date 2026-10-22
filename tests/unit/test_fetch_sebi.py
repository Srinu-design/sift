"""The SEBI listing parser: one row per circular link, with the date cell before it."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "fetch_sebi", Path(__file__).resolve().parents[2] / "corpus" / "fetch_sebi.py"
)
assert spec and spec.loader
fetch_sebi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_sebi)

PAGE = """<table>
<tr><th>Date</th><th>Title</th></tr>
<tr role='row' class='odd'> <td>Aug 20, 2026</td>
  <td><a href="https://www.sebi.gov.in/legal/circulars/aug-2026/power-of-attorney_103824.html"
   class="points"> Acceptance of digitally
   signed Power of Attorney</a> </tr>
<tr><td>Aug 19, 2026</td>
  <td><a href="https://www.sebi.gov.in/other/page.html">Not a circular</a></td></tr>
</table>"""


def test_rows_are_circular_links_with_their_date() -> None:
    assert fetch_sebi.parse_listing(PAGE) == [
        {
            "title": "Acceptance of digitally signed Power of Attorney",
            "page": "https://www.sebi.gov.in/legal/circulars/aug-2026/power-of-attorney_103824.html",
            "published_on": "2026-08-20",
        }
    ]
