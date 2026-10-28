"""PDF parsing with Docling. Skipped where the pdf extra is not installed, so CI (plain
`uv sync`) never runs them. The first run downloads Docling's layout models."""

import pytest

from sift.ingest.chunker import chunk
from sift.ingest.parsers import ParseError, parse
from tests.pdfgen import tiny_pdf

pytest.importorskip("docling")


def test_blocks_keep_their_page_and_chunks_stay_within_a_page() -> None:
    data = tiny_pdf(
        [
            [(20, "Master Direction on Payments"), (11, "Entities must keep escrow accounts.")],
            [(11, "Net worth shall be twenty five crore rupees by March.")],
        ]
    )
    doc = parse("docling-v1", data)
    by_page = {b.page: b.text for b in doc.blocks}
    assert "escrow accounts" in by_page[1]
    assert "twenty five crore" in by_page[2]
    assert [(c.page_start, c.page_end) for c in chunk(doc.blocks)] == [(1, 1), (2, 2)]


def test_a_broken_pdf_is_a_permanent_parse_error() -> None:
    with pytest.raises(ParseError):
        parse("docling-v1", b"%PDF-1.4 this is not really a pdf")
