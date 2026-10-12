"""Parsers turn a raw file into blocks: paragraphs, code and tables, each tagged with the
section it sits in. The chunker packs blocks into chunks; parsers never decide chunk size."""

from dataclasses import dataclass
from typing import Literal

BlockKind = Literal["text", "code", "table"]


class ParseError(Exception):
    """The file cannot be parsed. Permanent: retrying the same bytes cannot help (HTTP 422)."""


@dataclass(frozen=True, slots=True)
class Block:
    text: str  # for a table: one row per line, the header row first
    kind: BlockKind
    heading: str  # 'CREATE INDEX > Parameters > Building Indexes Concurrently'
    anchor: str | None  # the id of the section this block sits in, for citation links
    anchor_path: tuple[str, ...]  # every section id from the page root down to the block
    page: int | None = None  # PDFs only


@dataclass(frozen=True, slots=True)
class ParsedDoc:
    title: str
    blocks: list[Block]


HTML = "html-v1"
PDF = "docling-v1"


def parser_for(filename: str) -> str:
    """The parser a file gets, chosen at upload and recorded on its version."""
    name = filename.lower().split("?")[0]
    if name.endswith((".html", ".htm")):
        return HTML
    if name.endswith(".pdf"):
        return PDF
    raise ParseError(f"unsupported file type: {filename}")


def parse(parser: str, data: bytes) -> ParsedDoc:
    if parser == HTML:
        from sift.ingest.parsers import html

        return html.parse(data)
    if parser == PDF:
        try:
            from sift.ingest.parsers import pdf  # Docling is heavy: import it only when needed
        except ImportError as exc:  # permanent until a build with the pdf extra is deployed
            raise ParseError("PDF parsing is not installed in this build (the pdf extra)") from exc
        return pdf.parse(data)
    raise ParseError(f"unknown parser {parser!r}")
