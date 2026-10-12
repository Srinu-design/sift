"""PDF: Docling's blocks in reading order, every one tagged with its page number.

Docling does layout analysis, reading order, tables and OCR for scanned pages (MIT). PDFs
have no section ids, so citations point at pages: the chunker packs blocks within a page and
records page_start and page_end. Section headers Docling detects become the heading path.
Page headers and footers (Docling's furniture layer) are left out.
"""

from functools import lru_cache
from io import BytesIO
from typing import Any

from sift.ingest.parsers import Block, BlockKind, ParsedDoc, ParseError


@lru_cache
def _converter() -> Any:
    from docling.document_converter import DocumentConverter

    return DocumentConverter()  # loads the layout models once per process


def parse(data: bytes) -> ParsedDoc:
    from docling.datamodel.base_models import ConversionStatus, DocumentStream
    from docling_core.types.doc import (
        CodeItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    # Only a file Docling rejects is permanent (422). Anything it raises (a model that failed
    # to download or load) is the environment's fault: it propagates as a 500 and is retried.
    result = _converter().convert(
        DocumentStream(name="upload.pdf", stream=BytesIO(data)), raises_on_error=False
    )
    if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        raise ParseError(f"Docling could not read the PDF: {result.status.value}")

    doc = result.document
    title = ""
    stack: list[tuple[int, str]] = []  # (level, heading text)
    blocks: list[Block] = []
    for item, _ in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        page = int(prov[0].page_no)
        if isinstance(item, TitleItem):
            title = title or " ".join(item.text.split())
            continue
        if isinstance(item, SectionHeaderItem):
            level = int(item.level)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, " ".join(item.text.split())))
            continue
        kind: BlockKind
        if isinstance(item, TableItem):
            frame = item.export_to_dataframe(doc)
            rows = [" | ".join(str(c) for c in frame.columns)]
            rows += [" | ".join(str(v) for v in row) for row in frame.itertuples(index=False)]
            text, kind = "\n".join(rows), "table"
        elif isinstance(item, CodeItem):
            text, kind = item.text, "code"
        elif isinstance(item, TextItem):  # paragraphs, list items, captions
            text, kind = " ".join(item.text.split()), "text"
        else:
            continue  # pictures: out of scope (read as pictures)
        heading = " > ".join([t for t in [title, *(h for _, h in stack)] if t]) or "Document"
        if text.strip():
            blocks.append(Block(text, kind, heading, None, (), page))
    return ParsedDoc(title=title, blocks=blocks)
