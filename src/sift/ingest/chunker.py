"""Structure-aware chunking: sections first, then ~400-token windows with ~15% overlap.

- A chunk never spans two sections, so its citation points at one real section.
- Inside a section, blocks are packed up to TARGET tokens (heading included, because the
  heading is embedded with the content). The next chunk starts with the last ~OVERLAP tokens
  of the previous chunk's last paragraph.
- A code block or table that fits is never split. One that alone exceeds the budget splits
  at line (code) or row (table, header repeated) boundaries; a paragraph splits at sentences.
- Tokens are counted with the embedding model's own tokenizer, and any chunk over MAX_TOKENS
  fails the job: the model would silently truncate it.
"""

import hashlib
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import groupby

from tokenizers import Tokenizer

from sift.config import get_settings
from sift.ingest.parsers import Block, ParseError

# Ablation knobs (week 6), environment only: the defaults are the measured choice (ADR-0004).
TARGET_TOKENS = int(os.environ.get("CHUNK_TARGET_TOKENS", "400"))
OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "60"))  # 15% of TARGET
EMBED_HEADING = os.environ.get("CHUNK_EMBED_HEADING", "1") != "0"
MAX_TOKENS = 512  # bge-small-en-v1.5's limit, special tokens included
NAME = f"headings-{TARGET_TOKENS}-{round(100 * OVERLAP_TOKENS / TARGET_TOKENS)}" + (
    "" if EMBED_HEADING else "-nohead"
)

_SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+")


class ChunkTooLong(ParseError):
    pass


@lru_cache
def tokenizer() -> Tokenizer:
    settings = get_settings()
    tok = Tokenizer.from_pretrained(settings.embed_model, revision=settings.embed_revision)
    tok.no_truncation()  # its tokenizer.json truncates at 512, which would hide long chunks
    tok.no_padding()
    return tok


def count_tokens(text: str, special: bool = True) -> int:
    return len(tokenizer().encode(text, add_special_tokens=special).ids)


def embed_text(heading: str, content: str) -> str:
    """What the model embeds: the heading carries context ("this option") the content lacks."""
    return f"{heading}\n{content}" if EMBED_HEADING else content


@dataclass(frozen=True, slots=True)
class Chunk:
    ordinal: int
    heading: str
    anchor: str | None
    anchor_path: tuple[str, ...]
    content: str
    token_count: int  # of embed_text, special tokens included
    page_start: int | None = None
    page_end: int | None = None
    content_hash: bytes = field(default=b"", repr=False)  # SHA-256 of embed_text


@dataclass(slots=True)
class _Piece:
    text: str
    kind: str
    tokens: int
    path: tuple[str, ...]
    page: int | None


def _hard_split(text: str, budget: int) -> list[str]:
    """Last resort for one sentence or line longer than the budget: cut at token offsets."""
    enc = tokenizer().encode(text, add_special_tokens=False)
    out = []
    for i in range(0, len(enc.offsets), budget):
        window = enc.offsets[i : i + budget]
        out.append(text[window[0][0] : window[-1][1]])
    return out


def _pack_units(units: list[str], budget: int, joiner: str, header: str = "") -> list[str]:
    """Greedy packing of sentences, lines or rows; a header (table) starts every piece."""
    pieces: list[str] = []
    current: list[str] = []
    used = count_tokens(header, special=False) if header else 0
    base = used
    for unit in units:
        n = count_tokens(unit, special=False)
        if n > budget - base:
            if current:  # what came before goes first, or the text comes out of order
                pieces.append(joiner.join([header, *current] if header else current))
                current, used = [], base
            for part in _hard_split(unit, budget - base):
                units_part = [header, part] if header else [part]
                pieces.append(joiner.join(units_part))
            continue
        if current and used + n > budget:
            pieces.append(joiner.join([header, *current] if header else current))
            current, used = [], base
        current.append(unit)
        used += n
    if current:
        pieces.append(joiner.join([header, *current] if header else current))
    return pieces


def _split_block(block: Block, budget: int) -> list[str]:
    if count_tokens(block.text, special=False) <= budget:
        return [block.text]
    if block.kind == "code":
        return _pack_units(block.text.split("\n"), budget, "\n")
    if block.kind == "table":
        header, *rows = block.text.split("\n")
        return _pack_units(rows, budget, "\n", header=header)
    return _pack_units(_SENTENCE_END.split(block.text), budget, " ")


def _tail(text: str, n: int) -> str:
    """The last ~n tokens of text, starting at a word boundary."""
    enc = tokenizer().encode(text, add_special_tokens=False)
    if len(enc.offsets) <= n:
        return text
    start = enc.offsets[-n][0]
    while start < len(text) and not text[start - 1].isspace():
        start += 1
    return text[start:]


def chunk(blocks: list[Block]) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(heading: str, anchor: str | None, pieces: list[_Piece]) -> None:
        content = "\n\n".join(p.text for p in pieces)
        text = embed_text(heading, content)
        tokens = count_tokens(text)
        if tokens > MAX_TOKENS:
            raise ChunkTooLong(f"chunk {len(chunks)} under {heading!r} has {tokens} tokens")
        path: list[str] = []  # every section the chunk touches, outermost first
        for p in pieces:
            path.extend(a for a in p.path if a not in path)
        pages = [p.page for p in pieces if p.page is not None]
        chunks.append(
            Chunk(
                ordinal=len(chunks),
                heading=heading,
                anchor=anchor,
                anchor_path=tuple(path),
                content=content,
                token_count=tokens,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                content_hash=hashlib.sha256(text.encode()).digest(),
            )
        )

    # PDFs carry a page on every block: chunks never cross a page, so page citations are exact.
    for (heading, anchor, _), group in groupby(blocks, key=lambda b: (b.heading, b.anchor, b.page)):
        # -2: [CLS] and [SEP]; the heading and its newline are embedded with every chunk.
        budget = TARGET_TOKENS - 2 - count_tokens(heading + "\n", special=False)
        if budget < OVERLAP_TOKENS * 2:
            raise ChunkTooLong(f"heading too long to leave room for content: {heading[:80]!r}")
        current: list[_Piece] = []
        used = 0
        for block in group:
            for text in _split_block(block, budget):
                piece = _Piece(
                    text, block.kind, count_tokens(text, False), block.anchor_path, block.page
                )
                if current and used + piece.tokens > budget:
                    emit(heading, anchor, current)
                    last = current[-1]
                    current, used = [], 0
                    if last.kind == "text":  # never start a chunk with half a code block or table
                        tail = _tail(last.text, OVERLAP_TOKENS)
                        overlap = _Piece(
                            tail, "text", count_tokens(tail, False), last.path, last.page
                        )
                        if overlap.tokens + piece.tokens <= budget:
                            current, used = [overlap], overlap.tokens
                current.append(piece)
                used += piece.tokens
        if current:
            emit(heading, anchor, current)
    return chunks
