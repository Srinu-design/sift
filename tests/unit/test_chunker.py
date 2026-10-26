"""The chunker with the real bge-small tokenizer (a small download, cached)."""

import itertools

import pytest

from sift.ingest import chunker
from sift.ingest.chunker import MAX_TOKENS, OVERLAP_TOKENS, TARGET_TOKENS, ChunkTooLong, chunk
from sift.ingest.parsers import Block


def block(text: str, kind: str = "text", heading: str = "Doc > A", anchor: str = "A") -> Block:
    return Block(text, kind, heading, anchor, ("DOC", anchor))  # type: ignore[arg-type]


def sentence(n: int) -> str:
    return f"Sentence {n} says the vacuum process reclaims space held by dead rows. "


def test_chunks_never_cross_a_section_and_stay_under_the_target() -> None:
    blocks = [block(sentence(i) * 3) for i in range(40)]
    blocks += [block(sentence(i) * 3, heading="Doc > B", anchor="B") for i in range(40)]
    chunks = chunk(blocks)
    assert {c.anchor for c in chunks} == {"A", "B"}
    for c in chunks:
        assert c.token_count <= TARGET_TOKENS + OVERLAP_TOKENS
        assert c.token_count == chunker.count_tokens(chunker.embed_text(c.heading, c.content))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap() -> None:
    chunks = chunk([block(sentence(i)) for i in range(120)])
    assert len(chunks) >= 3
    for prev, nxt in itertools.pairwise(chunks):
        tail = prev.content[-60:]
        assert tail.split()[-3] in nxt.content[:600]  # the start of next repeats prev's end


def test_a_code_block_that_fits_is_never_split() -> None:
    code = "\n".join(f"SELECT {i} FROM t WHERE a = {i};" for i in range(25))
    prose = [block(sentence(i) * 4) for i in range(6)]
    chunks = chunk([*prose, block(code, "code"), *prose])
    with_code = [c for c in chunks if "SELECT" in c.content]
    assert len(with_code) == 1 and code in with_code[0].content  # whole, once, never a piece
    later = chunks[with_code[0].ordinal + 1]
    assert "SELECT" not in later.content  # overlap never carries half a code block forward


def test_an_oversized_table_splits_by_rows_with_the_header_repeated() -> None:
    header = "Function | Description"
    rows = [
        f"jsonb_fn_{i}(x) | Returns the {i}th element of the given jsonb array" for i in range(80)
    ]
    chunks = chunk([block("\n".join([header, *rows]), "table")])
    assert len(chunks) > 1
    assert all(c.content.startswith(header) for c in chunks)
    joined = "\n".join(c.content for c in chunks)
    assert all(r in joined for r in rows)


def test_a_huge_heading_fails_the_job() -> None:
    with pytest.raises(ChunkTooLong):
        chunk([block("text", heading="Very long heading " * 120)])


def test_the_guard_counts_special_tokens() -> None:
    assert MAX_TOKENS == 512
    assert chunker.count_tokens("hello") == chunker.count_tokens("hello", special=False) + 2


def test_the_hash_covers_the_heading() -> None:
    a, b = (
        chunk([block("same text", heading="X > 16")]),
        chunk([block("same text", heading="X > 17")]),
    )
    assert a[0].content_hash != b[0].content_hash


def test_pdf_chunks_never_cross_a_page() -> None:
    blocks = [Block(sentence(i), "text", "Circular", None, (), page=1 + i // 5) for i in range(15)]
    chunks = chunk(blocks)
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (2, 2), (3, 3)]


def test_an_overlong_unit_keeps_the_text_around_it_in_order() -> None:
    """Units before an over-budget sentence go out first, not after its hard-split parts."""
    long = " ".join(f"w{i}" for i in range(80))
    pieces = chunker._pack_units(["Alpha one.", "Beta two.", long, "Gamma three."], 30, " ")
    assert pieces[0] == "Alpha one. Beta two."
    assert pieces[-1] == "Gamma three."
    # The hard split cuts at token offsets, sometimes inside a word: compare without spaces.
    assert "".join(pieces[1:-1]).replace(" ", "") == long.replace(" ", "")
