"""HTML: split on h1-h3 sections, keep each section's anchor and heading path.

Written against the PostgreSQL docs (content in #docContent, sections as div.sect1/refsect1
with the id on the div), and general enough for pages whose headings carry their own id.
"""

from bs4 import BeautifulSoup, NavigableString, Tag

from sift.ingest.parsers import Block, BlockKind, ParsedDoc, ParseError

SECTION_HEADINGS = {"h1", "h2", "h3"}
BLOCKS = {"p", "pre", "table", "dt", "h4", "h5", "h6", "blockquote", "figcaption"}
NESTED_BLOCKS = {"p", "pre", "table", "ul", "ol", "dl", "div", "blockquote"}
# Navigation and page furniture; a.id_link is the "#" after each docs heading.
NOISE = ".navheader, .navfooter, div.toc, script, style, nav, a.id_link"
# Headings inside these are a box's title ("Note"), not a new section.
ADMONITIONS = {"note", "tip", "warning", "caution", "important", "sidebar"}


# Elements that separate words. Inline markup (code, acronym, links) must join its neighbours
# without a space ("TOASTed", not "TOAST ed"), but two paragraphs in a table cell must not.
_BREAKS = ["p", "br", "div", "li", "dt", "dd", "tr", "td", "th", "pre", "table"]


def _text(tag: Tag) -> str:
    for el in tag.find_all(_BREAKS):
        el.insert_after(" ")
    return " ".join(tag.get_text().split())


def _code(tag: Tag) -> str:
    lines = [line.rstrip() for line in tag.get_text().splitlines()]
    return "\n".join(lines).strip("\n")


def _table(tag: Tag) -> str:
    rows = []
    for tr in tag.find_all("tr"):
        cells = [_text(c) for c in tr.find_all(["th", "td"], recursive=False)]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _in_admonition(tag: Tag) -> bool:
    return any(set(p.get("class") or []) & ADMONITIONS for p in tag.parents)


def parse(data: bytes) -> ParsedDoc:
    soup = BeautifulSoup(data, "lxml")
    root = soup.select_one("#docContent") or soup.find("main") or soup.find("article") or soup.body
    if not isinstance(root, Tag):
        raise ParseError("no HTML body")
    for tag in root.select(NOISE):
        tag.decompose()

    first = root.find(list(SECTION_HEADINGS))
    first = first if isinstance(first, Tag) else None
    title_tag = soup.title
    title = _text(first) if first else (_text(title_tag) if title_tag else "")
    blocks: list[Block] = []
    stack: list[tuple[int, str, str | None]] = []  # (level, heading text, section id)

    def heading_path() -> str:
        names = [t for _, t, _ in stack]
        if title and (not names or names[0] != title):
            names.insert(0, title)
        return " > ".join(names)

    def emit(text: str, kind: BlockKind, path: tuple[str, ...]) -> None:
        if text.strip():
            anchor = stack[-1][2] if stack and stack[-1][2] else (path[-1] if path else None)
            blocks.append(Block(text, kind, heading_path(), anchor, path))

    def visit(node: Tag, path: tuple[str, ...]) -> None:
        if node.get("id") and node.name != "a" and node is not root:  # root's id: page furniture
            path = (*path, str(node["id"]))
        name = node.name
        if name in SECTION_HEADINGS and not _in_admonition(node):
            level = int(name[1])
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, _text(node), path[-1] if path else None))
            return
        if name == "pre":
            emit(_code(node), "code", path)
            return
        if name == "table":
            emit(_table(node), "table", path)
            return
        if name in BLOCKS or (name == "li" and not node.find(list(NESTED_BLOCKS))):
            emit(_text(node), "text", path)
            return
        if name in SECTION_HEADINGS:  # an admonition's title: keep it as text
            emit(_text(node), "text", path)
            return
        loose: list[str] = []  # text sitting directly in a container, outside any block
        for child in node.children:
            if isinstance(child, Tag):
                if loose:
                    emit(" ".join(" ".join(loose).split()), "text", path)
                    loose = []
                visit(child, path)
            elif isinstance(child, NavigableString) and child.strip():
                loose.append(str(child))
        if loose:
            emit(" ".join(" ".join(loose).split()), "text", path)

    visit(root, ())
    return ParsedDoc(title=title, blocks=blocks)
