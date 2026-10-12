from sift.ingest.parsers import ParseError, html, parser_for

PAGE = b"""<html><head><title>T</title></head><body>
<div id="docContent">
  <div class="navheader">Prev Up Next</div>
  <div class="refentry" id="SQL-CREATEINDEX">
    <div class="refnamediv"><h2>CREATE INDEX</h2>
      <p>CREATE INDEX &mdash; define a new index</p></div>
    <div class="refsect1" id="id-1.2"><h2>Parameters</h2>
      <div class="variablelist"><dl>
        <dt id="SQL-CREATEINDEX-PARAMS-UNIQUE">UNIQUE</dt>
        <dd><p>Checks for <acronym>TOAST</acronym>ed values and <code>NULL</code>s.</p></dd>
      </dl></div>
      <div class="refsect2" id="SQL-CREATEINDEX-CONCURRENTLY"><h3>Building Indexes Concurrently</h3>
        <p>Builds without locking out writes.</p>
        <div class="note"><h3>Note</h3><p>Cannot run in a transaction block.</p></div>
        <pre class="programlisting">CREATE INDEX CONCURRENTLY i
  ON t (a);</pre>
        <table><tr><th>Mode</th><th>Conflicts</th></tr>
          <tr><td><p>SHARE</p><p>row</p></td><td>ROW EXCLUSIVE</td></tr></table>
      </div>
    </div>
  </div>
  <div class="navfooter">Prev Up Next</div>
</div></body></html>"""


def test_sections_anchors_and_heading_paths() -> None:
    doc = html.parse(PAGE)
    assert doc.title == "CREATE INDEX"
    texts = [b.text for b in doc.blocks]
    assert not any("Prev Up Next" in t for t in texts)  # navigation is not content
    concurrent = next(b for b in doc.blocks if "locking out writes" in b.text)
    assert concurrent.heading == "CREATE INDEX > Parameters > Building Indexes Concurrently"
    assert concurrent.anchor == "SQL-CREATEINDEX-CONCURRENTLY"
    assert concurrent.anchor_path == ("SQL-CREATEINDEX", "id-1.2", "SQL-CREATEINDEX-CONCURRENTLY")


def test_a_note_title_does_not_start_a_section() -> None:
    doc = html.parse(PAGE)
    note = next(b for b in doc.blocks if "transaction block" in b.text)
    assert note.anchor == "SQL-CREATEINDEX-CONCURRENTLY"
    assert "Note" in [b.text for b in doc.blocks]


def test_inline_markup_joins_without_spaces() -> None:
    doc = html.parse(PAGE)
    assert "Checks for TOASTed values and NULLs." in [b.text for b in doc.blocks]


def test_code_keeps_lines_and_tables_keep_rows() -> None:
    doc = html.parse(PAGE)
    code = next(b for b in doc.blocks if b.kind == "code")
    assert code.text == "CREATE INDEX CONCURRENTLY i\n  ON t (a);"
    table = next(b for b in doc.blocks if b.kind == "table")
    assert table.text == "Mode | Conflicts\nSHARE row | ROW EXCLUSIVE"


def test_parser_choice_by_file_type() -> None:
    assert parser_for("a.HTML") == "html-v1"
    assert parser_for("circular.pdf") == "docling-v1"
    try:
        parser_for("notes.docx")
    except ParseError:
        pass
    else:
        raise AssertionError("an unsupported type must be a permanent parse error")
