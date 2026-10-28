"""Unit tests for deterministic citation validation and mapping."""

from dataclasses import replace

from sift.answer.citations import check, cited
from sift.answer.prompt import ModelAnswer, ModelCitation
from tests.conftest import make_hit

PG_HBA = make_hit(
    heading="Server Administration > Client Authentication > pg_hba.conf",
    anchor="AUTH-PG-HBA-CONF",
    anchor_path=("AUTH-PG-HBA-CONF",),
    source_uri="https://www.postgresql.org/docs/17/auth-pg-hba-conf.html",
    title="Client Authentication",
    content=(
        "Client authentication is controlled by a configuration file, "
        "which traditionally is named pg_hba.conf and is stored in the database cluster."
    ),
)


def test_valid_answer_passes() -> None:
    sources = [PG_HBA]
    ans = ModelAnswer(
        refused=False,
        answer="Client authentication uses pg_hba.conf [S1].",
        citations=[
            ModelCitation(
                source="S1",
                quote="Client authentication is controlled by a configuration file",
            )
        ],
    )
    assert check(ans, sources) == []


def test_unknown_id_in_text_fails() -> None:
    sources = [PG_HBA]
    ans = ModelAnswer(
        refused=False,
        answer="Client authentication uses pg_hba.conf [S2].",
        citations=[
            ModelCitation(
                source="S1",
                quote="Client authentication is controlled by a configuration file",
            )
        ],
    )
    errors = check(ans, sources)
    assert any("[S2]" in e for e in errors)


def test_unknown_id_in_citations_fails() -> None:
    sources = [PG_HBA]
    ans = ModelAnswer(
        refused=False,
        answer="Client authentication uses pg_hba.conf [S1].",
        citations=[
            ModelCitation(
                source="S3",
                quote="Client authentication is controlled by a configuration file",
            )
        ],
    )
    errors = check(ans, sources)
    assert any("S3" in e for e in errors)


def test_quote_not_in_chunk_fails() -> None:
    sources = [PG_HBA]
    ans = ModelAnswer(
        refused=False,
        answer="Client authentication uses pg_hba.conf [S1].",
        citations=[
            ModelCitation(
                source="S1",
                quote="This quote does not exist anywhere in the source text",
            )
        ],
    )
    errors = check(ans, sources)
    assert any("not found" in e for e in errors)


def test_quote_case_and_whitespace_normalised() -> None:
    sources = [PG_HBA]
    ans = ModelAnswer(
        refused=False,
        answer="Client authentication uses pg_hba.conf [S1].",
        citations=[
            ModelCitation(
                source="S1",
                quote='"`CLIENT   AUTHENTICATION\nis   controlled by a configuration file`"',
            )
        ],
    )
    assert check(ans, sources) == []


def test_quote_with_changed_punctuation_passes_fuzzy() -> None:
    sources = [PG_HBA]
    # ";" instead of "," after "configuration file" -> >90% contiguous match
    ans = ModelAnswer(
        refused=False,
        answer="Client authentication uses pg_hba.conf [S1].",
        citations=[
            ModelCitation(
                source="S1",
                quote="Client authentication is controlled by a configuration file;",
            )
        ],
    )
    assert check(ans, sources) == []


def test_uncited_sentence_fails_and_partial_disclaimer_passes() -> None:
    sources = [PG_HBA]
    bad = ModelAnswer(
        refused=False,
        answer="Client authentication uses pg_hba.conf [S1]. Second sentence has no citation.",
        citations=[
            ModelCitation(
                source="S1",
                quote="Client authentication is controlled by a configuration file",
            )
        ],
    )
    errors = check(bad, sources)
    assert any("Second sentence" in e for e in errors)

    partial = ModelAnswer(
        refused=False,
        answer=(
            "Client authentication uses pg_hba.conf [S1]. "
            "The sources do not cover LDAP certificate pinning."
        ),
        citations=[
            ModelCitation(
                source="S1",
                quote="Client authentication is controlled by a configuration file",
            )
        ],
    )
    assert check(partial, sources) == []


def test_refused_answer_skips_checks() -> None:
    sources = [PG_HBA]
    refused = ModelAnswer(
        refused=True,
        answer="",
        citations=[],
        refusal_reason="Not covered in sources.",
    )
    assert check(refused, sources) == []


def test_cited_maps_fields_and_dedupes() -> None:
    s1 = replace(
        PG_HBA,
        heading="Indexes > Index Types > GIN",
        anchor="GIN",
        doc_version="17",
        source_uri="https://www.postgresql.org/docs/17/gin.html",
        title="GIN Indexes",
    )
    s2 = replace(
        PG_HBA,
        chunk_id=2,
        heading="Appendix > Release Notes",
        anchor=None,
        anchor_path=(),
        page_start=42,
        page_end=42,
        doc_version=None,
        source_uri="https://www.postgresql.org/docs/release.pdf",
        title="Release Notes PDF",
    )
    ans = ModelAnswer(
        refused=False,
        answer="Use GIN [S2][S1]. GIN is fast [S1].",
        citations=[
            ModelCitation(source="S1", quote="Client authentication"),
            ModelCitation(source="S2", quote="Client authentication"),
        ],
    )
    out = cited(ans, [s1, s2])
    assert [c.n for c in out] == [1, 2]
    assert out[0].title == "GIN Indexes"
    assert out[0].version == "17"
    assert out[0].section == "GIN"
    assert out[0].url == "https://www.postgresql.org/docs/17/gin.html#GIN"
    assert out[0].page is None

    assert out[1].title == "Release Notes PDF"
    assert out[1].version is None
    assert out[1].section == "Release Notes"
    assert out[1].url == "https://www.postgresql.org/docs/release.pdf"
    assert out[1].page == 42
