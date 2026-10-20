"""Unit tests for metadata filters."""

from datetime import date

import pytest
from pydantic import ValidationError

from sift.retrieval.filters import Filters


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValidationError):
        Filters.model_validate({"unknown_field": "17"})


def test_iso_dates_parsed() -> None:
    f = Filters.model_validate(
        {
            "version": "17",
            "published_after": "2024-01-15",
            "published_before": "2024-12-31",
            "sources": ["pgdocs", "wiki"],
        }
    )
    assert f.version == "17"
    assert f.published_after == date(2024, 1, 15)
    assert f.published_before == date(2024, 12, 31)
    assert f.sources == ("pgdocs", "wiki")


def test_published_after_greater_than_before_rejected() -> None:
    with pytest.raises(ValidationError):
        Filters(
            published_after=date(2024, 6, 2),
            published_before=date(2024, 6, 1),
        )


def test_empty_sources_rejected() -> None:
    with pytest.raises(ValidationError):
        Filters(sources=())
    with pytest.raises(ValidationError):
        Filters.model_validate({"sources": []})


def test_params_output() -> None:
    empty = Filters()
    assert empty.params() == {
        "version": None,
        "published_after": None,
        "published_before": None,
        "sources": None,
    }

    populated = Filters(
        version="16",
        published_after=date(2023, 9, 14),
        published_before=date(2023, 9, 14),
        sources=("pgdocs",),
    )
    assert populated.params() == {
        "version": "16",
        "published_after": date(2023, 9, 14),
        "published_before": date(2023, 9, 14),
        "sources": ["pgdocs"],
    }
