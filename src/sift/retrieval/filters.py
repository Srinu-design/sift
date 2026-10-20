"""Metadata filter model and SQL predicate for chunk retrieval."""

from datetime import date
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

FILTER_SQL = """
  AND (CAST(:version AS text) IS NULL OR c.doc_version = :version)
  AND (CAST(:published_after AS date) IS NULL OR c.published_on >= :published_after)
  AND (CAST(:published_before AS date) IS NULL OR c.published_on <= :published_before)
  AND (CAST(:sources AS text[]) IS NULL OR c.source = ANY(CAST(:sources AS text[])))
"""


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str | None = None
    published_after: date | None = None
    published_before: date | None = None
    sources: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.sources is not None and len(self.sources) == 0:
            raise ValueError("sources must contain at least one item")
        if (
            self.published_after is not None
            and self.published_before is not None
            and self.published_after > self.published_before
        ):
            raise ValueError("published_after must not be after published_before")
        return self

    def params(self) -> dict[str, object]:
        return {
            "version": self.version,
            "published_after": self.published_after,
            "published_before": self.published_before,
            "sources": list(self.sources) if self.sources is not None else None,
        }
