"""
app/domain/models.py

Pydantic data models that form the canonical schema for every object
flowing through the research pipeline.  These are the single source of
truth for data shapes — providers, scrapers, and synthesisers all speak
this language.
"""
from datetime import datetime
from typing import Annotated

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Search layer
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """A single organic result returned by a search provider."""

    title: str = Field(..., min_length=1, description="Page title")
    url: Annotated[str, AnyHttpUrl] = Field(..., description="Canonical URL")
    snippet: str = Field(default="", description="Short excerpt from the SERP")
    position: int = Field(..., ge=1, description="1-based rank on the SERP")


class SearchResponse(BaseModel):
    """Validated response envelope from a search provider."""

    query: str = Field(..., min_length=1)
    results: list[SearchResult] = Field(default_factory=list)
    provider: str = Field(default="unknown")
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Scraping layer
# ---------------------------------------------------------------------------


class ScrapedPage(BaseModel):
    """Raw, cleaned content extracted from a single URL."""

    url: str = Field(..., description="Source URL")
    title: str = Field(default="")
    raw_text: str = Field(default="", description="Boilerplate-removed body text")
    word_count: int = Field(default=0, ge=0)
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    success: bool = Field(default=True)
    error: str | None = Field(default=None)

    @field_validator("word_count", mode="before")
    @classmethod
    def compute_word_count(cls, v: int, info: object) -> int:  # noqa: ANN001
        """Auto-compute word count from raw_text when not explicitly provided."""
        # If a caller already set it, trust them.
        if v:
            return v
        # Reach into the sibling field via the validation context.
        raw_text: str = getattr(info, "data", {}).get("raw_text", "")
        return len(raw_text.split())


# ---------------------------------------------------------------------------
# Synthesis / Report layer
# ---------------------------------------------------------------------------


class SourceCitation(BaseModel):
    """A single cited source referenced in the final report."""

    title: str
    url: str
    relevance_note: str = Field(
        default="",
        description="One-sentence note on why this source was included",
    )


class ResearchReport(BaseModel):
    """The final structured output of the research pipeline."""

    topic: str = Field(..., min_length=1, description="Original research query")
    executive_summary: str = Field(..., description="3-5 sentence TL;DR")
    body: str = Field(..., description="Full Markdown body of the report")
    key_findings: list[str] = Field(
        default_factory=list,
        description="Bullet-point key findings extracted by the LLM",
    )
    sources: list[SourceCitation] = Field(
        default_factory=list, description="All sources used to compile the report"
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: str = Field(default="")
    token_usage: dict[str, int] = Field(
        default_factory=dict,
        description="input_tokens / output_tokens from the LLM response",
    )
