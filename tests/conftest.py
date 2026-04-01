"""
tests/conftest.py

Shared pytest fixtures, marks, and configuration consumed by all test
modules.  pytest discovers this file automatically at collection time.

Fixtures defined here are available to every test without explicit import.
"""
import pytest

from app.domain.models import ResearchReport, ScrapedPage, SearchResponse, SearchResult, SourceCitation


# ---------------------------------------------------------------------------
# pytest-asyncio: event-loop scope
# ---------------------------------------------------------------------------
# asyncio_mode = "auto" is set in pyproject.toml [tool.pytest.ini_options].
# Individual fixtures that need an event loop use scope="function" (default).


# ---------------------------------------------------------------------------
# Domain-model factories
# ---------------------------------------------------------------------------

@pytest.fixture()
def make_search_result():
    """Factory fixture: returns a callable that builds SearchResult lists."""

    def _factory(n: int = 3, base_url: str = "https://example.com") -> list[SearchResult]:
        return [
            SearchResult(
                title=f"Result {i}",
                url=f"{base_url}/{i}",
                snippet=f"Snippet for result {i}.",
                position=i,
            )
            for i in range(1, n + 1)
        ]

    return _factory


@pytest.fixture()
def make_scraped_pages():
    """Factory fixture: returns a callable that builds ScrapedPage lists."""

    def _factory(
        n: int = 3,
        base_url: str = "https://example.com",
        word_count: int = 120,
    ) -> list[ScrapedPage]:
        return [
            ScrapedPage(
                url=f"{base_url}/{i}",
                title=f"Page {i} — Research Resource",
                raw_text=(
                    f"This is the main content for page {i}. " * 20
                ),
                word_count=word_count,
                success=True,
            )
            for i in range(1, n + 1)
        ]

    return _factory


@pytest.fixture()
def make_report():
    """Factory fixture: returns a callable that builds a ResearchReport."""

    def _factory(topic: str = "Test Research Topic", **overrides) -> ResearchReport:
        defaults = dict(
            topic=topic,
            executive_summary=(
                "This report summarises key findings about the research topic. "
                "Multiple credible sources were consulted during synthesis."
            ),
            body=(
                "## Introduction\n\nThe topic is well-studied in recent literature.\n\n"
                "## Key Areas\n\nSeveral areas stand out as particularly important.\n\n"
                "## Conclusion\n\nFurther research is recommended."
            ),
            key_findings=[
                "Finding one: significant progress in the field.",
                "Finding two: open challenges remain.",
                "Finding three: cross-disciplinary approaches show promise.",
                "Finding four: recent benchmarks exceed prior baselines.",
                "Finding five: industry adoption is accelerating.",
            ],
            sources=[
                SourceCitation(
                    title="Primary Source",
                    url="https://primary.example.com",
                    relevance_note="Core reference for the topic.",
                ),
                SourceCitation(
                    title="Secondary Source",
                    url="https://secondary.example.com",
                    relevance_note="Supporting evidence.",
                ),
            ],
            model_used="claude-sonnet-4-6",
            token_usage={"input_tokens": 3_200, "output_tokens": 820},
        )
        defaults.update(overrides)
        return ResearchReport(**defaults)

    return _factory


@pytest.fixture()
def make_search_response(make_search_result):
    """Factory fixture: wraps make_search_result in a SearchResponse."""

    def _factory(n: int = 3, query: str = "test query") -> SearchResponse:
        return SearchResponse(
            query=query,
            results=make_search_result(n),
            provider="serper",
        )

    return _factory
