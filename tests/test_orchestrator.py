"""
tests/test_orchestrator.py

Unit tests for ResearchOrchestrator.

All external providers (search + LLM) and the WebScraper are mocked via
dependency injection so no network calls are made.  Tests are organised
around:
  1. Happy-path pipeline flow (events emitted in correct order, report returned)
  2. No-results branch  → NoSearchResultsError
  3. All-pages-fail branch → NoScrapedPagesError
  4. Low-yield warning (partial scrape success)
  5. Event callback contract (correct PipelineStage sequence)
"""
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import LowYieldWarning, NoScrapedPagesError, NoSearchResultsError
from app.domain.models import ResearchReport, ScrapedPage, SearchResponse, SearchResult, SourceCitation
from app.services.orchestrator import PipelineEvent, PipelineStage, ResearchOrchestrator

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_search_result(n: int = 3) -> list[SearchResult]:
    return [
        SearchResult(title=f"Result {i}", url=f"https://example.com/{i}", position=i)
        for i in range(1, n + 1)
    ]


def _make_scraped_pages(n: int = 2) -> list[ScrapedPage]:
    return [
        ScrapedPage(
            url=f"https://example.com/{i}",
            title=f"Page {i}",
            raw_text=f"Content for page {i} — enough words to be useful.",
            word_count=10,
            success=True,
        )
        for i in range(1, n + 1)
    ]


def _make_report() -> ResearchReport:
    return ResearchReport(
        topic="Test Topic",
        executive_summary="A concise summary.",
        body="## Body\n\nSome content.",
        key_findings=["Finding A", "Finding B"],
        sources=[SourceCitation(title="Source 1", url="https://example.com/1")],
        model_used="claude-sonnet-4-6",
        token_usage={"input_tokens": 100, "output_tokens": 50},
    )


def _make_search_provider(results: list[SearchResult]) -> MagicMock:
    provider = MagicMock()
    provider.search = AsyncMock(
        return_value=SearchResponse(query="test", results=results, provider="serper")
    )
    return provider


def _make_llm_provider(report: ResearchReport) -> MagicMock:
    provider = MagicMock()
    provider.synthesise = AsyncMock(return_value=report)
    return provider


def _make_scraper(pages: list[ScrapedPage]) -> MagicMock:
    scraper = MagicMock()
    scraper.scrape_all = AsyncMock(return_value=pages)
    return scraper


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    @pytest.mark.asyncio
    async def test_run_returns_research_report(self) -> None:
        report = _make_report()
        orchestrator = ResearchOrchestrator(
            search_provider=_make_search_provider(_make_search_result()),
            llm_provider=_make_llm_provider(report),
            scraper=_make_scraper(_make_scraped_pages()),
        )
        result = await orchestrator.run(topic="Test Topic")
        assert result is report

    @pytest.mark.asyncio
    async def test_run_calls_search_with_correct_args(self) -> None:
        search = _make_search_provider(_make_search_result())
        orchestrator = ResearchOrchestrator(
            search_provider=search,
            llm_provider=_make_llm_provider(_make_report()),
            scraper=_make_scraper(_make_scraped_pages()),
        )
        await orchestrator.run(topic="My Query", num_results=7)
        search.search.assert_called_once_with("My Query", num_results=7)

    @pytest.mark.asyncio
    async def test_run_calls_scrape_all_with_search_results(self) -> None:
        results = _make_search_result(n=4)
        scraper = _make_scraper(_make_scraped_pages())
        orchestrator = ResearchOrchestrator(
            search_provider=_make_search_provider(results),
            llm_provider=_make_llm_provider(_make_report()),
            scraper=scraper,
        )
        await orchestrator.run(topic="Topic")
        scraper.scrape_all.assert_called_once_with(results)

    @pytest.mark.asyncio
    async def test_run_calls_synthesise_with_scraped_pages(self) -> None:
        pages = _make_scraped_pages(n=3)
        llm = _make_llm_provider(_make_report())
        orchestrator = ResearchOrchestrator(
            search_provider=_make_search_provider(_make_search_result()),
            llm_provider=llm,
            scraper=_make_scraper(pages),
        )
        await orchestrator.run(topic="Topic")
        llm.synthesise.assert_called_once_with("Topic", pages)


# ---------------------------------------------------------------------------
# No results branch
# ---------------------------------------------------------------------------

class TestNoResults:
    @pytest.mark.asyncio
    async def test_raises_no_search_results_error(self) -> None:
        orchestrator = ResearchOrchestrator(
            search_provider=_make_search_provider([]),   # empty results
            llm_provider=_make_llm_provider(_make_report()),
            scraper=_make_scraper(_make_scraped_pages()),
        )
        with pytest.raises(NoSearchResultsError) as exc_info:
            await orchestrator.run(topic="Niche Topic XYZ")
        assert "Niche Topic XYZ" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_llm_not_called_when_no_results(self) -> None:
        llm = _make_llm_provider(_make_report())
        orchestrator = ResearchOrchestrator(
            search_provider=_make_search_provider([]),
            llm_provider=llm,
            scraper=_make_scraper(_make_scraped_pages()),
        )
        with pytest.raises(NoSearchResultsError):
            await orchestrator.run(topic="Topic")
        llm.synthesise.assert_not_called()


# ---------------------------------------------------------------------------
# All pages fail branch
# ---------------------------------------------------------------------------

class TestAllPagesFail:
    @pytest.mark.asyncio
    async def test_raises_no_scraped_pages_error(self) -> None:
        orchestrator = ResearchOrchestrator(
            search_provider=_make_search_provider(_make_search_result()),
            llm_provider=_make_llm_provider(_make_report()),
            scraper=_make_scraper([]),   # zero pages scraped
        )
        with pytest.raises(NoScrapedPagesError) as exc_info:
            await orchestrator.run(topic="Topic")
        assert exc_info.value.attempted == 3  # matched _make_search_result default

    @pytest.mark.asyncio
    async def test_llm_not_called_when_no_pages(self) -> None:
        llm = _make_llm_provider(_make_report())
        orchestrator = ResearchOrchestrator(
            search_provider=_make_search_provider(_make_search_result()),
            llm_provider=llm,
            scraper=_make_scraper([]),
        )
        with pytest.raises(NoScrapedPagesError):
            await orchestrator.run(topic="Topic")
        llm.synthesise.assert_not_called()


# ---------------------------------------------------------------------------
# Low-yield warning
# ---------------------------------------------------------------------------

class TestLowYield:
    @pytest.mark.asyncio
    async def test_low_yield_warning_emitted(self) -> None:
        """Warn when fewer than 40 % of pages succeed, but still complete."""
        # 5 search results, only 1 scraped → 20 % yield → below 40 % threshold
        results = _make_search_result(n=5)
        pages = _make_scraped_pages(n=1)
        orchestrator = ResearchOrchestrator(
            search_provider=_make_search_provider(results),
            llm_provider=_make_llm_provider(_make_report()),
            scraper=_make_scraper(pages),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = await orchestrator.run(topic="Topic")

        assert any(issubclass(w.category, LowYieldWarning) for w in caught)
        assert report is not None  # pipeline still completes

    @pytest.mark.asyncio
    async def test_no_warning_when_yield_above_threshold(self) -> None:
        """No warning when ≥ 40 % of pages succeed."""
        results = _make_search_result(n=3)
        pages = _make_scraped_pages(n=3)  # 100 % yield
        orchestrator = ResearchOrchestrator(
            search_provider=_make_search_provider(results),
            llm_provider=_make_llm_provider(_make_report()),
            scraper=_make_scraper(pages),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await orchestrator.run(topic="Topic")

        assert not any(issubclass(w.category, LowYieldWarning) for w in caught)


# ---------------------------------------------------------------------------
# Event callback contract
# ---------------------------------------------------------------------------

class TestEventCallback:
    @pytest.mark.asyncio
    async def test_events_emitted_in_order_on_success(self) -> None:
        """The callback must receive SEARCHING → SCRAPING → SYNTHESISING → COMPLETE."""
        events: list[PipelineEvent] = []

        orchestrator = ResearchOrchestrator(
            search_provider=_make_search_provider(_make_search_result()),
            llm_provider=_make_llm_provider(_make_report()),
            scraper=_make_scraper(_make_scraped_pages()),
            on_event=events.append,
        )
        await orchestrator.run(topic="Topic")

        stages = [e.stage for e in events]
        assert PipelineStage.SEARCHING in stages
        assert PipelineStage.SCRAPING in stages
        assert PipelineStage.SYNTHESISING in stages
        assert PipelineStage.COMPLETE in stages
        # Verify ordering — SEARCHING must come before SCRAPING
        assert stages.index(PipelineStage.SEARCHING) < stages.index(PipelineStage.SCRAPING)
        assert stages.index(PipelineStage.SCRAPING) < stages.index(PipelineStage.SYNTHESISING)

    @pytest.mark.asyncio
    async def test_error_event_emitted_on_no_results(self) -> None:
        events: list[PipelineEvent] = []
        orchestrator = ResearchOrchestrator(
            search_provider=_make_search_provider([]),
            llm_provider=_make_llm_provider(_make_report()),
            scraper=_make_scraper([]),
            on_event=events.append,
        )
        with pytest.raises(NoSearchResultsError):
            await orchestrator.run(topic="Topic")

        stages = [e.stage for e in events]
        assert PipelineStage.ERROR in stages

    @pytest.mark.asyncio
    async def test_progress_fraction_increases_monotonically(self) -> None:
        """Progress values reported by events must never go backwards."""
        events: list[PipelineEvent] = []
        orchestrator = ResearchOrchestrator(
            search_provider=_make_search_provider(_make_search_result()),
            llm_provider=_make_llm_provider(_make_report()),
            scraper=_make_scraper(_make_scraped_pages()),
            on_event=events.append,
        )
        await orchestrator.run(topic="Topic")

        progress_values = [e.progress for e in events]
        for prev, curr in zip(progress_values, progress_values[1:]):
            assert curr >= prev, f"Progress went backwards: {prev} → {curr}"
