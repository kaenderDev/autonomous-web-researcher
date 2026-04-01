"""
tests/test_orchestrator.py

Unit and integration-style tests for ResearchOrchestrator.

All external providers (search + LLM) and WebScraper are injected as mocks
— zero network calls.  Tests are grouped by scenario:

A. Happy-path integration
   A1. Full pipeline: search → scrape → synthesise → report returned
   A2. Provider call-argument contracts verified

B. NoSearchResultsError branch
   B1. Exception raised with correct topic in message
   B2. Downstream stages NOT called

C. NoScrapedPagesError branch
   C1. Exception raised with correct `attempted` count
   C2. Downstream stages NOT called

D. LowYieldWarning (1-of-5 success)
   D1. Warning issued, pipeline still completes
   D2. No warning when yield is above threshold
   D3. Report is returned even at low yield

E. PipelineEvent callback contract
   E1. Stages emitted in correct order on success
   E2. ERROR stage emitted on NoSearchResultsError
   E3. ERROR stage emitted on NoScrapedPagesError
   E4. progress fractions are non-decreasing

F. Provider error propagation
   F1. SearchProviderError bubbles through unmodified
   F2. LLMProviderError bubbles through unmodified
"""
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import LowYieldWarning, NoScrapedPagesError, NoSearchResultsError
from app.domain.models import (
    ResearchReport,
    ScrapedPage,
    SearchResponse,
    SearchResult,
    SourceCitation,
)
from app.providers.base import LLMProviderError, SearchProviderError
from app.services.orchestrator import PipelineEvent, PipelineStage, ResearchOrchestrator

# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _results(n: int = 3) -> list[SearchResult]:
    return [
        SearchResult(title=f"Result {i}", url=f"https://example.com/{i}", position=i)
        for i in range(1, n + 1)
    ]


def _pages(n: int = 2) -> list[ScrapedPage]:
    return [
        ScrapedPage(
            url=f"https://example.com/{i}",
            title=f"Page {i}",
            raw_text=f"Substantive content for page {i} with enough words to be useful for synthesis.",
            word_count=14,
            success=True,
        )
        for i in range(1, n + 1)
    ]


def _report(topic: str = "Test Topic") -> ResearchReport:
    return ResearchReport(
        topic=topic,
        executive_summary="Summary of the research findings.",
        body="## Introduction\n\nBody content here.",
        key_findings=["Finding A", "Finding B", "Finding C", "Finding D", "Finding E"],
        sources=[SourceCitation(title="Source", url="https://example.com/1")],
        model_used="claude-sonnet-4-6",
        token_usage={"input_tokens": 1000, "output_tokens": 400},
    )


def _search_mock(result_list: list[SearchResult]) -> MagicMock:
    m = MagicMock()
    m.search = AsyncMock(
        return_value=SearchResponse(query="q", results=result_list, provider="serper")
    )
    return m


def _llm_mock(report: ResearchReport) -> MagicMock:
    m = MagicMock()
    m.synthesise = AsyncMock(return_value=report)
    return m


def _scraper_mock(page_list: list[ScrapedPage]) -> MagicMock:
    m = MagicMock()
    m.scrape_all = AsyncMock(return_value=page_list)
    return m


def _make_orchestrator(
    result_list: list[SearchResult],
    page_list: list[ScrapedPage],
    report: ResearchReport | None = None,
    on_event=None,
) -> ResearchOrchestrator:
    return ResearchOrchestrator(
        search_provider=_search_mock(result_list),
        llm_provider=_llm_mock(report or _report()),
        scraper=_scraper_mock(page_list),
        on_event=on_event,
    )


# ===========================================================================
# A. Happy-path integration
# ===========================================================================

class TestHappyPath:
    """A — full pipeline produces a ResearchReport with correct provenance."""

    @pytest.mark.asyncio
    async def test_a1_run_returns_research_report(self) -> None:
        expected = _report()
        orc = _make_orchestrator(_results(), _pages(), expected)
        result = await orc.run(topic="Test Topic")
        assert result is expected

    @pytest.mark.asyncio
    async def test_a1_report_topic_matches_input(self) -> None:
        orc = _make_orchestrator(_results(), _pages())
        result = await orc.run(topic="Quantum Computing")
        # The mock report always has topic="Test Topic", but the orchestrator
        # passes the caller's topic to synthesise — assert the call was made.
        orc._llm.synthesise.assert_called_once()
        call_args = orc._llm.synthesise.call_args
        assert call_args[0][0] == "Quantum Computing"

    @pytest.mark.asyncio
    async def test_a2_search_called_with_topic_and_num_results(self) -> None:
        search = _search_mock(_results())
        orc = ResearchOrchestrator(
            search_provider=search,
            llm_provider=_llm_mock(_report()),
            scraper=_scraper_mock(_pages()),
        )
        await orc.run(topic="Rust Language", num_results=12)
        search.search.assert_called_once_with("Rust Language", num_results=12)

    @pytest.mark.asyncio
    async def test_a2_scrape_all_receives_search_results(self) -> None:
        result_list = _results(n=5)
        scraper = _scraper_mock(_pages())
        orc = ResearchOrchestrator(
            search_provider=_search_mock(result_list),
            llm_provider=_llm_mock(_report()),
            scraper=scraper,
        )
        await orc.run(topic="Topic")
        scraper.scrape_all.assert_called_once_with(result_list)

    @pytest.mark.asyncio
    async def test_a2_synthesise_receives_scraped_pages(self) -> None:
        page_list = _pages(n=4)
        llm = _llm_mock(_report())
        orc = ResearchOrchestrator(
            search_provider=_search_mock(_results()),
            llm_provider=llm,
            scraper=_scraper_mock(page_list),
        )
        await orc.run(topic="Topic")
        llm.synthesise.assert_called_once_with("Topic", page_list)


# ===========================================================================
# B. NoSearchResultsError
# ===========================================================================

class TestNoSearchResults:
    """B — zero search results → NoSearchResultsError, downstream skipped."""

    @pytest.mark.asyncio
    async def test_b1_raises_no_search_results_error(self) -> None:
        orc = _make_orchestrator([], _pages())
        with pytest.raises(NoSearchResultsError) as exc_info:
            await orc.run(topic="Obscure Niche XYZ")
        assert "Obscure Niche XYZ" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_b2_scraper_not_called_on_no_results(self) -> None:
        scraper = _scraper_mock(_pages())
        orc = ResearchOrchestrator(
            search_provider=_search_mock([]),
            llm_provider=_llm_mock(_report()),
            scraper=scraper,
        )
        with pytest.raises(NoSearchResultsError):
            await orc.run(topic="Topic")
        scraper.scrape_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_b2_llm_not_called_on_no_results(self) -> None:
        llm = _llm_mock(_report())
        orc = ResearchOrchestrator(
            search_provider=_search_mock([]),
            llm_provider=llm,
            scraper=_scraper_mock(_pages()),
        )
        with pytest.raises(NoSearchResultsError):
            await orc.run(topic="Topic")
        llm.synthesise.assert_not_called()


# ===========================================================================
# C. NoScrapedPagesError
# ===========================================================================

class TestNoScrapedPages:
    """C — all scrapes fail → NoScrapedPagesError, LLM skipped."""

    @pytest.mark.asyncio
    async def test_c1_raises_no_scraped_pages_error(self) -> None:
        orc = _make_orchestrator(_results(n=5), [])   # 5 results, 0 pages
        with pytest.raises(NoScrapedPagesError) as exc_info:
            await orc.run(topic="Topic")
        assert exc_info.value.attempted == 5

    @pytest.mark.asyncio
    async def test_c1_error_message_contains_attempted_count(self) -> None:
        orc = _make_orchestrator(_results(n=8), [])
        with pytest.raises(NoScrapedPagesError) as exc_info:
            await orc.run(topic="Topic")
        assert "8" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_c2_llm_not_called_when_no_pages(self) -> None:
        llm = _llm_mock(_report())
        orc = ResearchOrchestrator(
            search_provider=_search_mock(_results()),
            llm_provider=llm,
            scraper=_scraper_mock([]),
        )
        with pytest.raises(NoScrapedPagesError):
            await orc.run(topic="Topic")
        llm.synthesise.assert_not_called()


# ===========================================================================
# D. LowYieldWarning — 1 of 5 succeeds
# ===========================================================================

class TestLowYield:
    """D — partial scrape success triggers warning but pipeline completes."""

    @pytest.mark.asyncio
    async def test_d1_low_yield_warning_issued_at_20_percent(self) -> None:
        """5 results, 1 page scraped (20%) → LowYieldWarning issued."""
        orc = _make_orchestrator(_results(n=5), _pages(n=1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = await orc.run(topic="Topic")

        low_yield_warnings = [w for w in caught if issubclass(w.category, LowYieldWarning)]
        assert low_yield_warnings, "Expected LowYieldWarning but none was issued"

    @pytest.mark.asyncio
    async def test_d3_pipeline_still_returns_report_at_low_yield(self) -> None:
        """Pipeline must complete even when yield is below threshold."""
        expected = _report()
        orc = _make_orchestrator(_results(n=5), _pages(n=1), expected)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = await orc.run(topic="Topic")
        assert result is expected

    @pytest.mark.asyncio
    async def test_d2_no_warning_at_100_percent_yield(self) -> None:
        orc = _make_orchestrator(_results(n=3), _pages(n=3))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await orc.run(topic="Topic")

        low_yield_warnings = [w for w in caught if issubclass(w.category, LowYieldWarning)]
        assert not low_yield_warnings

    @pytest.mark.asyncio
    async def test_d2_no_warning_at_exactly_40_percent_yield(self) -> None:
        """40 % is exactly the threshold — should NOT warn."""
        # 5 results, 2 pages = 40 %
        orc = _make_orchestrator(_results(n=5), _pages(n=2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await orc.run(topic="Topic")

        low_yield_warnings = [w for w in caught if issubclass(w.category, LowYieldWarning)]
        assert not low_yield_warnings

    @pytest.mark.asyncio
    async def test_d1_warning_at_39_percent_yield(self) -> None:
        """Below threshold (1/3 = 33 %) must warn."""
        orc = _make_orchestrator(_results(n=3), _pages(n=1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            await orc.run(topic="Topic")

        low_yield_warnings = [w for w in caught if issubclass(w.category, LowYieldWarning)]
        assert low_yield_warnings


# ===========================================================================
# E. PipelineEvent callback contract
# ===========================================================================

class TestEventCallback:
    """E — on_event receives the correct stages in the correct order."""

    @pytest.mark.asyncio
    async def test_e1_success_stages_emitted_in_order(self) -> None:
        events: list[PipelineEvent] = []
        orc = _make_orchestrator(_results(), _pages(), on_event=events.append)
        await orc.run(topic="Topic")

        stages = [e.stage for e in events]
        assert PipelineStage.SEARCHING in stages
        assert PipelineStage.SCRAPING in stages
        assert PipelineStage.SYNTHESISING in stages
        assert PipelineStage.COMPLETE in stages
        assert stages.index(PipelineStage.SEARCHING) < stages.index(PipelineStage.SCRAPING)
        assert stages.index(PipelineStage.SCRAPING) < stages.index(PipelineStage.SYNTHESISING)
        assert stages.index(PipelineStage.SYNTHESISING) < stages.index(PipelineStage.COMPLETE)

    @pytest.mark.asyncio
    async def test_e2_error_event_on_no_results(self) -> None:
        events: list[PipelineEvent] = []
        orc = _make_orchestrator([], _pages(), on_event=events.append)
        with pytest.raises(NoSearchResultsError):
            await orc.run(topic="Topic")
        assert any(e.stage == PipelineStage.ERROR for e in events)

    @pytest.mark.asyncio
    async def test_e3_error_event_on_no_pages(self) -> None:
        events: list[PipelineEvent] = []
        orc = _make_orchestrator(_results(), [], on_event=events.append)
        with pytest.raises(NoScrapedPagesError):
            await orc.run(topic="Topic")
        assert any(e.stage == PipelineStage.ERROR for e in events)

    @pytest.mark.asyncio
    async def test_e4_progress_fractions_non_decreasing(self) -> None:
        events: list[PipelineEvent] = []
        orc = _make_orchestrator(_results(), _pages(), on_event=events.append)
        await orc.run(topic="Topic")

        fractions = [e.progress for e in events]
        for prev, curr in zip(fractions, fractions[1:]):
            assert curr >= prev, f"Progress went backwards: {prev:.2f} → {curr:.2f}"

    @pytest.mark.asyncio
    async def test_e4_complete_event_has_progress_1(self) -> None:
        events: list[PipelineEvent] = []
        orc = _make_orchestrator(_results(), _pages(), on_event=events.append)
        await orc.run(topic="Topic")
        complete_events = [e for e in events if e.stage == PipelineStage.COMPLETE]
        assert complete_events
        assert complete_events[-1].progress == 1.0

    @pytest.mark.asyncio
    async def test_e1_events_carry_non_empty_message(self) -> None:
        events: list[PipelineEvent] = []
        orc = _make_orchestrator(_results(), _pages(), on_event=events.append)
        await orc.run(topic="Topic")
        assert all(e.message for e in events), "All events must have a non-empty message"


# ===========================================================================
# F. Provider error propagation
# ===========================================================================

class TestProviderErrorPropagation:
    """F — provider exceptions pass through the orchestrator unmodified."""

    @pytest.mark.asyncio
    async def test_f1_search_provider_error_propagates(self) -> None:
        search = MagicMock()
        search.search = AsyncMock(side_effect=SearchProviderError("Serper is down"))
        orc = ResearchOrchestrator(
            search_provider=search,
            llm_provider=_llm_mock(_report()),
            scraper=_scraper_mock(_pages()),
        )
        with pytest.raises(SearchProviderError, match="Serper is down"):
            await orc.run(topic="Topic")

    @pytest.mark.asyncio
    async def test_f2_llm_provider_error_propagates(self) -> None:
        llm = MagicMock()
        llm.synthesise = AsyncMock(side_effect=LLMProviderError("Claude is unavailable"))
        orc = ResearchOrchestrator(
            search_provider=_search_mock(_results()),
            llm_provider=llm,
            scraper=_scraper_mock(_pages()),
        )
        with pytest.raises(LLMProviderError, match="Claude is unavailable"):
            await orc.run(topic="Topic")
