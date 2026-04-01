"""
app/services/orchestrator.py

Coordinates the full research pipeline:

    1. Search   — SerperSearchProvider → SearchResponse
    2. Scrape   — WebScraper.scrape_all (concurrent) → list[ScrapedPage]
    3. Synthesise — AnthropicLLMProvider → ResearchReport

Architecture notes
──────────────────
• Depends only on the abstract provider interfaces (BaseSearchProvider,
  BaseLLMProvider).  Concrete classes are injected at construction time.
  This keeps the orchestrator fully testable with mocks.

• Progress events are emitted via an optional on_event callback.  The CLI
  registers a Rich Progress updater; tests can register a simple list.append.
  The orchestrator itself has zero knowledge of Rich — clean separation.

• Graceful degradation: if some (but not all) pages fail to scrape, a
  LowYieldWarning is issued and the pipeline continues with the pages that
  did succeed.  Only if *every* page fails is NoScrapedPagesError raised.

• Typed domain exceptions (NoSearchResultsError, NoScrapedPagesError) replace
  generic ValueError so the CLI can produce targeted user-facing messages.
"""
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from app.core.exceptions import LowYieldWarning, NoScrapedPagesError, NoSearchResultsError
from app.core.logging import get_logger
from app.domain.models import ResearchReport, ScrapedPage
from app.providers.base import BaseLLMProvider, BaseSearchProvider
from app.services.scraper import WebScraper

logger = get_logger(__name__)

# Fraction of URLs that must scrape successfully before a LowYieldWarning is
# issued.  E.g. 0.4 means "warn if fewer than 40 % of candidates succeeded".
_LOW_YIELD_THRESHOLD: float = 0.4


# ---------------------------------------------------------------------------
# Pipeline event protocol — consumed by the CLI's Rich Progress display
# ---------------------------------------------------------------------------

class PipelineStage(Enum):
    SEARCHING = auto()
    SCRAPING = auto()
    SYNTHESISING = auto()
    SAVING = auto()
    COMPLETE = auto()
    ERROR = auto()


@dataclass
class PipelineEvent:
    """
    Emitted at each stage boundary so the CLI (or tests) can react without
    coupling to the orchestrator's internals.
    """
    stage: PipelineStage
    message: str                      # short human-readable description
    detail: str = ""                  # optional extra context (e.g. counts)
    progress: float = 0.0             # 0.0 – 1.0 completion fraction


EventCallback = Callable[[PipelineEvent], None]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ResearchOrchestrator:
    """
    Executes the end-to-end research pipeline.

    Args:
        search_provider:  Concrete implementation of BaseSearchProvider.
        llm_provider:     Concrete implementation of BaseLLMProvider.
        scraper:          WebScraper instance; created internally if omitted.
        on_event:         Optional callback invoked at each pipeline stage.
                          Signature: (PipelineEvent) -> None
    """

    def __init__(
        self,
        search_provider: BaseSearchProvider,
        llm_provider: BaseLLMProvider,
        scraper: WebScraper | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self._search = search_provider
        self._llm = llm_provider
        self._scraper = scraper or WebScraper()
        self._on_event = on_event or (lambda _: None)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(
        self,
        topic: str,
        num_results: int = 10,
    ) -> ResearchReport:
        """
        Execute the full pipeline for *topic* and return a ResearchReport.

        Raises:
            NoSearchResultsError: Search provider returned zero organic results.
            NoScrapedPagesError:  Every candidate URL failed to scrape.
            SearchProviderError:  Unrecoverable search API failure.
            LLMProviderError:     Unrecoverable LLM API failure.
        """
        logger.info(f"[Orchestrator] 🚀 Starting — topic=[bold]{topic!r}[/bold]")
        self._emit(PipelineStage.SEARCHING, "Searching the web…", topic, 0.0)

        # ── Stage 1: Search ───────────────────────────────────────────────
        search_response = await self._search.search(topic, num_results=num_results)
        result_count = len(search_response.results)
        logger.info(f"[Orchestrator] 🔎 Search returned {result_count} result(s)")

        if result_count == 0:
            self._emit(PipelineStage.ERROR, "No results found", topic, 0.0)
            raise NoSearchResultsError(topic)

        self._emit(
            PipelineStage.SCRAPING,
            f"Scraping {result_count} page(s) concurrently…",
            f"{result_count} URLs queued",
            0.25,
        )

        # ── Stage 2: Scrape ───────────────────────────────────────────────
        scraped_pages: list[ScrapedPage] = await self._scraper.scrape_all(
            search_response.results
        )
        success_count = len(scraped_pages)
        failed_count = result_count - success_count

        if failed_count:
            logger.warning(
                f"[Orchestrator] ⚠ {failed_count}/{result_count} page(s) failed to scrape"
            )

        if success_count == 0:
            self._emit(PipelineStage.ERROR, "All pages failed to scrape", "", 0.25)
            raise NoScrapedPagesError(attempted=result_count)

        # Warn — but continue — if yield is below threshold.
        yield_rate = success_count / result_count
        if yield_rate < _LOW_YIELD_THRESHOLD:
            warnings.warn(
                f"Only {success_count}/{result_count} pages scraped successfully "
                f"({yield_rate:.0%}). The report may be incomplete.",
                LowYieldWarning,
                stacklevel=2,
            )

        logger.info(
            f"[Orchestrator] 🕷  Scraped [green]{success_count}[/green]/"
            f"{result_count} pages — "
            f"total words: {sum(p.word_count for p in scraped_pages):,}"
        )

        self._emit(
            PipelineStage.SYNTHESISING,
            "Synthesising report with Claude…",
            f"{success_count} pages · {sum(p.word_count for p in scraped_pages):,} words",
            0.55,
        )

        # ── Stage 3: Synthesise ────────────────────────────────────────────
        report: ResearchReport = await self._llm.synthesise(topic, scraped_pages)

        logger.info(
            f"[Orchestrator] ✅ Synthesis complete — "
            f"findings={len(report.key_findings)} "
            f"sources={len(report.sources)} "
            f"input_tokens={report.token_usage.get('input_tokens', '?')} "
            f"output_tokens={report.token_usage.get('output_tokens', '?')}"
        )

        self._emit(
            PipelineStage.COMPLETE,
            "Pipeline complete",
            f"{len(report.key_findings)} findings · {len(report.sources)} sources",
            1.0,
        )

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit(
        self,
        stage: PipelineStage,
        message: str,
        detail: str,
        progress: float,
    ) -> None:
        """Construct a PipelineEvent and invoke the registered callback."""
        self._on_event(PipelineEvent(stage=stage, message=message, detail=detail, progress=progress))
