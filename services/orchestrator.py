"""
app/services/orchestrator.py

The "brain" of ResearchAgent Pro.  Coordinates the full pipeline:

  1. Search  — query the search provider for relevant URLs
  2. Scrape  — concurrently fetch and clean page content
  3. Synthesise — send cleaned content to the LLM provider
  4. Render  — pass the structured report to the synthesiser

Depends entirely on the abstract provider interfaces (BaseSearchProvider,
BaseLLMProvider) — concrete implementations are injected at construction
time, keeping this class fully testable with mocks.
"""
from pathlib import Path

from app.core.constants import OutputFormat
from app.core.logging import get_logger
from app.domain.models import ResearchReport, ScrapedPage
from app.providers.base import BaseLLMProvider, BaseSearchProvider
from app.services.scraper import WebScraper
from app.services.synthesizer import ReportSynthesizer

logger = get_logger(__name__)


class ResearchOrchestrator:
    """
    Executes the end-to-end research pipeline.

    Args:
        search_provider: Concrete implementation of BaseSearchProvider.
        llm_provider:    Concrete implementation of BaseLLMProvider.
        scraper:         WebScraper instance (created internally if omitted).
        synthesizer:     ReportSynthesizer instance (created internally if omitted).
    """

    def __init__(
        self,
        search_provider: BaseSearchProvider,
        llm_provider: BaseLLMProvider,
        scraper: WebScraper | None = None,
        synthesizer: ReportSynthesizer | None = None,
    ) -> None:
        self._search = search_provider
        self._llm = llm_provider
        self._scraper = scraper or WebScraper()
        self._synthesizer = synthesizer or ReportSynthesizer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(
        self,
        topic: str,
        num_results: int = 10,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        output_path: Path | None = None,
        print_to_console: bool = True,
    ) -> ResearchReport:
        """
        Run the full research pipeline for a given topic.

        Args:
            topic:           Natural-language research query.
            num_results:     How many search results to fetch.
            output_format:   Render format for the final report.
            output_path:     Optional path to write the rendered report.
            print_to_console: Pretty-print report to terminal if True.

        Returns:
            A fully-populated ResearchReport.
        """
        logger.info(f"[Orchestrator] 🚀 Starting research — topic='{topic}'")

        # Step 1: Search
        search_response = await self._search.search(topic, num_results=num_results)
        logger.info(
            f"[Orchestrator] 🔎 Found {len(search_response.results)} results"
        )

        if not search_response.results:
            raise ValueError(f"No search results found for topic: '{topic}'")

        # Step 2: Scrape concurrently
        scraped_pages: list[ScrapedPage] = await self._scraper.scrape_all(
            search_response.results
        )
        logger.info(
            f"[Orchestrator] 🕷️  Scraped {len(scraped_pages)} usable pages"
        )

        if not scraped_pages:
            raise ValueError("All scraped pages failed — cannot synthesise a report.")

        # Step 3: Synthesise via LLM
        report: ResearchReport = await self._llm.synthesise(topic, scraped_pages)
        logger.info(
            f"[Orchestrator] ✅ Synthesis complete — "
            f"{len(report.key_findings)} findings · "
            f"{len(report.sources)} sources"
        )

        # Step 4: Render
        self._synthesizer.render(report, output_format, output_path)
        if print_to_console:
            self._synthesizer.print_to_console(report)

        return report
