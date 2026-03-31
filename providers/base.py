"""
app/providers/base.py

Abstract Base Classes that define the *contracts* every concrete provider
must honour.  New search engines or LLM backends slot in simply by
inheriting from these ABCs and wiring up via dependency injection in the
orchestrator.  No concrete code here — interfaces only.
"""
from abc import ABC, abstractmethod

from app.domain.models import ResearchReport, ScrapedPage, SearchResponse


class BaseSearchProvider(ABC):
    """Strategy interface for search-engine integrations."""

    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        """
        Execute a web search and return a normalised SearchResponse.

        Args:
            query:       The natural-language search query.
            num_results: Maximum number of results to return.

        Returns:
            A fully-validated SearchResponse instance.

        Raises:
            SearchProviderError: On any API-level failure after retries.
        """
        ...


class BaseLLMProvider(ABC):
    """Strategy interface for LLM / completion-API integrations."""

    @abstractmethod
    async def synthesise(
        self,
        topic: str,
        scraped_pages: list[ScrapedPage],
    ) -> ResearchReport:
        """
        Turn a list of scraped pages into a structured ResearchReport.

        Args:
            topic:         The original research topic / user query.
            scraped_pages: Cleaned page content gathered by the scraper.

        Returns:
            A fully-validated ResearchReport.

        Raises:
            LLMProviderError: On any API-level failure after retries.
        """
        ...


# ---------------------------------------------------------------------------
# Typed exceptions — callers catch these, not httpx/anthropic internals
# ---------------------------------------------------------------------------


class SearchProviderError(RuntimeError):
    """Raised when a search provider fails after exhausting all retries."""


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider fails after exhausting all retries."""
