"""
app/core/exceptions.py

Typed domain exceptions for the research pipeline.

Using specific exception types (instead of generic ValueError/RuntimeError)
lets the CLI catch and handle each failure mode with a precise message and
the right exit code, without relying on string matching.
"""


class ResearchAgentError(Exception):
    """Base class for all ResearchAgent Pro domain exceptions."""


class NoSearchResultsError(ResearchAgentError):
    """
    Raised by the orchestrator when the search provider returns zero results.

    This is a user-input problem (too specific a query, blocked region, etc.)
    rather than a transient API error, so it should NOT be retried.
    """

    def __init__(self, topic: str) -> None:
        self.topic = topic
        super().__init__(
            f"No search results found for topic: {topic!r}. "
            "Try broadening the query or checking your Serper API key."
        )


class NoScrapedPagesError(ResearchAgentError):
    """
    Raised by the orchestrator when every URL returned by the search provider
    fails to scrape (timeouts, 403s, JavaScript-only pages, etc.).
    """

    def __init__(self, attempted: int) -> None:
        self.attempted = attempted
        super().__init__(
            f"All {attempted} scraped page(s) failed. "
            "The search results may point to sites that block scrapers. "
            "Consider increasing --results to widen the candidate pool."
        )


class LowYieldWarning(UserWarning):
    """
    Issued (not raised) when fewer than a threshold fraction of pages
    scraped successfully, so the final report may be incomplete.
    """
