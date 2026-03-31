"""
app/core/constants.py

Immutable constants and enumerations used across the application.
Avoid magic strings: import from here instead.
"""
from enum import StrEnum


class OutputFormat(StrEnum):
    MARKDOWN = "markdown"
    JSON = "json"
    PDF = "pdf"


class SearchProvider(StrEnum):
    SERPER = "serper"
    GOOGLE = "google"


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"


# HTTP headers sent with every scraping request to avoid trivial bot blocks.
DEFAULT_SCRAPER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchAgentPro/1.0; "
        "+https://github.com/your-username/research-agent-pro)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SERPER_SEARCH_URL = "https://google.serper.dev/search"

# Boilerplate HTML tags stripped during content cleaning.
NOISE_TAGS: list[str] = [
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "noscript",
    "iframe",
    "advertisement",
]
