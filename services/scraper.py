"""
app/services/scraper.py

Async web scraper: fetch raw HTML with httpx, then extract only semantically
meaningful text using BeautifulSoup4.

Separation of concerns (strict):
  - Fetcher  → responsible for raw HTML acquisition only.
  - Parser   → responsible for text extraction and cleaning only.
  - Scraper  → orchestrates both, returns ScrapedPage domain objects.

The parser applies a tag whitelist strategy rather than a blocklist strategy:
instead of stripping noise tags and calling get_text() on the whole tree
(which still pulls in button labels, ARIA text, breadcrumbs, etc.), we
extract ONLY text from the semantic content tags:

    Headings : <h1> … <h6>
    Body copy: <p>
    Lists    : <li>
    Tables   : <td>, <th>       (secondary — included for data-rich pages)
    Quotes   : <blockquote>

This gives cleaner signal-to-noise in the content fed to the synthesiser.

Additional cleaning steps applied in order:
  1. Remove containers identified as noise (nav, header, footer, aside,
     script, style, noscript, iframe, form) BEFORE tag extraction so that
     headings and paragraphs inside those containers are also excluded.
  2. Deduplicate text blocks (navigation menus often repeat the same link
     text multiple times; we keep only the first occurrence).
  3. Discard blocks shorter than MIN_BLOCK_LENGTH characters — these are
     almost always button labels, ARIA descriptions, or lone punctuation.
  4. Collapse excessive whitespace within each block.
"""
import asyncio
import re
from collections import OrderedDict

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

import logging
from app.core.config import settings
from app.core.constants import DEFAULT_SCRAPER_HEADERS, NOISE_TAGS
from app.core.logging import get_logger
from app.domain.models import ScrapedPage, SearchResult

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Only text from these tags is extracted during parsing.
_CONTENT_TAGS: tuple[str, ...] = (
    "h1", "h2", "h3", "h4", "h5", "h6",  # headings
    "p",                                    # body paragraphs
    "li",                                   # list items
    "td", "th",                             # table cells
    "blockquote",                           # pull quotes / citations
)

# Heading tags are always kept regardless of length — a two-word heading
# still carries structural meaning (section title, topic label, etc.).
_HEADING_TAGS: frozenset[str] = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

# Non-heading text blocks shorter than this are discarded as UI chrome
# (button labels, ARIA descriptions, lone punctuation, etc.).
_MIN_BLOCK_LENGTH: int = 25

# Regex to collapse runs of whitespace inside a block.
_WHITESPACE_RE = re.compile(r"\s+")

# HTTP errors that warrant a retry.
def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


# ---------------------------------------------------------------------------
# WebScraper
# ---------------------------------------------------------------------------

class WebScraper:
    """
    Concurrently scrapes a list of SearchResult URLs, extracts clean text,
    and returns a list of ScrapedPage domain objects.

    The httpx.AsyncClient is created once and reused across all requests in
    a single `scrape_all` call, then closed when the call completes.
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._injected_client: httpx.AsyncClient | None = client
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_scrapers)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def scrape_all(self, results: list[SearchResult]) -> list[ScrapedPage]:
        """
        Scrape all URLs concurrently (bounded by the semaphore), filter out
        failures, and return only successfully scraped pages.
        """
        # Use an injected client (tests) or create a fresh one for this batch.
        client = self._injected_client or httpx.AsyncClient(
            headers=DEFAULT_SCRAPER_HEADERS,
            timeout=httpx.Timeout(settings.http_timeout),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=settings.max_concurrent_scrapers),
        )

        try:
            tasks = [self._scrape_one(result, client) for result in results]
            pages: list[ScrapedPage] = await asyncio.gather(*tasks)
        finally:
            if self._injected_client is None:
                await client.aclose()

        successful = [p for p in pages if p.success]
        logger.info(
            f"[Scraper] Completed — "
            f"{len(successful)}/{len(results)} pages scraped successfully"
        )
        return successful

    # ------------------------------------------------------------------
    # Private: fetch pipeline
    # ------------------------------------------------------------------

    async def _scrape_one(
        self, result: SearchResult, client: httpx.AsyncClient
    ) -> ScrapedPage:
        """Acquire semaphore slot, fetch, parse, and return a ScrapedPage."""
        async with self._semaphore:
            try:
                html = await self._fetch(result.url, client)
                page = self._parse(result.url, html)
                logger.debug(
                    f"[Scraper] ✓ {result.url!r} — "
                    f"title={page.title!r} words={page.word_count}"
                )
                return page
            except Exception as exc:
                logger.warning(f"[Scraper] ✗ {result.url!r} — {type(exc).__name__}: {exc}")
                return ScrapedPage(url=result.url, success=False, error=str(exc))

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(settings.max_retries),
        before_sleep=before_sleep_log(logger, logging.WARNING),  # type: ignore[arg-type]
        reraise=True,
    )
    async def _fetch(self, url: str, client: httpx.AsyncClient) -> str:
        """GET a URL and return the raw HTML text."""
        logger.debug(f"[Scraper] GET {url!r}")
        response = await client.get(url)
        response.raise_for_status()
        logger.debug(f"[Scraper] HTTP {response.status_code} — {url!r}")
        return response.text

    # ------------------------------------------------------------------
    # Private: HTML parsing — tag-whitelist extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(url: str, html: str) -> ScrapedPage:
        """
        Parse raw HTML into a clean ScrapedPage.

        Strategy:
          1. Build a BeautifulSoup tree with the lxml parser (fastest).
          2. Extract the <title> before any mutation.
          3. Decompose all noise containers (nav, footer, header, …) so
             that any semantic tags nested inside them are removed too.
          4. Walk only the whitelisted content tags and collect non-empty
             text blocks.
          5. Deduplicate blocks (preserving order) and drop blocks that
             are too short to be real content.
          6. Join blocks with double newlines to preserve visual structure.
        """
        soup = BeautifulSoup(html, "lxml")

        # Step 1: Extract title before mutating the tree.
        title = WebScraper._extract_title(soup)

        # Step 2: Decompose all noise containers — this also removes any
        # content tags that happen to live inside them.
        for noise_tag_name in NOISE_TAGS:
            for noise_node in soup.find_all(noise_tag_name):
                if isinstance(noise_node, Tag):
                    noise_node.decompose()

        # Step 3: Collect (tag_name, text) pairs from whitelisted content tags.
        raw_blocks = WebScraper._extract_content_blocks(soup)

        # Step 4: Deduplicate on text (OrderedDict preserves insertion order).
        seen: OrderedDict[str, str] = OrderedDict()  # text -> tag_name
        for tag_name, text in raw_blocks:
            if text not in seen:
                seen[text] = tag_name

        # Step 5: Filter by minimum length, but always keep headings.
        filtered = [
            text for text, tag_name in seen.items()
            if tag_name in _HEADING_TAGS or len(text) >= _MIN_BLOCK_LENGTH
        ]

        raw_text = "\n\n".join(filtered)
        word_count = len(raw_text.split())

        return ScrapedPage(
            url=url,
            title=title,
            raw_text=raw_text,
            word_count=word_count,
            success=True,
        )

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        """
        Extract the best available page title.

        Preference order:
          1. <title> tag (most reliable cross-site)
          2. First <h1> (useful when <title> is generic like "Home")
          3. Empty string as a safe fallback
        """
        title_tag = soup.find("title")
        if isinstance(title_tag, Tag):
            title = title_tag.get_text(strip=True)
            if title:
                return title

        h1_tag = soup.find("h1")
        if isinstance(h1_tag, Tag):
            h1_text = h1_tag.get_text(strip=True)
            if h1_text:
                return h1_text

        return ""

    @staticmethod
    def _extract_content_blocks(soup: BeautifulSoup) -> list[tuple[str, str]]:
        """
        Walk the document tree and collect (tag_name, text) pairs from
        whitelisted tags only.

        Each tag's text is normalised:
          - Internal whitespace collapsed to single spaces
          - Leading/trailing whitespace stripped
          - Empty results dropped

        Returns a list of (tag_name, cleaned_text) tuples in document order.
        """
        blocks: list[tuple[str, str]] = []

        for tag in soup.find_all(_CONTENT_TAGS):
            if not isinstance(tag, Tag):
                continue

            # Skip list items that are parents of other list items to avoid
            # duplicating nested list content (the children will be visited).
            if tag.name == "li" and tag.find("li"):
                continue

            # Collect the tag's direct text, ignoring child tag text that
            # will be picked up when those child tags are visited directly.
            # For most content tags there are no nested content tags, but for
            # <li> containing <p> this prevents double-extraction.
            text_parts: list[str] = []
            for child in tag.children:
                if isinstance(child, NavigableString):
                    text_parts.append(str(child))
                elif isinstance(child, Tag) and child.name not in _CONTENT_TAGS:
                    # Inline elements (a, span, em, strong, code, etc.)
                    text_parts.append(child.get_text())

            raw = " ".join(text_parts)
            cleaned = _WHITESPACE_RE.sub(" ", raw).strip()

            if cleaned:
                blocks.append((tag.name, cleaned))

        return blocks
