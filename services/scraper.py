"""
app/services/scraper.py

Async web scraper using httpx + BeautifulSoup.  Runs concurrently behind
an asyncio.Semaphore, strips boilerplate HTML, and returns clean
ScrapedPage objects ready for the synthesiser.

Separation of concerns:
  - This module only *fetches* and *cleans* content.
  - It does NOT interpret, summarise, or score content — that belongs
    to the synthesiser.
"""
import asyncio
import re

import httpx
from bs4 import BeautifulSoup, Tag
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.constants import DEFAULT_SCRAPER_HEADERS, NOISE_TAGS
from app.core.logging import get_logger
from app.domain.models import ScrapedPage, SearchResult

logger = get_logger(__name__)

_RETRY_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


class WebScraper:
    """
    Concurrently scrapes a list of URLs, cleans each page, and returns
    a list of ScrapedPage domain objects.
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_scrapers)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def scrape_all(self, results: list[SearchResult]) -> list[ScrapedPage]:
        """Scrape all URLs concurrently, bounded by the semaphore."""
        tasks = [self._scrape_one(r) for r in results]
        pages = await asyncio.gather(*tasks, return_exceptions=False)
        successful = [p for p in pages if p.success]
        logger.info(
            f"[Scraper] Scraped {len(successful)}/{len(results)} pages successfully"
        )
        return successful

    # ------------------------------------------------------------------
    # Private: fetch
    # ------------------------------------------------------------------

    async def _scrape_one(self, result: SearchResult) -> ScrapedPage:
        async with self._semaphore:
            try:
                html = await self._fetch(result.url)
                return self._parse(result.url, html)
            except Exception as exc:
                logger.warning(f"[Scraper] Failed to scrape {result.url}: {exc}")
                return ScrapedPage(url=result.url, success=False, error=str(exc))

    @retry(
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(settings.max_retries),
        reraise=True,
    )
    async def _fetch(self, url: str) -> str:
        async with (self._client or httpx.AsyncClient()) as client:
            resp = client if self._client else client
            response = await resp.get(
                url,
                headers=DEFAULT_SCRAPER_HEADERS,
                timeout=settings.http_timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
            logger.debug(f"[Scraper] HTTP {response.status_code} — {url}")
            return response.text

    # ------------------------------------------------------------------
    # Private: parse & clean
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(url: str, html: str) -> ScrapedPage:
        soup = BeautifulSoup(html, "lxml")

        # 1. Strip noise tags (scripts, styles, nav…)
        for tag in NOISE_TAGS:
            for node in soup.find_all(tag):
                node.decompose()

        # 2. Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if isinstance(title_tag, Tag) else ""

        # 3. Prefer <main> or <article>, fall back to <body>
        content_node = (
            soup.find("main")
            or soup.find("article")
            or soup.find("body")
        )
        raw_text = content_node.get_text(separator="\n") if content_node else ""

        # 4. Collapse excessive whitespace
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

        word_count = len(raw_text.split())
        logger.debug(f"[Scraper] Parsed '{title}' — {word_count} words")

        return ScrapedPage(
            url=url,
            title=title,
            raw_text=raw_text,
            word_count=word_count,
            success=True,
        )
