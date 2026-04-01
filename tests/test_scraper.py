"""
tests/test_scraper.py

Unit and async tests for WebScraper.

Coverage plan
─────────────
A. _parse (pure function — zero I/O)
   A1. Content extraction — h1-h6, p, li, blockquote present in output
   A2. Noise exclusion  — nav, header, footer, aside, script, noscript absent
   A3. Title extraction — <title>, h1 fallback, empty fallback
   A4. Metadata         — word_count, url, success flag, error=None
   A5. Deduplication    — repeated blocks appear exactly once
   A6. Min-length filter — short non-heading blocks discarded; headings kept

B. scrape_all (async — _scrape_one mocked throughout)
   B1. Successful pages returned; failed pages filtered
   B2. All-fail  → empty list returned, no exception raised
   B3. 1-of-5 partial success — exactly the one success returned
   B4. Semaphore respected — peak concurrency ≤ max_concurrent_scrapers
   B5. URL identity preserved per page

C. _scrape_one (async — _fetch mocked)
   C1. HTTP exception → ScrapedPage(success=False, error=...)
   C2. Successful fetch → _parse called, result returned

D. ScrapedPage domain model edge cases

All tests are fully isolated: no real HTTP calls are made.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.models import ScrapedPage, SearchResult
from app.services.scraper import WebScraper

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

FULL_PAGE_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Research Page Title</title></head>
<body>
  <nav>
    <h2>Navigation heading — MUST be excluded</h2>
    <ul><li>Short nav link</li></ul>
  </nav>
  <header>
    <h1>Site Banner inside header — MUST be excluded</h1>
  </header>
  <main>
    <h1>Primary Article Heading</h1>
    <p>This is the opening paragraph containing enough words to pass the minimum length filter.</p>
    <h2>Section Two Subheading</h2>
    <ul>
      <li>First bullet point with sufficient text to pass the filter threshold.</li>
      <li>Second bullet point that is also long enough to survive deduplication.</li>
    </ul>
    <blockquote>An important quoted passage that provides direct evidence for the claim.</blockquote>
    <p>A closing paragraph that rounds out the section with additional context.</p>
  </main>
  <aside><p>Sidebar content — aside is a NOISE_TAG and must be stripped out.</p></aside>
  <footer><p>Copyright 2025 ResearchAgent — footer is NOISE_TAG.</p></footer>
  <script>console.log("script must not appear");</script>
  <noscript>Enable JavaScript — noscript must not appear.</noscript>
</body>
</html>
"""

NO_TITLE_HTML = (
    "<html><body>"
    "<h1>Heading Used As Title Fallback</h1>"
    "<p>Some content paragraph that is long enough to survive filtering.</p>"
    "</body></html>"
)

DUPLICATE_HTML = (
    "<html><head><title>T</title></head><body>"
    "<p>This paragraph text appears twice and must only show once in output.</p>"
    "<p>This paragraph text appears twice and must only show once in output.</p>"
    "</body></html>"
)

SHORT_BLOCKS_HTML = (
    "<html><head><title>T</title></head><body>"
    "<p>Hi</p>"
    "<h3>Short</h3>"
    "<p>This paragraph is comfortably long enough to survive the minimum character filter.</p>"
    "</body></html>"
)


def _result(url: str = "https://example.com", pos: int = 1) -> SearchResult:
    return SearchResult(title="Page", url=url, snippet="", position=pos)


def _good_page(url: str = "https://example.com") -> ScrapedPage:
    return ScrapedPage(
        url=url,
        title="Good Page",
        raw_text="Substantive content about the research topic with many words here.",
        word_count=12,
        success=True,
    )


def _bad_page(url: str = "https://example.com", error: str = "timeout") -> ScrapedPage:
    return ScrapedPage(url=url, success=False, error=error)


# ===========================================================================
# A. _parse — pure-function tests (synchronous)
# ===========================================================================

class TestParseContentExtraction:
    """A1 — Content from whitelisted tags must appear in raw_text."""

    def test_h1_in_main_extracted(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert "Primary Article Heading" in page.raw_text

    def test_h2_in_main_extracted(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert "Section Two Subheading" in page.raw_text

    def test_paragraph_text_extracted(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert "opening paragraph" in page.raw_text

    def test_list_items_extracted(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert "First bullet point" in page.raw_text
        assert "Second bullet point" in page.raw_text

    def test_blockquote_extracted(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert "important quoted passage" in page.raw_text

    def test_closing_paragraph_extracted(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert "closing paragraph" in page.raw_text


class TestParseNoiseExclusion:
    """A2 — Content inside noise containers must NOT appear in raw_text."""

    def test_nav_heading_excluded(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert "Navigation heading" not in page.raw_text

    def test_header_h1_excluded(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert "Site Banner inside header" not in page.raw_text

    def test_aside_content_excluded(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert "Sidebar content" not in page.raw_text

    def test_footer_content_excluded(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert "Copyright 2025" not in page.raw_text

    def test_script_content_excluded(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert "console.log" not in page.raw_text

    def test_noscript_content_excluded(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert "Enable JavaScript" not in page.raw_text


class TestParseTitleExtraction:
    """A3 — Title priority: <title> > first <h1> > empty string."""

    def test_title_tag_preferred(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert page.title == "Research Page Title"

    def test_h1_used_when_no_title_tag(self) -> None:
        page = WebScraper._parse("https://example.com", NO_TITLE_HTML)
        assert page.title == "Heading Used As Title Fallback"

    def test_empty_string_when_neither_present(self) -> None:
        html = "<html><body><p>Content without any title element at all.</p></body></html>"
        page = WebScraper._parse("https://example.com", html)
        assert page.title == ""


class TestParseMetadata:
    """A4 — Metadata fields populated correctly on successful parse."""

    def test_success_flag_true(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert page.success is True

    def test_url_preserved_verbatim(self) -> None:
        url = "https://specific.example.com/path?q=1"
        page = WebScraper._parse(url, FULL_PAGE_HTML)
        assert page.url == url

    def test_word_count_matches_raw_text(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert page.word_count == len(page.raw_text.split())

    def test_word_count_positive(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert page.word_count > 0

    def test_error_is_none(self) -> None:
        page = WebScraper._parse("https://example.com", FULL_PAGE_HTML)
        assert page.error is None


class TestParseDeduplication:
    """A5 — Identical text blocks must appear exactly once."""

    def test_duplicate_paragraph_collapsed_to_one(self) -> None:
        page = WebScraper._parse("https://example.com", DUPLICATE_HTML)
        count = page.raw_text.count(
            "This paragraph text appears twice and must only show once in output."
        )
        assert count == 1


class TestParseMinLengthFilter:
    """A6 — Short non-heading blocks discarded; headings always kept."""

    def test_short_paragraph_discarded(self) -> None:
        page = WebScraper._parse("https://example.com", SHORT_BLOCKS_HTML)
        # "Hi" is 2 chars — well below the threshold
        assert "Hi" not in page.raw_text

    def test_long_paragraph_kept(self) -> None:
        page = WebScraper._parse("https://example.com", SHORT_BLOCKS_HTML)
        assert "comfortably long enough" in page.raw_text

    def test_short_heading_kept_despite_length(self) -> None:
        """Headings (h1-h6) are exempt from the minimum-length filter."""
        page = WebScraper._parse("https://example.com", SHORT_BLOCKS_HTML)
        # "Short" is only 5 chars — below _MIN_BLOCK_LENGTH — but is an h3.
        assert "Short" in page.raw_text


# ===========================================================================
# B. scrape_all — async concurrency tests (_scrape_one mocked)
# ===========================================================================

class TestScrapeAll:
    """B — scrape_all orchestration, filtering, concurrency, identity."""

    @pytest.mark.asyncio
    async def test_b1a_successful_pages_returned(self) -> None:
        scraper = WebScraper()
        with patch.object(scraper, "_scrape_one", new=AsyncMock(return_value=_good_page())):
            pages = await scraper.scrape_all([_result()])
        assert len(pages) == 1
        assert pages[0].success is True

    @pytest.mark.asyncio
    async def test_b1b_failed_pages_filtered_out(self) -> None:
        scraper = WebScraper()
        with patch.object(scraper, "_scrape_one", new=AsyncMock(return_value=_bad_page())):
            pages = await scraper.scrape_all([_result()])
        assert pages == []

    @pytest.mark.asyncio
    async def test_b2_all_fail_returns_empty_list_no_exception(self) -> None:
        """All-fail scenario must return [] without raising any exception."""
        scraper = WebScraper()
        bads = [_bad_page(f"https://example.com/{i}") for i in range(5)]
        with patch.object(scraper, "_scrape_one", new=AsyncMock(side_effect=bads)):
            pages = await scraper.scrape_all([_result()] * 5)
        assert pages == []

    @pytest.mark.asyncio
    async def test_b3_one_of_five_succeeds(self) -> None:
        """1-of-5 partial success: exactly one page in the result.

        This is the canonical LowYieldWarning integration scenario:
        5 URLs requested, only result #3 scrapes without error.
        """
        results = [_result(f"https://example.com/{i}", i) for i in range(1, 6)]
        side_effects = [
            _bad_page("https://example.com/1", "403 Forbidden"),
            _bad_page("https://example.com/2", "connection reset"),
            _good_page("https://example.com/3"),    # ← only success
            _bad_page("https://example.com/4", "timeout"),
            _bad_page("https://example.com/5", "ssl error"),
        ]
        scraper = WebScraper()
        with patch.object(
            scraper,
            "_scrape_one",
            new=AsyncMock(side_effect=side_effects),
        ):
            output = await scraper.scrape_all(results)

        assert len(output) == 1
        assert output[0].url == "https://example.com/3"
        assert output[0].success is True

    @pytest.mark.asyncio
    async def test_b3_variant_three_of_five_succeed(self) -> None:
        """3 success + 2 failure → exactly 3 pages returned."""
        results = [_result(f"https://example.com/{i}", i) for i in range(1, 6)]
        side_effects = [
            _good_page("https://example.com/1"),
            _bad_page("https://example.com/2"),
            _good_page("https://example.com/3"),
            _bad_page("https://example.com/4"),
            _good_page("https://example.com/5"),
        ]
        scraper = WebScraper()
        with patch.object(
            scraper,
            "_scrape_one",
            new=AsyncMock(side_effect=side_effects),
        ):
            output = await scraper.scrape_all(results)

        assert len(output) == 3
        assert all(p.success for p in output)

    @pytest.mark.asyncio
    async def test_b4_semaphore_limits_peak_concurrency(self) -> None:
        """Peak concurrent _scrape_one calls must never exceed the semaphore limit.

        Strategy: inject a slow coroutine that increments a shared counter
        while running.  After all tasks complete, verify peak ≤ limit.
        """
        from app.core.config import settings

        limit = settings.max_concurrent_scrapers
        n_tasks = limit * 3
        peak_concurrent: int = 0
        current_concurrent: int = 0
        counter_lock = asyncio.Lock()

        async def metered_scrape(*_args, **_kwargs) -> ScrapedPage:
            nonlocal peak_concurrent, current_concurrent
            async with counter_lock:
                current_concurrent += 1
                if current_concurrent > peak_concurrent:
                    peak_concurrent = current_concurrent
            await asyncio.sleep(0.01)
            async with counter_lock:
                current_concurrent -= 1
            return _good_page()

        results = [_result(f"https://example.com/{i}", i) for i in range(n_tasks)]
        scraper = WebScraper()
        with patch.object(scraper, "_scrape_one", new=metered_scrape):
            await scraper.scrape_all(results)

        assert peak_concurrent <= limit, (
            f"Peak concurrency {peak_concurrent} exceeded semaphore limit {limit}"
        )

    @pytest.mark.asyncio
    async def test_b5_url_identity_preserved(self) -> None:
        """Each ScrapedPage in the output retains the URL of its SearchResult."""
        results = [
            _result("https://alpha.example.com", 1),
            _result("https://beta.example.com", 2),
        ]
        pages = [
            _good_page("https://alpha.example.com"),
            _good_page("https://beta.example.com"),
        ]
        scraper = WebScraper()
        with patch.object(scraper, "_scrape_one", new=AsyncMock(side_effect=pages)):
            output = await scraper.scrape_all(results)

        urls = {p.url for p in output}
        assert urls == {"https://alpha.example.com", "https://beta.example.com"}


# ===========================================================================
# C. _scrape_one — async unit tests (_fetch mocked)
# ===========================================================================

class TestScrapeOne:
    """C — _scrape_one error handling and parse delegation."""

    @pytest.mark.asyncio
    async def test_c1_http_exception_produces_failure_page(self) -> None:
        """Any exception from _fetch must become ScrapedPage(success=False)."""
        import httpx

        scraper = WebScraper()
        with patch.object(
            scraper,
            "_fetch",
            new=AsyncMock(side_effect=httpx.TimeoutException("timed out")),
        ):
            page = await scraper._scrape_one(_result(), MagicMock())

        assert page.success is False
        assert "timed" in (page.error or "").lower()

    @pytest.mark.asyncio
    async def test_c1_generic_exception_produces_failure_page(self) -> None:
        """Non-httpx exceptions also produce failure pages (no re-raise)."""
        scraper = WebScraper()
        with patch.object(
            scraper,
            "_fetch",
            new=AsyncMock(side_effect=RuntimeError("unexpected")),
        ):
            page = await scraper._scrape_one(_result(), MagicMock())

        assert page.success is False
        assert page.error is not None

    @pytest.mark.asyncio
    async def test_c2_successful_fetch_returns_parsed_page(self) -> None:
        """Successful _fetch → _parse is called → ScrapedPage returned."""
        scraper = WebScraper()
        with patch.object(
            scraper,
            "_fetch",
            new=AsyncMock(return_value=FULL_PAGE_HTML),
        ):
            page = await scraper._scrape_one(
                _result("https://example.com"), MagicMock()
            )

        assert page.success is True
        assert page.url == "https://example.com"
        assert page.title == "Research Page Title"


# ===========================================================================
# D. ScrapedPage domain model edge cases
# ===========================================================================

class TestScrapedPageModel:

    def test_default_raw_text_is_empty_string(self) -> None:
        page = ScrapedPage(url="https://x.com", success=False, error="err")
        assert page.raw_text == ""

    def test_default_word_count_is_zero(self) -> None:
        page = ScrapedPage(url="https://x.com", success=False)
        assert page.word_count == 0

    def test_error_field_stored_verbatim(self) -> None:
        msg = "SSL: CERTIFICATE_VERIFY_FAILED"
        page = ScrapedPage(url="https://x.com", success=False, error=msg)
        assert page.error == msg

    def test_success_true_with_content(self) -> None:
        page = ScrapedPage(
            url="https://x.com",
            title="Title",
            raw_text="Content",
            word_count=1,
            success=True,
        )
        assert page.success is True
        assert page.error is None
