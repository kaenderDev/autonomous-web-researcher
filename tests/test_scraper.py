"""
tests/test_scraper.py

Unit tests for WebScraper._parse (pure function — no I/O required) and
the scrape_all concurrency/filtering logic.

All tests are fully isolated: no HTTP calls, no filesystem access.
"""
import pytest

from app.domain.models import ScrapedPage, SearchResult
from app.services.scraper import WebScraper

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

SAMPLE_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Test Page Title</title></head>
<body>
  <nav><a href="/">Home</a> | <a href="/about">About</a></nav>
  <header><h1>Site Banner (inside header — should be excluded)</h1></header>
  <main>
    <h1>Main Heading</h1>
    <p>This is the first relevant paragraph with enough words to pass the filter.</p>
    <ul>
      <li>First list item with sufficient length to survive filtering.</li>
      <li>Second list item that is also long enough to survive.</li>
    </ul>
    <h2>Sub-heading for a section</h2>
    <p>A second paragraph providing more context about the research topic.</p>
  </main>
  <footer><p>Copyright 2025 — should be stripped because footer is in NOISE_TAGS</p></footer>
  <script>alert('noise — must be excluded')</script>
</body>
</html>
"""

# HTML with noise that also contains headings — ensures noise decomposition
# removes headings inside noisy containers and not from the main content.
NOISY_NAV_HTML = """\
<html>
<head><title>Nav Test</title></head>
<body>
  <nav>
    <h2>Navigation heading (must NOT appear in output)</h2>
    <ul><li>Nav link</li></ul>
  </nav>
  <main>
    <h2>Content heading (must appear in output)</h2>
    <p>Content paragraph with sufficient characters to survive min-length filter.</p>
  </main>
</body>
</html>
"""

SAMPLE_RESULT = SearchResult(
    title="Test Page",
    url="https://example.com/test",
    snippet="Test snippet",
    position=1,
)


# ---------------------------------------------------------------------------
# _parse: content extraction
# ---------------------------------------------------------------------------


class TestParseContentExtraction:
    def test_extracts_main_heading(self) -> None:
        page = WebScraper._parse("https://example.com", SAMPLE_HTML)
        assert "Main Heading" in page.raw_text

    def test_extracts_paragraph_text(self) -> None:
        page = WebScraper._parse("https://example.com", SAMPLE_HTML)
        assert "first relevant paragraph" in page.raw_text

    def test_extracts_list_items(self) -> None:
        page = WebScraper._parse("https://example.com", SAMPLE_HTML)
        assert "First list item" in page.raw_text

    def test_extracts_subheadings(self) -> None:
        page = WebScraper._parse("https://example.com", SAMPLE_HTML)
        assert "Sub-heading for a section" in page.raw_text


# ---------------------------------------------------------------------------
# _parse: noise exclusion
# ---------------------------------------------------------------------------


class TestParseNoiseExclusion:
    def test_excludes_script_content(self) -> None:
        page = WebScraper._parse("https://example.com", SAMPLE_HTML)
        assert "alert" not in page.raw_text
        assert "noise — must be excluded" not in page.raw_text

    def test_excludes_nav_text(self) -> None:
        page = WebScraper._parse("https://example.com", SAMPLE_HTML)
        # The nav links "Home" and "About" are too short to survive the
        # min-length filter even if nav were not stripped, but also the
        # nav container itself should be decomposed.
        assert "Home | About" not in page.raw_text

    def test_excludes_footer_content(self) -> None:
        page = WebScraper._parse("https://example.com", SAMPLE_HTML)
        assert "Copyright 2025" not in page.raw_text

    def test_nav_headings_excluded_content_headings_included(self) -> None:
        """Headings inside <nav> must NOT appear; headings in <main> must appear."""
        page = WebScraper._parse("https://example.com", NOISY_NAV_HTML)
        assert "Navigation heading" not in page.raw_text
        assert "Content heading" in page.raw_text

    def test_header_banner_excluded(self) -> None:
        """<h1> inside <header> must be excluded since header is a noise container."""
        page = WebScraper._parse("https://example.com", SAMPLE_HTML)
        assert "Site Banner" not in page.raw_text


# ---------------------------------------------------------------------------
# _parse: title extraction
# ---------------------------------------------------------------------------


class TestParseTitleExtraction:
    def test_extracts_title_tag(self) -> None:
        page = WebScraper._parse("https://example.com", SAMPLE_HTML)
        assert page.title == "Test Page Title"

    def test_falls_back_to_h1_when_no_title_tag(self) -> None:
        html = "<html><body><h1>Fallback Title from H1</h1><p>Some content here that is long enough.</p></body></html>"
        page = WebScraper._parse("https://example.com", html)
        assert page.title == "Fallback Title from H1"

    def test_empty_title_when_none_available(self) -> None:
        html = "<html><body><p>Some paragraph with no title at all.</p></body></html>"
        page = WebScraper._parse("https://example.com", html)
        assert page.title == ""


# ---------------------------------------------------------------------------
# _parse: word count and success flag
# ---------------------------------------------------------------------------


class TestParseMetadata:
    def test_success_flag_is_true(self) -> None:
        page = WebScraper._parse("https://example.com", SAMPLE_HTML)
        assert page.success is True

    def test_word_count_matches_raw_text(self) -> None:
        page = WebScraper._parse("https://example.com", SAMPLE_HTML)
        expected = len(page.raw_text.split())
        assert page.word_count == expected

    def test_url_is_preserved(self) -> None:
        page = WebScraper._parse("https://example.com/path", SAMPLE_HTML)
        assert page.url == "https://example.com/path"


# ---------------------------------------------------------------------------
# _parse: deduplication
# ---------------------------------------------------------------------------


class TestParseDeduplication:
    def test_duplicate_blocks_collapsed(self) -> None:
        """Repeated paragraphs should appear only once in raw_text."""
        repeated = (
            "<html><head><title>T</title></head><body>"
            "<p>Repeated content that is long enough to pass the filter.</p>"
            "<p>Repeated content that is long enough to pass the filter.</p>"
            "</body></html>"
        )
        page = WebScraper._parse("https://example.com", repeated)
        count = page.raw_text.count(
            "Repeated content that is long enough to pass the filter."
        )
        assert count == 1


# ---------------------------------------------------------------------------
# _parse: minimum block length filtering
# ---------------------------------------------------------------------------


class TestParseMinLengthFilter:
    def test_very_short_blocks_discarded(self) -> None:
        """Text blocks shorter than _MIN_BLOCK_LENGTH should not appear."""
        html = (
            "<html><head><title>T</title></head><body>"
            "<p>Hi</p>"  # too short
            "<p>This paragraph is long enough to survive the minimum length filter.</p>"
            "</body></html>"
        )
        page = WebScraper._parse("https://example.com", html)
        assert "Hi" not in page.raw_text
        assert "long enough to survive" in page.raw_text


# ---------------------------------------------------------------------------
# ScrapedPage failure model
# ---------------------------------------------------------------------------


class TestScrapedPageFailure:
    def test_failed_page_has_correct_shape(self) -> None:
        page = ScrapedPage(
            url="https://bad.example.com",
            success=False,
            error="connection refused",
        )
        assert page.success is False
        assert page.error == "connection refused"
        assert page.raw_text == ""


# ---------------------------------------------------------------------------
# scrape_all: concurrency and filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scrape_all_returns_only_successful_pages() -> None:
    """scrape_all must filter out failed pages before returning."""
    from unittest.mock import AsyncMock, patch

    good_page = ScrapedPage(
        url="https://good.example.com",
        title="Good",
        raw_text="Relevant research content about the main topic discussed here.",
        word_count=10,
        success=True,
    )
    bad_page = ScrapedPage(
        url="https://bad.example.com",
        success=False,
        error="timeout",
    )

    results = [SAMPLE_RESULT, SAMPLE_RESULT]
    scraper = WebScraper()

    with patch.object(
        scraper,
        "_scrape_one",
        new=AsyncMock(side_effect=[good_page, bad_page]),
    ):
        pages = await scraper.scrape_all(results)

    assert len(pages) == 1
    assert pages[0].success is True


@pytest.mark.asyncio
async def test_scrape_all_handles_all_failures_gracefully() -> None:
    """scrape_all should return an empty list if every page fails."""
    from unittest.mock import AsyncMock, patch

    bad = ScrapedPage(url="https://x.com", success=False, error="dns error")
    scraper = WebScraper()

    with patch.object(scraper, "_scrape_one", new=AsyncMock(return_value=bad)):
        pages = await scraper.scrape_all([SAMPLE_RESULT])

    assert pages == []

