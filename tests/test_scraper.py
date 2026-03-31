"""
tests/test_scraper.py

Unit tests for WebScraper.  Uses pytest-httpx to mock HTTP calls
without touching the network.
"""
import pytest

from app.domain.models import SearchResult
from app.services.scraper import WebScraper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
  <nav>Navigation (should be removed)</nav>
  <main>
    <h1>Hello World</h1>
    <p>This is a relevant paragraph with enough words to test extraction.</p>
  </main>
  <script>alert('noise')</script>
  <footer>Footer (should be removed)</footer>
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
# Tests
# ---------------------------------------------------------------------------


def test_parse_extracts_main_content() -> None:
    """_parse should extract text from <main> and strip noise tags."""
    scraper = WebScraper()
    page = scraper._parse("https://example.com/test", SAMPLE_HTML)

    assert page.success is True
    assert page.title == "Test Page"
    assert "Hello World" in page.raw_text
    assert "relevant paragraph" in page.raw_text


def test_parse_strips_noise_tags() -> None:
    """_parse should remove <nav>, <footer>, and <script> content."""
    scraper = WebScraper()
    page = scraper._parse("https://example.com/test", SAMPLE_HTML)

    assert "Navigation" not in page.raw_text
    assert "Footer" not in page.raw_text
    assert "alert" not in page.raw_text


def test_parse_computes_word_count() -> None:
    """word_count should reflect the actual token count of raw_text."""
    scraper = WebScraper()
    page = scraper._parse("https://example.com/test", SAMPLE_HTML)

    expected = len(page.raw_text.split())
    assert page.word_count == expected


def test_parse_failed_page_structure() -> None:
    """A ScrapedPage with success=False should carry the error message."""
    from app.domain.models import ScrapedPage

    page = ScrapedPage(url="https://bad.example.com", success=False, error="timeout")
    assert page.success is False
    assert page.error == "timeout"


@pytest.mark.asyncio
async def test_scrape_all_returns_only_successful_pages() -> None:
    """scrape_all should filter out pages that failed to scrape."""
    from unittest.mock import AsyncMock, patch

    from app.domain.models import ScrapedPage

    good_page = ScrapedPage(
        url="https://good.example.com",
        title="Good",
        raw_text="Some content here",
        word_count=3,
        success=True,
    )
    bad_page = ScrapedPage(
        url="https://bad.example.com",
        success=False,
        error="connection refused",
    )

    scraper = WebScraper()
    with patch.object(
        scraper,
        "_scrape_one",
        new=AsyncMock(side_effect=[good_page, bad_page]),
    ):
        results = [SAMPLE_RESULT, SAMPLE_RESULT]
        pages = await scraper.scrape_all(results)

    assert len(pages) == 1
    assert pages[0].url == "https://good.example.com"
