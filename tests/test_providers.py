"""
tests/test_providers.py

Unit tests for SerperSearchProvider and AnthropicLLMProvider.
All external HTTP calls and SDK calls are mocked — no network required.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.models import ScrapedPage, SearchResponse
from app.providers.base import SearchProviderError
from app.providers.search import SerperSearchProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SERPER_RESPONSE = {
    "organic": [
        {
            "title": "Rust Programming Language",
            "link": "https://www.rust-lang.org",
            "snippet": "A language empowering everyone.",
        },
        {
            "title": "Why Rust?",
            "link": "https://blog.rust-lang.org/why-rust",
            "snippet": "Memory safety without GC.",
        },
    ]
}

SCRAPER_PAGES = [
    ScrapedPage(
        url="https://www.rust-lang.org",
        title="Rust Programming Language",
        raw_text="Rust is a systems programming language.",
        word_count=6,
        success=True,
    )
]


# ---------------------------------------------------------------------------
# SerperSearchProvider tests
# ---------------------------------------------------------------------------


class TestSerperSearchProvider:
    def test_parse_maps_organic_results(self) -> None:
        """_parse should create one SearchResult per organic item."""
        provider = SerperSearchProvider()
        response = provider._parse("rust systems programming", SERPER_RESPONSE)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 2
        assert response.results[0].position == 1
        assert response.results[0].url == "https://www.rust-lang.org"
        assert response.provider == "serper"

    def test_parse_empty_organic_returns_empty_list(self) -> None:
        """If Serper returns no organic results, the list should be empty."""
        provider = SerperSearchProvider()
        response = provider._parse("obscure query", {"organic": []})
        assert response.results == []

    @pytest.mark.asyncio
    async def test_search_raises_on_http_failure(self) -> None:
        """search() should raise SearchProviderError after retries exhaust."""
        import httpx

        provider = SerperSearchProvider()
        with patch.object(
            provider,
            "_fetch_with_retry",
            new=AsyncMock(side_effect=httpx.TimeoutException("timeout")),
        ):
            with pytest.raises(SearchProviderError, match="Serper search failed"):
                await provider.search("any query")


# ---------------------------------------------------------------------------
# AnthropicLLMProvider tests
# ---------------------------------------------------------------------------


class TestAnthropicLLMProvider:
    def _make_provider(self, mock_response_text: str):  # type: ignore[return]
        from app.providers.llm import AnthropicLLMProvider

        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=mock_response_text)]
        mock_message.usage.input_tokens = 500
        mock_message.usage.output_tokens = 200
        mock_client.messages.create = AsyncMock(return_value=mock_message)
        return AnthropicLLMProvider(client=mock_client)

    @pytest.mark.asyncio
    async def test_synthesise_returns_research_report(self) -> None:
        """synthesise() should parse valid JSON into a ResearchReport."""
        from app.domain.models import ResearchReport

        payload = {
            "executive_summary": "Rust is a safe systems language.",
            "body": "## Why Rust\n\nRust prevents memory errors...",
            "key_findings": ["Memory safety", "Zero-cost abstractions"],
            "sources": [
                {
                    "title": "Rust Lang",
                    "url": "https://www.rust-lang.org",
                    "relevance_note": "Official site",
                }
            ],
        }
        provider = self._make_provider(json.dumps(payload))
        report = await provider.synthesise("Rust in systems programming", SCRAPER_PAGES)

        assert isinstance(report, ResearchReport)
        assert report.topic == "Rust in systems programming"
        assert "Memory safety" in report.key_findings
        assert report.token_usage["input_tokens"] == 500

    @pytest.mark.asyncio
    async def test_synthesise_strips_markdown_fences(self) -> None:
        """synthesise() should handle responses wrapped in ```json fences."""
        from app.domain.models import ResearchReport

        payload = {
            "executive_summary": "Summary.",
            "body": "Body text.",
            "key_findings": [],
            "sources": [],
        }
        fenced = f"```json\n{json.dumps(payload)}\n```"
        provider = self._make_provider(fenced)
        report = await provider.synthesise("test topic", SCRAPER_PAGES)
        assert isinstance(report, ResearchReport)
