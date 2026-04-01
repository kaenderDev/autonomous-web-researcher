"""
tests/test_providers.py

Unit tests for SerperSearchProvider and AnthropicLLMProvider.

All external API calls are mocked — no network required.  Tests are
organised around three concerns per provider:
  1. Happy-path response mapping (raw payload → domain model)
  2. Edge-case handling (empty results, missing fields, malformed URLs)
  3. Error propagation (transient errors → retry → SearchProviderError)
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.domain.models import ScrapedPage, SearchResponse
from app.providers.base import LLMProviderError, SearchProviderError
from app.providers.search import SerperSearchProvider, _is_retryable

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SERPER_ORGANIC_RESPONSE = {
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
        raw_text="Rust is a systems programming language focused on safety.",
        word_count=9,
        success=True,
    )
]


# ---------------------------------------------------------------------------
# _is_retryable predicate
# ---------------------------------------------------------------------------


class TestIsRetryable:
    def test_timeout_is_retryable(self) -> None:
        assert _is_retryable(httpx.TimeoutException("timeout")) is True

    def test_network_error_is_retryable(self) -> None:
        assert _is_retryable(httpx.NetworkError("net")) is True

    def test_429_is_retryable(self) -> None:
        resp = httpx.Response(429)
        exc = httpx.HTTPStatusError("429", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is True

    def test_503_is_retryable(self) -> None:
        resp = httpx.Response(503)
        exc = httpx.HTTPStatusError("503", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is True

    def test_404_is_not_retryable(self) -> None:
        resp = httpx.Response(404)
        exc = httpx.HTTPStatusError("404", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is False

    def test_401_is_not_retryable(self) -> None:
        resp = httpx.Response(401)
        exc = httpx.HTTPStatusError("401", request=MagicMock(), response=resp)
        assert _is_retryable(exc) is False

    def test_value_error_is_not_retryable(self) -> None:
        assert _is_retryable(ValueError("bad")) is False


# ---------------------------------------------------------------------------
# SerperSearchProvider
# ---------------------------------------------------------------------------


class TestSerperSearchProvider:

    # -- Happy path ----------------------------------------------------------

    def test_parse_maps_organic_results_to_domain_model(self) -> None:
        provider = SerperSearchProvider(client=MagicMock())
        response = provider._parse("rust systems programming", SERPER_ORGANIC_RESPONSE)

        assert isinstance(response, SearchResponse)
        assert response.query == "rust systems programming"
        assert response.provider == "serper"
        assert len(response.results) == 2

    def test_parse_result_position_is_1_based(self) -> None:
        provider = SerperSearchProvider(client=MagicMock())
        response = provider._parse("query", SERPER_ORGANIC_RESPONSE)

        assert response.results[0].position == 1
        assert response.results[1].position == 2

    def test_parse_maps_url_title_snippet(self) -> None:
        provider = SerperSearchProvider(client=MagicMock())
        response = provider._parse("query", SERPER_ORGANIC_RESPONSE)

        first = response.results[0]
        assert first.url == "https://www.rust-lang.org"
        assert first.title == "Rust Programming Language"
        assert first.snippet == "A language empowering everyone."

    # -- Edge cases ----------------------------------------------------------

    def test_parse_empty_organic_returns_empty_list(self) -> None:
        provider = SerperSearchProvider(client=MagicMock())
        response = provider._parse("obscure query xyz", {"organic": []})

        assert response.results == []

    def test_parse_skips_results_with_missing_url(self) -> None:
        raw = {
            "organic": [
                {"title": "No URL entry", "link": "", "snippet": ""},
                {"title": "Valid entry", "link": "https://valid.example.com", "snippet": "ok"},
            ]
        }
        provider = SerperSearchProvider(client=MagicMock())
        response = provider._parse("query", raw)

        assert len(response.results) == 1
        assert response.results[0].url == "https://valid.example.com"

    def test_parse_skips_results_with_non_http_url(self) -> None:
        raw = {
            "organic": [
                {"title": "FTP link", "link": "ftp://files.example.com", "snippet": ""},
                {"title": "Valid", "link": "https://good.example.com", "snippet": "ok"},
            ]
        }
        provider = SerperSearchProvider(client=MagicMock())
        response = provider._parse("query", raw)

        assert len(response.results) == 1
        assert "ftp" not in response.results[0].url

    def test_parse_uses_url_as_title_fallback(self) -> None:
        raw = {
            "organic": [
                {"title": "", "link": "https://notitle.example.com", "snippet": "x"}
            ]
        }
        provider = SerperSearchProvider(client=MagicMock())
        response = provider._parse("query", raw)

        assert response.results[0].title == "https://notitle.example.com"

    # -- Error propagation ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_search_wraps_exception_in_search_provider_error(self) -> None:
        provider = SerperSearchProvider(client=MagicMock())
        with patch.object(
            provider,
            "_fetch_with_retry",
            new=AsyncMock(side_effect=httpx.TimeoutException("timeout")),
        ):
            with pytest.raises(SearchProviderError, match="Serper search failed"):
                await provider.search("any query")


# ---------------------------------------------------------------------------
# AnthropicLLMProvider
# ---------------------------------------------------------------------------


def _make_llm_provider(response_text: str):  # type: ignore[return]
    """Build an AnthropicLLMProvider backed by a fully mocked AsyncAnthropic client."""
    from app.providers.llm import AnthropicLLMProvider

    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=response_text)]
    mock_message.stop_reason = "end_turn"
    mock_message.usage.input_tokens = 500
    mock_message.usage.output_tokens = 200
    mock_client.messages.create = AsyncMock(return_value=mock_message)
    return AnthropicLLMProvider(client=mock_client)


def _valid_payload(**overrides) -> dict:  # type: ignore[type-arg]
    base: dict = {  # type: ignore[type-arg]
        "executive_summary": "Rust is a memory-safe systems language.",
        "body": "## Why Rust\n\nRust prevents memory errors at compile time.",
        "key_findings": ["Memory safety", "Zero-cost abstractions", "No GC"],
        "sources": [
            {
                "title": "Rust Lang",
                "url": "https://www.rust-lang.org",
                "relevance_note": "Official website.",
            }
        ],
    }
    base.update(overrides)
    return base


class TestAnthropicLLMProvider:

    # -- Happy path ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_synthesise_returns_research_report(self) -> None:
        from app.domain.models import ResearchReport

        provider = _make_llm_provider(json.dumps(_valid_payload()))
        report = await provider.synthesise("Rust in systems programming", SCRAPER_PAGES)

        assert isinstance(report, ResearchReport)
        assert report.topic == "Rust in systems programming"

    @pytest.mark.asyncio
    async def test_synthesise_maps_key_findings(self) -> None:
        provider = _make_llm_provider(json.dumps(_valid_payload()))
        report = await provider.synthesise("topic", SCRAPER_PAGES)

        assert "Memory safety" in report.key_findings
        assert "Zero-cost abstractions" in report.key_findings

    @pytest.mark.asyncio
    async def test_synthesise_maps_sources(self) -> None:
        provider = _make_llm_provider(json.dumps(_valid_payload()))
        report = await provider.synthesise("topic", SCRAPER_PAGES)

        assert len(report.sources) == 1
        assert report.sources[0].url == "https://www.rust-lang.org"

    @pytest.mark.asyncio
    async def test_synthesise_records_token_usage(self) -> None:
        provider = _make_llm_provider(json.dumps(_valid_payload()))
        report = await provider.synthesise("topic", SCRAPER_PAGES)

        assert report.token_usage["input_tokens"] == 500
        assert report.token_usage["output_tokens"] == 200

    # -- Fence stripping -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_synthesise_strips_backtick_json_fence(self) -> None:
        fenced = f"```json\n{json.dumps(_valid_payload())}\n```"
        provider = _make_llm_provider(fenced)
        report = await provider.synthesise("topic", SCRAPER_PAGES)
        assert report.executive_summary  # successfully parsed

    @pytest.mark.asyncio
    async def test_synthesise_strips_fence_without_language_hint(self) -> None:
        fenced = f"```\n{json.dumps(_valid_payload())}\n```"
        provider = _make_llm_provider(fenced)
        report = await provider.synthesise("topic", SCRAPER_PAGES)
        assert report.executive_summary

    @pytest.mark.asyncio
    async def test_synthesise_strips_uppercase_json_fence(self) -> None:
        fenced = f"```JSON\n{json.dumps(_valid_payload())}\n```"
        provider = _make_llm_provider(fenced)
        report = await provider.synthesise("topic", SCRAPER_PAGES)
        assert report.executive_summary

    # -- Error propagation ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_synthesise_raises_on_non_json_response(self) -> None:
        provider = _make_llm_provider("This is not JSON at all, just prose.")
        with pytest.raises(LLMProviderError, match="non-JSON"):
            await provider.synthesise("topic", SCRAPER_PAGES)

    @pytest.mark.asyncio
    async def test_synthesise_raises_on_missing_required_keys(self) -> None:
        incomplete = {"executive_summary": "Only a summary, nothing else."}
        provider = _make_llm_provider(json.dumps(incomplete))
        with pytest.raises(LLMProviderError, match="missing required keys"):
            await provider.synthesise("topic", SCRAPER_PAGES)

    # -- Budget calculation --------------------------------------------------

    def test_budget_per_page_decreases_with_more_pages(self) -> None:
        from app.providers.llm import AnthropicLLMProvider

        budget_3 = AnthropicLLMProvider._budget_per_page(3)
        budget_30 = AnthropicLLMProvider._budget_per_page(30)
        assert budget_3 > budget_30

    def test_budget_per_page_never_goes_below_minimum(self) -> None:
        from app.providers.llm import AnthropicLLMProvider, _MIN_CHARS_PER_PAGE

        budget = AnthropicLLMProvider._budget_per_page(1000)
        assert budget >= _MIN_CHARS_PER_PAGE
