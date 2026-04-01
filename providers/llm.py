"""
app/providers/llm.py

Concrete Anthropic/Claude LLM provider.  Strictly implements BaseLLMProvider.

Design decisions:
  - System prompt uses explicit JSON schema instructions and prohibits preamble.
  - Fence stripping uses a regex that handles ``` json (with space), ```JSON
    (uppercase), and unterminated fences so the parser never sees raw markdown.
  - Per-page content budget is calculated dynamically: the total available
    character budget is divided equally across all pages so that large scrapes
    do not overflow the model's context window.
  - Retry covers connection errors, rate limits, AND 5xx server errors
    (APIStatusError with status >= 500) — the most common transient failures
    in production Anthropic API usage.
  - Pre-flight validation checks that all required keys are present before
    constructing the domain model, giving a clear error rather than a
    cryptic KeyError.
"""
import json
import logging
import re

import anthropic
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.models import ResearchReport, ScrapedPage, SourceCitation
from app.providers.base import BaseLLMProvider, LLMProviderError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters allocated to scraped content per page.  Each page's raw_text is
# truncated to this length before being sent to the model.  The budget scales
# inversely with the number of pages so total prompt size stays bounded.
_BASE_CHARS_PER_PAGE: int = 8_000
_MIN_CHARS_PER_PAGE: int = 1_500  # never go below this, even with many pages

# Required top-level keys in the model's JSON response.
_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"executive_summary", "body", "key_findings", "sources"}
)

# Regex that strips optional ``` fences with optional language hints.
# Handles: ```json, ``` json, ```JSON, ```, and unterminated fences.
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?|```\s*$", re.MULTILINE)

_SYNTHESIS_SYSTEM_PROMPT = """\
You are an elite research analyst. You receive a research topic and a
collection of scraped web page contents. Your task is to synthesise a
comprehensive, accurate, and well-structured research report.

Return ONLY a single valid JSON object — no markdown fences, no preamble,
no trailing commentary. The object must match this exact schema:

{
  "executive_summary": "<3-5 sentence TL;DR of the entire topic>",
  "body": "<full Markdown report — use ### headings, **bold**, and - lists>",
  "key_findings": [
    "<concise finding 1>",
    "<concise finding 2>",
    "... (5-10 total)"
  ],
  "sources": [
    {
      "title": "<page title>",
      "url": "<exact URL from the source>",
      "relevance_note": "<one sentence explaining why this source matters>"
    }
  ]
}

Hard rules:
1. Be strictly factual — only assert what the provided sources state.
2. Cite sources inline in the body using [Title](url) Markdown links.
3. key_findings must be 5-10 standalone, actionable bullet points.
4. Never invent URLs, statistics, or quotes not present in the source content.
5. If a source's content is insufficient, state that briefly — do not fabricate.
"""


# ---------------------------------------------------------------------------
# Retry predicate
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient Anthropic API conditions only."""
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.RateLimitError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return False


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class AnthropicLLMProvider(BaseLLMProvider):
    """
    Uses Claude (via the official Anthropic async SDK) to synthesise scraped
    page content into a structured ResearchReport domain object.

    Inject a pre-built AsyncAnthropic client for testing; production callers
    rely on the default which reads ANTHROPIC_API_KEY from settings.
    """

    def __init__(self, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._client: anthropic.AsyncAnthropic = client or anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )

    # ------------------------------------------------------------------
    # Public interface — satisfies BaseLLMProvider contract
    # ------------------------------------------------------------------

    async def synthesise(
        self,
        topic: str,
        scraped_pages: list[ScrapedPage],
    ) -> ResearchReport:
        """
        Transform a list of ScrapedPage objects into a ResearchReport by
        calling the Claude API with a structured synthesis prompt.

        Raises:
            LLMProviderError: If the API call fails after all retries, or if
                              the response cannot be parsed into a valid report.
        """
        logger.info(
            f"[LLM] Starting synthesis — "
            f"model={settings.llm_model} topic='{topic}' pages={len(scraped_pages)}"
        )

        chars_per_page = self._budget_per_page(len(scraped_pages))
        user_content = self._build_user_message(topic, scraped_pages, chars_per_page)
        logger.debug(
            f"[LLM] User message built — "
            f"{len(user_content):,} chars / {chars_per_page:,} chars per page"
        )

        try:
            raw = await self._call_with_retry(user_content)
            report = self._parse_report(topic, raw)
            logger.info(
                f"[LLM] Synthesis complete — "
                f"findings={len(report.key_findings)} sources={len(report.sources)} "
                f"input_tokens={report.token_usage.get('input_tokens', '?')} "
                f"output_tokens={report.token_usage.get('output_tokens', '?')}"
            )
            return report
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                f"Synthesis failed for topic='{topic}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Private: API call with retry
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(settings.max_retries),
        before_sleep=before_sleep_log(logger, logging.WARNING),  # type: ignore[arg-type]
        reraise=True,
    )
    async def _call_with_retry(self, user_content: str) -> dict:  # type: ignore[type-arg]
        """
        Call the Anthropic messages endpoint and return the parsed JSON dict.
        Strips markdown fences defensively and validates required keys.

        Raises:
            json.JSONDecodeError: If the model returns non-JSON content.
            LLMProviderError:     If required keys are missing from the response.
        """
        response = await self._client.messages.create(
            model=settings.llm_model,
            max_tokens=4_096,
            system=_SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        logger.debug(
            f"[LLM] API response — "
            f"stop_reason={response.stop_reason} "
            f"input={response.usage.input_tokens} "
            f"output={response.usage.output_tokens}"
        )

        text = response.content[0].text.strip()
        text = _FENCE_RE.sub("", text).strip()

        try:
            parsed: dict = json.loads(text)  # type: ignore[type-arg]
        except json.JSONDecodeError as exc:
            logger.error(f"[LLM] Failed to parse JSON — raw response:\n{text[:500]}")
            raise LLMProviderError(
                f"Model returned non-JSON content: {exc}"
            ) from exc

        # Pre-flight: ensure all required keys are present
        missing = _REQUIRED_KEYS - parsed.keys()
        if missing:
            raise LLMProviderError(
                f"Model response missing required keys: {sorted(missing)}"
            )

        parsed["_usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return parsed

    # ------------------------------------------------------------------
    # Private: prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _budget_per_page(num_pages: int) -> int:
        """
        Compute the character budget allocated to each page's raw_text.

        The budget shrinks proportionally as more pages are included so that
        the total scraped content fits comfortably within the model's context.
        Never drops below _MIN_CHARS_PER_PAGE.
        """
        if num_pages == 0:
            return _BASE_CHARS_PER_PAGE
        budget = _BASE_CHARS_PER_PAGE // max(1, num_pages // 3)
        return max(budget, _MIN_CHARS_PER_PAGE)

    @staticmethod
    def _build_user_message(
        topic: str,
        pages: list[ScrapedPage],
        chars_per_page: int,
    ) -> str:
        """
        Assemble the user-turn message that feeds each scraped page into the
        synthesis prompt.  Each source is wrapped in its own XML-style block
        so the model can clearly delineate source boundaries.
        """
        header = (
            f"Research Topic: {topic}\n"
            f"Total sources provided: {len(pages)}\n"
            f"{'=' * 72}\n"
        )
        blocks: list[str] = [header]

        for i, page in enumerate(pages, start=1):
            content_preview = page.raw_text[:chars_per_page]
            truncated = len(page.raw_text) > chars_per_page

            block = (
                f"<source index=\"{i}\">\n"
                f"  <title>{page.title or 'Untitled'}</title>\n"
                f"  <url>{page.url}</url>\n"
                f"  <word_count>{page.word_count}</word_count>\n"
                f"  <content{'  truncated=\"true\"' if truncated else ''}>\n"
                f"{content_preview}\n"
                f"  </content>\n"
                f"</source>"
            )
            blocks.append(block)

        blocks.append(
            f"\n{'=' * 72}\n"
            "Using the sources above, produce the research report JSON now."
        )
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Private: response mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_report(topic: str, raw: dict) -> ResearchReport:  # type: ignore[type-arg]
        """Map the validated raw dict to a ResearchReport domain object."""
        usage: dict[str, int] = raw.pop("_usage", {})

        sources: list[SourceCitation] = []
        for entry in raw.get("sources", []):
            try:
                sources.append(SourceCitation(**entry))
            except Exception as err:
                logger.warning(f"[LLM] Skipping malformed source entry {entry!r}: {err}")

        return ResearchReport(
            topic=topic,
            executive_summary=raw["executive_summary"],
            body=raw["body"],
            key_findings=raw.get("key_findings", []),
            sources=sources,
            model_used=settings.llm_model,
            token_usage=usage,
        )
