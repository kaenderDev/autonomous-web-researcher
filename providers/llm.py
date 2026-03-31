"""
app/providers/llm.py

Anthropic/Claude concrete LLM provider.  Implements BaseLLMProvider,
builds the synthesis prompt, calls the API with adaptive thinking, and
maps the response to a ResearchReport domain model.
"""
import json

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.models import ResearchReport, ScrapedPage, SourceCitation
from app.providers.base import BaseLLMProvider, LLMProviderError

logger = get_logger(__name__)

_SYNTHESIS_SYSTEM_PROMPT = """\
You are an elite research analyst.  You receive a research topic and a
collection of scraped web page contents.  Your task is to synthesise a
comprehensive, accurate, and well-structured research report in strict
JSON format matching the schema below.

Schema (return ONLY this JSON — no markdown fences, no preamble):
{
  "executive_summary": "<3-5 sentence TL;DR>",
  "body": "<full Markdown report body — use ### headings, lists, and emphasis>",
  "key_findings": ["<finding 1>", "<finding 2>", "..."],
  "sources": [
    {"title": "<page title>", "url": "<url>", "relevance_note": "<one sentence>"}
  ]
}

Rules:
- Be factual; only claim what the sources support.
- Cite sources inline using [Title](url) Markdown links within the body.
- key_findings must be 5-10 concise, standalone bullet points.
- Do not hallucinate URLs or statistics not present in the provided content.
"""


class AnthropicLLMProvider(BaseLLMProvider):
    """
    Uses Claude (via the official Anthropic SDK) to synthesise scraped
    content into a structured ResearchReport.
    """

    def __init__(self, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._client = client or anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def synthesise(
        self,
        topic: str,
        scraped_pages: list[ScrapedPage],
    ) -> ResearchReport:
        logger.info(
            f"[LLM] Starting synthesis — topic='{topic}' "
            f"pages={len(scraped_pages)} model={settings.llm_model}"
        )
        user_content = self._build_user_message(topic, scraped_pages)
        try:
            raw_json = await self._call_with_retry(user_content)
            return self._parse_report(topic, raw_json)
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                f"Synthesis failed for topic '{topic}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(
            (anthropic.APIConnectionError, anthropic.RateLimitError)
        ),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(settings.max_retries),
        reraise=True,
    )
    async def _call_with_retry(self, user_content: str) -> dict:  # type: ignore[type-arg]
        response = await self._client.messages.create(
            model=settings.llm_model,
            max_tokens=4096,
            system=_SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        logger.debug(
            f"[LLM] Usage — input={response.usage.input_tokens} "
            f"output={response.usage.output_tokens}"
        )
        text = response.content[0].text.strip()
        # Strip accidental markdown fences if the model slips.
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed: dict = json.loads(text)  # type: ignore[type-arg]
        parsed["_usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return parsed

    @staticmethod
    def _build_user_message(topic: str, pages: list[ScrapedPage]) -> str:
        sections: list[str] = [
            f"Research Topic: {topic}\n",
            f"Number of sources: {len(pages)}\n",
            "=" * 60,
        ]
        for i, page in enumerate(pages, 1):
            sections.append(
                f"\n[Source {i}] {page.title}\nURL: {page.url}\n"
                f"Word count: {page.word_count}\n\n"
                f"{page.raw_text[:3000]}"  # Truncate to keep within context limits
            )
        return "\n".join(sections)

    @staticmethod
    def _parse_report(topic: str, raw: dict) -> ResearchReport:  # type: ignore[type-arg]
        usage: dict[str, int] = raw.pop("_usage", {})
        sources = [SourceCitation(**s) for s in raw.get("sources", [])]
        return ResearchReport(
            topic=topic,
            executive_summary=raw["executive_summary"],
            body=raw["body"],
            key_findings=raw.get("key_findings", []),
            sources=sources,
            model_used=settings.llm_model,
            token_usage=usage,
        )
