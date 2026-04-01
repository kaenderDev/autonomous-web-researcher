"""
app/providers/search.py

Concrete Serper.dev search provider.  Strictly implements BaseSearchProvider.

Design decisions:
  - The httpx.AsyncClient is managed as a persistent instance on `self` so it
    can be reused across retries and injected cleanly in tests.  We never wrap
    an already-open client in `async with` — that would close it prematurely.
  - Retry targets only transient network/rate-limit conditions; 4xx client
    errors (except 429) are NOT retried because they indicate bad inputs and
    would waste quota.
  - URL validation is applied during _parse to silently drop malformed entries
    rather than letting a Pydantic ValidationError bubble up from the loop.
"""
import logging

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.constants import SERPER_SEARCH_URL
from app.core.logging import get_logger
from app.domain.models import SearchResponse, SearchResult
from app.providers.base import BaseSearchProvider, SearchProviderError

logger = get_logger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """
    Decide whether an exception warrants a retry attempt.

    Retryable:  network errors, timeouts, 429 Too Many Requests, 5xx server errors.
    Not retryable: 4xx client errors (bad key, malformed query, etc.)
    """
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class SerperSearchProvider(BaseSearchProvider):
    """
    Hits the Serper.dev Google Search API and maps results to the canonical
    SearchResponse / SearchResult domain models.

    The provider owns the lifetime of its httpx.AsyncClient.  Pass a
    pre-built client only in tests; production callers use the default.
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        # Store (but do not own lifecycle of) an externally injected client, or
        # create a reusable client with sensible connection pool defaults.
        self._client: httpx.AsyncClient = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.http_timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self._owns_client: bool = client is None  # only close what we created

    # ------------------------------------------------------------------
    # Public interface — satisfies BaseSearchProvider contract
    # ------------------------------------------------------------------

    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        """
        Execute a search against Serper.dev and return a normalised
        SearchResponse.  Raises SearchProviderError after all retries fail.
        """
        logger.info(f"[Search] provider=serper query='{query}' num_results={num_results}")
        try:
            raw = await self._fetch_with_retry(query, num_results)
            response = self._parse(query, raw)
            logger.info(
                f"[Search] Completed — {len(response.results)} results "
                f"for query='{query}'"
            )
            return response
        except SearchProviderError:
            raise  # already wrapped — let it through unchanged
        except Exception as exc:
            raise SearchProviderError(
                f"Serper search failed after retries for query='{query}': {exc}"
            ) from exc
        finally:
            if self._owns_client:
                await self._client.aclose()

    # ------------------------------------------------------------------
    # Private: HTTP fetch with exponential backoff
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(settings.max_retries),
        before_sleep=before_sleep_log(logger, logging.WARNING),  # type: ignore[arg-type]
        reraise=True,
    )
    async def _fetch_with_retry(self, query: str, num_results: int) -> dict:  # type: ignore[type-arg]
        """POST to Serper.dev and return the raw JSON dict."""
        headers = {
            "X-API-KEY": settings.serper_api_key.get_secret_value(),
            "Content-Type": "application/json",
        }
        payload: dict[str, int | str] = {
            "q": query,
            "num": num_results,
            "gl": "us",   # geo-locale: United States
            "hl": "en",   # language: English
        }
        logger.debug(f"[Search] POST {SERPER_SEARCH_URL} payload={payload}")
        response = await self._client.post(
            SERPER_SEARCH_URL,
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        logger.debug(f"[Search] HTTP {response.status_code} received from Serper")
        return response.json()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Private: response mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(query: str, raw: dict) -> SearchResponse:  # type: ignore[type-arg]
        """
        Map the raw Serper JSON payload to the SearchResponse domain model.

        Only organic results are considered.  Each entry is validated
        individually; malformed URLs are skipped rather than raising.
        """
        organic: list[dict] = raw.get("organic", [])  # type: ignore[type-arg]
        results: list[SearchResult] = []

        for idx, item in enumerate(organic):
            url: str = item.get("link", "").strip()
            title: str = item.get("title", "").strip()

            # Skip entries with missing or obviously invalid URLs
            if not url or not url.startswith(("http://", "https://")):
                logger.debug(f"[Search] Skipping result #{idx + 1} — invalid URL: {url!r}")
                continue

            try:
                results.append(
                    SearchResult(
                        title=title or url,           # fall back to URL if title absent
                        url=url,
                        snippet=item.get("snippet", "").strip(),
                        position=len(results) + 1,    # re-number after skips
                    )
                )
            except Exception as validation_err:
                logger.warning(
                    f"[Search] Skipping result #{idx + 1} — "
                    f"validation error: {validation_err}"
                )

        logger.debug(f"[Search] Parsed {len(results)}/{len(organic)} valid organic results")
        return SearchResponse(query=query, results=results, provider="serper")
