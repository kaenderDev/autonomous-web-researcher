"""
app/providers/search.py

Concrete Serper.dev search provider.  Implements BaseSearchProvider and
adds exponential-backoff retry logic via `tenacity`.
"""
import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.constants import SERPER_SEARCH_URL
from app.core.logging import get_logger
from app.domain.models import SearchResponse, SearchResult
from app.providers.base import BaseSearchProvider, SearchProviderError

logger = get_logger(__name__)

_RETRY_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


class SerperSearchProvider(BaseSearchProvider):
    """
    Hits the Serper.dev Google Search API and maps results to the
    canonical SearchResponse / SearchResult domain models.
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client  # injected in tests; created lazily in prod

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        logger.info(f"[Search] query='{query}' num_results={num_results}")
        try:
            raw = await self._fetch_with_retry(query, num_results)
            return self._parse(query, raw)
        except Exception as exc:
            raise SearchProviderError(
                f"Serper search failed for query '{query}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(settings.max_retries),
        reraise=True,
    )
    async def _fetch_with_retry(self, query: str, num_results: int) -> dict:  # type: ignore[type-arg]
        headers = {
            "X-API-KEY": settings.serper_api_key.get_secret_value(),
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}

        async with (self._client or httpx.AsyncClient()) as client:
            resp = client if self._client else client
            response = await resp.post(
                SERPER_SEARCH_URL,
                json=payload,
                headers=headers,
                timeout=settings.http_timeout,
            )
            response.raise_for_status()
            logger.debug(f"[Search] HTTP {response.status_code} from Serper")
            return response.json()  # type: ignore[no-any-return]

    @staticmethod
    def _parse(query: str, raw: dict) -> SearchResponse:  # type: ignore[type-arg]
        organic: list[dict] = raw.get("organic", [])  # type: ignore[type-arg]
        results = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                position=idx + 1,
            )
            for idx, item in enumerate(organic)
        ]
        logger.info(f"[Search] Parsed {len(results)} organic results")
        return SearchResponse(query=query, results=results, provider="serper")
