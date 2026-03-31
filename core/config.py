"""
app/core/config.py

Centralised, type-safe configuration loaded from environment variables
(or a .env file) via pydantic-settings.  Import the `settings` singleton
anywhere in the application — never read `os.environ` directly.
"""
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings validated at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- LLM -----------------------------------------------------------------
    anthropic_api_key: SecretStr = Field(..., description="Anthropic API key")
    llm_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model string used for synthesis",
    )

    # -- Search --------------------------------------------------------------
    serper_api_key: SecretStr = Field(..., description="Serper.dev API key")
    max_search_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of search results fetched per query",
    )

    # -- Scraping ------------------------------------------------------------
    max_concurrent_scrapers: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Semaphore limit for concurrent HTTP scraping tasks",
    )
    http_timeout: float = Field(
        default=20.0,
        gt=0,
        description="Per-request HTTP timeout in seconds",
    )
    max_retries: int = Field(
        default=4,
        ge=0,
        le=10,
        description="Maximum retry attempts with exponential backoff",
    )

    # -- Logging -------------------------------------------------------------
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()  # type: ignore[call-arg]


# Convenience alias used throughout the codebase
settings: Settings = get_settings()
