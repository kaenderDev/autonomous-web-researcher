"""
app/core/logging.py

Configures a single, application-wide logger backed by Rich for
human-readable, colourised output.  Call `get_logger(__name__)` in
every module instead of using the root logger directly.
"""
import logging
from functools import lru_cache

from rich.logging import RichHandler

from app.core.config import settings

_LOG_FORMAT = "%(message)s"
_DATE_FORMAT = "[%X]"


def _configure_root_logger() -> None:
    """Wire up the root logger exactly once."""
    logging.basicConfig(
        level=settings.log_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_path=True,
                markup=True,
            )
        ],
    )


_configure_root_logger()


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with the application log level applied."""
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)
    return logger
