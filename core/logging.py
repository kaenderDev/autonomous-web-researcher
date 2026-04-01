"""
app/core/logging.py

Application-wide logging configuration built on Rich.

Design goals
────────────
1. Every log line is colourised and timestamped by Rich's RichHandler.
2. Logs go to stderr so they never intermix with piped stdout output.
3. Third-party libraries (httpx, urllib3, charset_normalizer) are silenced
   at WARNING level — their DEBUG traffic is not useful to operators.
4. tenacity is kept at WARNING so retry attempts ARE visible (important for
   diagnosing rate-limit or connectivity problems in production).
5. A single `get_logger(name)` factory creates module loggers and caches
   them so repeated calls return the same instance without re-configuration.
6. A `get_console()` helper returns the shared Rich Console (stderr) that
   both the logger handler and the CLI Progress widget must share.
   Using the same Console instance prevents Rich from corrupting its own
   live-display when a log line is emitted mid-animation.
"""
import logging
import sys
from functools import lru_cache

from rich.console import Console
from rich.logging import RichHandler

from app.core.config import settings

# ---------------------------------------------------------------------------
# Shared Rich console — stderr so logs don't pollute piped stdout
# ---------------------------------------------------------------------------

_CONSOLE = Console(stderr=True, highlight=False)

# ---------------------------------------------------------------------------
# Third-party loggers that produce too much noise at DEBUG/INFO
# ---------------------------------------------------------------------------

_QUIET_LOGGERS: list[str] = [
    "httpx",
    "httpcore",
    "urllib3",
    "charset_normalizer",
    "asyncio",
]

# tenacity is intentionally NOT in this list — we want retry warnings visible.


def _configure_root_logger() -> None:
    """
    Wire up the root logger exactly once.

    - RichHandler renders level, timestamp, module path, and message.
    - show_path=False  → omit the file:line suffix (too noisy in Rich markup).
    - markup=True      → allows callers to embed [bold], [red], etc. in messages.
    - rich_tracebacks  → pretty exception rendering on ERROR/CRITICAL.
    - Keywords list    → Rich highlights these words automatically.
    """
    handler = RichHandler(
        console=_CONSOLE,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        keywords=["RETRY", "FAILED", "ERROR", "SKIP", "SUCCESS", "✓", "✗", "🚀", "🔎", "🕷", "✅"],
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%H:%M:%S]"))

    root = logging.getLogger()
    # Avoid adding duplicate handlers if _configure_root_logger() is somehow
    # called more than once (e.g. during test collection with importlib reload).
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(settings.log_level)

    # Silence chatty third-party libraries.
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


_configure_root_logger()


def get_console() -> Console:
    """
    Return the shared application Rich Console (stderr).

    The CLI's Progress and Live widgets MUST use this same Console instance
    so Rich can correctly suspend/resume its live display around log lines.
    """
    return _CONSOLE


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger.

    The logger inherits the root handler and level; calling setLevel here
    on the child logger lets individual modules opt into a stricter level
    (e.g. a noisy utility module can be set to WARNING while root stays INFO).
    """
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)
    return logger
