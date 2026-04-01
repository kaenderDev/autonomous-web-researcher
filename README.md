# ResearchAgent Pro 🤖🔍

An autonomous AI research agent that transforms a single search query into a
comprehensive, timestamped Markdown report — powered by **Claude** (Anthropic)
and **Serper.dev** (Google Search API).

---

## How it works

```
--topic "..."
      │
      ▼
┌─────────────┐     SearchResult[]    ┌──────────────┐     ScrapedPage[]
│   Serper    │ ───────────────────▶  │  WebScraper  │ ──────────────────▶ ┐
│   Search    │                       │  (asyncio,   │                      │
│  Provider   │                       │   httpx,     │                      │
└─────────────┘                       │   BS4)       │                      │
                                      └──────────────┘                      │
                                                                             │
                                      ┌──────────────┐    ResearchReport    │
                                      │   Anthropic  │ ◀────────────────────┘
                                      │     LLM      │
                                      │   Provider   │
                                      └──────┬───────┘
                                             │
                                             ▼
                                      reports/{slug}_{timestamp}.md
```

**Pipeline stages (visible in the terminal spinner):**

| Stage | What happens |
|---|---|
| 🔎 Searching | Queries Serper.dev for organic results |
| 🕷 Scraping | Fetches all URLs concurrently (semaphore-limited) |
| 🤖 Synthesising | Sends cleaned content to Claude for structured analysis |
| 💾 Saving | Writes a timestamped `.md` (or `.json`) report to disk |

---

## Installation

### Prerequisites

- Python **3.10+**
- API keys for [Anthropic](https://console.anthropic.com) and [Serper.dev](https://serper.dev)

### Steps

```bash
# 1 — Clone and enter the project
git clone https://github.com/your-username/research-agent-pro.git
cd research-agent-pro

# 2 — Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3a — Install runtime dependencies only
pip install -r requirements.txt

# 3b — OR install with dev toolchain (pytest, ruff, mypy)
pip install -e ".[dev]"

# 4 — Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY and SERPER_API_KEY
```

---

## Usage

### Basic research run

```bash
python -m app --topic "The future of Rust in systems programming"
```

This will:
1. Search the web for 10 results (default)
2. Scrape each page concurrently
3. Synthesise a report with Claude
4. Save it to `reports/the_future_of_rust_in_systems_programming_20250401_143022.md`
5. Display the full report in the terminal

### All flags

```
Usage: python -m app [OPTIONS]

Options:
  -t, --topic TEXT           Research topic or question to investigate [required]
  -n, --results INTEGER      Number of search results to fetch (1-50) [default: 10]
  -f, --format [markdown|json]
                             Output format for the saved report [default: markdown]
  -o, --output-dir PATH      Directory to save the report (created if absent)
                             [default: reports]
      --no-display           Skip the terminal report display after saving
      --help                 Show this message and exit.
```

### Examples

```bash
# Wider search, JSON output, custom save directory
python -m app \
  --topic "Quantum computing breakthroughs 2025" \
  --results 15 \
  --format json \
  --output-dir ./my_reports

# CI / scripted mode — no terminal display, exit codes checked by caller
python -m app --topic "LLM agent architectures" --no-display
echo "Exit: $?"   # 0 = success, 2 = no results, 3 = scrape failed, 4/5 = API errors

# Via the installed entry point (after pip install -e .)
research-agent --topic "WebAssembly in the browser"
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success — report saved |
| `2` | No search results (try broadening the query) |
| `3` | All pages failed to scrape |
| `4` | Search API error (Serper) |
| `5` | LLM API error (Anthropic) |
| `130` | Interrupted (Ctrl+C) |

---

## Project structure

```
research-agent-pro/
├── app/
│   ├── __main__.py          # Enables: python -m app
│   ├── main.py              # Typer CLI — progress spinner, error panels
│   ├── core/
│   │   ├── config.py        # Pydantic BaseSettings (reads .env)
│   │   ├── constants.py     # Shared literals, enums, HTTP headers
│   │   ├── exceptions.py    # Typed domain exceptions (NoSearchResultsError, …)
│   │   └── logging.py       # Rich logging — stderr, retry-visible, noise-suppressed
│   ├── domain/
│   │   └── models.py        # Pydantic models: SearchResult, ScrapedPage, ResearchReport
│   ├── providers/
│   │   ├── base.py          # Abstract Base Classes (Strategy pattern)
│   │   ├── search.py        # SerperSearchProvider — retries, URL validation
│   │   └── llm.py           # AnthropicLLMProvider — fence stripping, budget scaling
│   └── services/
│       ├── orchestrator.py  # Pipeline coordinator — emits PipelineEvents
│       ├── scraper.py       # WebScraper — tag-whitelist extraction, semaphore
│       └── synthesizer.py   # ReportSynthesizer — Markdown/JSON render, auto-save
├── tests/
│   ├── conftest.py          # Shared fixtures (make_report, make_scraped_pages, …)
│   ├── test_scraper.py      # 30+ tests: parse, noise exclusion, 1-of-5, semaphore
│   ├── test_orchestrator.py # 25+ tests: happy path, error branches, LowYieldWarning
│   ├── test_providers.py    # SerperSearchProvider + AnthropicLLMProvider unit tests
│   └── test_synthesizer.py  # Slug generation, Markdown/JSON render, save-to-disk
├── requirements.txt         # Runtime pinned dependencies
├── pyproject.toml           # Build, ruff, mypy, pytest configuration
└── .env.example             # Environment variable template
```

---

## Running tests

```bash
# Full suite with coverage
pytest

# Single module
pytest tests/test_orchestrator.py -v

# Only the 1-of-5 yield scenario
pytest tests/test_orchestrator.py -k "low_yield" -v

# Only the semaphore test
pytest tests/test_scraper.py -k "semaphore" -v

# Coverage report in browser
open htmlcov/index.html
```

---

## Design notes

### Architecture decisions

**Strategy Pattern for providers.** `BaseSearchProvider` and `BaseLLMProvider` are pure ABCs. Swapping Serper for Bing, or Claude for GPT-4, requires only a new concrete class — the orchestrator and tests are untouched.

**Tag-whitelist scraping.** Instead of stripping noise tags and calling `get_text()` on the whole tree, the scraper extracts only from `h1–h6`, `p`, `li`, `td`, `th`, and `blockquote`. Noise containers (`nav`, `header`, `footer`, `aside`, etc.) are decomposed first so their content tags are removed atomically.

**PipelineEvent callbacks.** The orchestrator emits events via an `on_event` callback rather than calling Rich directly. The CLI subscribes a Progress updater; tests subscribe `list.append`. Clean separation — the orchestrator has zero knowledge of Rich.

**Graceful degradation.** If some (but not all) pages fail to scrape, a `LowYieldWarning` is issued and the pipeline continues with the pages that did succeed. Only if *every* page fails is `NoScrapedPagesError` raised.

**Dynamic token budgeting.** The LLM prompt allocates characters per page proportionally: fewer pages → more context each, many pages → less each, never below `_MIN_CHARS_PER_PAGE`. This keeps the total prompt within the model's context window regardless of how many URLs were scraped.

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

*Built with Claude Code — showcasing autonomous research pipelines and senior software architecture.*
