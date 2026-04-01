"""
app/main.py

CLI entrypoint for ResearchAgent Pro.

Usage examples
──────────────
    python -m app.main --topic "The future of Rust in systems programming"
    python -m app.main -t "Quantum computing 2025" --results 15 --format json
    python -m app.main -t "LLM agents" --output-dir ./my_reports --no-display

The CLI wires together:
  • Rich Progress (multi-stage spinner) — subscribed to orchestrator events
  • ResearchOrchestrator                — the async pipeline
  • ReportSynthesizer                   — Markdown save + terminal display
  • Typed error handlers                — clean exit codes per failure mode
"""
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.text import Text

from app.core.constants import OutputFormat
from app.core.exceptions import (
    NoScrapedPagesError,
    NoSearchResultsError,
    ResearchAgentError,
)
from app.core.logging import get_console, get_logger
from app.providers.base import LLMProviderError, SearchProviderError
from app.providers.llm import AnthropicLLMProvider
from app.providers.search import SerperSearchProvider
from app.services.orchestrator import (
    EventCallback,
    PipelineEvent,
    PipelineStage,
    ResearchOrchestrator,
)
from app.services.synthesizer import ReportSynthesizer

logger = get_logger(__name__)

# Use the shared console so Rich Progress and log lines share the same
# internal state — prevents corrupted output during live animations.
_console: Console = get_console()

app = typer.Typer(
    name="research-agent",
    help="Autonomous AI research agent powered by Claude & Serper.",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,  # we handle exceptions ourselves
)

# ---------------------------------------------------------------------------
# Stage labels shown in the Progress bar
# ---------------------------------------------------------------------------

_STAGE_LABELS: dict[PipelineStage, str] = {
    PipelineStage.SEARCHING:    "Searching the web",
    PipelineStage.SCRAPING:     "Scraping pages",
    PipelineStage.SYNTHESISING: "Synthesising with Claude",
    PipelineStage.SAVING:       "Saving report",
    PipelineStage.COMPLETE:     "Complete",
    PipelineStage.ERROR:        "Failed",
}


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@app.command()
def main(
    topic: Annotated[
        str,
        typer.Option("--topic", "-t", help="Research topic or question to investigate"),
    ],
    num_results: Annotated[
        int,
        typer.Option(
            "--results", "-n",
            help="Number of search results to fetch (1–50)",
            min=1, max=50,
        ),
    ] = 10,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format", "-f",
            help="Output format for the saved report [markdown|json]",
        ),
    ] = OutputFormat.MARKDOWN,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o",
            help="Directory to save the report (created if absent)",
        ),
    ] = Path("reports"),
    no_display: Annotated[
        bool,
        typer.Option(
            "--no-display",
            help="Skip the terminal report display after saving",
        ),
    ] = False,
) -> None:
    """
    Run the autonomous research pipeline and save a timestamped report.

    Example:

        research-agent --topic "Rust in systems programming" --results 12
    """
    _print_banner(topic)

    exit_code = asyncio.run(
        _run_pipeline(
            topic=topic,
            num_results=num_results,
            output_format=output_format,
            output_dir=output_dir,
            display=not no_display,
        )
    )
    raise SystemExit(exit_code)


# ---------------------------------------------------------------------------
# Async pipeline runner
# ---------------------------------------------------------------------------

async def _run_pipeline(
    topic: str,
    num_results: int,
    output_format: OutputFormat,
    output_dir: Path,
    display: bool,
) -> int:
    """
    Orchestrate the full pipeline inside a Rich Progress context.

    Returns:
        0 on success, non-zero on any failure.
    """
    synthesizer = ReportSynthesizer()

    with _build_progress() as progress:
        task_id = progress.add_task(
            description="[cyan]Initialising…",
            total=100,
        )

        def on_event(event: PipelineEvent) -> None:
            """Update the Rich Progress bar from orchestrator pipeline events."""
            label = _STAGE_LABELS.get(event.stage, event.message)
            detail = f"  [dim]{event.detail}[/dim]" if event.detail else ""
            progress.update(
                task_id,
                description=f"[cyan]{label}[/cyan]{detail}",
                completed=int(event.progress * 100),
            )
            if event.stage not in (PipelineStage.ERROR, PipelineStage.COMPLETE):
                logger.info(
                    f"[Pipeline] {label}"
                    + (f" — {event.detail}" if event.detail else "")
                )

        orchestrator = ResearchOrchestrator(
            search_provider=SerperSearchProvider(),
            llm_provider=AnthropicLLMProvider(),
            on_event=on_event,
        )

        try:
            report = await orchestrator.run(topic=topic, num_results=num_results)
        except KeyboardInterrupt:
            progress.stop()
            _console.print("\n[yellow]⚠ Interrupted by user.[/yellow]")
            return 130
        except NoSearchResultsError as exc:
            progress.stop()
            _print_error("No Search Results", str(exc))
            return 2
        except NoScrapedPagesError as exc:
            progress.stop()
            _print_error("Scraping Failed", str(exc))
            return 3
        except SearchProviderError as exc:
            progress.stop()
            _print_error("Search API Error", str(exc))
            return 4
        except LLMProviderError as exc:
            progress.stop()
            _print_error("LLM API Error", str(exc))
            return 5
        except ResearchAgentError as exc:
            progress.stop()
            _print_error("Pipeline Error", str(exc))
            return 1
        except Exception as exc:  # noqa: BLE001
            progress.stop()
            logger.exception(f"Unexpected error: {exc}")
            _print_error("Unexpected Error", str(exc))
            return 1

        # ── Stage: Save ──────────────────────────────────────────────────
        progress.update(
            task_id,
            description="[cyan]Saving report…[/cyan]",
            completed=90,
        )
        on_event(PipelineEvent(
            stage=PipelineStage.SAVING,
            message="Saving report…",
            detail=str(output_dir),
            progress=0.90,
        ))

        saved_path = synthesizer.save_report(report, output_dir, output_format)

        progress.update(
            task_id,
            description="[bold green]✅ Done[/bold green]",
            completed=100,
        )

    # ── Display ────────────────────────────────────────────────────────────
    if display:
        synthesizer.print_to_console(report, saved_path=saved_path)
    else:
        _print_summary(report, saved_path)

    return 0


# ---------------------------------------------------------------------------
# Rich UI helpers
# ---------------------------------------------------------------------------

def _build_progress() -> Progress:
    """
    Build the Rich Progress widget used during pipeline execution.

    Layout: [spinner] [description ............] [bar] [elapsed]
    """
    return Progress(
        SpinnerColumn(spinner_name="dots2", style="bright_cyan"),
        TextColumn("[progress.description]{task.description}", justify="left"),
        BarColumn(bar_width=28, style="bright_blue", complete_style="bright_green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=_console,
        transient=False,   # keep the final state visible after completion
        refresh_per_second=12,
    )


def _print_banner(topic: str) -> None:
    """Print the startup banner before the pipeline begins."""
    _console.print()
    _console.print(Rule("[bold bright_blue]ResearchAgent Pro[/bold bright_blue]", style="bright_blue"))
    _console.print(
        Panel(
            Text(topic, style="bold italic white", justify="center"),
            title="[dim]Research Topic[/dim]",
            border_style="bright_blue",
            padding=(0, 4),
        )
    )
    _console.print(
        f"  [dim]Started:[/dim]  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    _console.print()


def _print_summary(report, saved_path: Path) -> None:
    """Print a compact one-panel summary when --no-display is set."""
    _console.print()
    _console.print(
        Panel(
            f"[bold]{report.executive_summary}[/bold]\n\n"
            f"[dim]Findings:[/dim] {len(report.key_findings)}   "
            f"[dim]Sources:[/dim] {len(report.sources)}   "
            f"[dim]Tokens:[/dim] "
            f"{report.token_usage.get('input_tokens', '?')} in / "
            f"{report.token_usage.get('output_tokens', '?')} out\n"
            f"[dim]Saved →[/dim] [bold green]{saved_path}[/bold green]",
            title=f"[bold cyan]{report.topic}[/bold cyan]",
            border_style="green",
            padding=(1, 2),
        )
    )


def _print_error(title: str, message: str) -> None:
    """Render a prominent error panel and log the message."""
    logger.error(f"[{title}] {message}")
    _console.print()
    _console.print(
        Panel(
            f"[bold red]{message}[/bold red]",
            title=f"[bold red]✗ {title}[/bold red]",
            border_style="red",
            padding=(0, 2),
        )
    )
    _console.print()


if __name__ == "__main__":
    app()
