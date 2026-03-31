"""
app/main.py

CLI entrypoint for ResearchAgent Pro.  Uses Typer for a clean,
self-documenting command-line interface.

Usage:
    python -m app.main --topic "The future of Rust in systems programming"
    python -m app.main --topic "Quantum computing 2025" --format json --output report.json
"""
import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from app.core.constants import OutputFormat
from app.providers.llm import AnthropicLLMProvider
from app.providers.search import SerperSearchProvider
from app.services.orchestrator import ResearchOrchestrator

app = typer.Typer(
    name="research-agent",
    help="Autonomous AI research agent powered by Claude & Serper.",
    add_completion=False,
)

_console = Console()


@app.command()
def main(
    topic: Annotated[
        str,
        typer.Option("--topic", "-t", help="Research topic or question to investigate"),
    ],
    num_results: Annotated[
        int,
        typer.Option("--results", "-n", help="Number of search results to fetch", min=1, max=50),
    ] = 10,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format: markdown | json"),
    ] = OutputFormat.MARKDOWN,
    output_path: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Path to write the report file"),
    ] = None,
    no_console: Annotated[
        bool,
        typer.Option("--no-console", help="Suppress terminal report output"),
    ] = False,
) -> None:
    """Run the autonomous research pipeline for a given topic."""
    _console.print(
        f"\n[bold green]ResearchAgent Pro[/bold green] — researching: "
        f"[italic cyan]{topic}[/italic cyan]\n"
    )

    orchestrator = ResearchOrchestrator(
        search_provider=SerperSearchProvider(),
        llm_provider=AnthropicLLMProvider(),
    )

    asyncio.run(
        orchestrator.run(
            topic=topic,
            num_results=num_results,
            output_format=output_format,
            output_path=output_path,
            print_to_console=not no_console,
        )
    )


if __name__ == "__main__":
    app()
