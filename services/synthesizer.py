"""
app/services/synthesizer.py

Renders a ResearchReport to Markdown or JSON, and saves it to disk.

Responsibilities (and only these):
  • _to_markdown / _to_json  — convert domain model → string
  • save_report              — write file with auto-generated timestamped path
  • print_to_console         — Rich-formatted terminal display

This module deliberately has no knowledge of the pipeline or HTTP layer.
"""
import json
import re
from datetime import datetime
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from app.core.constants import OutputFormat
from app.core.logging import get_console, get_logger
from app.domain.models import ResearchReport

logger = get_logger(__name__)

# The synthesiser uses the *shared* console so Rich's internal state
# (e.g. a Progress spinner in main.py) is not corrupted.
_console: Console = get_console()

# Maximum characters in the topic slug used for file naming.
_SLUG_MAX_LEN: int = 60


class ReportSynthesizer:
    """Renders and persists a ResearchReport."""

    # ------------------------------------------------------------------
    # Public: save to disk
    # ------------------------------------------------------------------

    def save_report(
        self,
        report: ResearchReport,
        output_dir: Path = Path("reports"),
        output_format: OutputFormat = OutputFormat.MARKDOWN,
    ) -> Path:
        """
        Render the report and write it to *output_dir* with an
        auto-generated, timestamped filename.

        Filename pattern:
            reports/{slug}_{YYYYMMDD_HHMMSS}.md   (or .json)

        Args:
            report:        The fully-synthesised ResearchReport.
            output_dir:    Directory to write the file into (created if absent).
            output_format: markdown (default) or json.

        Returns:
            The absolute Path of the written file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        slug = self._topic_to_slug(report.topic)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        ext = ".md" if output_format == OutputFormat.MARKDOWN else ".json"
        filename = f"{slug}_{timestamp}{ext}"
        output_path = output_dir / filename

        rendered = (
            self._to_markdown(report)
            if output_format == OutputFormat.MARKDOWN
            else self._to_json(report)
        )

        output_path.write_text(rendered, encoding="utf-8")
        logger.info(f"[Synthesizer] 💾 Report saved → [bold]{output_path}[/bold]")
        return output_path

    # ------------------------------------------------------------------
    # Public: explicit path save (used by tests / programmatic callers)
    # ------------------------------------------------------------------

    def render_to_path(
        self,
        report: ResearchReport,
        output_path: Path,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
    ) -> str:
        """
        Render the report and write it to an explicit *output_path*.

        Returns the rendered string (for testing / chaining).
        """
        rendered = (
            self._to_markdown(report)
            if output_format == OutputFormat.MARKDOWN
            else self._to_json(report)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        logger.info(f"[Synthesizer] 💾 Report saved → [bold]{output_path}[/bold]")
        return rendered

    # ------------------------------------------------------------------
    # Public: terminal display
    # ------------------------------------------------------------------

    def print_to_console(self, report: ResearchReport, saved_path: Path | None = None) -> None:
        """
        Pretty-print the report to the terminal using Rich panels and Markdown.

        Displays in order: header, executive summary, key findings, full body,
        sources list, and a footer line with metadata.
        """
        _console.print()
        _console.print(Rule(style="dim"))
        _console.print(
            Panel(
                Text(report.topic, style="bold cyan", justify="center"),
                title="[bold white]ResearchAgent Pro — Report[/bold white]",
                border_style="bright_blue",
                padding=(1, 4),
            )
        )

        # Executive summary
        _console.print(
            Panel(
                report.executive_summary,
                title="[bold green]Executive Summary[/bold green]",
                border_style="green",
                padding=(0, 2),
            )
        )

        # Key findings as a bulleted Rich text block
        if report.key_findings:
            findings_text = "\n".join(f"  • {f}" for f in report.key_findings)
            _console.print(
                Panel(
                    findings_text,
                    title="[bold yellow]Key Findings[/bold yellow]",
                    border_style="yellow",
                    padding=(0, 2),
                )
            )

        # Full body as rendered Markdown
        _console.print(Rule("[dim]Full Report[/dim]", style="dim"))
        _console.print(Markdown(report.body))

        # Sources
        if report.sources:
            _console.print(Rule("[dim]Sources[/dim]", style="dim"))
            for i, src in enumerate(report.sources, 1):
                note = f"  [dim]{src.relevance_note}[/dim]" if src.relevance_note else ""
                _console.print(f"  [cyan]{i}.[/cyan] [link={src.url}]{src.title}[/link]{note}")

        # Footer
        _console.print()
        _console.print(Rule(style="dim"))
        meta_parts = [
            f"[dim]Generated:[/dim] {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"[dim]Model:[/dim] {report.model_used}",
            f"[dim]Tokens:[/dim] {report.token_usage.get('input_tokens', '?')} in / "
            f"{report.token_usage.get('output_tokens', '?')} out",
        ]
        if saved_path:
            meta_parts.append(f"[dim]Saved:[/dim] [bold]{saved_path}[/bold]")
        _console.print("  " + "   │   ".join(meta_parts))
        _console.print(Rule(style="dim"))
        _console.print()

    # ------------------------------------------------------------------
    # Private: format renderers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_markdown(report: ResearchReport) -> str:
        """Render report to a complete, self-contained Markdown document."""
        ts = report.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        lines: list[str] = [
            f"# {report.topic}",
            "",
            f"> Generated {ts} using `{report.model_used}`  ",
            f"> {report.token_usage.get('input_tokens', '?')} input tokens · "
            f"{report.token_usage.get('output_tokens', '?')} output tokens",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            report.executive_summary,
            "",
            "---",
            "",
            "## Key Findings",
            "",
        ]
        for finding in report.key_findings:
            lines.append(f"- {finding}")
        lines += [
            "",
            "---",
            "",
            "## Full Report",
            "",
            report.body,
            "",
            "---",
            "",
            "## Sources",
            "",
        ]
        for i, source in enumerate(report.sources, 1):
            note = f" — {source.relevance_note}" if source.relevance_note else ""
            lines.append(f"{i}. [{source.title}]({source.url}){note}")
        lines.append("")  # trailing newline
        return "\n".join(lines)

    @staticmethod
    def _to_json(report: ResearchReport) -> str:
        """Render report to pretty-printed JSON."""
        return json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)

    @staticmethod
    def _topic_to_slug(topic: str) -> str:
        """
        Convert a free-text topic into a safe filename slug.

        'The future of Rust in systems programming'
         → 'the_future_of_rust_in_systems_programming'
        """
        slug = topic.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)      # strip non-word chars
        slug = re.sub(r"[\s-]+", "_", slug)        # spaces/hyphens → underscore
        slug = slug.strip("_")
        return slug[:_SLUG_MAX_LEN]
