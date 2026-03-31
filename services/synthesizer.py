"""
app/services/synthesizer.py

Responsible for *rendering* a finished ResearchReport to a target output
format (Markdown, JSON).  This service does NOT call any LLM — LLM work
is delegated to BaseLLMProvider implementations.

The synthesiser sits at the end of the pipeline and is the only layer
that knows about file I/O and formatting concerns.
"""
import json
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from app.core.constants import OutputFormat
from app.core.logging import get_logger
from app.domain.models import ResearchReport

logger = get_logger(__name__)
_console = Console()


class ReportSynthesizer:
    """Renders a ResearchReport to the requested output format."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def render(
        self,
        report: ResearchReport,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        output_path: Path | None = None,
    ) -> str:
        """
        Render the report to a string in the requested format, and
        optionally write it to disk.

        Args:
            report:        The fully-synthesised ResearchReport.
            output_format: One of markdown | json.
            output_path:   If provided, write the rendered string here.

        Returns:
            The rendered report as a string.
        """
        rendered = (
            self._to_markdown(report)
            if output_format == OutputFormat.MARKDOWN
            else self._to_json(report)
        )

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered, encoding="utf-8")
            logger.info(f"[Synthesizer] Report written to {output_path}")

        return rendered

    def print_to_console(self, report: ResearchReport) -> None:
        """Pretty-print the report to the terminal using Rich."""
        _console.print(
            Panel(
                f"[bold cyan]{report.topic}[/bold cyan]",
                title="ResearchAgent Pro — Report",
                expand=False,
            )
        )
        _console.print(
            Panel(
                report.executive_summary,
                title="Executive Summary",
                border_style="green",
            )
        )
        _console.print(Markdown(report.body))
        _console.print(
            f"\n[dim]Generated at {report.generated_at.isoformat()} "
            f"using {report.model_used}[/dim]"
        )
        _console.print(
            f"[dim]Tokens — input: {report.token_usage.get('input_tokens', '?')} "
            f"output: {report.token_usage.get('output_tokens', '?')}[/dim]"
        )

    # ------------------------------------------------------------------
    # Private: format renderers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_markdown(report: ResearchReport) -> str:
        lines: list[str] = [
            f"# {report.topic}\n",
            f"*Generated at {report.generated_at.isoformat()} · {report.model_used}*\n",
            "---\n",
            "## Executive Summary\n",
            f"{report.executive_summary}\n",
            "---\n",
            "## Key Findings\n",
        ]
        for finding in report.key_findings:
            lines.append(f"- {finding}")
        lines += ["\n---\n", "## Full Report\n", report.body, "\n---\n", "## Sources\n"]
        for source in report.sources:
            note = f" — {source.relevance_note}" if source.relevance_note else ""
            lines.append(f"- [{source.title}]({source.url}){note}")
        return "\n".join(lines)

    @staticmethod
    def _to_json(report: ResearchReport) -> str:
        return json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
