"""
tests/test_synthesizer.py

Unit tests for ReportSynthesizer.

Covers:
  1. _topic_to_slug — slug generation edge cases
  2. _to_markdown   — structure and content of the rendered document
  3. _to_json       — valid JSON, all keys present
  4. save_report    — file creation, path structure, timestamp format
  5. render_to_path — explicit path write
"""
import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from app.domain.models import ResearchReport, SourceCitation
from app.services.synthesizer import ReportSynthesizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _report(**overrides) -> ResearchReport:
    defaults = dict(
        topic="The future of Rust in systems programming",
        executive_summary="Rust is becoming dominant in systems software.",
        body="## Why Rust\n\nRust prevents memory errors.",
        key_findings=["Memory safety", "Zero-cost abstractions"],
        sources=[
            SourceCitation(
                title="Rust Lang",
                url="https://www.rust-lang.org",
                relevance_note="Official site.",
            )
        ],
        model_used="claude-sonnet-4-6",
        token_usage={"input_tokens": 200, "output_tokens": 80},
    )
    defaults.update(overrides)
    return ResearchReport(**defaults)


# ---------------------------------------------------------------------------
# _topic_to_slug
# ---------------------------------------------------------------------------

class TestTopicToSlug:
    def test_basic_slug(self) -> None:
        slug = ReportSynthesizer._topic_to_slug("The future of Rust")
        assert slug == "the_future_of_rust"

    def test_special_characters_stripped(self) -> None:
        slug = ReportSynthesizer._topic_to_slug("AI & ML: What's next?")
        # Ampersand, colon, apostrophe, question mark stripped
        assert "&" not in slug
        assert ":" not in slug
        assert "?" not in slug

    def test_spaces_become_underscores(self) -> None:
        slug = ReportSynthesizer._topic_to_slug("hello world foo")
        assert " " not in slug
        assert "_" in slug

    def test_slug_truncated_at_max_length(self) -> None:
        long_topic = "word " * 30
        slug = ReportSynthesizer._topic_to_slug(long_topic)
        assert len(slug) <= 60

    def test_leading_trailing_underscores_stripped(self) -> None:
        slug = ReportSynthesizer._topic_to_slug("  hello  ")
        assert not slug.startswith("_")
        assert not slug.endswith("_")

    def test_consecutive_spaces_collapsed(self) -> None:
        slug = ReportSynthesizer._topic_to_slug("foo   bar   baz")
        assert "__" not in slug


# ---------------------------------------------------------------------------
# _to_markdown
# ---------------------------------------------------------------------------

class TestToMarkdown:
    def test_contains_topic_as_h1(self) -> None:
        md = ReportSynthesizer._to_markdown(_report())
        assert "# The future of Rust in systems programming" in md

    def test_contains_executive_summary(self) -> None:
        md = ReportSynthesizer._to_markdown(_report())
        assert "Rust is becoming dominant" in md

    def test_key_findings_as_bullets(self) -> None:
        md = ReportSynthesizer._to_markdown(_report())
        assert "- Memory safety" in md
        assert "- Zero-cost abstractions" in md

    def test_source_rendered_as_markdown_link(self) -> None:
        md = ReportSynthesizer._to_markdown(_report())
        assert "[Rust Lang](https://www.rust-lang.org)" in md

    def test_model_name_in_metadata(self) -> None:
        md = ReportSynthesizer._to_markdown(_report())
        assert "claude-sonnet-4-6" in md

    def test_body_included_verbatim(self) -> None:
        md = ReportSynthesizer._to_markdown(_report())
        assert "## Why Rust" in md
        assert "Rust prevents memory errors." in md

    def test_ends_with_newline(self) -> None:
        md = ReportSynthesizer._to_markdown(_report())
        assert md.endswith("\n")


# ---------------------------------------------------------------------------
# _to_json
# ---------------------------------------------------------------------------

class TestToJson:
    def test_valid_json(self) -> None:
        json_str = ReportSynthesizer._to_json(_report())
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_all_required_keys_present(self) -> None:
        parsed = json.loads(ReportSynthesizer._to_json(_report()))
        for key in ("topic", "executive_summary", "body", "key_findings", "sources"):
            assert key in parsed, f"Missing key: {key}"

    def test_sources_serialised_as_list(self) -> None:
        parsed = json.loads(ReportSynthesizer._to_json(_report()))
        assert isinstance(parsed["sources"], list)
        assert parsed["sources"][0]["url"] == "https://www.rust-lang.org"

    def test_pretty_printed(self) -> None:
        json_str = ReportSynthesizer._to_json(_report())
        assert "\n" in json_str  # not a single-line compact JSON


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

class TestSaveReport:
    def test_creates_output_directory(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "nested" / "reports"
        assert not out_dir.exists()
        ReportSynthesizer().save_report(_report(), output_dir=out_dir)
        assert out_dir.is_dir()

    def test_file_created_in_output_dir(self, tmp_path: Path) -> None:
        path = ReportSynthesizer().save_report(_report(), output_dir=tmp_path)
        assert path.parent == tmp_path
        assert path.exists()

    def test_markdown_file_has_md_extension(self, tmp_path: Path) -> None:
        from app.core.constants import OutputFormat
        path = ReportSynthesizer().save_report(
            _report(), output_dir=tmp_path, output_format=OutputFormat.MARKDOWN
        )
        assert path.suffix == ".md"

    def test_json_file_has_json_extension(self, tmp_path: Path) -> None:
        from app.core.constants import OutputFormat
        path = ReportSynthesizer().save_report(
            _report(), output_dir=tmp_path, output_format=OutputFormat.JSON
        )
        assert path.suffix == ".json"

    def test_filename_contains_slug(self, tmp_path: Path) -> None:
        path = ReportSynthesizer().save_report(_report(), output_dir=tmp_path)
        assert "future_of_rust" in path.stem

    def test_filename_contains_timestamp(self, tmp_path: Path) -> None:
        path = ReportSynthesizer().save_report(_report(), output_dir=tmp_path)
        # Timestamp pattern: YYYYMMDD_HHMMSS (14 digits + underscore)
        assert re.search(r"\d{8}_\d{6}", path.stem), f"No timestamp in {path.stem!r}"

    def test_file_content_is_valid_markdown(self, tmp_path: Path) -> None:
        path = ReportSynthesizer().save_report(_report(), output_dir=tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "# The future of Rust" in content

    def test_returns_absolute_or_resolvable_path(self, tmp_path: Path) -> None:
        path = ReportSynthesizer().save_report(_report(), output_dir=tmp_path)
        assert path.exists()


# ---------------------------------------------------------------------------
# render_to_path
# ---------------------------------------------------------------------------

class TestRenderToPath:
    def test_writes_file_to_explicit_path(self, tmp_path: Path) -> None:
        out = tmp_path / "custom_name.md"
        ReportSynthesizer().render_to_path(_report(), out)
        assert out.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "dir" / "report.md"
        ReportSynthesizer().render_to_path(_report(), out)
        assert out.exists()

    def test_returns_rendered_string(self, tmp_path: Path) -> None:
        out = tmp_path / "report.md"
        content = ReportSynthesizer().render_to_path(_report(), out)
        assert isinstance(content, str)
        assert "# The future of Rust" in content
