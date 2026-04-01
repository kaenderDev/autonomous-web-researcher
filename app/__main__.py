"""
app/__main__.py

Entry point for `python -m app`.

Allows the package to be invoked directly from the project root without
installing the console_scripts entry point:

    python -m app --topic "Rust in systems programming"
    python -m app --help

This delegates immediately to the Typer application defined in app.main,
which owns all argument parsing, progress display, and error handling.
"""
from app.main import app

if __name__ == "__main__":
    app()
