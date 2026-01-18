"""Tests verifying CLI delegates file loading to DocumentProcessor.

These tests use SOURCE CODE INSPECTION to ensure proper TDD RED verification.
"""
from pathlib import Path


def test_cli_does_not_manually_read_files():
    """Verify CLI source code doesn't manually read file contents."""
    cli_source = Path("crawl4r/cli/main.py").read_text()

    # Check for direct file reading patterns
    forbidden_patterns = [
        ".read_text()",
        ".read_bytes()",
    ]

    for pattern in forbidden_patterns:
        assert pattern not in cli_source, (
            f"CLI should NOT use '{pattern}' - delegate file loading to DocumentProcessor."
        )


def test_cli_does_not_use_load_markdown_file():
    """Verify CLI source code doesn't call _load_markdown_file directly."""
    cli_source = Path("crawl4r/cli/main.py").read_text()
    assert "_load_markdown_file" not in cli_source, (
        "CLI should delegate file loading to DocumentProcessor, not call _load_markdown_file."
    )


def test_cli_uses_document_processor():
    """Verify CLI imports and uses DocumentProcessor for file processing."""
    cli_source = Path("crawl4r/cli/main.py").read_text()
    assert "DocumentProcessor" in cli_source, (
        "CLI should use DocumentProcessor for file processing."
    )
