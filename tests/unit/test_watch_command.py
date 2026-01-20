"""Unit tests for watch command.

Tests the watch command functionality including:
- Help text display
- Folder option handling
- Command registration
"""

from typer.testing import CliRunner

from crawl4r.cli.app import app


def test_watch_help_shows_options() -> None:
    """Test that watch command help displays required options."""
    runner = CliRunner()
    result = runner.invoke(app, ["watch", "--help"])
    assert result.exit_code == 0
    assert "--folder" in result.output
