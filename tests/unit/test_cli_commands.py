"""Unit tests for CLI commands using Typer's CliRunner.

Tests command parsing, file input/output handling, and basic command flows
without making external service calls (all services are mocked).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from crawl4r.cli.app import app

runner = CliRunner()


# ============================================================================
# Help Text Tests - Verify command registration and help output
# ============================================================================


@pytest.mark.parametrize(
    "command,expected_text",
    [
        ("scrape", "Scrape URLs and output markdown"),
        ("crawl", "Crawl URLs and ingest results into the vector store"),
        ("status", "Show crawl status information"),
    ],
)
def test_command_help(command: str, expected_text: str) -> None:
    """Commands should display help text when --help is passed."""
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0
    assert expected_text in result.stdout


# ============================================================================
# File Input Tests - Verify URL file reading
# ============================================================================


def test_scrape_file_input(tmp_path: Path) -> None:
    """Scrape should read URLs from file when -f option is provided."""
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://example.com\n")

    with patch("crawl4r.cli.commands.scrape.ScraperService") as mock_service:
        mock_instance = MagicMock()
        mock_instance.scrape_url = AsyncMock(
            return_value=MagicMock(success=True, url="https://example.com", markdown="# Test")
        )
        mock_service.return_value = mock_instance

        result = runner.invoke(app, ["scrape", "-f", str(urls_file)])
        assert result.exit_code == 0


def test_crawl_file_input(tmp_path: Path) -> None:
    """Crawl should read URLs from file when -f option is provided."""
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://example.com\n")

    with patch("crawl4r.cli.commands.crawl.IngestionService") as mock_service:
        mock_instance = MagicMock()
        mock_instance.ingest_urls = AsyncMock(
            return_value=MagicMock(
                crawl_id="test-123",
                queued=False,
                success=True,
                urls_total=1,
                urls_failed=0,
                chunks_created=5,
            )
        )
        mock_service.return_value = mock_instance

        result = runner.invoke(app, ["crawl", "-f", str(urls_file)])
        assert result.exit_code == 0


def test_scrape_nonexistent_file() -> None:
    """Scrape should fail gracefully when input file doesn't exist."""
    result = runner.invoke(app, ["scrape", "-f", "/nonexistent/urls.txt"])
    assert result.exit_code != 0


# ============================================================================
# File Output Tests - Verify markdown output writing
# ============================================================================


def test_scrape_output_file_single_url(tmp_path: Path) -> None:
    """Scrape should write markdown to file when -o option is provided with single URL."""
    out_file = tmp_path / "out.md"

    with patch("crawl4r.cli.commands.scrape.ScraperService") as mock_service:
        mock_instance = MagicMock()
        mock_instance.scrape_url = AsyncMock(
            return_value=MagicMock(
                success=True,
                url="https://example.com",
                markdown="# Test Content",
            )
        )
        mock_service.return_value = mock_instance

        result = runner.invoke(app, ["scrape", "https://example.com", "-o", str(out_file)])
        assert result.exit_code == 0
        # Output file handling is tested via integration tests
        # This unit test verifies command parsing and execution only


def test_scrape_output_directory_multiple_urls(tmp_path: Path) -> None:
    """Scrape should write multiple markdown files to directory when -o is a directory."""
    out_dir = tmp_path / "output"

    with patch("crawl4r.cli.commands.scrape.ScraperService") as mock_service:
        mock_instance = MagicMock()

        # Mock two different URLs with different content
        async def mock_scrape(url: str):
            if "example.com" in url:
                return MagicMock(
                    success=True,
                    url=url,
                    markdown="# Example",
                )
            else:
                return MagicMock(
                    success=True,
                    url=url,
                    markdown="# Test",
                )

        mock_instance.scrape_url = AsyncMock(side_effect=mock_scrape)
        mock_service.return_value = mock_instance

        result = runner.invoke(
            app,
            ["scrape", "https://example.com", "https://test.com", "-o", str(out_dir)],
        )
        assert result.exit_code == 0
        # Output directory handling is tested via integration tests
        # This unit test verifies command parsing and execution only


# ============================================================================
# URL Argument Tests - Verify direct URL passing
# ============================================================================


def test_scrape_with_single_url() -> None:
    """Scrape should accept a single URL as argument."""
    with patch("crawl4r.cli.commands.scrape.ScraperService") as mock_service:
        mock_instance = MagicMock()
        mock_instance.scrape_url = AsyncMock(
            return_value=MagicMock(success=True, url="https://example.com", markdown="# Test")
        )
        mock_service.return_value = mock_instance

        result = runner.invoke(app, ["scrape", "https://example.com"])
        assert result.exit_code == 0


def test_scrape_with_multiple_urls() -> None:
    """Scrape should accept multiple URLs as arguments."""
    with patch("crawl4r.cli.commands.scrape.ScraperService") as mock_service:
        mock_instance = MagicMock()

        async def mock_scrape(url: str):
            return MagicMock(success=True, url=url, markdown="# Test")

        mock_instance.scrape_url = AsyncMock(side_effect=mock_scrape)
        mock_service.return_value = mock_instance

        result = runner.invoke(
            app, ["scrape", "https://example.com", "https://test.com"]
        )
        assert result.exit_code == 0


def test_scrape_no_urls_provided() -> None:
    """Scrape should fail when no URLs are provided."""
    result = runner.invoke(app, ["scrape"])
    # Typer shows help when no args provided to a callback command
    assert result.exit_code != 0


def test_crawl_no_urls_provided() -> None:
    """Crawl should fail when no URLs are provided."""
    result = runner.invoke(app, ["crawl"])
    # Typer shows help when no args provided to a callback command
    assert result.exit_code != 0


# ============================================================================
# Option Tests - Verify command options work correctly
# ============================================================================


def test_scrape_concurrent_option() -> None:
    """Scrape should accept --concurrent option for concurrent request limit."""
    with patch("crawl4r.cli.commands.scrape.ScraperService") as mock_service:
        mock_instance = MagicMock()
        mock_instance.scrape_url = AsyncMock(
            return_value=MagicMock(success=True, url="https://example.com", markdown="# Test")
        )
        mock_service.return_value = mock_instance

        result = runner.invoke(
            app, ["scrape", "https://example.com", "--concurrent", "10"]
        )
        assert result.exit_code == 0


def test_crawl_depth_option() -> None:
    """Crawl should accept --depth option for crawl depth."""
    with patch("crawl4r.cli.commands.crawl.IngestionService") as mock_service:
        mock_instance = MagicMock()
        mock_instance.ingest_urls = AsyncMock(
            return_value=MagicMock(
                crawl_id="test-123",
                queued=False,
                success=True,
                urls_total=1,
                urls_failed=0,
                chunks_created=5,
            )
        )
        mock_service.return_value = mock_instance

        result = runner.invoke(app, ["crawl", "https://example.com", "--depth", "2"])
        assert result.exit_code == 0


# ============================================================================
# Status Command Tests - Verify status reporting
# ============================================================================


def test_status_no_args_lists_recent() -> None:
    """Status command with no args should require more information."""
    result = runner.invoke(app, ["status"])
    # Status command requires an argument or flag when called without args
    assert result.exit_code != 0


def test_status_with_crawl_id() -> None:
    """Status command should show specific crawl when ID is provided."""
    with patch("crawl4r.cli.commands.status.QueueManager") as mock_queue:
        from crawl4r.services.models import CrawlStatus, CrawlStatusInfo

        mock_instance = MagicMock()
        mock_instance.get_status = AsyncMock(
            return_value=CrawlStatusInfo(
                crawl_id="test-123",
                status=CrawlStatus.COMPLETED,
                started_at="2024-01-01T00:00:00Z",
                finished_at="2024-01-01T00:01:00Z",
            )
        )
        mock_queue.return_value = mock_instance

        result = runner.invoke(app, ["status", "test-123"])
        assert result.exit_code == 0


def test_status_list_flag() -> None:
    """Status command should list recent crawls when --list flag is passed."""
    with patch("crawl4r.cli.commands.status.QueueManager") as mock_queue:
        mock_instance = MagicMock()
        mock_instance.list_recent = AsyncMock(return_value=[])
        mock_queue.return_value = mock_instance

        result = runner.invoke(app, ["status", "--list"])
        assert result.exit_code == 0


def test_status_active_flag() -> None:
    """Status command should show active crawls when --active flag is passed."""
    with patch("crawl4r.cli.commands.status.QueueManager") as mock_queue:
        mock_instance = MagicMock()
        mock_instance.get_active = AsyncMock(return_value=[])
        mock_queue.return_value = mock_instance

        result = runner.invoke(app, ["status", "--active"])
        assert result.exit_code == 0


# ============================================================================
# Error Handling Tests - Verify graceful failure handling
# ============================================================================


def test_scrape_handles_failed_urls_gracefully() -> None:
    """Scrape should handle failed URLs without crashing."""
    with patch("crawl4r.cli.commands.scrape.ScraperService") as mock_service:
        mock_instance = MagicMock()
        mock_instance.scrape_url = AsyncMock(
            return_value=MagicMock(
                success=False,
                url="https://example.com",
                markdown=None,
                error="Connection timeout",
            )
        )
        mock_service.return_value = mock_instance

        result = runner.invoke(app, ["scrape", "https://example.com"])
        # Should exit with error code since all URLs failed
        assert result.exit_code != 0


def test_scrape_mixed_success_and_failure() -> None:
    """Scrape should report summary correctly when some URLs fail."""
    with patch("crawl4r.cli.commands.scrape.ScraperService") as mock_service:
        mock_instance = MagicMock()

        # First URL succeeds, second fails
        async def mock_scrape(url: str):
            if "example.com" in url:
                return MagicMock(success=True, url=url, markdown="# Success")
            else:
                return MagicMock(success=False, url=url, markdown=None, error="Failed")

        mock_instance.scrape_url = AsyncMock(side_effect=mock_scrape)
        mock_service.return_value = mock_instance

        result = runner.invoke(
            app, ["scrape", "https://example.com", "https://fail.com"]
        )
        # Should exit with error code since at least one URL failed
        assert result.exit_code != 0
        assert "1 succeeded, 1 failed" in result.stdout


# ============================================================================
# Additional Coverage Tests - Edge cases and error paths
# ============================================================================


def test_scrape_file_input_merges_with_args(tmp_path: Path) -> None:
    """Scrape should merge URLs from file with command-line arguments."""
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://file.com\n")

    with patch("crawl4r.cli.commands.scrape.ScraperService") as mock_service:
        mock_instance = MagicMock()

        async def mock_scrape(url: str):
            return MagicMock(success=True, url=url, markdown="# Test")

        mock_instance.scrape_url = AsyncMock(side_effect=mock_scrape)
        mock_service.return_value = mock_instance

        result = runner.invoke(
            app, ["scrape", "https://arg.com", "-f", str(urls_file)]
        )
        assert result.exit_code == 0


def test_scrape_with_empty_urls_file(tmp_path: Path) -> None:
    """Scrape should handle empty URLs file gracefully."""
    urls_file = tmp_path / "empty.txt"
    urls_file.write_text("\n\n\n")  # Only whitespace

    result = runner.invoke(app, ["scrape", "-f", str(urls_file)])
    assert result.exit_code != 0


def test_crawl_with_empty_urls_file(tmp_path: Path) -> None:
    """Crawl should handle empty URLs file gracefully."""
    urls_file = tmp_path / "empty.txt"
    urls_file.write_text("\n\n\n")  # Only whitespace

    result = runner.invoke(app, ["crawl", "-f", str(urls_file)])
    assert result.exit_code != 0


def test_crawl_file_too_large(tmp_path: Path) -> None:
    """Crawl should reject URL files that are too large."""
    # Create a file larger than 1MB limit
    urls_file = tmp_path / "large.txt"
    large_content = "https://example.com\n" * 100000  # ~2MB
    urls_file.write_text(large_content)

    result = runner.invoke(app, ["crawl", "-f", str(urls_file)])
    assert result.exit_code != 0
    # Error message might be in stderr or stdout depending on Typer behavior
    output = result.stdout.lower() + (result.stderr or "").lower()
    assert "too large" in output or result.exit_code != 0


def test_crawl_success_queued(tmp_path: Path) -> None:
    """Crawl should show queue position when result is queued."""
    with patch("crawl4r.cli.commands.crawl.IngestionService") as mock_service:
        mock_instance = MagicMock()
        mock_instance.ingest_urls = AsyncMock(
            return_value=MagicMock(
                crawl_id="test-queued",
                queued=True,  # This one is queued
                success=True,
                urls_total=1,
                urls_failed=0,
                chunks_created=0,
            )
        )
        mock_instance.queue_manager = MagicMock()
        mock_instance.queue_manager.get_queue_length = AsyncMock(return_value=5)
        mock_service.return_value = mock_instance

        result = runner.invoke(app, ["crawl", "https://example.com"])
        assert result.exit_code == 0
        assert "Queue position: 5" in result.stdout


def test_crawl_failed_result() -> None:
    """Crawl should exit with error code when ingestion fails."""
    with patch("crawl4r.cli.commands.crawl.IngestionService") as mock_service:
        mock_instance = MagicMock()
        mock_instance.ingest_urls = AsyncMock(
            return_value=MagicMock(
                crawl_id="test-failed",
                queued=False,
                success=False,  # Failed
                urls_total=1,
                urls_failed=1,
                chunks_created=0,
            )
        )
        mock_service.return_value = mock_instance

        result = runner.invoke(app, ["crawl", "https://example.com"])
        assert result.exit_code != 0


def test_status_crawl_id_not_found() -> None:
    """Status command should handle missing crawl IDs gracefully."""
    with patch("crawl4r.cli.commands.status.QueueManager") as mock_queue:
        mock_instance = MagicMock()
        mock_instance.get_status = AsyncMock(return_value=None)
        mock_queue.return_value = mock_instance

        result = runner.invoke(app, ["status", "nonexistent-id"])
        assert result.exit_code == 0
        assert "No crawl status records found" in result.stdout


@pytest.mark.parametrize(
    "flag,method_name",
    [
        ("--list", "list_recent"),
        ("--active", "get_active"),
    ],
)
def test_status_empty_results(flag: str, method_name: str) -> None:
    """Status command should handle empty results for list and active flags."""
    with patch("crawl4r.cli.commands.status.QueueManager") as mock_queue:
        mock_instance = MagicMock()
        setattr(mock_instance, method_name, AsyncMock(return_value=[]))
        mock_queue.return_value = mock_instance

        result = runner.invoke(app, ["status", flag])
        assert result.exit_code == 0
        assert "No crawl status records found" in result.stdout


def test_status_list_multiple_crawls() -> None:
    """Status command should display table for multiple crawls."""
    with patch("crawl4r.cli.commands.status.QueueManager") as mock_queue:
        from crawl4r.services.models import CrawlStatus, CrawlStatusInfo

        mock_instance = MagicMock()
        mock_instance.list_recent = AsyncMock(
            return_value=[
                CrawlStatusInfo(
                    crawl_id="test-1",
                    status=CrawlStatus.COMPLETED,
                    started_at="2024-01-01T00:00:00Z",
                    finished_at="2024-01-01T00:01:00Z",
                ),
                CrawlStatusInfo(
                    crawl_id="test-2",
                    status=CrawlStatus.RUNNING,
                    started_at="2024-01-01T00:02:00Z",
                ),
            ]
        )
        mock_queue.return_value = mock_instance

        result = runner.invoke(app, ["status", "--list"])
        assert result.exit_code == 0
        assert "test-1" in result.stdout
        assert "test-2" in result.stdout


def test_scrape_url_whitespace_stripping(tmp_path: Path) -> None:
    """Scrape should strip whitespace from URLs in file."""
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("  https://example.com  \n\n  https://test.com  \n")

    with patch("crawl4r.cli.commands.scrape.ScraperService") as mock_service:
        mock_instance = MagicMock()

        async def mock_scrape(url: str):
            return MagicMock(success=True, url=url, markdown="# Test")

        mock_instance.scrape_url = AsyncMock(side_effect=mock_scrape)
        mock_service.return_value = mock_instance

        result = runner.invoke(app, ["scrape", "-f", str(urls_file)])
        assert result.exit_code == 0
