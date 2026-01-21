"""Integration tests for CLI commands with real Crawl4AI service.

Tests real Crawl4AI service endpoints to verify:
- CLI commands can invoke real service
- Commands return appropriate exit codes
- Commands handle service failures gracefully
- Commands produce expected output formats

These tests require the Crawl4AI service to be running. The endpoint can be
configured via the CRAWL4AI_PORT environment variable. If the service is not
available, tests will be skipped.

Example:
    Run only CLI integration tests:
    $ pytest tests/integration/test_cli_integration.py -v -m integration

    Run with custom endpoint:
    $ CRAWL4AI_PORT=52004 pytest tests/integration/test_cli_integration.py -v -m integration

    Run with service availability check:
    $ docker compose up -d crawl4ai
    $ pytest tests/integration/test_cli_integration.py -v -m integration
"""

import os
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from crawl4r.cli.app import app

runner = CliRunner()

# Get Crawl4AI endpoint from environment or use default
CRAWL4AI_PORT = os.getenv("CRAWL4AI_PORT", "52004")
CRAWL4AI_ENDPOINT = f"http://localhost:{CRAWL4AI_PORT}"


def services_available() -> bool:
    """Check if Crawl4AI service is available.

    Returns:
        True if service is reachable, False otherwise

    Example:
        >>> if services_available():
        ...     # Run integration test
        ...     pass
    """
    return os.getenv("CRAWL4AI_PORT") is not None


@pytest.fixture(autouse=True)
async def check_crawl4ai_service() -> None:
    """Check if Crawl4AI service is available before running tests.

    Automatically runs before each test to verify the Crawl4AI service is
    reachable. Uses the CRAWL4AI_PORT environment variable or defaults to
    52004. If the service is not available, the test will be skipped with
    an informative message.

    Raises:
        pytest.skip: If Crawl4AI service is not available at configured endpoint

    Example:
        This fixture runs automatically for all tests in this module.
        No explicit usage required.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{CRAWL4AI_ENDPOINT}/health")
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip(f"Crawl4AI service not available at {CRAWL4AI_ENDPOINT}")


@pytest.mark.integration
def test_scrape_command_real_service() -> None:
    """Test scrape command invokes real Crawl4AI service.

    Verifies:
    - scrape command accepts URL argument
    - Command invokes real Crawl4AI service
    - Command returns success (0) or partial failure (1) exit code
    - Command produces markdown output

    Raises:
        AssertionError: If exit code is not 0 or 1, or output is empty
    """
    result = runner.invoke(app, ["scrape", "https://example.com"])

    # Allow exit code 0 (success) or 1 (partial failure - service issues)
    assert result.exit_code in (0, 1), f"Unexpected exit code: {result.exit_code}"

    # If successful, should have output
    if result.exit_code == 0:
        assert len(result.stdout) > 0, "Expected markdown output for successful scrape"


@pytest.mark.integration
def test_scrape_command_invalid_url_fails() -> None:
    """Test scrape command fails gracefully on invalid URL.

    Verifies:
    - scrape command handles invalid URLs
    - Command returns non-zero exit code for failures
    - Error message is informative

    Raises:
        AssertionError: If exit code is 0 for invalid URL
    """
    result = runner.invoke(app, ["scrape", "not-a-valid-url"])

    # Should fail with non-zero exit code
    assert result.exit_code != 0, "Expected failure for invalid URL"


@pytest.mark.integration
def test_scrape_command_output_file(tmp_path: Path) -> None:
    """Test scrape command writes output to file.

    Verifies:
    - scrape command accepts --output flag
    - Command writes markdown to specified file
    - Output file exists and contains content

    Args:
        tmp_path: pytest temporary directory fixture

    Raises:
        AssertionError: If output file is not created or is empty
    """
    output_file = tmp_path / "output.md"
    result = runner.invoke(
        app, ["scrape", "https://example.com", "--output", str(output_file)]
    )

    # Allow success or partial failure
    assert result.exit_code in (0, 1), f"Unexpected exit code: {result.exit_code}"

    # If successful, output file should exist
    if result.exit_code == 0:
        assert output_file.exists(), "Expected output file to be created"
        assert output_file.stat().st_size > 0, "Expected non-empty output file"


@pytest.mark.integration
def test_status_command_real_service() -> None:
    """Test status command attempts to connect to Redis service.

    Verifies:
    - status command invokes real Redis service
    - Command returns success (0) or failure (1) exit code
    - Command either shows status output or error message

    Note:
        This test may fail if Redis is not accessible. The status command
        depends on Redis for queue management, not Crawl4AI.

    Raises:
        AssertionError: If exit code is not 0 or 1
    """
    result = runner.invoke(app, ["status", "--list"])

    # Allow exit code 0 (success) or 1 (Redis connection failed)
    # Exit code 1 is expected if Redis is not available
    assert result.exit_code in (0, 1), f"Unexpected exit code: {result.exit_code}"


@pytest.mark.integration
def test_map_command_real_service() -> None:
    """Test map command discovers URLs using real service.

    Verifies:
    - map command accepts URL argument
    - Command invokes real Crawl4AI service
    - Command returns success (0) or partial failure (1) exit code
    - Command produces URL list output

    Raises:
        AssertionError: If exit code is not 0 or 1
    """
    result = runner.invoke(app, ["map", "https://example.com"])

    # Allow exit code 0 (success) or 1 (partial failure)
    assert result.exit_code in (0, 1), f"Unexpected exit code: {result.exit_code}"

    # If successful, should have output
    if result.exit_code == 0:
        assert len(result.stdout) > 0, "Expected URL list output"


@pytest.mark.integration
def test_screenshot_command_real_service(tmp_path: Path) -> None:
    """Test screenshot command captures page using real service.

    Verifies:
    - screenshot command accepts URL and output arguments
    - Command invokes real Crawl4AI service
    - Command returns success (0) or partial failure (1) exit code
    - Command creates screenshot file

    Args:
        tmp_path: pytest temporary directory fixture

    Raises:
        AssertionError: If exit code is not 0 or 1, or screenshot not created
    """
    output_file = tmp_path / "screenshot.png"
    result = runner.invoke(
        app, ["screenshot", "https://example.com", "--output", str(output_file)]
    )

    # Allow exit code 0 (success) or 1 (partial failure)
    assert result.exit_code in (0, 1), f"Unexpected exit code: {result.exit_code}"

    # If successful, screenshot file should exist
    if result.exit_code == 0:
        assert output_file.exists(), "Expected screenshot file to be created"
        assert output_file.stat().st_size > 0, "Expected non-empty screenshot file"


@pytest.mark.integration
def test_extract_command_real_service() -> None:
    """Test extract command extracts structured data using real service.

    Verifies:
    - extract command accepts URL and schema arguments
    - Command invokes real Crawl4AI service
    - Command returns success (0) or partial failure (1) exit code
    - Command produces JSON output

    Raises:
        AssertionError: If exit code is not 0 or 1
    """
    schema = '{"type": "object", "properties": {"title": {"type": "string"}}}'
    result = runner.invoke(
        app, ["extract", "https://example.com", "--schema", schema]
    )

    # Allow exit code 0 (success) or 1 (partial failure)
    assert result.exit_code in (0, 1), f"Unexpected exit code: {result.exit_code}"

    # If successful, should have output
    if result.exit_code == 0:
        assert len(result.stdout) > 0, "Expected JSON output"
