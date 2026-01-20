"""Tests for crawl4r.readers.crawl.models module."""

from datetime import datetime

from crawl4r.readers.crawl.models import CrawlResult


def test_crawl_result_success_creation() -> None:
    """Verify CrawlResult can represent successful crawl.

    Ensures:
    - Required fields: url, markdown, success
    - Optional fields: title, description, status_code
    - timestamp defaults to current ISO8601 time string
    """
    result = CrawlResult(
        url="https://example.com",
        markdown="# Example\n\nContent here",
        title="Example Domain",
        description="Example description",
        status_code=200,
        success=True,
    )

    assert result.url == "https://example.com"
    assert result.markdown == "# Example\n\nContent here"
    assert result.title == "Example Domain"
    assert result.description == "Example description"
    assert result.status_code == 200
    assert result.success is True
    # Timestamp is now an ISO8601 string (for API compatibility)
    assert isinstance(result.timestamp, str)
    # Verify it's parseable as a datetime
    parsed_ts = datetime.fromisoformat(result.timestamp)
    assert parsed_ts is not None


def test_crawl_result_failure_creation() -> None:
    """Verify CrawlResult can represent failed crawl.

    Ensures:
    - error field captures failure reason
    - success=False
    - markdown can be empty on failure
    """
    result = CrawlResult(
        url="https://example.com",
        markdown="",
        success=False,
        error="Connection timeout",
        status_code=0,
    )

    assert result.success is False
    assert result.error == "Connection timeout"
    assert result.markdown == ""
