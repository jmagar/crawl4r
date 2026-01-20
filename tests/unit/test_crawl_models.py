"""Tests for crawl4r.readers.crawl.models module."""

import pytest
from datetime import datetime, timezone

from crawl4r.readers.crawl.models import CrawlResult


def test_crawl_result_success_creation() -> None:
    """Verify CrawlResult can represent successful crawl.

    Ensures:
    - Required fields: url, markdown, success
    - Optional fields: title, description, status_code
    - timestamp defaults to current time
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
    assert isinstance(result.timestamp, datetime)
    assert result.timestamp.tzinfo == timezone.utc


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
