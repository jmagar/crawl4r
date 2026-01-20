"""Tests for crawl4r.readers.crawl.metadata_builder module."""

import pytest
from datetime import datetime

from crawl4r.readers.crawl.metadata_builder import MetadataBuilder
from crawl4r.readers.crawl.models import CrawlResult
from crawl4r.core.metadata import MetadataKeys


def test_build_creates_complete_metadata() -> None:
    """Verify builder creates metadata with all expected fields."""
    builder = MetadataBuilder()

    result = CrawlResult(
        url="https://example.com/page",
        markdown="# Test Page",
        title="Test Page",
        description="A test page",
        status_code=200,
        success=True,
    )

    metadata = builder.build(result)

    assert metadata[MetadataKeys.SOURCE_URL] == "https://example.com/page"
    assert metadata[MetadataKeys.SOURCE_TYPE] == "web_crawl"
    assert metadata[MetadataKeys.TITLE] == "Test Page"
    assert metadata[MetadataKeys.DESCRIPTION] == "A test page"
    assert metadata[MetadataKeys.STATUS_CODE] == 200
    assert MetadataKeys.CRAWL_TIMESTAMP in metadata


def test_build_handles_missing_optional_fields() -> None:
    """Verify builder handles missing title/description gracefully."""
    builder = MetadataBuilder()

    result = CrawlResult(
        url="https://example.com",
        markdown="# Content",
        success=True,
        status_code=200,
    )

    metadata = builder.build(result)

    assert metadata[MetadataKeys.SOURCE_URL] == "https://example.com"
    assert metadata[MetadataKeys.TITLE] == ""
    assert metadata[MetadataKeys.DESCRIPTION] == ""
