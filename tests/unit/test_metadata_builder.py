"""Tests for crawl4r.readers.crawl.metadata_builder module."""

from crawl4r.readers.crawl.metadata_builder import MetadataBuilder
from crawl4r.readers.crawl.models import CrawlResult


def test_build_creates_complete_metadata() -> None:
    """Verify builder creates metadata with all 9 expected fields."""
    builder = MetadataBuilder()

    result = CrawlResult(
        url="https://example.com/page",
        markdown="# Test Page",
        title="Test Page",
        description="A test page",
        status_code=200,
        success=True,
        internal_links_count=5,
        external_links_count=3,
    )

    metadata = builder.build(result)

    # Verify all 9 required fields
    assert metadata["source"] == "https://example.com/page"
    assert metadata["source_url"] == "https://example.com/page"
    assert metadata["title"] == "Test Page"
    assert metadata["description"] == "A test page"
    assert metadata["status_code"] == 200
    assert "crawl_timestamp" in metadata
    assert metadata["internal_links_count"] == 5
    assert metadata["external_links_count"] == 3
    assert metadata["source_type"] == "web_crawl"


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

    assert metadata["source_url"] == "https://example.com"
    assert metadata["title"] == ""
    assert metadata["description"] == ""
    assert metadata["internal_links_count"] == 0
    assert metadata["external_links_count"] == 0
