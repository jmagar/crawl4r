"""Unit tests for service data models."""

from crawl4r.services.models import CrawlStatus, IngestResult, ScrapeResult


def test_crawl_status_enum_values() -> None:
    assert CrawlStatus.QUEUED.value == "QUEUED"
    assert CrawlStatus.RUNNING.value == "RUNNING"
    assert CrawlStatus.COMPLETED.value == "COMPLETED"
    assert CrawlStatus.FAILED.value == "FAILED"


def test_scrape_result_shape() -> None:
    result = ScrapeResult(
        url="https://example.com",
        success=True,
        markdown="# Title",
        status_code=200,
        error=None,
        metadata={"title": "Example"},
    )
    assert result.url == "https://example.com"
    assert result.markdown.startswith("#")
    assert result.success is True


def test_ingest_result_counts() -> None:
    result = IngestResult(
        crawl_id="crawl_test",
        success=False,
        urls_total=2,
        urls_failed=1,
        chunks_created=3,
        queued=False,
        error="1 failure",
    )
    assert result.urls_total == 2
    assert result.urls_failed == 1
    assert result.chunks_created == 3
    assert result.success is False
