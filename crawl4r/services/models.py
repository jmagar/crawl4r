"""Service-layer data models for crawl operations."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CrawlStatus(str, Enum):
    """Status values for crawl lifecycle tracking."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ScrapeResult:
    """Result for a single URL scrape.

    Args:
        url: Source URL for the scrape
        success: Whether the scrape completed successfully
        error: Error message if scrape failed
        markdown: Extracted markdown content
        status_code: HTTP status code from the crawl request
        metadata: Additional metadata extracted from the page
    """

    url: str
    success: bool
    error: str | None = None
    markdown: str | None = None
    status_code: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class MapResult:
    """Result for a crawl map operation.

    Args:
        url: Seed URL for the crawl map
        success: Whether map discovery succeeded
        error: Error message if mapping failed
        links: Discovered URLs from the crawl
        internal_count: Number of internal links
        external_count: Number of external links
        depth_reached: Maximum crawl depth reached
    """

    url: str
    success: bool
    error: str | None = None
    links: list[str] | None = None
    internal_count: int | None = None
    external_count: int | None = None
    depth_reached: int | None = None


@dataclass(frozen=True)
class ExtractResult:
    """Result for structured data extraction.

    Args:
        url: Source URL for the extraction
        success: Whether extraction succeeded
        error: Error message if extraction failed
        data: Extracted structured data payload
    """

    url: str
    success: bool
    error: str | None = None
    data: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScreenshotResult:
    """Result for a screenshot capture.

    Args:
        url: Source URL for the screenshot
        success: Whether screenshot capture succeeded
        error: Error message if capture failed
        file_path: Path to saved screenshot file
        file_size: Size of screenshot file in bytes
    """

    url: str
    success: bool
    error: str | None = None
    file_path: str | None = None
    file_size: int | None = None


@dataclass(frozen=True)
class IngestResult:
    """Result for ingesting one or more URLs.

    Args:
        crawl_id: Identifier for the crawl request
        success: Whether ingestion succeeded
        error: Error message if ingestion failed
        urls_total: Total number of URLs requested
        urls_failed: Number of URLs that failed ingestion
        chunks_created: Total chunks created for embedding
        queued: Whether the request was queued instead of processed
    """

    crawl_id: str
    success: bool
    urls_total: int
    urls_failed: int
    chunks_created: int
    queued: bool
    error: str | None = None


@dataclass(frozen=True)
class CrawlStatusInfo:
    """Status details for a crawl request.

    Args:
        crawl_id: Identifier for the crawl request
        status: Current crawl status
        error: Error message if crawl failed
        started_at: ISO timestamp for start time
        finished_at: ISO timestamp for completion time
    """

    crawl_id: str
    status: CrawlStatus
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
