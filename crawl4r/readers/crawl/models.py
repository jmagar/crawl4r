"""Data models for web crawling operations."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class CrawlResult:
    """Result of crawling a single URL.

    Attributes:
        url: Original URL that was crawled
        markdown: Extracted markdown content
        success: Whether crawl succeeded
        title: Page title (optional)
        description: Page description (optional)
        status_code: HTTP status code
        error: Error message if crawl failed
        timestamp: When crawl occurred (UTC)
    """

    url: str
    markdown: str
    success: bool
    title: str | None = None
    description: str | None = None
    status_code: int = 0
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
