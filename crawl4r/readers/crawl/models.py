"""Data models for web crawling operations."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _default_timestamp() -> str:
    """Generate default ISO8601 timestamp string."""
    return datetime.now(timezone.utc).isoformat()


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
        timestamp: When crawl occurred (ISO8601 string)
        internal_links_count: Number of internal links found
        external_links_count: Number of external links found
        detected_language: Detected language code (optional, e.g., 'en', 'es')
        language_confidence: Confidence score for language detection (optional, 0.0-1.0)
    """

    url: str
    markdown: str
    success: bool
    title: str | None = None
    description: str | None = None
    status_code: int = 0
    error: str | None = None
    timestamp: str = field(default_factory=_default_timestamp)
    internal_links_count: int = 0
    external_links_count: int = 0
    detected_language: str | None = None
    language_confidence: float | None = None
