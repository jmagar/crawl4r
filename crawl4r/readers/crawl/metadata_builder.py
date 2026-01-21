"""Builds document metadata from crawl results."""

from crawl4r.readers.crawl.models import CrawlResult


class MetadataBuilder:
    """Builds LlamaIndex document metadata from CrawlResult.

    Produces flat metadata dictionaries with 9 fields required for
    Qdrant compatibility and downstream processing.
    """

    def build(self, result: CrawlResult) -> dict[str, str | int]:
        """Build metadata dictionary from crawl result.

        Args:
            result: CrawlResult from web crawl

        Returns:
            Metadata dictionary with all required fields:
            - source: Original URL
            - source_url: Same as source (indexed for deduplication)
            - title: Page title (empty string if missing)
            - description: Page description (empty string if missing)
            - status_code: HTTP status code
            - crawl_timestamp: ISO8601 timestamp string
            - internal_links_count: Count of internal links
            - external_links_count: Count of external links
            - source_type: Always "web_crawl"
            - detected_language: ISO 639-1 language code (if present)
            - language_confidence: Confidence score 0.0-1.0 (if present)
        """
        metadata = {
            "source": result.url,
            "source_url": result.url,
            "title": result.title or "",
            "description": result.description or "",
            "status_code": result.status_code,
            "crawl_timestamp": result.timestamp,
            "internal_links_count": result.internal_links_count,
            "external_links_count": result.external_links_count,
            "source_type": "web_crawl",
        }

        # Add language metadata if present
        if result.detected_language is not None:
            metadata["detected_language"] = result.detected_language
        if result.language_confidence is not None:
            metadata["language_confidence"] = result.language_confidence

        return metadata
