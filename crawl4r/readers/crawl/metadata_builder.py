"""Builds document metadata from crawl results."""

from crawl4r.core.metadata import MetadataKeys
from crawl4r.readers.crawl.models import CrawlResult


class MetadataBuilder:
    """Builds LlamaIndex document metadata from CrawlResult."""

    def build(self, result: CrawlResult) -> dict[str, str | int]:
        """Build metadata dictionary from crawl result.

        Args:
            result: CrawlResult from web crawl

        Returns:
            Metadata dictionary with source_url, title, description, etc.
        """
        return {
            MetadataKeys.SOURCE_URL: result.url,
            MetadataKeys.SOURCE_TYPE: "web_crawl",
            MetadataKeys.TITLE: result.title or "",
            MetadataKeys.DESCRIPTION: result.description or "",
            MetadataKeys.STATUS_CODE: result.status_code,
            MetadataKeys.CRAWL_TIMESTAMP: result.timestamp.isoformat(),
        }
