"""Web crawling components for Crawl4r."""

from crawl4r.readers.crawl.http_client import HttpCrawlClient
from crawl4r.readers.crawl.metadata_builder import MetadataBuilder
from crawl4r.readers.crawl.models import CrawlResult
from crawl4r.readers.crawl.url_validator import UrlValidator, ValidationError

__all__ = [
    "CrawlResult",
    "HttpCrawlClient",
    "MetadataBuilder",
    "UrlValidator",
    "ValidationError",
]
