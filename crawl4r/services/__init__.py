"""Service layer for crawl and ingestion operations."""

from crawl4r.services.ingestion import IngestionService, generate_crawl_id
from crawl4r.services.mapper import MapperService
from crawl4r.services.models import (
    CrawlStatus,
    CrawlStatusInfo,
    ExtractResult,
    IngestResult,
    MapResult,
    ScrapeResult,
    ScreenshotResult,
)
from crawl4r.services.queue import QueueManager
from crawl4r.services.scraper import ScraperService

__all__ = [
    "CrawlStatus",
    "CrawlStatusInfo",
    "ExtractResult",
    "generate_crawl_id",
    "IngestionService",
    "IngestResult",
    "MapperService",
    "MapResult",
    "QueueManager",
    "ScrapeResult",
    "ScreenshotResult",
    "ScraperService",
]
