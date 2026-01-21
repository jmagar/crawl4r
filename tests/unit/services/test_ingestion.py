from unittest.mock import AsyncMock

import pytest

from crawl4r.services.ingestion import IngestionService, generate_crawl_id


def test_generate_crawl_id_format() -> None:
    crawl_id = generate_crawl_id()
    assert crawl_id.startswith("crawl_")


@pytest.mark.asyncio
async def test_ingest_urls_returns_result() -> None:
    service = IngestionService(
        scraper=AsyncMock(),
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=AsyncMock(),
    )
    service.scraper.scrape_urls = AsyncMock(return_value=[])
    result = await service.ingest_urls(["https://example.com"])
    assert result.crawl_id.startswith("crawl_")
    assert result.success is True
