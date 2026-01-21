"""Test URL validation in services."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4r.services.ingestion import IngestionService
from crawl4r.services.scraper import ScraperService


@pytest.mark.asyncio
async def test_scraper_rejects_invalid_url():
    """ScraperService should reject invalid URLs with ScrapeResult(success=False)."""
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("not-a-url")
    assert result.success is False
    assert result.error is not None
    assert "Invalid URL" in result.error


@pytest.mark.asyncio
async def test_scraper_rejects_private_ip():
    """ScraperService should reject private IP addresses."""
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("http://192.168.1.1/admin")
    assert result.success is False
    assert result.error is not None
    assert "Invalid URL" in result.error


@pytest.mark.asyncio
async def test_scraper_rejects_localhost():
    """ScraperService should reject localhost URLs."""
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("http://localhost:8080/secret")
    assert result.success is False
    assert result.error is not None
    assert "Invalid URL" in result.error


@pytest.mark.asyncio
async def test_scraper_accepts_valid_url():
    """ScraperService should accept valid URLs."""
    service = ScraperService(endpoint_url="http://localhost:52004")

    # Mock the health check and fetch to avoid actual HTTP calls
    service._check_health = AsyncMock()
    service._fetch_markdown = AsyncMock(
        return_value=MagicMock(success=True, url="https://example.com")
    )

    await service.scrape_url("https://example.com")
    # Should not fail at validation stage
    service._fetch_markdown.assert_called_once()


@pytest.mark.asyncio
async def test_ingestion_validates_urls_upfront():
    """IngestionService should validate all URLs before processing."""
    # Create service with mocked dependencies
    scraper = MagicMock(spec=ScraperService)
    embeddings = MagicMock()
    vector_store = MagicMock()
    queue_manager = MagicMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.set_status = AsyncMock()
    queue_manager.release_lock = AsyncMock()

    service = IngestionService(
        scraper=scraper,
        embeddings=embeddings,
        vector_store=vector_store,
        queue_manager=queue_manager,
    )

    urls = ["not-a-url", "http://192.168.1.1/admin", "https://example.com"]
    result = await service.ingest_urls(urls)

    # Should fail because of invalid URLs
    assert result.success is False
    assert result.error is not None
    assert "Invalid URL" in result.error or "invalid" in result.error.lower()


@pytest.mark.asyncio
async def test_ingestion_allows_all_valid_urls():
    """IngestionService should allow batch when all URLs are valid."""
    # Create service with mocked dependencies
    scraper = MagicMock(spec=ScraperService)
    scraper.scrape_urls = AsyncMock(return_value=[])
    embeddings = MagicMock()
    vector_store = MagicMock()
    queue_manager = MagicMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.set_status = AsyncMock()
    queue_manager.release_lock = AsyncMock()

    service = IngestionService(
        scraper=scraper,
        embeddings=embeddings,
        vector_store=vector_store,
        queue_manager=queue_manager,
    )

    urls = ["https://example.com", "https://example.org"]
    await service.ingest_urls(urls)

    # Should not fail at validation stage
    scraper.scrape_urls.assert_called_once_with(urls, max_concurrent=5)


def test_ingestion_validate_url_helper():
    """IngestionService should have a validate_url helper method."""
    service = IngestionService()

    # Valid URLs
    assert service.validate_url("https://example.com") is True
    assert service.validate_url("http://example.com") is True

    # Invalid URLs
    assert service.validate_url("not-a-url") is False
    assert service.validate_url("ftp://example.com") is False
    assert service.validate_url("http://192.168.1.1") is False
    assert service.validate_url("http://localhost:8080") is False
    assert service.validate_url("") is False
