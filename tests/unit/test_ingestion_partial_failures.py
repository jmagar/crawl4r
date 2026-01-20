"""Unit tests for IngestionService partial failure handling."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, Mock

from crawl4r.services.ingestion import IngestionService
from crawl4r.services.models import ScrapeResult


@pytest.mark.asyncio
async def test_ingestion_continues_on_single_failure() -> None:
    """IngestionService should continue processing when one URL fails."""
    service = IngestionService()

    # Mock scraper to return mixed results
    service.scraper = Mock()
    service.scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(
                url="https://ok.com", success=True, markdown="# Good content"
            ),
            ScrapeResult(url="https://bad.com", success=False, error="Network error"),
        ]
    )

    # Mock other dependencies to prevent actual work
    service.queue_manager = Mock()
    service.queue_manager.acquire_lock = AsyncMock(return_value=True)
    service.queue_manager.release_lock = AsyncMock()
    service.queue_manager.set_status = AsyncMock()

    service.embeddings = Mock()
    service.embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 1024])

    service.vector_store = Mock()
    service.vector_store.delete_by_url = AsyncMock()
    service.vector_store.upsert_vectors_batch = AsyncMock()

    result = await service.ingest_urls(["https://ok.com", "https://bad.com"])

    # Should report partial failure
    assert result.success is False
    assert result.urls_total == 2
    assert result.urls_failed == 1
    assert result.error is not None
    assert "failed" in result.error.lower()


@pytest.mark.asyncio
async def test_ingestion_success_when_all_urls_succeed() -> None:
    """IngestionService should report success when all URLs succeed."""
    service = IngestionService()

    # Mock scraper to return all successful results
    service.scraper = Mock()
    service.scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(url="https://ok1.com", success=True, markdown="# Content 1"),
            ScrapeResult(url="https://ok2.com", success=True, markdown="# Content 2"),
        ]
    )

    # Mock other dependencies
    service.queue_manager = Mock()
    service.queue_manager.acquire_lock = AsyncMock(return_value=True)
    service.queue_manager.release_lock = AsyncMock()
    service.queue_manager.set_status = AsyncMock()

    service.embeddings = Mock()
    service.embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 1024])

    service.vector_store = Mock()
    service.vector_store.delete_by_url = AsyncMock()
    service.vector_store.upsert_vectors_batch = AsyncMock()

    result = await service.ingest_urls(["https://ok1.com", "https://ok2.com"])

    # Should report full success
    assert result.success is True
    assert result.urls_total == 2
    assert result.urls_failed == 0
    assert result.error is None


@pytest.mark.asyncio
async def test_ingestion_processes_successful_urls_when_some_fail() -> None:
    """IngestionService should process successful URLs even when others fail."""
    service = IngestionService()

    # Mock scraper to return mixed results
    service.scraper = Mock()
    service.scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(url="https://ok.com", success=True, markdown="# Good"),
            ScrapeResult(url="https://bad.com", success=False, error="Failed"),
        ]
    )

    # Mock other dependencies and track calls
    service.queue_manager = Mock()
    service.queue_manager.acquire_lock = AsyncMock(return_value=True)
    service.queue_manager.release_lock = AsyncMock()
    service.queue_manager.set_status = AsyncMock()

    service.embeddings = Mock()
    service.embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 1024])

    service.vector_store = Mock()
    service.vector_store.delete_by_url = AsyncMock()
    service.vector_store.upsert_vectors_batch = AsyncMock()

    await service.ingest_urls(["https://ok.com", "https://bad.com"])

    # Should have processed the successful URL
    assert service.embeddings.embed_batch.called
    assert service.vector_store.upsert_vectors_batch.called


@pytest.mark.asyncio
async def test_ingestion_reports_all_failures() -> None:
    """IngestionService should correctly count all failures."""
    service = IngestionService()

    # Mock scraper to return all failures
    service.scraper = Mock()
    service.scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(url="https://bad1.com", success=False, error="Error 1"),
            ScrapeResult(url="https://bad2.com", success=False, error="Error 2"),
            ScrapeResult(url="https://bad3.com", success=False, error="Error 3"),
        ]
    )

    # Mock other dependencies
    service.queue_manager = Mock()
    service.queue_manager.acquire_lock = AsyncMock(return_value=True)
    service.queue_manager.release_lock = AsyncMock()
    service.queue_manager.set_status = AsyncMock()

    result = await service.ingest_urls(
        ["https://bad1.com", "https://bad2.com", "https://bad3.com"]
    )

    # Should report all failures
    assert result.success is False
    assert result.urls_total == 3
    assert result.urls_failed == 3
    assert result.chunks_created == 0


@pytest.mark.asyncio
async def test_ingestion_handles_empty_markdown_as_failure() -> None:
    """IngestionService should treat empty markdown as a failure."""
    service = IngestionService()

    # Mock scraper to return success with empty markdown
    service.scraper = Mock()
    service.scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(url="https://empty.com", success=True, markdown=None),
        ]
    )

    # Mock other dependencies
    service.queue_manager = Mock()
    service.queue_manager.acquire_lock = AsyncMock(return_value=True)
    service.queue_manager.release_lock = AsyncMock()
    service.queue_manager.set_status = AsyncMock()

    result = await service.ingest_urls(["https://empty.com"])

    # Should treat as failure
    assert result.success is False
    assert result.urls_failed == 1
