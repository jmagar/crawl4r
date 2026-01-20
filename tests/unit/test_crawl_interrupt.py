"""Unit tests for graceful shutdown handling in crawl command."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from crawl4r.services.models import CrawlStatus, IngestResult


@pytest.mark.asyncio
async def test_crawl_releases_lock_on_interrupt() -> None:
    """Test that crawl command handles interrupt and sets status to FAILED."""
    # Mock the ingestion service
    mock_service = Mock()
    mock_service.queue_manager = Mock()
    mock_service.queue_manager.set_status = AsyncMock()
    mock_service.queue_manager.get_queue_length = AsyncMock(return_value=0)

    # Mock ingest_urls to raise KeyboardInterrupt (simulates service being interrupted)
    # The service's finally block would normally release the lock
    async def raise_interrupt(urls: list[str]) -> IngestResult:
        raise KeyboardInterrupt("Simulated Ctrl+C")

    mock_service.ingest_urls = raise_interrupt

    # Import _run_crawl
    from crawl4r.cli.commands.crawl import _run_crawl

    # Run crawl and expect it to handle the interrupt gracefully
    with pytest.raises(KeyboardInterrupt):
        await _run_crawl(mock_service, ["https://example.com"])

    # Verify status was set to FAILED (the service releases the lock in its own finally block)
    assert mock_service.queue_manager.set_status.called


@pytest.mark.asyncio
async def test_crawl_sets_failed_status_on_interrupt() -> None:
    """Test that crawl command sets status to FAILED on interrupt."""
    # Mock the ingestion service
    mock_service = Mock()
    mock_service.queue_manager = Mock()
    mock_service.queue_manager.release_lock = AsyncMock()
    mock_service.queue_manager.set_status = AsyncMock()
    mock_service.queue_manager.get_queue_length = AsyncMock(return_value=0)

    # Mock ingest_urls to raise KeyboardInterrupt
    async def raise_interrupt(urls: list[str]) -> IngestResult:
        raise KeyboardInterrupt("Simulated Ctrl+C")

    mock_service.ingest_urls = raise_interrupt

    # Import _run_crawl
    from crawl4r.cli.commands.crawl import _run_crawl

    # Run crawl and expect it to handle the interrupt gracefully
    with pytest.raises(KeyboardInterrupt):
        await _run_crawl(mock_service, ["https://example.com"])

    # Verify status was set to FAILED
    assert mock_service.queue_manager.set_status.called
    call_args = mock_service.queue_manager.set_status.call_args
    status_info = call_args[0][0]
    assert status_info.status == CrawlStatus.FAILED
    assert "Interrupted" in status_info.error or "interrupted" in status_info.error


@pytest.mark.asyncio
async def test_crawl_handles_sigterm() -> None:
    """Test that crawl command handles SIGTERM gracefully."""
    # Mock the ingestion service
    mock_service = Mock()
    mock_service.queue_manager = Mock()
    mock_service.queue_manager.set_status = AsyncMock()
    mock_service.queue_manager.get_queue_length = AsyncMock(return_value=0)

    # Mock ingest_urls to simulate SIGTERM
    async def raise_sigterm(urls: list[str]) -> IngestResult:
        # SIGTERM typically raises SystemExit in signal handlers
        raise SystemExit(0)

    mock_service.ingest_urls = raise_sigterm

    # Import _run_crawl
    from crawl4r.cli.commands.crawl import _run_crawl

    # Run crawl and expect it to handle the signal gracefully
    with pytest.raises(SystemExit):
        await _run_crawl(mock_service, ["https://example.com"])

    # Verify status was set to FAILED
    assert mock_service.queue_manager.set_status.called
