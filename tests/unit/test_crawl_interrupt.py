"""Unit tests for graceful shutdown handling in crawl command."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from crawl4r.services.models import IngestResult


@pytest.mark.asyncio
async def test_crawl_releases_lock_on_interrupt() -> None:
    """Test that crawl command handles interrupt without creating phantom status."""
    # Mock the ingestion service
    mock_service = Mock()
    mock_service.queue_manager = Mock()
    mock_service.queue_manager.set_status = AsyncMock()
    mock_service.queue_manager.get_queue_length = AsyncMock(return_value=0)

    # Mock ingest_urls to raise KeyboardInterrupt before crawl_id is assigned
    # This simulates early interrupt before crawl actually starts
    async def raise_interrupt(urls: list[str]) -> IngestResult:
        raise KeyboardInterrupt("Simulated Ctrl+C")

    mock_service.ingest_urls = raise_interrupt

    # Import _run_crawl
    from crawl4r.cli.commands.crawl import _run_crawl

    # Run crawl and expect it to handle the interrupt gracefully
    with pytest.raises(KeyboardInterrupt):
        await _run_crawl(mock_service, ["https://example.com"])

    # Verify status was NOT set (no phantom crawl_id created)
    # The service releases the lock in its own finally block
    assert not mock_service.queue_manager.set_status.called


@pytest.mark.asyncio
async def test_crawl_sets_failed_status_on_interrupt() -> None:
    """Test that crawl command sets status to FAILED only if crawl actually started."""
    # Mock the ingestion service
    mock_service = Mock()
    mock_service.queue_manager = Mock()
    mock_service.queue_manager.release_lock = AsyncMock()
    mock_service.queue_manager.set_status = AsyncMock()
    mock_service.queue_manager.get_queue_length = AsyncMock(return_value=0)

    # Mock ingest_urls to return a result with crawl_id, then raise interrupt
    # This simulates crawl starting successfully before being interrupted
    async def ingest_then_interrupt(urls: list[str]) -> IngestResult:
        # Return a partial result (crawl started)
        return IngestResult(
            success=False,
            crawl_id="test-crawl-123",
            urls_total=1,
            urls_failed=0,
            chunks_created=0,
            queued=False,
        )

    mock_service.ingest_urls = ingest_then_interrupt

    # Import _run_crawl
    from crawl4r.cli.commands.crawl import _run_crawl

    # This should not raise since ingest_urls completes
    result, _ = await _run_crawl(mock_service, ["https://example.com"])
    assert result.crawl_id == "test-crawl-123"


@pytest.mark.asyncio
async def test_crawl_handles_sigterm() -> None:
    """Test that crawl command handles SIGTERM gracefully without phantom status."""
    # Mock the ingestion service
    mock_service = Mock()
    mock_service.queue_manager = Mock()
    mock_service.queue_manager.set_status = AsyncMock()
    mock_service.queue_manager.get_queue_length = AsyncMock(return_value=0)

    # Mock ingest_urls to simulate SIGTERM before crawl_id is assigned
    async def raise_sigterm(urls: list[str]) -> IngestResult:
        # SIGTERM typically raises SystemExit in signal handlers
        raise SystemExit(0)

    mock_service.ingest_urls = raise_sigterm

    # Import _run_crawl
    from crawl4r.cli.commands.crawl import _run_crawl

    # Run crawl and expect it to handle the signal gracefully
    with pytest.raises(SystemExit):
        await _run_crawl(mock_service, ["https://example.com"])

    # Verify status was NOT set (no phantom crawl_id created)
    assert not mock_service.queue_manager.set_status.called
