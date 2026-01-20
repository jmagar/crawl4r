"""Unit tests for QueueManager Redis fallback behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from crawl4r.services.queue import QueueManager


@pytest.mark.asyncio
async def test_queue_fallback_when_redis_unavailable() -> None:
    """QueueManager should detect Redis unavailability gracefully."""
    queue = QueueManager(redis_url="redis://localhost:53379")

    # Mock the client to simulate Redis being down
    queue._client = Mock()
    queue._client.ping = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

    # Should have a method to check availability
    is_available = await queue.is_available()

    assert is_available is False


@pytest.mark.asyncio
async def test_queue_available_when_redis_responds() -> None:
    """QueueManager should detect Redis availability."""
    queue = QueueManager(redis_url="redis://localhost:53379")

    # Mock the client to simulate Redis being available
    queue._client = Mock()
    queue._client.ping = AsyncMock(return_value=True)

    is_available = await queue.is_available()

    assert is_available is True


@pytest.mark.asyncio
async def test_queue_acquire_lock_fails_gracefully_when_redis_down() -> None:
    """QueueManager should fail gracefully when Redis is unavailable for lock."""
    queue = QueueManager(redis_url="redis://localhost:53379")

    # Mock the client to simulate Redis being down
    queue._client = Mock()
    queue._client.set = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

    # Should return False instead of raising
    result = await queue.acquire_lock("test-owner")

    assert result is False


@pytest.mark.asyncio
async def test_queue_enqueue_no_crash_when_redis_down() -> None:
    """QueueManager should not crash when enqueuing with Redis down."""
    queue = QueueManager(redis_url="redis://localhost:53379")

    # Mock the client to simulate Redis being down
    queue._client = Mock()
    queue._client.lpush = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

    # Should not raise - either succeed silently or return status
    try:
        await queue.enqueue_crawl("crawl_123", ["https://example.com"])
        # If it doesn't raise, that's acceptable
    except Exception as exc:
        # Should not be a raw ConnectionError
        assert not isinstance(exc, ConnectionError)


@pytest.mark.asyncio
async def test_queue_set_status_no_crash_when_redis_down() -> None:
    """QueueManager should not crash when setting status with Redis down."""
    from crawl4r.services.models import CrawlStatus, CrawlStatusInfo

    queue = QueueManager(redis_url="redis://localhost:53379")

    # Mock the client to simulate Redis being down
    queue._client = Mock()
    queue._client.set = AsyncMock(side_effect=ConnectionError("Redis unavailable"))
    queue._client.lpush = AsyncMock(side_effect=ConnectionError("Redis unavailable"))
    queue._client.ltrim = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

    info = CrawlStatusInfo(crawl_id="test", status=CrawlStatus.RUNNING)

    # Should not raise - either succeed silently or return status
    try:
        await queue.set_status(info)
        # If it doesn't raise, that's acceptable
    except Exception as exc:
        # Should not be a raw ConnectionError
        assert not isinstance(exc, ConnectionError)


@pytest.mark.asyncio
async def test_queue_get_status_returns_none_when_redis_down() -> None:
    """QueueManager should return None when Redis is unavailable for status lookup."""
    queue = QueueManager(redis_url="redis://localhost:53379")

    # Mock the client to simulate Redis being down
    queue._client = Mock()
    queue._client.get = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

    status = await queue.get_status("crawl_123")

    assert status is None
