"""Tests for stale lock recovery in QueueManager."""

from unittest.mock import AsyncMock, Mock

import pytest

from crawl4r.services.models import CrawlStatus, CrawlStatusInfo
from crawl4r.services.queue import QueueManager


@pytest.mark.asyncio
async def test_acquire_lock_recovers_failed_holder() -> None:
    """Test that acquire_lock recovers locks held by FAILED crawls."""
    queue = QueueManager(redis_url="redis://localhost:53379")
    queue._client = Mock()  # type: ignore[attr-defined]

    # First set() call returns False (lock is held)
    # Second set() call returns True (lock acquired after recovery)
    queue._client.set = AsyncMock(side_effect=[False, True])

    # get() returns the current lock holder
    queue._client.get = AsyncMock(return_value=b"crawl_1")

    # get_status() returns FAILED status for crawl_1
    queue.get_status = AsyncMock(  # type: ignore[method-assign]
        return_value=CrawlStatusInfo(
            crawl_id="crawl_1",
            status=CrawlStatus.FAILED,
            error="boom",
            started_at="2024-01-15T10:00:00Z",
            finished_at="2024-01-15T10:05:00Z",
        )
    )

    # Mock delete to track if stale lock was removed
    queue._client.delete = AsyncMock()

    # Attempt to acquire lock as crawl_2
    acquired = await queue.acquire_lock("crawl_2")

    # Should successfully acquire after recovering from failed holder
    assert acquired is True

    # Should have deleted the stale lock
    queue._client.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_acquire_lock_respects_running_holder() -> None:
    """Test that acquire_lock does NOT recover locks held by RUNNING crawls."""
    queue = QueueManager(redis_url="redis://localhost:53379")
    queue._client = Mock()  # type: ignore[attr-defined]

    # set() returns False (lock is held)
    queue._client.set = AsyncMock(return_value=False)

    # get() returns the current lock holder
    queue._client.get = AsyncMock(return_value=b"crawl_1")

    # get_status() returns RUNNING status for crawl_1
    queue.get_status = AsyncMock(  # type: ignore[method-assign]
        return_value=CrawlStatusInfo(
            crawl_id="crawl_1",
            status=CrawlStatus.RUNNING,
            error=None,
            started_at="2024-01-15T10:00:00Z",
            finished_at=None,
        )
    )

    # Mock delete to verify it's NOT called
    queue._client.delete = AsyncMock()

    # Attempt to acquire lock as crawl_2
    acquired = await queue.acquire_lock("crawl_2")

    # Should fail to acquire (lock is legitimately held)
    assert acquired is False

    # Should NOT have deleted the lock
    queue._client.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_acquire_lock_handles_missing_status() -> None:
    """Test that acquire_lock handles missing status gracefully."""
    queue = QueueManager(redis_url="redis://localhost:53379")
    queue._client = Mock()  # type: ignore[attr-defined]

    # set() returns False (lock is held)
    queue._client.set = AsyncMock(return_value=False)

    # get() returns the current lock holder
    queue._client.get = AsyncMock(return_value=b"crawl_unknown")

    # get_status() returns None (status not found)
    queue.get_status = AsyncMock(return_value=None)  # type: ignore[method-assign]

    # Mock delete to verify it's NOT called
    queue._client.delete = AsyncMock()

    # Attempt to acquire lock
    acquired = await queue.acquire_lock("crawl_2")

    # Should fail to acquire (can't verify status)
    assert acquired is False

    # Should NOT have deleted the lock
    queue._client.delete.assert_not_awaited()
