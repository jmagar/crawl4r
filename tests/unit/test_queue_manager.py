"""Unit tests for QueueManager with mocked Redis operations."""

from unittest.mock import AsyncMock

import pytest

from crawl4r.services.models import CrawlStatus, CrawlStatusInfo
from crawl4r.services.queue import QueueManager


@pytest.mark.asyncio
async def test_acquire_lock_success() -> None:
    """Test acquiring a lock when available."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.set = AsyncMock(return_value=True)

    assert await manager.acquire_lock("crawl_id") is True


@pytest.mark.asyncio
async def test_acquire_lock_failure() -> None:
    """Test acquiring a lock when already held."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.set = AsyncMock(return_value=False)
    manager._client.get = AsyncMock(return_value=b"other_crawl_id")
    manager._client.eval = AsyncMock(return_value=0)

    # Mock get_status to return None (no status for the holder)
    async def mock_get_status(crawl_id: str) -> CrawlStatusInfo | None:
        return None

    manager.get_status = mock_get_status  # type: ignore[method-assign]

    assert await manager.acquire_lock("crawl_id") is False


@pytest.mark.asyncio
async def test_enqueue_dequeue_round_trip() -> None:
    """Test enqueueing and dequeueing a crawl request."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.lpush = AsyncMock(return_value=1)
    manager._client.brpop = AsyncMock(
        return_value=(b"crawl_queue", b"crawl_id|https://example.com")
    )

    result = await manager.enqueue_crawl("crawl_id", ["https://example.com"])
    assert result is True

    item = await manager.dequeue_crawl()
    assert item == ("crawl_id", ["https://example.com"])


@pytest.mark.asyncio
async def test_set_get_status() -> None:
    """Test setting and retrieving crawl status."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.set = AsyncMock(return_value=True)
    manager._client.lpush = AsyncMock(return_value=1)
    manager._client.ltrim = AsyncMock(return_value=True)
    manager._client.get = AsyncMock(
        return_value=b'{"crawl_id": "crawl_id", "status": "RUNNING", "error": null, "started_at": "2024-01-15T10:30:00Z", "finished_at": null}'
    )

    status_info = CrawlStatusInfo(
        crawl_id="crawl_id",
        status=CrawlStatus.RUNNING,
        started_at="2024-01-15T10:30:00Z",
    )

    await manager.set_status(status_info)
    status = await manager.get_status("crawl_id")

    assert status is not None
    assert status.crawl_id == "crawl_id"
    assert status.status == CrawlStatus.RUNNING
    assert status.started_at == "2024-01-15T10:30:00Z"


@pytest.mark.asyncio
async def test_list_recent_and_get_active() -> None:
    """Test listing recent crawls and filtering active ones."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()

    # Mock lrange to return crawl IDs
    manager._client.lrange = AsyncMock(return_value=[b"crawl_1", b"crawl_2"])

    # Mock get to return status data
    async def mock_get(key: str) -> bytes:
        if key.endswith("crawl_1"):
            return b'{"crawl_id": "crawl_1", "status": "RUNNING", "error": null, "started_at": "2024-01-15T10:30:00Z", "finished_at": null}'
        return b'{"crawl_id": "crawl_2", "status": "COMPLETED", "error": null, "started_at": "2024-01-15T10:30:00Z", "finished_at": "2024-01-15T10:35:00Z"}'

    manager._client.get = mock_get

    recent = await manager.list_recent(limit=10)
    assert len(recent) == 2
    assert recent[0].crawl_id == "crawl_1"
    assert recent[1].crawl_id == "crawl_2"

    active = await manager.get_active()
    assert len(active) == 1
    assert active[0].crawl_id == "crawl_1"
    assert active[0].status == CrawlStatus.RUNNING


@pytest.mark.asyncio
async def test_stale_lock_recovery() -> None:
    """Test automatic recovery of stale lock from failed crawl."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()

    # First set call fails (lock already held)
    # Second set call would succeed after recovery
    manager._client.set = AsyncMock(return_value=False)
    manager._client.get = AsyncMock(return_value=b"crawl_old")

    # Lua eval succeeds (lock recovered)
    manager._client.eval = AsyncMock(return_value=1)

    # Mock get_status to return FAILED status
    async def mock_get_status(crawl_id: str) -> CrawlStatusInfo | None:
        if crawl_id == "crawl_old":
            return CrawlStatusInfo(
                crawl_id="crawl_old",
                status=CrawlStatus.FAILED,
                error="Previous crawl failed",
                started_at="2024-01-15T10:00:00Z",
                finished_at="2024-01-15T10:15:00Z",
            )
        return None

    manager.get_status = mock_get_status  # type: ignore[method-assign]

    # Should recover stale lock
    result = await manager.acquire_lock("crawl_new")
    assert result is True


@pytest.mark.asyncio
async def test_release_lock_when_owned() -> None:
    """Test releasing a lock when owned by the caller."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.get = AsyncMock(return_value=b"crawl_id")
    manager._client.delete = AsyncMock(return_value=1)

    await manager.release_lock("crawl_id")
    manager._client.delete.assert_called_once()


@pytest.mark.asyncio
async def test_release_lock_when_not_owned() -> None:
    """Test releasing a lock when not owned by the caller."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.get = AsyncMock(return_value=b"other_crawl_id")
    manager._client.delete = AsyncMock(return_value=0)

    await manager.release_lock("crawl_id")
    manager._client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_dequeue_crawl_returns_none_on_timeout() -> None:
    """Test dequeue returns None when no items available."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.brpop = AsyncMock(return_value=None)

    result = await manager.dequeue_crawl(timeout=1)
    assert result is None


@pytest.mark.asyncio
async def test_get_status_returns_none_when_not_found() -> None:
    """Test get_status returns None when crawl ID not found."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.get = AsyncMock(return_value=None)

    status = await manager.get_status("nonexistent_crawl")
    assert status is None


@pytest.mark.asyncio
async def test_is_available_success() -> None:
    """Test Redis availability check when Redis is reachable."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.ping = AsyncMock(return_value=True)

    assert await manager.is_available() is True


@pytest.mark.asyncio
async def test_is_available_failure() -> None:
    """Test Redis availability check when Redis is unreachable."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.ping = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

    assert await manager.is_available() is False


@pytest.mark.asyncio
async def test_acquire_lock_redis_error() -> None:
    """Test lock acquisition handles Redis connection errors gracefully."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.set = AsyncMock(side_effect=ConnectionError("Redis unavailable"))

    result = await manager.acquire_lock("crawl_id")
    assert result is False


@pytest.mark.asyncio
async def test_enqueue_crawl_redis_error() -> None:
    """Test enqueue handles Redis errors gracefully."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.lpush = AsyncMock(side_effect=TimeoutError("Redis timeout"))

    result = await manager.enqueue_crawl("crawl_id", ["https://example.com"])
    assert result is False


@pytest.mark.asyncio
async def test_get_status_redis_error() -> None:
    """Test get_status handles Redis errors gracefully."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.get = AsyncMock(side_effect=OSError("Network error"))

    status = await manager.get_status("crawl_id")
    assert status is None


@pytest.mark.asyncio
async def test_get_queue_length() -> None:
    """Test getting the current queue length."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.llen = AsyncMock(return_value=5)

    length = await manager.get_queue_length()
    assert length == 5


@pytest.mark.asyncio
async def test_close_connection() -> None:
    """Test closing the Redis connection."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.aclose = AsyncMock(return_value=None)

    await manager.close()
    manager._client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_enqueue_multiple_urls() -> None:
    """Test enqueueing crawl with multiple URLs."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.lpush = AsyncMock(return_value=1)
    manager._client.brpop = AsyncMock(
        return_value=(
            b"crawl_queue",
            b"crawl_id|https://example.com,https://example.org",
        )
    )

    await manager.enqueue_crawl(
        "crawl_id", ["https://example.com", "https://example.org"]
    )
    item = await manager.dequeue_crawl()

    assert item == ("crawl_id", ["https://example.com", "https://example.org"])


@pytest.mark.asyncio
async def test_set_status_redis_error() -> None:
    """Test set_status handles Redis errors gracefully without raising."""
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.set = AsyncMock(side_effect=ConnectionError("Redis down"))

    status_info = CrawlStatusInfo(
        crawl_id="crawl_id",
        status=CrawlStatus.RUNNING,
        started_at="2024-01-15T10:30:00Z",
    )

    # Should not raise exception, just log warning
    await manager.set_status(status_info)
