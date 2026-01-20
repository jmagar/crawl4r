from unittest.mock import AsyncMock

import pytest

from crawl4r.services.queue import QueueManager


@pytest.mark.asyncio
async def test_enqueue_dequeue_round_trip() -> None:
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.lpush = AsyncMock(return_value=1)
    manager._client.brpop = AsyncMock(
        return_value=(b"crawl_queue", b"crawl_id|https://example.com")
    )

    await manager.enqueue_crawl("crawl_id", ["https://example.com"])
    item = await manager.dequeue_crawl()
    assert item == ("crawl_id", ["https://example.com"])
