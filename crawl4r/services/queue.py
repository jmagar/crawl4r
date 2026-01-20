"""Queue manager for Redis-backed crawl coordination."""

from __future__ import annotations

import inspect
import json
from typing import Awaitable, TypeVar

import redis.asyncio as redis

from crawl4r.services.models import CrawlStatus, CrawlStatusInfo

T = TypeVar("T")

LOCK_KEY = "crawl4r:queue_lock"
QUEUE_KEY = "crawl4r:crawl_queue"
STATUS_PREFIX = "crawl4r:crawl_status:"
RECENT_LIST_KEY = "crawl4r:crawl_recent"
LOCK_TTL_SECONDS = 3600
STATUS_TTL_SECONDS = 86400


class QueueManager:
    """Manage crawl queue state in Redis."""

    def __init__(self, redis_url: str) -> None:
        """Initialize the queue manager.

        Args:
            redis_url: Redis connection URL
        """
        self._client: redis.Redis = redis.from_url(redis_url, decode_responses=False)

    async def _await(self, result: Awaitable[T] | T) -> T:
        if inspect.isawaitable(result):
            return await result
        return result

    async def acquire_lock(self, owner: str) -> bool:
        """Acquire a queue lock for exclusive crawl processing.

        Args:
            owner: Identifier for the lock holder

        Returns:
            True if the lock was acquired
        """
        result = await self._await(
            self._client.set(LOCK_KEY, owner, nx=True, ex=LOCK_TTL_SECONDS)
        )
        return bool(result)

    async def release_lock(self, owner: str) -> None:
        """Release the queue lock if owned by the caller.

        Args:
            owner: Identifier for the lock holder
        """
        current = await self._await(self._client.get(LOCK_KEY))
        if current == owner.encode():
            await self._await(self._client.delete(LOCK_KEY))

    async def enqueue_crawl(self, crawl_id: str, urls: list[str]) -> None:
        """Enqueue a crawl request for later processing.

        Args:
            crawl_id: Crawl identifier
            urls: URLs to crawl
        """
        payload = f"{crawl_id}|{','.join(urls)}"
        await self._await(self._client.lpush(QUEUE_KEY, payload))

    async def dequeue_crawl(self, timeout: int = 5) -> tuple[str, list[str]] | None:
        """Dequeue the next crawl request.

        Args:
            timeout: Seconds to block while waiting for work

        Returns:
            Tuple of crawl_id and URL list when available
        """
        item = await self._await(self._client.brpop([QUEUE_KEY], timeout=timeout))
        if not item:
            return None
        _, value = item
        decoded = value.decode() if isinstance(value, bytes) else value
        crawl_id, raw_urls = decoded.split("|", maxsplit=1)
        urls = [url for url in raw_urls.split(",") if url]
        return crawl_id, urls

    async def set_status(self, info: CrawlStatusInfo) -> None:
        """Persist crawl status information.

        Args:
            info: Crawl status details
        """
        payload = json.dumps(
            {
                "crawl_id": info.crawl_id,
                "status": info.status.value,
                "error": info.error,
                "started_at": info.started_at,
                "finished_at": info.finished_at,
            }
        )
        await self._await(
            self._client.set(
                f"{STATUS_PREFIX}{info.crawl_id}", payload, ex=STATUS_TTL_SECONDS
            )
        )
        await self._await(self._client.lpush(RECENT_LIST_KEY, info.crawl_id))
        await self._await(self._client.ltrim(RECENT_LIST_KEY, 0, 49))

    async def get_status(self, crawl_id: str) -> CrawlStatusInfo | None:
        """Fetch crawl status information.

        Args:
            crawl_id: Crawl identifier

        Returns:
            CrawlStatusInfo if present
        """
        raw = await self._await(self._client.get(f"{STATUS_PREFIX}{crawl_id}"))
        if raw is None:
            return None
        raw_value = raw.decode() if isinstance(raw, bytes) else raw
        data = json.loads(raw_value)
        return CrawlStatusInfo(
            crawl_id=data["crawl_id"],
            status=CrawlStatus(data["status"]),
            error=data.get("error"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
        )

    async def list_recent(self, limit: int = 10) -> list[CrawlStatusInfo]:
        """Return recent crawl statuses.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of CrawlStatusInfo entries
        """
        crawl_ids = await self._await(
            self._client.lrange(RECENT_LIST_KEY, 0, limit - 1)
        )
        statuses: list[CrawlStatusInfo] = []
        for raw_id in crawl_ids:
            crawl_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            status = await self.get_status(crawl_id)
            if status:
                statuses.append(status)
        return statuses

    async def get_active(self) -> list[CrawlStatusInfo]:
        """Return active crawl statuses.

        Returns:
            List of CrawlStatusInfo entries that are queued or running
        """
        recent = await self.list_recent(limit=50)
        return [
            status
            for status in recent
            if status.status in {CrawlStatus.QUEUED, CrawlStatus.RUNNING}
        ]

    async def close(self) -> None:
        """Close the Redis client connection."""
        await self._await(self._client.aclose())

    async def get_queue_length(self) -> int:
        """Return the current crawl queue length."""
        length = await self._await(self._client.llen(QUEUE_KEY))
        return int(length)
