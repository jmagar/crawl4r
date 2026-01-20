"""Queue manager for Redis-backed crawl coordination."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable
from typing import TypeVar

import redis.asyncio as redis

from crawl4r.services.models import CrawlStatus, CrawlStatusInfo

logger = logging.getLogger(__name__)

T = TypeVar("T")

LOCK_KEY = "crawl4r:queue_lock"
QUEUE_KEY = "crawl4r:crawl_queue"
STATUS_PREFIX = "crawl4r:crawl_status:"
RECENT_LIST_KEY = "crawl4r:crawl_recent"
LOCK_TTL_SECONDS = 3600
STATUS_TTL_SECONDS = 86400

# Lua script for atomic stale lock recovery
# Checks if lock holder hasn't changed before deleting and re-acquiring
RECOVER_LOCK_SCRIPT = """
local current_holder = redis.call('GET', KEYS[1])
if current_holder == ARGV[1] then
    redis.call('DEL', KEYS[1])
    return redis.call('SET', KEYS[1], ARGV[2], 'NX', 'EX', ARGV[3])
end
return 0
"""


class QueueManager:
    """Manage crawl queue state in Redis."""

    def __init__(self, redis_url: str) -> None:
        """Initialize the queue manager.

        Args:
            redis_url: Redis connection URL
        """
        self._client: redis.Redis = redis.from_url(redis_url, decode_responses=False)

    async def is_available(self) -> bool:
        """Check if Redis connection is available.

        Returns:
            True if Redis is reachable, False otherwise
        """
        try:
            await self._await(self._client.ping())
            return True
        except (ConnectionError, TimeoutError, OSError):
            logger.warning("Redis is unavailable - queue operations will be degraded")
            return False

    async def _await(self, result: Awaitable[T] | T) -> T:
        if inspect.isawaitable(result):
            return await result
        return result

    async def acquire_lock(self, owner: str) -> bool:
        """Acquire a queue lock for exclusive crawl processing.

        If the lock is already held, checks if the holder has FAILED status
        and recovers the lock automatically in that case using an atomic
        Lua script to prevent race conditions.

        Args:
            owner: Identifier for the lock holder

        Returns:
            True if the lock was acquired, False on failure or Redis unavailable
        """
        try:
            result = await self._await(
                self._client.set(LOCK_KEY, owner, nx=True, ex=LOCK_TTL_SECONDS)
            )

            # If lock was not acquired, check if holder has failed
            if not result:
                holder = await self._await(self._client.get(LOCK_KEY))
                if holder:
                    holder_id = holder.decode() if isinstance(holder, bytes) else holder
                    status = await self.get_status(holder_id)

                    # Recover lock if holder has FAILED status
                    if status and status.status == CrawlStatus.FAILED:
                        logger.info(
                            "Recovering stale lock from FAILED crawl: %s", holder_id
                        )

                        # Atomic recovery: only delete if holder hasn't changed
                        # Lua script ensures check-and-delete-and-set is atomic
                        lua_result = await self._await(
                            self._client.eval(
                                RECOVER_LOCK_SCRIPT,
                                1,  # number of keys
                                LOCK_KEY,  # KEYS[1]
                                holder_id,  # ARGV[1] - expected holder
                                owner,  # ARGV[2] - new owner
                                str(LOCK_TTL_SECONDS),  # ARGV[3] - TTL
                            )
                        )
                        result = bool(lua_result)

            return bool(result)
        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.warning("Failed to acquire lock due to Redis error: %s", exc)
            return False

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

        Note:
            Silently fails if Redis is unavailable (logs warning)
        """
        try:
            payload = f"{crawl_id}|{','.join(urls)}"
            await self._await(self._client.lpush(QUEUE_KEY, payload))
        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.warning("Failed to enqueue crawl %s: %s", crawl_id, exc)

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

        Note:
            Silently fails if Redis is unavailable (logs warning)
        """
        try:
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
        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.warning("Failed to set status for crawl %s: %s", info.crawl_id, exc)

    async def get_status(self, crawl_id: str) -> CrawlStatusInfo | None:
        """Fetch crawl status information.

        Args:
            crawl_id: Crawl identifier

        Returns:
            CrawlStatusInfo if present, None if not found or Redis unavailable
        """
        try:
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
        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.warning("Failed to get status for crawl %s: %s", crawl_id, exc)
            return None

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
