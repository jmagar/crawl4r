"""Shared retry policy for consistent error handling across services.

This module provides a reusable RetryPolicy class that encapsulates retry logic
with exponential backoff, used across TEI, Qdrant, and Crawl4AI clients.

Example:
    Basic retry with default delays:
        >>> policy = RetryPolicy()
        >>> result = await policy.execute_async(some_async_operation)

    Custom retry configuration:
        >>> policy = RetryPolicy(
        ...     max_retries=5,
        ...     delays=[1.0, 2.0, 4.0, 8.0, 16.0]
        ... )
        >>> result = await policy.execute_async(operation)
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

T = TypeVar("T")


class RetryPolicy:
    """Configurable retry policy with exponential backoff.

    Handles transient network errors, timeouts, and 5xx HTTP errors with
    exponential backoff and jitter to avoid thundering herd.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        delays: List of delay values in seconds for each retry attempt
                (default: [1.0, 2.0, 4.0])
    """

    def __init__(
        self,
        max_retries: int = 3,
        delays: list[float] | None = None,
    ):
        """Initialize retry policy.

        Args:
            max_retries: Maximum number of retry attempts
            delays: Optional custom delay sequence (seconds). If not provided,
                   uses [1.0, 2.0, 4.0] exponential backoff.
        """
        self.max_retries = max_retries
        self.delays = delays or [1.0, 2.0, 4.0]
        self._logger = logging.getLogger(__name__)

    async def execute_async(
        self,
        operation: Callable[[], Awaitable[T]],
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
        operation_name: str = "operation",
    ) -> T:
        """Execute async operation with retry logic.

        Args:
            operation: Async callable to execute
            retryable_exceptions: Tuple of exception types to retry
                                 (default: network/timeout errors)
            operation_name: Human-readable operation name for logging

        Returns:
            Result from successful operation execution

        Raises:
            Exception: Last exception if all retries exhausted
        """
        if retryable_exceptions is None:
            retryable_exceptions = (
                httpx.NetworkError,
                httpx.ConnectError,
                httpx.TimeoutException,
            )

        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return await operation()
            except retryable_exceptions as e:
                last_exception = e

                if attempt < self.max_retries - 1:
                    delay = self._get_delay(attempt)
                    self._logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {delay:.1f}s: {e}",
                    )
                    await asyncio.sleep(delay)
                else:
                    self._logger.error(
                        f"{operation_name} failed after {self.max_retries} attempts: {e}"
                    )
            except httpx.HTTPStatusError as e:
                # Only retry 5xx errors
                if e.response.status_code >= 500 and attempt < self.max_retries - 1:
                    delay = self._get_delay(attempt)
                    self._logger.warning(
                        f"{operation_name} got {e.response.status_code}, "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    # 4xx errors or max retries exhausted
                    raise

        # All retries exhausted
        if last_exception:
            raise last_exception
        raise RuntimeError(f"{operation_name} failed unexpectedly")

    def _get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with jitter.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds with added jitter
        """
        base_delay = self.delays[min(attempt, len(self.delays) - 1)]
        jitter = random.uniform(0, base_delay * 0.1)
        return base_delay + jitter
