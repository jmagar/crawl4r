"""Circuit breaker pattern implementation.

This module implements the circuit breaker pattern to prevent cascading failures
when external services (TEI, Qdrant) become unavailable. The circuit breaker
monitors failure rates and can temporarily block calls to failing services,
allowing them time to recover.

Circuit Breaker States:
    CLOSED: Normal operation, all calls allowed
    OPEN: Service failing, calls rejected immediately (fail-fast)
    HALF_OPEN: Testing recovery, limited calls allowed

State Transitions:
    CLOSED -> OPEN: When failure count reaches threshold
    OPEN -> HALF_OPEN: After reset timeout expires
    HALF_OPEN -> CLOSED: On successful call
    HALF_OPEN -> OPEN: On failed call

Example:
    >>> cb = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
    >>> async def risky_operation():
    ...     # Call external service
    ...     return await client.fetch_data()
    >>> result = await cb.call(risky_operation)
"""

import asyncio
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker rejects a call."""

    pass


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures.

    The circuit breaker monitors consecutive failures and transitions between
    three states (CLOSED, OPEN, HALF_OPEN) to protect against cascading
    failures when external services become unavailable.

    Attributes:
        failure_threshold: Number of consecutive failures before opening circuit
        reset_timeout: Seconds to wait before transitioning OPEN -> HALF_OPEN
        state: Current circuit state (CLOSED, OPEN, HALF_OPEN)
        failure_count: Number of consecutive failures
        opened_at: Timestamp when circuit was opened (None if not open)

    Example:
        >>> cb = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
        >>> if cb.can_execute():
        ...     try:
        ...         result = await external_service()
        ...         cb.record_success()
        ...     except Exception:
        ...         cb.record_failure()
    """

    def __init__(
        self, failure_threshold: int = 5, reset_timeout: float = 60.0
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening
                circuit. Must be positive integer.
            reset_timeout: Seconds to wait before attempting recovery. Must be
                positive float.

        Raises:
            ValueError: If failure_threshold or reset_timeout is not positive.

        Example:
            >>> cb = CircuitBreaker(failure_threshold=3, reset_timeout=30.0)
            >>> cb.state
            'CLOSED'
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if reset_timeout <= 0:
            raise ValueError("reset_timeout must be positive")

        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._state = CircuitState.CLOSED
        self.failure_count = 0
        self.opened_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state as string.

        Automatically transitions OPEN -> HALF_OPEN if reset timeout has expired.

        Returns:
            Current state: "CLOSED", "OPEN", or "HALF_OPEN"
        """
        # Check if we should transition from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN:
            if (
                self.opened_at is not None
                and time.time() - self.opened_at >= self.reset_timeout
            ):
                self._state = CircuitState.HALF_OPEN
        return self._state.value

    def can_execute(self) -> bool:
        """Check if a call is allowed in the current state.

        In CLOSED state, all calls are allowed.
        In OPEN state, checks if reset timeout has expired:
            - If expired, transitions to HALF_OPEN and allows call
            - If not expired, rejects call
        In HALF_OPEN state, allows a single test call.

        Returns:
            True if call should proceed, False if it should be rejected

        Example:
            >>> cb = CircuitBreaker(failure_threshold=1)
            >>> cb.can_execute()
            True
            >>> cb.record_failure()
            >>> cb.can_execute()  # Circuit opened
            False
        """
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if enough time has passed to try recovery
            if (
                self.opened_at is not None
                and time.time() - self.opened_at >= self.reset_timeout
            ):
                # Transition to HALF_OPEN to test recovery
                self._state = CircuitState.HALF_OPEN
                return True
            return False

        # HALF_OPEN state: allow single test call
        return True

    def record_success(self) -> None:
        """Record a successful call and reset failure counter.

        In CLOSED state: Resets failure counter to 0
        In HALF_OPEN state: Transitions to CLOSED and resets counter
        In OPEN state: Should not be called (call was rejected)

        Example:
            >>> cb = CircuitBreaker()
            >>> cb.record_failure()
            >>> cb.failure_count
            1
            >>> cb.record_success()
            >>> cb.failure_count
            0
        """
        self.failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call and increment failure counter.

        In CLOSED state:
            - Increments failure counter
            - If threshold reached, transitions to OPEN
        In HALF_OPEN state:
            - Transitions back to OPEN (recovery failed)
            - Records new opened_at timestamp
        In OPEN state: Should not be called (call was rejected)

        Example:
            >>> cb = CircuitBreaker(failure_threshold=2)
            >>> cb.record_failure()
            >>> cb.state
            'CLOSED'
            >>> cb.record_failure()
            >>> cb.state
            'OPEN'
        """
        self.failure_count += 1

        if self._state == CircuitState.CLOSED:
            # Check if we've reached the failure threshold
            if self.failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self.opened_at = time.time()
        elif self._state == CircuitState.HALF_OPEN:
            # Recovery failed, go back to OPEN
            self._state = CircuitState.OPEN
            self.opened_at = time.time()

    async def call(self, func: Callable[[], Any]) -> T:
        """Execute an async function with circuit breaker protection.

        This method wraps an async function call with circuit breaker logic.
        It checks if the call is allowed, executes the function, and records
        the success or failure.

        Args:
            func: Async function to execute (takes no arguments)

        Returns:
            Result of the function call

        Raises:
            CircuitBreakerError: If circuit is OPEN and call is rejected
            Exception: Any exception raised by the wrapped function

        Example:
            >>> cb = CircuitBreaker()
            >>> async def fetch_data():
            ...     return {"key": "value"}
            >>> result = await cb.call(fetch_data)
            >>> result
            {'key': 'value'}
        """
        async with self._lock:
            if not self.can_execute():
                raise CircuitBreakerError(
                    f"Circuit breaker is {self.state}, rejecting call"
                )

        try:
            result = await func()
            async with self._lock:
                self.record_success()
            return result
        except Exception as e:
            async with self._lock:
                self.record_failure()
            raise e
