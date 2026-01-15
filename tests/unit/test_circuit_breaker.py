"""Unit tests for circuit breaker pattern.

This module contains comprehensive tests for the CircuitBreaker class,
which implements the circuit breaker pattern to prevent cascading failures
when external services (TEI, Qdrant) become unavailable.

Circuit Breaker States:
    - CLOSED: Normal operation, requests allowed
    - OPEN: Service failing, requests rejected immediately
    - HALF_OPEN: Testing if service recovered, single request allowed

Test Coverage:
    - Initialization with configurable thresholds
    - State transitions (CLOSED → OPEN → HALF_OPEN → CLOSED/OPEN)
    - Failure counting and threshold detection
    - Reset timeout for OPEN → HALF_OPEN transition
    - Success counter reset after successful call
    - Integration with async functions
    - Thread safety for concurrent calls
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest


class TestCircuitBreakerInitialization:
    """Test circuit breaker initialization with various configurations."""

    def test_default_initialization(self) -> None:
        """Circuit breaker initializes with default failure threshold of 5."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker()
        assert cb.failure_threshold == 5
        assert cb.reset_timeout == 60
        assert cb.state == "CLOSED"

    def test_custom_failure_threshold(self) -> None:
        """Circuit breaker accepts custom failure threshold."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)
        assert cb.failure_threshold == 3

    def test_custom_reset_timeout(self) -> None:
        """Circuit breaker accepts custom reset timeout."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(reset_timeout=30)
        assert cb.reset_timeout == 30

    def test_invalid_failure_threshold_raises(self) -> None:
        """Circuit breaker rejects non-positive failure threshold."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreaker(failure_threshold=0)

        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            CircuitBreaker(failure_threshold=-1)

    def test_invalid_reset_timeout_raises(self) -> None:
        """Circuit breaker rejects non-positive reset timeout."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        with pytest.raises(ValueError, match="reset_timeout must be positive"):
            CircuitBreaker(reset_timeout=0)

        with pytest.raises(ValueError, match="reset_timeout must be positive"):
            CircuitBreaker(reset_timeout=-1)


class TestCircuitBreakerClosedState:
    """Test circuit breaker behavior in CLOSED state."""

    def test_allows_calls_when_closed(self) -> None:
        """Circuit breaker allows calls in CLOSED state."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker()
        assert cb.can_execute() is True

    def test_counts_failures_in_closed_state(self) -> None:
        """Circuit breaker counts consecutive failures in CLOSED state."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)
        assert cb.failure_count == 0

        cb.record_failure()
        assert cb.failure_count == 1

        cb.record_failure()
        assert cb.failure_count == 2

    def test_transitions_to_open_at_threshold(self) -> None:
        """Circuit breaker transitions CLOSED → OPEN at failure threshold."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == "CLOSED"

        cb.record_failure()
        cb.record_failure()
        assert cb.state == "CLOSED"

        cb.record_failure()
        assert cb.state == "OPEN"

    def test_resets_failure_count_on_success(self) -> None:
        """Circuit breaker resets failure count on successful call."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == "CLOSED"


class TestCircuitBreakerOpenState:
    """Test circuit breaker behavior in OPEN state."""

    def test_rejects_calls_when_open(self) -> None:
        """Circuit breaker rejects calls immediately in OPEN state."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.can_execute() is False

    def test_records_open_time(self) -> None:
        """Circuit breaker records timestamp when transitioning to OPEN."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1)
        assert cb.opened_at is None

        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.opened_at is not None
        assert isinstance(cb.opened_at, float)

    def test_transitions_to_half_open_after_timeout(self) -> None:
        """Circuit breaker transitions OPEN → HALF_OPEN after reset timeout."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1, reset_timeout=1)
        cb.record_failure()
        assert cb.state == "OPEN"

        time.sleep(1.1)
        assert cb.can_execute() is True
        assert cb.state == "HALF_OPEN"

    def test_stays_open_before_timeout(self) -> None:
        """Circuit breaker stays OPEN before reset timeout expires."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1, reset_timeout=5)
        cb.record_failure()
        assert cb.state == "OPEN"

        time.sleep(0.5)
        assert cb.can_execute() is False
        assert cb.state == "OPEN"


class TestCircuitBreakerHalfOpenState:
    """Test circuit breaker behavior in HALF_OPEN state."""

    def test_allows_single_test_call(self) -> None:
        """Circuit breaker allows single test call in HALF_OPEN state."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1, reset_timeout=1)
        cb.record_failure()
        time.sleep(1.1)
        assert cb.state == "HALF_OPEN"
        assert cb.can_execute() is True

    def test_transitions_to_closed_on_success(self) -> None:
        """Circuit breaker transitions HALF_OPEN → CLOSED on success."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1, reset_timeout=1)
        cb.record_failure()
        time.sleep(1.1)
        assert cb.state == "HALF_OPEN"

        cb.record_success()
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    def test_transitions_to_open_on_failure(self) -> None:
        """Circuit breaker transitions HALF_OPEN → OPEN on failure."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1, reset_timeout=1)
        cb.record_failure()
        time.sleep(1.1)
        assert cb.state == "HALF_OPEN"

        cb.record_failure()
        assert cb.state == "OPEN"

    def test_resets_opened_at_on_half_open_failure(self) -> None:
        """Circuit breaker resets opened_at timestamp on HALF_OPEN failure."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1, reset_timeout=1)
        cb.record_failure()
        first_opened_at = cb.opened_at
        time.sleep(1.1)
        assert cb.state == "HALF_OPEN"

        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.opened_at is not None
        assert cb.opened_at > first_opened_at  # type: ignore


class TestCircuitBreakerAsyncIntegration:
    """Test circuit breaker integration with async functions."""

    @pytest.mark.asyncio
    async def test_wraps_async_function(self) -> None:
        """Circuit breaker can wrap and execute async functions."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker()

        async def async_operation() -> str:
            await asyncio.sleep(0.01)
            return "success"

        result = await cb.call(async_operation)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_records_success_for_async_function(self) -> None:
        """Circuit breaker records success for successful async calls."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker()
        cb.record_failure()
        assert cb.failure_count == 1

        async def async_operation() -> str:
            return "success"

        await cb.call(async_operation)
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_records_failure_for_async_exception(self) -> None:
        """Circuit breaker records failure when async function raises."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)
        assert cb.failure_count == 0

        async def failing_operation() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await cb.call(failing_operation)

        assert cb.failure_count == 1

    @pytest.mark.asyncio
    async def test_raises_when_circuit_open(self) -> None:
        """Circuit breaker raises exception when circuit is OPEN."""
        from rag_ingestion.circuit_breaker import CircuitBreaker, CircuitBreakerError

        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == "OPEN"

        async def async_operation() -> str:
            return "success"

        with pytest.raises(CircuitBreakerError, match="Circuit breaker is OPEN"):
            await cb.call(async_operation)

    @pytest.mark.asyncio
    async def test_multiple_consecutive_failures(self) -> None:
        """Circuit breaker opens after multiple consecutive async failures."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3)

        async def failing_operation() -> None:
            raise RuntimeError("Service unavailable")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing_operation)
            assert cb.state == "CLOSED"

        with pytest.raises(RuntimeError):
            await cb.call(failing_operation)
        assert cb.state == "OPEN"


class TestCircuitBreakerThreadSafety:
    """Test circuit breaker thread safety for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_can_execute_calls(self) -> None:
        """Circuit breaker handles concurrent can_execute() calls safely."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker()

        async def check_can_execute() -> bool:
            return cb.can_execute()

        results = await asyncio.gather(*[check_can_execute() for _ in range(100)])
        assert all(results)

    @pytest.mark.asyncio
    async def test_concurrent_record_failure_calls(self) -> None:
        """Circuit breaker handles concurrent record_failure() calls safely."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=10)

        def record_failure() -> None:
            cb.record_failure()

        await asyncio.gather(*[asyncio.to_thread(record_failure) for _ in range(5)])
        assert cb.failure_count == 5

    @pytest.mark.asyncio
    async def test_concurrent_async_calls(self) -> None:
        """Circuit breaker handles concurrent async function calls safely."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=10)
        call_count = 0

        async def tracked_operation() -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return "success"

        results = await asyncio.gather(*[cb.call(tracked_operation) for _ in range(10)])
        assert len(results) == 10
        assert all(r == "success" for r in results)
        assert call_count == 10
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_state_transitions_thread_safe(self) -> None:
        """Circuit breaker state transitions are thread-safe."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=5)

        async def mixed_operation(should_fail: bool) -> str:
            if should_fail:
                raise RuntimeError("Intentional failure")
            return "success"

        tasks = []
        for i in range(10):
            if i < 4:
                tasks.append(cb.call(lambda: mixed_operation(True)))
            else:
                tasks.append(cb.call(lambda: mixed_operation(False)))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert cb.state in ["CLOSED", "OPEN"]


class TestCircuitBreakerEdgeCases:
    """Test circuit breaker edge cases and boundary conditions."""

    def test_failure_threshold_of_one(self) -> None:
        """Circuit breaker works with failure threshold of 1."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1)
        assert cb.state == "CLOSED"

        cb.record_failure()
        assert cb.state == "OPEN"

    def test_reset_after_multiple_open_close_cycles(self) -> None:
        """Circuit breaker resets correctly after multiple open/close cycles."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, reset_timeout=1)

        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"

        time.sleep(1.1)
        assert cb.can_execute() is True
        cb.record_success()
        assert cb.state == "CLOSED"

        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"

    def test_success_in_closed_state_is_noop(self) -> None:
        """Recording success in CLOSED state with no failures is safe."""
        from rag_ingestion.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker()
        assert cb.failure_count == 0

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == "CLOSED"

    @pytest.mark.asyncio
    async def test_circuit_breaker_error_contains_state(self) -> None:
        """CircuitBreakerError includes current state information."""
        from rag_ingestion.circuit_breaker import CircuitBreaker, CircuitBreakerError

        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()

        async def operation() -> str:
            return "test"

        try:
            await cb.call(operation)
            pytest.fail("Expected CircuitBreakerError")
        except CircuitBreakerError as e:
            assert "OPEN" in str(e)
