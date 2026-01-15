"""Tests for TEI client integration with circuit breaker pattern.

This test module verifies the integration between TEIClient and CircuitBreaker
to ensure resilient embedding generation with automatic failure recovery.

Test Coverage:
    - TEIClient initialization with circuit breaker
    - Circuit breaker wrapping of embed_single()
    - Circuit breaker wrapping of embed_batch()
    - Circuit breaker state transitions based on TEI failures
    - Circuit breaker rejection when OPEN
    - Circuit breaker recovery when HALF_OPEN
    - Success resets failure counter
    - Integration preserves all existing TEI client functionality
"""

import asyncio
from unittest.mock import MagicMock, patch

import httpx
import pytest

from rag_ingestion.circuit_breaker import CircuitBreaker, CircuitBreakerError
from rag_ingestion.tei_client import TEIClient


class TestTEIClientCircuitBreakerInitialization:
    """Test TEIClient initialization with circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_tei_client_has_circuit_breaker_attribute(self) -> None:
        """TEIClient should have a circuit_breaker attribute after initialization."""
        client = TEIClient("http://test:80")

        # TEIClient should have circuit_breaker attribute
        assert hasattr(client, "circuit_breaker")
        assert isinstance(client.circuit_breaker, CircuitBreaker)

    @pytest.mark.asyncio
    async def test_circuit_breaker_initialized_with_defaults(self) -> None:
        """Circuit breaker should be initialized with default threshold and timeout."""
        client = TEIClient("http://test:80")

        # Default values should match circuit breaker defaults
        assert client.circuit_breaker.failure_threshold == 5
        assert client.circuit_breaker.reset_timeout == 60.0

    @pytest.mark.asyncio
    async def test_circuit_breaker_initialized_with_custom_values(self) -> None:
        """Circuit breaker should accept custom threshold and timeout values."""
        client = TEIClient(
            "http://test:80",
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=30.0
        )

        # Custom values should be passed to circuit breaker
        assert client.circuit_breaker.failure_threshold == 3
        assert client.circuit_breaker.reset_timeout == 30.0


class TestEmbedSingleWithCircuitBreaker:
    """Test embed_single() wrapped with circuit breaker protection."""

    @pytest.mark.asyncio
    async def test_embed_single_succeeds_when_circuit_closed(self) -> None:
        """embed_single should work normally when circuit is CLOSED."""
        client = TEIClient("http://test:80")

        with patch("httpx.AsyncClient.post") as mock_post:
            # Mock successful TEI response
            mock_response = MagicMock()
            mock_response.json.return_value = [[[0.1] * 1024]]
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            # Should succeed and record success
            result = await client.embed_single("test text")

            assert len(result) == 1024
            assert client.circuit_breaker.failure_count == 0
            assert client.circuit_breaker.state == "CLOSED"

    @pytest.mark.asyncio
    async def test_embed_single_records_failure_on_error(self) -> None:
        """embed_single should record failure when TEI request fails."""
        client = TEIClient("http://test:80", max_retries=1)

        with patch("httpx.AsyncClient.post") as mock_post:
            # Mock connection error
            mock_post.side_effect = httpx.ConnectError("Connection failed")

            # Should raise error and record failure
            with pytest.raises(httpx.ConnectError):
                await client.embed_single("test text")

            # Circuit breaker should track the failure
            assert client.circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_embed_single_opens_circuit_after_threshold(self) -> None:
        """Circuit should open after failure threshold is reached."""
        client = TEIClient(
            "http://test:80",
            max_retries=1,
            circuit_breaker_threshold=2
        )

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection failed")

            # First failure - circuit still CLOSED
            with pytest.raises(httpx.ConnectError):
                await client.embed_single("test text")
            assert client.circuit_breaker.state == "CLOSED"

            # Second failure - circuit should OPEN
            with pytest.raises(httpx.ConnectError):
                await client.embed_single("test text")
            assert client.circuit_breaker.state == "OPEN"

    @pytest.mark.asyncio
    async def test_embed_single_rejects_when_circuit_open(self) -> None:
        """embed_single should reject calls immediately when circuit is OPEN."""
        client = TEIClient(
            "http://test:80",
            circuit_breaker_threshold=1,
            max_retries=1
        )

        with patch("httpx.AsyncClient.post") as mock_post:
            # Fail once to open circuit
            mock_post.side_effect = httpx.ConnectError("Connection failed")
            with pytest.raises(httpx.ConnectError):
                await client.embed_single("test text")

            # Circuit is now OPEN
            assert client.circuit_breaker.state == "OPEN"

            # Next call should be rejected immediately without calling TEI
            mock_post.reset_mock()
            with pytest.raises(CircuitBreakerError):
                await client.embed_single("test text")

            # TEI should NOT have been called (circuit rejected it)
            mock_post.assert_not_called()


class TestEmbedBatchWithCircuitBreaker:
    """Test embed_batch() wrapped with circuit breaker protection."""

    @pytest.mark.asyncio
    async def test_embed_batch_succeeds_when_circuit_closed(self) -> None:
        """embed_batch should work normally when circuit is CLOSED."""
        client = TEIClient("http://test:80")

        with patch("httpx.AsyncClient.post") as mock_post:
            # Mock successful TEI batch response
            mock_response = MagicMock()
            mock_response.json.return_value = [[[0.1] * 1024, [0.2] * 1024]]
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            # Should succeed and record success
            result = await client.embed_batch(["text1", "text2"])

            assert len(result) == 2
            assert client.circuit_breaker.failure_count == 0
            assert client.circuit_breaker.state == "CLOSED"

    @pytest.mark.asyncio
    async def test_embed_batch_records_failure_on_error(self) -> None:
        """embed_batch should record failure when TEI request fails."""
        client = TEIClient("http://test:80", max_retries=1)

        with patch("httpx.AsyncClient.post") as mock_post:
            # Mock timeout error
            mock_post.side_effect = httpx.TimeoutException("Request timeout")

            # Should raise error and record failure
            with pytest.raises(httpx.TimeoutException):
                await client.embed_batch(["text1", "text2"])

            # Circuit breaker should track the failure
            assert client.circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_embed_batch_rejects_when_circuit_open(self) -> None:
        """embed_batch should reject calls immediately when circuit is OPEN."""
        client = TEIClient(
            "http://test:80",
            circuit_breaker_threshold=1,
            max_retries=1
        )

        with patch("httpx.AsyncClient.post") as mock_post:
            # Fail once to open circuit
            mock_post.side_effect = httpx.TimeoutException("Request timeout")
            with pytest.raises(httpx.TimeoutException):
                await client.embed_batch(["text1", "text2"])

            # Circuit is now OPEN
            assert client.circuit_breaker.state == "OPEN"

            # Next call should be rejected immediately
            mock_post.reset_mock()
            with pytest.raises(CircuitBreakerError):
                await client.embed_batch(["text1", "text2"])

            # TEI should NOT have been called
            mock_post.assert_not_called()


class TestCircuitBreakerStateTransitions:
    """Test circuit breaker state transitions with TEI client."""

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open_after_timeout(self) -> None:
        """Circuit should transition to HALF_OPEN after reset timeout expires."""
        client = TEIClient(
            "http://test:80",
            circuit_breaker_threshold=1,
            circuit_breaker_timeout=0.1,  # 100ms timeout
            max_retries=1
        )

        with patch("httpx.AsyncClient.post") as mock_post:
            # Fail once to open circuit
            mock_post.side_effect = httpx.ConnectError("Connection failed")
            with pytest.raises(httpx.ConnectError):
                await client.embed_single("test text")

            assert client.circuit_breaker.state == "OPEN"

            # Wait for timeout to expire
            await asyncio.sleep(0.15)

            # Circuit should now be HALF_OPEN (checked via state property)
            assert client.circuit_breaker.state == "HALF_OPEN"

    @pytest.mark.asyncio
    async def test_circuit_allows_test_call_when_half_open(self) -> None:
        """Circuit should allow a test call when HALF_OPEN."""
        client = TEIClient(
            "http://test:80",
            circuit_breaker_threshold=1,
            circuit_breaker_timeout=0.1,
            max_retries=1
        )

        with patch("httpx.AsyncClient.post") as mock_post:
            # Open circuit
            mock_post.side_effect = httpx.ConnectError("Connection failed")
            with pytest.raises(httpx.ConnectError):
                await client.embed_single("test text")

            # Wait for HALF_OPEN
            await asyncio.sleep(0.15)
            assert client.circuit_breaker.state == "HALF_OPEN"

            # Now return success for test call
            mock_response = MagicMock()
            mock_response.json.return_value = [[[0.1] * 1024]]
            mock_response.raise_for_status = MagicMock()
            mock_post.side_effect = None
            mock_post.return_value = mock_response

            # Test call should succeed
            result = await client.embed_single("test text")
            assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_successful_call_closes_circuit_from_half_open(self) -> None:
        """Successful call should close circuit from HALF_OPEN state."""
        client = TEIClient(
            "http://test:80",
            circuit_breaker_threshold=1,
            circuit_breaker_timeout=0.1,
            max_retries=1
        )

        with patch("httpx.AsyncClient.post") as mock_post:
            # Open circuit
            mock_post.side_effect = httpx.ConnectError("Connection failed")
            with pytest.raises(httpx.ConnectError):
                await client.embed_single("test text")

            # Wait for HALF_OPEN
            await asyncio.sleep(0.15)

            # Successful test call
            mock_response = MagicMock()
            mock_response.json.return_value = [[[0.1] * 1024]]
            mock_response.raise_for_status = MagicMock()
            mock_post.side_effect = None
            mock_post.return_value = mock_response

            await client.embed_single("test text")

            # Circuit should be CLOSED again
            assert client.circuit_breaker.state == "CLOSED"
            assert client.circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_failed_test_call_reopens_circuit(self) -> None:
        """Failed test call should reopen circuit from HALF_OPEN state."""
        client = TEIClient(
            "http://test:80",
            circuit_breaker_threshold=1,
            circuit_breaker_timeout=0.1,
            max_retries=1
        )

        with patch("httpx.AsyncClient.post") as mock_post:
            # Open circuit
            mock_post.side_effect = httpx.ConnectError("Connection failed")
            with pytest.raises(httpx.ConnectError):
                await client.embed_single("test text")

            # Wait for HALF_OPEN
            await asyncio.sleep(0.15)
            assert client.circuit_breaker.state == "HALF_OPEN"

            # Test call fails again
            mock_post.side_effect = httpx.ConnectError("Still failing")
            with pytest.raises(httpx.ConnectError):
                await client.embed_single("test text")

            # Circuit should be OPEN again
            assert client.circuit_breaker.state == "OPEN"


class TestSuccessResetsFailureCounter:
    """Test that successful calls reset the failure counter."""

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self) -> None:
        """Successful call should reset failure counter to 0."""
        client = TEIClient("http://test:80", max_retries=1)

        with patch("httpx.AsyncClient.post") as mock_post:
            # Simulate some failures (but not enough to open circuit)
            mock_post.side_effect = httpx.ConnectError("Connection failed")
            with pytest.raises(httpx.ConnectError):
                await client.embed_single("test text")

            assert client.circuit_breaker.failure_count == 1

            # Now simulate success
            mock_response = MagicMock()
            mock_response.json.return_value = [[[0.1] * 1024]]
            mock_response.raise_for_status = MagicMock()
            mock_post.side_effect = None
            mock_post.return_value = mock_response

            await client.embed_single("test text")

            # Failure count should be reset
            assert client.circuit_breaker.failure_count == 0


class TestIntegrationPreservesExistingFunctionality:
    """Test that circuit breaker integration preserves all TEI client features."""

    @pytest.mark.asyncio
    async def test_preserves_dimension_validation(self) -> None:
        """Circuit breaker should not interfere with dimension validation."""
        client = TEIClient("http://test:80", dimensions=512)

        with patch("httpx.AsyncClient.post") as mock_post:
            # Mock response with wrong dimensions
            mock_response = MagicMock()
            mock_response.json.return_value = [[[0.1] * 1024]]  # 1024 instead of 512
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            # Should raise ValueError for dimension mismatch
            with pytest.raises(ValueError, match="Expected 512 dimensions"):
                await client.embed_single("test text")

    @pytest.mark.asyncio
    async def test_preserves_empty_text_validation(self) -> None:
        """Circuit breaker should not interfere with empty text validation."""
        client = TEIClient("http://test:80")

        # Should raise ValueError before even checking circuit breaker
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await client.embed_single("")

    @pytest.mark.asyncio
    async def test_preserves_batch_size_validation(self) -> None:
        """Circuit breaker should not interfere with batch size validation."""
        client = TEIClient("http://test:80", batch_size_limit=10)

        # Should raise ValueError before checking circuit breaker
        with pytest.raises(ValueError, match="Batch size exceeds limit"):
            await client.embed_batch(["text"] * 11)

    @pytest.mark.asyncio
    async def test_preserves_retry_logic(self) -> None:
        """Circuit breaker should work with existing retry logic."""
        client = TEIClient("http://test:80", max_retries=3)

        with patch("httpx.AsyncClient.post") as mock_post:
            # Mock transient failure followed by success
            mock_response = MagicMock()
            mock_response.json.return_value = [[[0.1] * 1024]]
            mock_response.raise_for_status = MagicMock()

            # Fail twice, then succeed
            mock_post.side_effect = [
                httpx.ConnectError("Connection failed"),
                httpx.ConnectError("Connection failed"),
                mock_response
            ]

            # Should eventually succeed after retries
            result = await client.embed_single("test text")
            assert len(result) == 1024

            # Should have been called 3 times (2 failures + 1 success)
            assert mock_post.call_count == 3
