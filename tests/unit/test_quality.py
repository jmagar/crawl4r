"""Tests for quality verification module.

Tests startup validation including TEI connection checks and dimension verification.
"""

from unittest.mock import AsyncMock, patch

import pytest


class TestTEIConnectionValidation:
    """Test TEI connection validation during startup."""

    @pytest.mark.asyncio
    async def test_validate_tei_connection(self) -> None:
        """Verify successful TEI connection validation passes."""
        # Import will fail since module doesn't exist yet
        from rag_ingestion.quality import QualityVerifier

        # Mock TEI client with successful embedding response
        tei_client = AsyncMock()
        tei_client.embed_single.return_value = [0.1] * 1024

        # Create verifier and validate connection
        verifier = QualityVerifier()
        result = await verifier.validate_tei_connection(tei_client)

        # Verify validation passed
        assert result is True
        tei_client.embed_single.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_validate_tei_retries(self) -> None:
        """Verify TEI validation retries with exponential backoff."""
        from rag_ingestion.quality import QualityVerifier

        # Mock TEI client: 2 failures, then success
        tei_client = AsyncMock()
        tei_client.embed_single.side_effect = [
            RuntimeError("Connection failed"),
            RuntimeError("Connection failed"),
            [0.1] * 1024,  # Success on 3rd attempt
        ]

        # Mock asyncio.sleep to avoid actual delays
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            verifier = QualityVerifier()
            result = await verifier.validate_tei_connection(tei_client)

            # Verify succeeded after retries
            assert result is True
            assert tei_client.embed_single.call_count == 3

            # Verify retry delays: 5s, 10s (successful on 3rd, no 3rd delay)
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(5)  # First retry delay
            mock_sleep.assert_any_call(10)  # Second retry delay

    @pytest.mark.asyncio
    async def test_validate_tei_exits_on_failure(self) -> None:
        """Verify validation exits with code 1 after max retries."""
        from rag_ingestion.quality import QualityVerifier

        # Mock TEI client: all attempts fail
        tei_client = AsyncMock()
        tei_client.embed_single.side_effect = RuntimeError("Connection failed")

        # Mock sys.exit to capture exit call
        with patch("sys.exit") as mock_exit, patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            verifier = QualityVerifier()
            await verifier.validate_tei_connection(tei_client)

            # Verify sys.exit(1) was called
            mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_validate_tei_checks_dimensions(self) -> None:
        """Verify validation checks embedding dimensions match expected."""
        from rag_ingestion.quality import QualityVerifier

        # Mock TEI client with correct 1024-dimensional embedding
        tei_client = AsyncMock()
        tei_client.embed_single.return_value = [0.1] * 1024

        verifier = QualityVerifier()
        result = await verifier.validate_tei_connection(tei_client)

        # Verify validation passed with correct dimensions
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_tei_rejects_wrong_dimensions(self) -> None:
        """Verify validation fails with wrong embedding dimensions."""
        from rag_ingestion.quality import QualityVerifier

        # Mock TEI client with wrong 768-dimensional embedding
        tei_client = AsyncMock()
        tei_client.embed_single.return_value = [0.1] * 768

        verifier = QualityVerifier()

        # Verify validation raises error for wrong dimensions
        with pytest.raises(ValueError, match="dimension"):
            await verifier.validate_tei_connection(tei_client)
