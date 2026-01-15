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


class TestQdrantConnectionValidation:
    """Test Qdrant connection validation during startup."""

    @pytest.mark.asyncio
    async def test_validate_qdrant_connection(self) -> None:
        """Verify successful Qdrant connection validation passes."""
        from rag_ingestion.quality import QualityVerifier

        # Mock vector store with successful collection info
        vector_store = AsyncMock()
        vector_store.get_collection_info.return_value = {
            "vector_size": 1024,
            "distance": "Cosine",
        }

        # Create verifier and validate connection
        verifier = QualityVerifier()
        result = await verifier.validate_qdrant_connection(vector_store)

        # Verify validation passed
        assert result is True
        vector_store.get_collection_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_qdrant_retries(self) -> None:
        """Verify Qdrant validation retries with exponential backoff."""
        from rag_ingestion.quality import QualityVerifier

        # Mock vector store: 2 failures, then success
        vector_store = AsyncMock()
        vector_store.get_collection_info.side_effect = [
            RuntimeError("Connection failed"),
            RuntimeError("Connection failed"),
            {"vector_size": 1024, "distance": "Cosine"},  # Success on 3rd
        ]

        # Mock asyncio.sleep to avoid actual delays
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            verifier = QualityVerifier()
            result = await verifier.validate_qdrant_connection(vector_store)

            # Verify succeeded after retries
            assert result is True
            assert vector_store.get_collection_info.call_count == 3

            # Verify retry delays: 5s, 10s (successful on 3rd, no 3rd delay)
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(5)  # First retry delay
            mock_sleep.assert_any_call(10)  # Second retry delay

    @pytest.mark.asyncio
    async def test_validate_qdrant_exits_on_failure(self) -> None:
        """Verify validation exits with code 1 after max retries."""
        from rag_ingestion.quality import QualityVerifier

        # Mock vector store: all attempts fail
        vector_store = AsyncMock()
        vector_store.get_collection_info.side_effect = RuntimeError(
            "Connection failed"
        )

        # Mock sys.exit to capture exit call
        with patch("sys.exit") as mock_exit, patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            verifier = QualityVerifier()
            await verifier.validate_qdrant_connection(vector_store)

            # Verify sys.exit(1) was called
            mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_validate_qdrant_checks_dimensions(self) -> None:
        """Verify validation checks vector size matches expected."""
        from rag_ingestion.quality import QualityVerifier

        # Mock vector store with correct 1024-dimensional vectors
        vector_store = AsyncMock()
        vector_store.get_collection_info.return_value = {
            "vector_size": 1024,
            "distance": "Cosine",
        }

        verifier = QualityVerifier()
        result = await verifier.validate_qdrant_connection(vector_store)

        # Verify validation passed with correct dimensions
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_qdrant_rejects_wrong_dimensions(self) -> None:
        """Verify validation fails with wrong vector dimensions."""
        from rag_ingestion.quality import QualityVerifier

        # Mock vector store with wrong 768-dimensional vectors
        vector_store = AsyncMock()
        vector_store.get_collection_info.return_value = {
            "vector_size": 768,
            "distance": "Cosine",
        }

        verifier = QualityVerifier()

        # Verify validation raises error for wrong dimensions
        with pytest.raises(ValueError, match="dimension"):
            await verifier.validate_qdrant_connection(vector_store)


class TestRuntimeQualityChecks:
    """Test runtime quality checks for embeddings."""

    def test_check_embedding_dimensions(self) -> None:
        """Verify dimension checking passes for correct dimensions."""
        from rag_ingestion.quality import QualityVerifier

        verifier = QualityVerifier()
        embedding = [0.1] * 1024

        # Should not raise for correct dimensions
        verifier.check_embedding_dimensions(embedding)

    def test_check_embedding_rejects_wrong_dims(self) -> None:
        """Verify dimension checking rejects wrong dimensions."""
        from rag_ingestion.quality import QualityVerifier

        verifier = QualityVerifier()
        embedding = [0.1] * 512  # Wrong dimensions

        # Should raise ValueError
        with pytest.raises(ValueError, match="dimension"):
            verifier.check_embedding_dimensions(embedding)

    def test_sample_embeddings_for_normalization(self) -> None:
        """Verify 5% sampling of embeddings."""
        from rag_ingestion.quality import QualityVerifier

        verifier = QualityVerifier()
        embeddings = [[0.1] * 1024 for _ in range(100)]

        # Sample 5% (default)
        sampled = verifier.sample_embeddings(embeddings)

        # Should return approximately 5 embeddings (5% of 100)
        assert len(sampled) == 5

    def test_check_normalization(self) -> None:
        """Verify normalization checking passes for L2-normalized embedding."""
        from rag_ingestion.quality import QualityVerifier

        verifier = QualityVerifier()
        # Create L2-normalized embedding (norm = 1.0)
        import math

        n = 1024
        embedding = [1.0 / math.sqrt(n)] * n

        # Should pass without warnings
        verifier.check_normalization(embedding)

    def test_check_normalization_with_tolerance(self) -> None:
        """Verify normalization checking passes within tolerance."""
        from rag_ingestion.quality import QualityVerifier

        verifier = QualityVerifier()
        # Create embedding with norm slightly above 1.0 (within ±0.01 tolerance)
        import math

        n = 1024
        # Scale to get norm = 1.008 (within 0.01 tolerance)
        embedding = [1.008 / math.sqrt(n)] * n

        # Should pass within tolerance
        verifier.check_normalization(embedding, tolerance=0.01)

    def test_check_normalization_warns(self) -> None:
        """Verify normalization checking logs warning for non-normalized."""
        from rag_ingestion.quality import QualityVerifier

        verifier = QualityVerifier()
        # Create embedding with norm = 0.9 (outside tolerance)
        import math

        n = 1024
        embedding = [0.9 / math.sqrt(n)] * n

        # Should log warning
        with patch.object(verifier.logger, "warning") as mock_warning:
            verifier.check_normalization(embedding)

            # Verify warning was logged
            mock_warning.assert_called_once()
            call_args = mock_warning.call_args[0][0]
            assert "not L2-normalized" in call_args or "norm" in call_args
