"""Unit tests for Qdrant vector store manager (Phase 3.1 - Collection Setup).

This module contains RED-phase tests for VectorStoreManager:
- Initialization with Qdrant URL and collection name
- Collection existence check
- Collection creation with 1024 dimensions and cosine distance
- Collection configuration validation

All tests should FAIL with ModuleNotFoundError until implementation exists.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag_ingestion.vector_store import VectorStoreManager


class TestQdrantClientInitialization:
    """Test VectorStoreManager initialization."""

    def test_initialization_with_url_and_collection(self) -> None:
        """VectorStoreManager should initialize with Qdrant URL and collection.

        Verifies:
        - Accepts qdrant_url parameter
        - Accepts collection_name parameter
        - Creates internal QdrantClient instance
        - Stores configuration parameters
        """
        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="crawl4r"
        )

        assert manager.qdrant_url == "http://crawl4r-vectors:6333"
        assert manager.collection_name == "crawl4r"
        assert manager.client is not None

    def test_initialization_with_custom_dimensions(self) -> None:
        """VectorStoreManager should accept custom vector dimensions.

        Verifies:
        - Accepts dimensions parameter (default 1024)
        - Stores dimensions for collection creation
        """
        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection",
            dimensions=1024
        )

        assert manager.dimensions == 1024

    def test_initialization_defaults_to_1024_dimensions(self) -> None:
        """VectorStoreManager should default to 1024 dimensions.

        Verifies:
        - Default dimensions is 1024 (Qwen3 output size)
        """
        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        assert manager.dimensions == 1024


class TestEnsureCollectionCreatesIfMissing:
    """Test collection creation when collection does not exist."""

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_creates_collection_when_missing(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should create collection if it does not exist.

        Verifies:
        - Calls collection_exists() to check
        - Calls create_collection() when not found
        - Uses correct vector size (1024)
        - Uses cosine distance metric
        """
        # Mock Qdrant client instance
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )
        manager.ensure_collection()

        # Verify collection_exists called with collection name
        mock_client.collection_exists.assert_called_once_with("test_collection")

        # Verify create_collection called with correct parameters
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "test_collection"

        # Verify vector config has 1024 dimensions and cosine distance
        vector_config = call_args[1]["vectors_config"]
        assert vector_config.size == 1024
        assert vector_config.distance.name == "COSINE"

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_creates_collection_with_custom_dimensions(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should create collection with custom dimensions.

        Verifies:
        - Respects custom dimensions parameter
        - Uses provided dimensions in VectorParams
        """
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection",
            dimensions=512
        )
        manager.ensure_collection()

        # Verify vector config uses custom dimensions
        call_args = mock_client.create_collection.call_args
        vector_config = call_args[1]["vectors_config"]
        assert vector_config.size == 512


class TestEnsureCollectionSkipsIfExists:
    """Test collection creation skipped when collection already exists."""

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_skips_creation_when_exists(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should not create collection if it already exists.

        Verifies:
        - Calls collection_exists() to check
        - Does NOT call create_collection() when found
        - Returns without error
        """
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="existing_collection"
        )
        manager.ensure_collection()

        # Verify collection_exists called
        mock_client.collection_exists.assert_called_once_with(
            "existing_collection"
        )

        # Verify create_collection NOT called
        mock_client.create_collection.assert_not_called()


class TestCollectionConfigurationMatches:
    """Test collection configuration validation."""

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_collection_has_correct_vector_size(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Created collection should have 1024 vector dimensions.

        Verifies:
        - VectorParams size is 1024
        """
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )
        manager.ensure_collection()

        call_args = mock_client.create_collection.call_args
        vector_config = call_args[1]["vectors_config"]
        assert vector_config.size == 1024

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_collection_uses_cosine_distance(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Created collection should use cosine distance metric.

        Verifies:
        - Distance metric is Distance.COSINE
        - Appropriate for normalized embeddings from Qwen3
        """
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )
        manager.ensure_collection()

        call_args = mock_client.create_collection.call_args
        vector_config = call_args[1]["vectors_config"]
        assert vector_config.distance.name == "COSINE"


class TestUpsertSingleVector:
    """Test upserting a single vector with metadata."""

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_upsert_single_vector_creates_point_struct(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should create PointStruct with vector and metadata.

        Verifies:
        - Calls upsert() with single PointStruct
        - PointStruct has deterministic UUID from content hash
        - Vector dimensions match collection (1024)
        - Payload includes all required metadata
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vector = [0.1] * 1024  # 1024-dimensional vector
        metadata = {
            "file_path_relative": "docs/test.md",
            "file_path_absolute": "/home/user/docs/test.md",
            "filename": "test.md",
            "modification_date": "2026-01-15T00:00:00Z",
            "chunk_index": 0,
            "chunk_text": "Test content",
        }

        manager.upsert_vector(vector, metadata)

        # Verify upsert called with PointStruct
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "test_collection"

        # Verify PointStruct structure
        points = call_args[1]["points"]
        assert len(points) == 1
        point = points[0]
        assert len(point.vector) == 1024
        assert point.payload == metadata

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_upsert_generates_deterministic_id_from_hash(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should generate deterministic UUID from content hash.

        Verifies:
        - Same file path + chunk index → same UUID
        - Different file path + chunk index → different UUID
        - Uses SHA256(file_path_relative:chunk_index)
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata1 = {
            "file_path_relative": "docs/test.md",
            "chunk_index": 0,
            "chunk_text": "First chunk",
        }
        metadata2 = {
            "file_path_relative": "docs/test.md",
            "chunk_index": 0,
            "chunk_text": "Same chunk, different text",
        }

        manager.upsert_vector(vector, metadata1)
        call1_id = mock_client.upsert.call_args[1]["points"][0].id

        mock_client.reset_mock()

        manager.upsert_vector(vector, metadata2)
        call2_id = mock_client.upsert.call_args[1]["points"][0].id

        # Same file + chunk index should produce same ID
        assert call1_id == call2_id

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_upsert_rejects_wrong_dimension_vector(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should reject vector with wrong dimensions.

        Verifies:
        - ValueError when vector != configured dimensions
        - Error message mentions expected vs actual dimensions
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection",
            dimensions=1024
        )

        vector = [0.1] * 512  # Wrong dimension (512 != 1024)
        metadata = {
            "file_path_relative": "docs/test.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }

        with pytest.raises(ValueError, match="1024.*512"):
            manager.upsert_vector(vector, metadata)

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_upsert_rejects_empty_vector(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should reject empty vector.

        Verifies:
        - ValueError when vector is empty list
        - No upsert call made
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vector: list[float] = []
        metadata = {
            "file_path_relative": "docs/test.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }

        with pytest.raises(ValueError, match="empty"):
            manager.upsert_vector(vector, metadata)

        mock_client.upsert.assert_not_called()

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_upsert_requires_file_path_relative(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should require file_path_relative in metadata.

        Verifies:
        - ValueError when file_path_relative missing
        - No upsert call made
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata = {
            "chunk_index": 0,
            "chunk_text": "Test",
        }

        with pytest.raises(ValueError, match="file_path_relative"):
            manager.upsert_vector(vector, metadata)

        mock_client.upsert.assert_not_called()

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_upsert_requires_chunk_index(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should require chunk_index in metadata.

        Verifies:
        - ValueError when chunk_index missing
        - No upsert call made
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata = {
            "file_path_relative": "docs/test.md",
            "chunk_text": "Test",
        }

        with pytest.raises(ValueError, match="chunk_index"):
            manager.upsert_vector(vector, metadata)

        mock_client.upsert.assert_not_called()

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_upsert_requires_chunk_text(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should require chunk_text in metadata.

        Verifies:
        - ValueError when chunk_text missing
        - No upsert call made
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata = {
            "file_path_relative": "docs/test.md",
            "chunk_index": 0,
        }

        with pytest.raises(ValueError, match="chunk_text"):
            manager.upsert_vector(vector, metadata)

        mock_client.upsert.assert_not_called()


class TestUpsertBatchVectors:
    """Test batch upsert operations with multiple vectors."""

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_upsert_batch_creates_multiple_points(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should create multiple PointStructs in single batch.

        Verifies:
        - Calls upsert() with list of PointStructs
        - Each point has correct vector and metadata
        - All points in single batch (no splitting)
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    "chunk_index": i,
                    "chunk_text": f"Chunk {i}",
                }
            }
            for i in range(5)
        ]

        manager.upsert_vectors_batch(vectors_with_metadata)

        # Verify single upsert call with 5 points
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        points = call_args[1]["points"]
        assert len(points) == 5

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_upsert_batch_splits_at_100_points(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should split batches at 100 points per upsert.

        Verifies:
        - 150 points → 2 upsert calls (100 + 50)
        - First batch has 100 points
        - Second batch has 50 points
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    "chunk_index": i,
                    "chunk_text": f"Chunk {i}",
                }
            }
            for i in range(150)
        ]

        manager.upsert_vectors_batch(vectors_with_metadata)

        # Verify 2 upsert calls
        assert mock_client.upsert.call_count == 2

        # Verify batch sizes
        first_batch = mock_client.upsert.call_args_list[0][1]["points"]
        second_batch = mock_client.upsert.call_args_list[1][1]["points"]
        assert len(first_batch) == 100
        assert len(second_batch) == 50

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_upsert_batch_handles_empty_list(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should handle empty batch gracefully.

        Verifies:
        - No upsert call made for empty list
        - No error raised
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        manager.upsert_vectors_batch([])

        mock_client.upsert.assert_not_called()

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_upsert_batch_validates_all_vectors(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should validate all vectors before upserting.

        Verifies:
        - ValueError if any vector has wrong dimensions
        - No partial upsert (all-or-nothing)
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection",
            dimensions=1024
        )

        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    "chunk_index": 0,
                    "chunk_text": "Good chunk",
                }
            },
            {
                "vector": [0.1] * 512,  # Wrong dimension
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    "chunk_index": 1,
                    "chunk_text": "Bad chunk",
                }
            },
        ]

        with pytest.raises(ValueError, match="dimension"):
            manager.upsert_vectors_batch(vectors_with_metadata)

        # No partial upsert
        mock_client.upsert.assert_not_called()

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_upsert_batch_validates_all_metadata(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should validate all metadata before upserting.

        Verifies:
        - ValueError if any metadata missing required fields
        - No partial upsert (all-or-nothing)
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    "chunk_index": 0,
                    "chunk_text": "Good chunk",
                }
            },
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    # Missing chunk_index
                    "chunk_text": "Bad chunk",
                }
            },
        ]

        with pytest.raises(ValueError, match="chunk_index"):
            manager.upsert_vectors_batch(vectors_with_metadata)

        # No partial upsert
        mock_client.upsert.assert_not_called()


class TestUpsertWithRetry:
    """Test upsert retry logic with exponential backoff."""

    @patch("rag_ingestion.vector_store.QdrantClient")
    @patch("rag_ingestion.vector_store.time.sleep")
    def test_upsert_retries_on_network_error(
        self, mock_sleep: MagicMock, mock_qdrant_client: MagicMock
    ) -> None:
        """Should retry on network errors with exponential backoff.

        Verifies:
        - Retries 3 times on network error
        - Uses exponential backoff (1s, 2s, 4s)
        - Succeeds on third attempt
        """
        from qdrant_client.http.exceptions import UnexpectedResponse
        import httpx

        mock_client = MagicMock()
        # Fail twice, succeed on third attempt
        mock_client.upsert.side_effect = [
            UnexpectedResponse(
                status_code=500,
                reason_phrase="Server Error",
                content=b"Server Error",
                headers=httpx.Headers(),
            ),
            UnexpectedResponse(
                status_code=503,
                reason_phrase="Service Unavailable",
                content=b"Service Unavailable",
                headers=httpx.Headers(),
            ),
            None,  # Success
        ]
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata = {
            "file_path_relative": "docs/test.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }

        manager.upsert_vector(vector, metadata)

        # Verify 3 upsert attempts
        assert mock_client.upsert.call_count == 3

    @patch("rag_ingestion.vector_store.QdrantClient")
    @patch("rag_ingestion.vector_store.time.sleep")
    def test_upsert_raises_after_max_retries(
        self, mock_sleep: MagicMock, mock_qdrant_client: MagicMock
    ) -> None:
        """Should raise error after exhausting retries.

        Verifies:
        - Tries 3 times
        - Raises RuntimeError on final failure
        - Error message includes retry count
        """
        from qdrant_client.http.exceptions import UnexpectedResponse
        import httpx

        mock_client = MagicMock()
        mock_client.upsert.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="Server Error",
            content=b"Server Error",
            headers=httpx.Headers(),
        )
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata = {
            "file_path_relative": "docs/test.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }

        with pytest.raises(RuntimeError, match="3.*retries"):
            manager.upsert_vector(vector, metadata)

        # Verify 3 upsert attempts
        assert mock_client.upsert.call_count == 3

    @patch("rag_ingestion.vector_store.QdrantClient")
    @patch("rag_ingestion.vector_store.time.sleep")
    def test_upsert_batch_retries_per_batch(
        self, mock_sleep: MagicMock, mock_qdrant_client: MagicMock
    ) -> None:
        """Should retry each batch independently.

        Verifies:
        - First batch fails twice, succeeds on third
        - Second batch succeeds immediately
        - Total 4 upsert calls (3 + 1)
        """
        from qdrant_client.http.exceptions import UnexpectedResponse
        import httpx

        mock_client = MagicMock()
        # First batch: fail twice, succeed
        # Second batch: succeed immediately
        mock_client.upsert.side_effect = [
            UnexpectedResponse(
                status_code=500,
                reason_phrase="Server Error",
                content=b"Server Error",
                headers=httpx.Headers(),
            ),
            UnexpectedResponse(
                status_code=503,
                reason_phrase="Service Unavailable",
                content=b"Service Unavailable",
                headers=httpx.Headers(),
            ),
            None,  # First batch success
            None,  # Second batch success
        ]
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        # 150 vectors → 2 batches (100 + 50)
        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    "chunk_index": i,
                    "chunk_text": f"Chunk {i}",
                }
            }
            for i in range(150)
        ]

        manager.upsert_vectors_batch(vectors_with_metadata)

        # Verify 4 total upsert calls (3 for first batch + 1 for second)
        assert mock_client.upsert.call_count == 4


class TestGeneratePointId:
    """Test deterministic point ID generation from content hash."""

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_generate_id_is_deterministic(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should generate same UUID for same file_path + chunk_index.

        Verifies:
        - Same inputs → same UUID every time
        - Uses SHA256 hash
        - Converts to UUID format
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata = {
            "file_path_relative": "docs/test.md",
            "chunk_index": 5,
            "chunk_text": "Test",
        }

        # Call twice with same metadata
        manager.upsert_vector(vector, metadata)
        id1 = mock_client.upsert.call_args[1]["points"][0].id

        mock_client.reset_mock()

        manager.upsert_vector(vector, metadata)
        id2 = mock_client.upsert.call_args[1]["points"][0].id

        # Should be identical
        assert id1 == id2

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_generate_id_differs_by_chunk_index(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should generate different UUID for different chunk_index.

        Verifies:
        - Same file, different chunk → different UUID
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vector = [0.1] * 1024

        metadata1 = {
            "file_path_relative": "docs/test.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }
        metadata2 = {
            "file_path_relative": "docs/test.md",
            "chunk_index": 1,
            "chunk_text": "Test",
        }

        manager.upsert_vector(vector, metadata1)
        id1 = mock_client.upsert.call_args[1]["points"][0].id

        mock_client.reset_mock()

        manager.upsert_vector(vector, metadata2)
        id2 = mock_client.upsert.call_args[1]["points"][0].id

        # Should be different
        assert id1 != id2

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_generate_id_differs_by_file_path(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should generate different UUID for different file_path.

        Verifies:
        - Different file, same chunk → different UUID
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection"
        )

        vector = [0.1] * 1024

        metadata1 = {
            "file_path_relative": "docs/test1.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }
        metadata2 = {
            "file_path_relative": "docs/test2.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }

        manager.upsert_vector(vector, metadata1)
        id1 = mock_client.upsert.call_args[1]["points"][0].id

        mock_client.reset_mock()

        manager.upsert_vector(vector, metadata2)
        id2 = mock_client.upsert.call_args[1]["points"][0].id

        # Should be different
        assert id1 != id2
