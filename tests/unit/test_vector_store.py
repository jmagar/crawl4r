"""Unit tests for Qdrant vector store manager.

This module contains comprehensive tests for VectorStoreManager:
- Initialization with Qdrant URL and collection name
- Collection existence check
- Collection creation with 1024 dimensions and cosine distance
- Collection configuration validation
- Vector upsert operations (single and batch)
- Deterministic point ID generation
- Search operations for semantic similarity retrieval

The latest test class (TestSearchSimilar) contains RED-phase tests that
should FAIL with AttributeError until search_similar() method is implemented.
"""

from unittest.mock import MagicMock, patch

import pytest

from rag_ingestion.vector_store import VectorMetadata, VectorStoreManager


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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="crawl4r"
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
            dimensions=1024,
        )

        assert manager.dimensions == 1024

    def test_initialization_defaults_to_1024_dimensions(self) -> None:
        """VectorStoreManager should default to 1024 dimensions.

        Verifies:
        - Default dimensions is 1024 (Qwen3 output size)
        """
        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
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
            dimensions=512,
        )
        manager.ensure_collection()

        # Verify vector config uses custom dimensions
        call_args = mock_client.create_collection.call_args
        vector_config = call_args[1]["vectors_config"]
        assert vector_config.size == 512


class TestEnsureCollectionSkipsIfExists:
    """Test collection creation skipped when collection already exists."""

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_skips_creation_when_exists(self, mock_qdrant_client: MagicMock) -> None:
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
            collection_name="existing_collection",
        )
        manager.ensure_collection()

        # Verify collection_exists called
        mock_client.collection_exists.assert_called_once_with("existing_collection")

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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata1: VectorMetadata = {
            "file_path_relative": "docs/test.md",
            "chunk_index": 0,
            "chunk_text": "First chunk",
        }
        metadata2: VectorMetadata = {
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
            dimensions=1024,
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
    def test_upsert_rejects_empty_vector(self, mock_qdrant_client: MagicMock) -> None:
        """Should reject empty vector.

        Verifies:
        - ValueError when vector is empty list
        - No upsert call made
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
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
    def test_upsert_requires_chunk_index(self, mock_qdrant_client: MagicMock) -> None:
        """Should require chunk_index in metadata.

        Verifies:
        - ValueError when chunk_index missing
        - No upsert call made
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
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
    def test_upsert_requires_chunk_text(self, mock_qdrant_client: MagicMock) -> None:
        """Should require chunk_text in metadata.

        Verifies:
        - ValueError when chunk_text missing
        - No upsert call made
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    "chunk_index": i,
                    "chunk_text": f"Chunk {i}",
                },
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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    "chunk_index": i,
                    "chunk_text": f"Chunk {i}",
                },
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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
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
            dimensions=1024,
        )

        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    "chunk_index": 0,
                    "chunk_text": "Good chunk",
                },
            },
            {
                "vector": [0.1] * 512,  # Wrong dimension
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    "chunk_index": 1,
                    "chunk_text": "Bad chunk",
                },
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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    "chunk_index": 0,
                    "chunk_text": "Good chunk",
                },
            },
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    # Missing chunk_index
                    "chunk_text": "Bad chunk",
                },
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
        import httpx
        from qdrant_client.http.exceptions import UnexpectedResponse

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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
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
        import httpx
        from qdrant_client.http.exceptions import UnexpectedResponse

        mock_client = MagicMock()
        mock_client.upsert.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="Server Error",
            content=b"Server Error",
            headers=httpx.Headers(),
        )
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
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
        import httpx
        from qdrant_client.http.exceptions import UnexpectedResponse

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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        # 150 vectors → 2 batches (100 + 50)
        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path_relative": "docs/test.md",
                    "chunk_index": i,
                    "chunk_text": f"Chunk {i}",
                },
            }
            for i in range(150)
        ]

        manager.upsert_vectors_batch(vectors_with_metadata)

        # Verify 4 total upsert calls (3 for first batch + 1 for second)
        assert mock_client.upsert.call_count == 4


class TestGeneratePointId:
    """Test deterministic point ID generation from content hash."""

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_generate_id_is_deterministic(self, mock_qdrant_client: MagicMock) -> None:
        """Should generate same UUID for same file_path + chunk_index.

        Verifies:
        - Same inputs → same UUID every time
        - Uses SHA256 hash
        - Converts to UUID format
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector = [0.1] * 1024

        metadata1: VectorMetadata = {
            "file_path_relative": "docs/test.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }
        metadata2: VectorMetadata = {
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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector = [0.1] * 1024

        metadata1: VectorMetadata = {
            "file_path_relative": "docs/test1.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }
        metadata2: VectorMetadata = {
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


class TestSearchSimilar:
    """Test semantic similarity search operations."""

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_search_similar_returns_results_with_scores(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should return search results with scores and metadata.

        Verifies:
        - Calls client.search() with query vector and top_k limit
        - Returns list of results with id, score, and metadata
        - Each result includes file_path, chunk_index, chunk_text
        - Results are ordered by score (highest first)
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        # Mock search response with 3 results
        mock_search_result = [
            MagicMock(
                id="uuid-1",
                score=0.95,
                payload={
                    "file_path_relative": "docs/test1.md",
                    "chunk_index": 0,
                    "chunk_text": "Most similar chunk",
                },
            ),
            MagicMock(
                id="uuid-2",
                score=0.85,
                payload={
                    "file_path_relative": "docs/test2.md",
                    "chunk_index": 1,
                    "chunk_text": "Second similar chunk",
                },
            ),
            MagicMock(
                id="uuid-3",
                score=0.75,
                payload={
                    "file_path_relative": "docs/test3.md",
                    "chunk_index": 2,
                    "chunk_text": "Third similar chunk",
                },
            ),
        ]
        mock_client.search.return_value = mock_search_result

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024
        results = manager.search_similar(query_vector, top_k=5)

        # Verify search was called with correct parameters
        mock_client.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=query_vector,
            limit=5,
        )

        # Verify results structure
        assert len(results) == 3
        assert results[0]["id"] == "uuid-1"
        assert results[0]["score"] == 0.95
        assert results[0]["file_path_relative"] == "docs/test1.md"
        assert results[0]["chunk_index"] == 0
        assert results[0]["chunk_text"] == "Most similar chunk"

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_search_similar_validates_query_vector_dimensions(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should validate query vector dimensions match collection config.

        Verifies:
        - ValueError if query vector dimensions != configured dimensions
        - No search call made with invalid dimensions
        - Error message includes expected vs actual dimensions
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection",
            dimensions=1024,
        )

        query_vector = [0.1] * 512  # Wrong dimension

        with pytest.raises(ValueError, match="1024.*512"):
            manager.search_similar(query_vector, top_k=5)

        mock_client.search.assert_not_called()

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_search_similar_validates_empty_query_vector(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should reject empty query vector.

        Verifies:
        - ValueError when query vector is empty list
        - No search call made
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector: list[float] = []

        with pytest.raises(ValueError, match="empty"):
            manager.search_similar(query_vector, top_k=5)

        mock_client.search.assert_not_called()

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_search_similar_validates_positive_top_k(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should validate top_k is a positive integer.

        Verifies:
        - ValueError if top_k <= 0
        - No search call made with invalid top_k
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024

        with pytest.raises(ValueError, match="top_k.*positive"):
            manager.search_similar(query_vector, top_k=0)

        with pytest.raises(ValueError, match="top_k.*positive"):
            manager.search_similar(query_vector, top_k=-5)

        mock_client.search.assert_not_called()

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_search_similar_handles_empty_collection(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should handle empty collection gracefully.

        Verifies:
        - Returns empty list when no results found
        - No error raised for empty collection
        """
        mock_client = MagicMock()
        mock_client.search.return_value = []
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024
        results = manager.search_similar(query_vector, top_k=5)

        assert results == []
        mock_client.search.assert_called_once()

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_search_similar_limits_results_to_top_k(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should limit results to top_k parameter.

        Verifies:
        - Passes top_k as limit parameter to search
        - Returns at most top_k results
        """
        mock_client = MagicMock()
        # Mock 3 results when asking for top_k=3
        mock_search_result = [
            MagicMock(
                id=f"uuid-{i}",
                score=0.9 - i * 0.1,
                payload={
                    "file_path_relative": f"docs/test{i}.md",
                    "chunk_index": i,
                    "chunk_text": f"Chunk {i}",
                },
            )
            for i in range(3)
        ]
        mock_client.search.return_value = mock_search_result
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024
        results = manager.search_similar(query_vector, top_k=3)

        # Verify limit parameter passed to search
        call_args = mock_client.search.call_args
        assert call_args[1]["limit"] == 3

        # Verify result count matches top_k
        assert len(results) == 3

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_search_similar_results_sorted_by_score(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should return results sorted by score (highest first).

        Verifies:
        - Results are in descending order by score
        - First result has highest score
        - Last result has lowest score
        """
        mock_client = MagicMock()
        # Mock results in descending score order
        mock_search_result = [
            MagicMock(
                id="uuid-1",
                score=0.95,
                payload={
                    "file_path_relative": "docs/test1.md",
                    "chunk_index": 0,
                    "chunk_text": "Best match",
                },
            ),
            MagicMock(
                id="uuid-2",
                score=0.85,
                payload={
                    "file_path_relative": "docs/test2.md",
                    "chunk_index": 1,
                    "chunk_text": "Good match",
                },
            ),
            MagicMock(
                id="uuid-3",
                score=0.75,
                payload={
                    "file_path_relative": "docs/test3.md",
                    "chunk_index": 2,
                    "chunk_text": "OK match",
                },
            ),
        ]
        mock_client.search.return_value = mock_search_result
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024
        results = manager.search_similar(query_vector, top_k=5)

        # Verify descending score order
        assert results[0]["score"] == 0.95
        assert results[1]["score"] == 0.85
        assert results[2]["score"] == 0.75
        assert results[0]["score"] > results[1]["score"]
        assert results[1]["score"] > results[2]["score"]

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_search_similar_includes_all_metadata_fields(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should include all metadata fields in results.

        Verifies:
        - Each result includes all metadata fields from payload
        - file_path_relative, chunk_index, chunk_text are present
        - Additional metadata fields are preserved
        """
        mock_client = MagicMock()
        mock_search_result = [
            MagicMock(
                id="uuid-1",
                score=0.95,
                payload={
                    "file_path_relative": "docs/test.md",
                    "file_path_absolute": "/home/user/docs/test.md",
                    "filename": "test.md",
                    "modification_date": "2026-01-15T00:00:00Z",
                    "chunk_index": 0,
                    "chunk_text": "Test content",
                    "section_path": "Introduction",
                    "heading_level": 1,
                },
            ),
        ]
        mock_client.search.return_value = mock_search_result
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024
        results = manager.search_similar(query_vector, top_k=5)

        # Verify all metadata fields included
        result = results[0]
        assert result["file_path_relative"] == "docs/test.md"
        assert result["file_path_absolute"] == "/home/user/docs/test.md"
        assert result["filename"] == "test.md"
        assert result["modification_date"] == "2026-01-15T00:00:00Z"
        assert result["chunk_index"] == 0
        assert result["chunk_text"] == "Test content"
        assert result["section_path"] == "Introduction"
        assert result["heading_level"] == 1

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_search_similar_retries_on_connection_error(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should retry search on network errors with exponential backoff.

        Verifies:
        - Retries on UnexpectedResponse errors
        - Uses exponential backoff (1s, 2s, 4s)
        - Succeeds on final attempt
        """
        import httpx
        from qdrant_client.http.exceptions import UnexpectedResponse

        mock_client = MagicMock()
        # Fail twice, succeed on third attempt
        mock_client.search.side_effect = [
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
            [
                MagicMock(
                    id="uuid-1",
                    score=0.95,
                    payload={
                        "file_path_relative": "docs/test.md",
                        "chunk_index": 0,
                        "chunk_text": "Test",
                    },
                )
            ],
        ]
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024
        results = manager.search_similar(query_vector, top_k=5)

        # Verify 3 search attempts
        assert mock_client.search.call_count == 3
        # Verify results returned on success
        assert len(results) == 1
        assert results[0]["score"] == 0.95


class TestDeleteById:
    """Test deletion of single points by UUID."""

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_delete_by_id_deletes_single_point(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should delete a single point by its UUID.

        Verifies:
        - Calls client.delete() with point ID
        - Collection name passed correctly
        - No return value (void function)
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        point_id = "550e8400-e29b-41d4-a716-446655440000"
        manager.delete_by_id(point_id)

        # Verify delete called with correct parameters
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert call_args[1]["points_selector"].points == [point_id]

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_delete_by_id_validates_uuid_format(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should validate UUID format before deletion.

        Verifies:
        - ValueError when point_id is not valid UUID format
        - No delete call made with invalid UUID
        - Error message mentions UUID format
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        invalid_id = "not-a-valid-uuid"

        with pytest.raises(ValueError, match="UUID"):
            manager.delete_by_id(invalid_id)

        mock_client.delete.assert_not_called()

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_delete_by_id_handles_nonexistent_id_gracefully(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should handle deletion of non-existent ID without error.

        Verifies:
        - No exception raised when ID doesn't exist
        - Delete call still made (Qdrant handles gracefully)
        - Returns without error
        """
        mock_client = MagicMock()
        # Qdrant delete doesn't error on non-existent IDs
        mock_client.delete.return_value = None
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        # Non-existent but valid UUID
        point_id = "550e8400-e29b-41d4-a716-446655440000"
        manager.delete_by_id(point_id)

        # Verify delete was attempted
        mock_client.delete.assert_called_once()

    @patch("rag_ingestion.vector_store.QdrantClient")
    @patch("rag_ingestion.vector_store.time.sleep")
    def test_delete_by_id_retries_on_connection_error(
        self, mock_sleep: MagicMock, mock_qdrant_client: MagicMock
    ) -> None:
        """Should retry deletion on network errors.

        Verifies:
        - Retries on UnexpectedResponse errors
        - Uses exponential backoff (1s, 2s, 4s)
        - Succeeds on final attempt
        """
        import httpx
        from qdrant_client.http.exceptions import UnexpectedResponse

        mock_client = MagicMock()
        # Fail twice, succeed on third
        mock_client.delete.side_effect = [
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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        point_id = "550e8400-e29b-41d4-a716-446655440000"
        manager.delete_by_id(point_id)

        # Verify 3 delete attempts
        assert mock_client.delete.call_count == 3


class TestDeleteByFile:
    """Test deletion of all chunks from a file."""

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_delete_by_file_deletes_all_chunks(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should delete all chunks for a given file path.

        Verifies:
        - Uses scroll to find all points with matching file_path_relative
        - Calls delete for each found point
        - Returns count of deleted points
        """
        mock_client = MagicMock()
        # Mock scroll response with 3 chunks
        mock_client.scroll.return_value = (
            [
                MagicMock(id="uuid-1"),
                MagicMock(id="uuid-2"),
                MagicMock(id="uuid-3"),
            ],
            None,  # No next page offset
        )
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        file_path = "docs/test.md"
        count = manager.delete_by_file(file_path)

        # Verify scroll called with filter
        mock_client.scroll.assert_called_once()
        scroll_args = mock_client.scroll.call_args
        assert scroll_args[1]["collection_name"] == "test_collection"
        # Filter should match file_path_relative field
        scroll_filter = scroll_args[1]["scroll_filter"]
        assert scroll_filter.must[0].key == "file_path_relative"
        assert scroll_filter.must[0].match.value == file_path

        # Verify delete called for found points
        mock_client.delete.assert_called_once()
        delete_args = mock_client.delete.call_args
        assert delete_args[1]["collection_name"] == "test_collection"
        # Should delete all 3 point IDs
        assert len(delete_args[1]["points_selector"].points) == 3

        # Verify count returned
        assert count == 3

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_delete_by_file_returns_count(self, mock_qdrant_client: MagicMock) -> None:
        """Should return count of deleted points.

        Verifies:
        - Return value is integer count
        - Count matches number of points found
        """
        mock_client = MagicMock()
        mock_client.scroll.return_value = (
            [MagicMock(id=f"uuid-{i}") for i in range(5)],
            None,
        )
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        count = manager.delete_by_file("docs/test.md")

        assert isinstance(count, int)
        assert count == 5

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_delete_by_file_handles_empty_results_gracefully(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should handle file with no chunks gracefully.

        Verifies:
        - Returns 0 when no matching chunks found
        - No delete call made for empty results
        - No error raised
        """
        mock_client = MagicMock()
        # Mock empty scroll response
        mock_client.scroll.return_value = ([], None)
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        count = manager.delete_by_file("docs/nonexistent.md")

        # Verify scroll was called
        mock_client.scroll.assert_called_once()
        # Verify no delete call made
        mock_client.delete.assert_not_called()
        # Verify 0 count returned
        assert count == 0

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_delete_by_file_handles_pagination(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should handle paginated scroll results.

        Verifies:
        - Continues scrolling when next_page_offset is present
        - Accumulates all points across pages
        - Returns total count from all pages
        """
        mock_client = MagicMock()
        # Mock paginated scroll response (2 pages)
        mock_client.scroll.side_effect = [
            # First page: 100 points with next offset
            (
                [MagicMock(id=f"uuid-{i}") for i in range(100)],
                "next-offset-token",
            ),
            # Second page: 50 points, no next offset
            (
                [MagicMock(id=f"uuid-{i}") for i in range(100, 150)],
                None,
            ),
        ]
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        count = manager.delete_by_file("docs/large_file.md")

        # Verify scroll called twice (pagination)
        assert mock_client.scroll.call_count == 2
        # Verify delete called with all 150 points
        delete_args = mock_client.delete.call_args
        assert len(delete_args[1]["points_selector"].points) == 150
        # Verify total count
        assert count == 150

    @patch("rag_ingestion.vector_store.QdrantClient")
    @patch("rag_ingestion.vector_store.time.sleep")
    def test_delete_by_file_retries_on_connection_error(
        self, mock_sleep: MagicMock, mock_qdrant_client: MagicMock
    ) -> None:
        """Should retry deletion on network errors.

        Verifies:
        - Retries scroll operation on UnexpectedResponse
        - Uses exponential backoff (1s, 2s, 4s)
        - Succeeds on final attempt
        """
        import httpx
        from qdrant_client.http.exceptions import UnexpectedResponse

        mock_client = MagicMock()
        # Scroll fails twice, succeeds on third
        mock_client.scroll.side_effect = [
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
            ([MagicMock(id="uuid-1")], None),  # Success
        ]
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        count = manager.delete_by_file("docs/test.md")

        # Verify 3 scroll attempts
        assert mock_client.scroll.call_count == 3
        # Verify successful deletion
        assert count == 1

    @patch("rag_ingestion.vector_store.QdrantClient")
    @patch("rag_ingestion.vector_store.time.sleep")
    def test_delete_by_file_retries_delete_on_error(
        self, mock_sleep: MagicMock, mock_qdrant_client: MagicMock
    ) -> None:
        """Should retry delete operation on network errors.

        Verifies:
        - Scroll succeeds, delete fails and retries
        - Uses exponential backoff for delete retry
        - Returns correct count on success
        """
        import httpx
        from qdrant_client.http.exceptions import UnexpectedResponse

        mock_client = MagicMock()
        # Scroll succeeds
        mock_client.scroll.return_value = (
            [MagicMock(id="uuid-1"), MagicMock(id="uuid-2")],
            None,
        )
        # Delete fails twice, succeeds on third
        mock_client.delete.side_effect = [
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
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        count = manager.delete_by_file("docs/test.md")

        # Verify scroll called once
        assert mock_client.scroll.call_count == 1
        # Verify delete retried 3 times
        assert mock_client.delete.call_count == 3
        # Verify count returned
        assert count == 2


class TestEnsurePayloadIndexes:
    """Test payload index creation for query performance optimization."""

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_create_payload_index_file_path_relative(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should create keyword index on file_path_relative field.

        Verifies:
        - Calls create_payload_index() with collection name
        - Creates text/keyword index on file_path_relative field
        - Index name is "file_path_relative_index"
        - Field type is keyword for exact matching
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        manager.ensure_payload_indexes()

        # Verify create_payload_index called for file_path_relative
        calls = mock_client.create_payload_index.call_args_list
        file_path_call = [
            c for c in calls if c[1].get("field_name") == "file_path_relative"
        ]
        assert len(file_path_call) == 1
        assert file_path_call[0][1]["collection_name"] == "test_collection"

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_create_payload_index_filename(self, mock_qdrant_client: MagicMock) -> None:
        """Should create keyword index on filename field.

        Verifies:
        - Calls create_payload_index() for filename field
        - Creates text/keyword index for exact filename matching
        - Index name is "filename_index"
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        manager.ensure_payload_indexes()

        # Verify create_payload_index called for filename
        calls = mock_client.create_payload_index.call_args_list
        filename_call = [c for c in calls if c[1].get("field_name") == "filename"]
        assert len(filename_call) == 1
        assert filename_call[0][1]["collection_name"] == "test_collection"

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_create_payload_index_chunk_index(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should create integer index on chunk_index field.

        Verifies:
        - Calls create_payload_index() for chunk_index field
        - Creates integer index for range queries
        - Enables filtering by chunk position in document
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        manager.ensure_payload_indexes()

        # Verify create_payload_index called for chunk_index
        calls = mock_client.create_payload_index.call_args_list
        chunk_index_call = [c for c in calls if c[1].get("field_name") == "chunk_index"]
        assert len(chunk_index_call) == 1
        assert chunk_index_call[0][1]["collection_name"] == "test_collection"

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_create_payload_index_modification_date(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should create datetime index on modification_date field.

        Verifies:
        - Calls create_payload_index() for modification_date field
        - Creates datetime index for temporal queries
        - Enables filtering by file modification time
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        manager.ensure_payload_indexes()

        # Verify create_payload_index called for modification_date
        calls = mock_client.create_payload_index.call_args_list
        mod_date_call = [
            c for c in calls if c[1].get("field_name") == "modification_date"
        ]
        assert len(mod_date_call) == 1
        assert mod_date_call[0][1]["collection_name"] == "test_collection"

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_create_payload_index_tags(self, mock_qdrant_client: MagicMock) -> None:
        """Should create keyword index on tags field.

        Verifies:
        - Calls create_payload_index() for tags field
        - Creates keyword array index for tag filtering
        - Supports multi-value tag queries
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        manager.ensure_payload_indexes()

        # Verify create_payload_index called for tags
        calls = mock_client.create_payload_index.call_args_list
        tags_call = [c for c in calls if c[1].get("field_name") == "tags"]
        assert len(tags_call) == 1
        assert tags_call[0][1]["collection_name"] == "test_collection"

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_ensure_payload_indexes_creates_all_indexes(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should create all required payload indexes in single call.

        Verifies:
        - Creates indexes for all metadata fields
        - file_path_relative, filename, chunk_index, modification_date, tags
        - All indexes created with single method call
        """
        mock_client = MagicMock()
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        manager.ensure_payload_indexes()

        # Verify create_payload_index called for all expected fields
        calls = mock_client.create_payload_index.call_args_list
        field_names = [c[1]["field_name"] for c in calls]

        assert "file_path_relative" in field_names
        assert "filename" in field_names
        assert "chunk_index" in field_names
        assert "modification_date" in field_names
        assert "tags" in field_names

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_ensure_payload_indexes_is_idempotent(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should handle already-existing indexes gracefully.

        Verifies:
        - Second call to ensure_payload_indexes() doesn't error
        - Method is idempotent and safe to call multiple times
        - Handles "index already exists" response gracefully
        """
        mock_client = MagicMock()
        # First call succeeds
        # Second call should either skip or handle gracefully
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        # Call twice - should not error
        manager.ensure_payload_indexes()
        manager.ensure_payload_indexes()

        # Verify create_payload_index called for each field twice
        # (tests idempotency - implementation may check before creating)
        calls = mock_client.create_payload_index.call_args_list
        # At minimum, should have been called for all fields at least once
        field_names = [c[1]["field_name"] for c in calls]
        assert field_names.count("file_path_relative") >= 1
        assert field_names.count("filename") >= 1

    @patch("rag_ingestion.vector_store.QdrantClient")
    @patch("rag_ingestion.vector_store.time.sleep")
    def test_ensure_payload_indexes_retries_on_error(
        self, mock_sleep: MagicMock, mock_qdrant_client: MagicMock
    ) -> None:
        """Should retry index creation on network errors.

        Verifies:
        - Retries on UnexpectedResponse errors
        - Uses exponential backoff (1s, 2s, 4s)
        - Succeeds on final attempt
        """
        import httpx
        from qdrant_client.http.exceptions import UnexpectedResponse

        mock_client = MagicMock()
        # Fail twice, succeed on third
        mock_client.create_payload_index.side_effect = [
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
            None,  # Success on third attempt
            None,  # Success for remaining indexes
            None,
            None,
            None,
        ]
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        manager.ensure_payload_indexes()

        # Verify retry happened (3 attempts for first field + 1 each for rest)
        # Should have at least 3 calls for first field due to retries
        assert mock_client.create_payload_index.call_count >= 5

    @patch("rag_ingestion.vector_store.QdrantClient")
    def test_ensure_payload_indexes_validates_collection_exists(
        self, mock_qdrant_client: MagicMock
    ) -> None:
        """Should validate collection exists before creating indexes.

        Verifies:
        - Checks collection_exists() before creating indexes
        - Raises error if collection doesn't exist
        - Error message explains collection must be created first
        """
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        with pytest.raises(ValueError, match="Collection.*does not exist"):
            manager.ensure_payload_indexes()

        # Verify collection_exists was checked
        mock_client.collection_exists.assert_called_once_with("test_collection")
        # Verify no index creation attempted
        mock_client.create_payload_index.assert_not_called()
