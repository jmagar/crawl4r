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

import itertools
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from qdrant_client.http.exceptions import ApiException, UnexpectedResponse

from crawl4r.storage.qdrant import VectorMetadata, VectorStoreManager

pytestmark = pytest.mark.asyncio


def _create_async_client() -> MagicMock:
    client = MagicMock()
    client.collection_exists = AsyncMock(return_value=True)
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.query_points = AsyncMock()
    client.delete = AsyncMock()
    client.scroll = AsyncMock(return_value=([], None))
    client.create_payload_index = AsyncMock()
    return client


class TestQdrantClientInitialization:
    """Test VectorStoreManager initialization."""

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_initialization_with_url_and_collection(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """VectorStoreManager should initialize with Qdrant URL and collection.

        Verifies:
        - Accepts qdrant_url parameter
        - Accepts collection_name parameter
        - Creates internal QdrantClient instance
        - Stores configuration parameters
        """
        mock_async_qdrant_client.return_value = MagicMock()

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="crawl4r"
        )

        assert manager.qdrant_url == "http://crawl4r-vectors:6333"
        assert manager.collection_name == "crawl4r"
        assert manager.client is not None

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_initialization_with_custom_dimensions(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """VectorStoreManager should accept custom vector dimensions.

        Verifies:
        - Accepts dimensions parameter (default 1024)
        - Stores dimensions for collection creation
        """
        mock_async_qdrant_client.return_value = MagicMock()

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection",
            dimensions=1024,
        )

        assert manager.dimensions == 1024

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_initialization_defaults_to_1024_dimensions(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """VectorStoreManager should default to 1024 dimensions.

        Verifies:
        - Default dimensions is 1024 (Qwen3 output size)
        """
        mock_async_qdrant_client.return_value = MagicMock()

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        assert manager.dimensions == 1024


class TestEnsureCollectionCreatesIfMissing:
    """Test collection creation when collection does not exist."""

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_creates_collection_when_missing(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should create collection if it does not exist.

        Verifies:
        - Calls collection_exists() to check
        - Calls create_collection() when not found
        - Uses correct vector size (1024)
        - Uses cosine distance metric
        """
        # Mock Qdrant client instance
        mock_client = _create_async_client()
        mock_client.collection_exists = AsyncMock(return_value=False)
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )
        await manager.ensure_collection()

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

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_creates_collection_with_custom_dimensions(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should create collection with custom dimensions.

        Verifies:
        - Respects custom dimensions parameter
        - Uses provided dimensions in VectorParams
        """
        mock_client = _create_async_client()
        mock_client.collection_exists = AsyncMock(return_value=False)
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection",
            dimensions=512,
        )
        await manager.ensure_collection()

        # Verify vector config uses custom dimensions
        call_args = mock_client.create_collection.call_args
        vector_config = call_args[1]["vectors_config"]
        assert vector_config.size == 512


class TestEnsureCollectionSkipsIfExists:
    """Test collection creation skipped when collection already exists."""

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_skips_creation_when_exists(self, mock_async_qdrant_client: MagicMock) -> None:
        """Should not create collection if it already exists.

        Verifies:
        - Calls collection_exists() to check
        - Does NOT call create_collection() when found
        - Returns without error
        """
        mock_client = _create_async_client()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="existing_collection",
        )
        await manager.ensure_collection()

        # Verify collection_exists called
        mock_client.collection_exists.assert_called_once_with("existing_collection")

        # Verify create_collection NOT called
        mock_client.create_collection.assert_not_called()


class TestUpsertSingleVector:
    """Test upserting a single vector with metadata."""

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_upsert_single_vector_creates_point_struct(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should create PointStruct with vector and metadata.

        Verifies:
        - Calls upsert() with single PointStruct
        - PointStruct has deterministic UUID from content hash
        - Vector dimensions match collection (1024)
        - Payload includes all required metadata
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector = [0.1] * 1024  # 1024-dimensional vector
        metadata = {
            "file_path": "/home/user/docs/test.md",
            "file_path_absolute": "/home/user/docs/test.md",
            "file_name": "test.md",
            "last_modified_date": "2026-01-15T00:00:00Z",
            "chunk_index": 0,
            "chunk_text": "Test content",
        }

        await manager.upsert_vector(vector, metadata)

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

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_upsert_generates_deterministic_id_from_hash(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should generate deterministic UUID from content hash.

        Verifies:
        - Same file path + chunk index → same UUID
        - Different file path + chunk index → different UUID
        - Uses SHA256(file_path:chunk_index)
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata1: VectorMetadata = {
            "file_path": "/home/user/docs/test.md",
            "chunk_index": 0,
            "chunk_text": "First chunk",
        }
        metadata2: VectorMetadata = {
            "file_path": "/home/user/docs/test.md",
            "chunk_index": 0,
            "chunk_text": "Same chunk, different text",
        }

        await manager.upsert_vector(vector, metadata1)
        call1_id = mock_client.upsert.call_args[1]["points"][0].id

        mock_client.reset_mock()

        await manager.upsert_vector(vector, metadata2)
        call2_id = mock_client.upsert.call_args[1]["points"][0].id

        # Same file + chunk index should produce same ID
        assert call1_id == call2_id

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_upsert_rejects_wrong_dimension_vector(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should reject vector with wrong dimensions.

        Verifies:
        - ValueError when vector != configured dimensions
        - Error message mentions expected vs actual dimensions
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection",
            dimensions=1024,
        )

        vector = [0.1] * 512  # Wrong dimension (512 != 1024)
        metadata = {
            "file_path": "/home/user/docs/test.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }

        with pytest.raises(ValueError, match="1024.*512"):
            await manager.upsert_vector(vector, metadata)

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_upsert_rejects_empty_vector(self, mock_async_qdrant_client: MagicMock) -> None:
        """Should reject empty vector.

        Verifies:
        - ValueError when vector is empty list
        - No upsert call made
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector: list[float] = []
        metadata = {
            "file_path": "/home/user/docs/test.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }

        with pytest.raises(ValueError, match="empty"):
            await manager.upsert_vector(vector, metadata)

        mock_client.upsert.assert_not_called()

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_upsert_requires_file_path(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should require file_path in metadata.

        Verifies:
        - ValueError when file_path missing
        - No upsert call made
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata = {
            "chunk_index": 0,
            "chunk_text": "Test",
        }

        with pytest.raises(ValueError, match="file_path"):
            await manager.upsert_vector(vector, metadata)

        mock_client.upsert.assert_not_called()

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_upsert_requires_chunk_index(self, mock_async_qdrant_client: MagicMock) -> None:
        """Should require chunk_index in metadata.

        Verifies:
        - ValueError when chunk_index missing
        - No upsert call made
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata = {
            "file_path": "/home/user/docs/test.md",
            "chunk_text": "Test",
        }

        with pytest.raises(ValueError, match="chunk_index"):
            await manager.upsert_vector(vector, metadata)

        mock_client.upsert.assert_not_called()

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_upsert_requires_chunk_text(self, mock_async_qdrant_client: MagicMock) -> None:
        """Should require chunk_text in metadata.

        Verifies:
        - ValueError when chunk_text missing
        - No upsert call made
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata = {
            "file_path": "/home/user/docs/test.md",
            "chunk_index": 0,
        }

        with pytest.raises(ValueError, match="chunk_text"):
            await manager.upsert_vector(vector, metadata)

        mock_client.upsert.assert_not_called()


class TestUpsertBatchVectors:
    """Test batch upsert operations with multiple vectors."""

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_upsert_batch_creates_multiple_points(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should create multiple PointStructs in single batch.

        Verifies:
        - Calls upsert() with list of PointStructs
        - Each point has correct vector and metadata
        - All points in single batch (no splitting)
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path": "/home/user/docs/test.md",
                    "chunk_index": i,
                    "chunk_text": f"Chunk {i}",
                },
            }
            for i in range(5)
        ]

        await manager.upsert_vectors_batch(vectors_with_metadata)

        # Verify single upsert call with 5 points
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        points = call_args[1]["points"]
        assert len(points) == 5

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_upsert_batch_splits_at_100_points(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should split batches at 100 points per upsert.

        Verifies:
        - 150 points → 2 upsert calls (100 + 50)
        - First batch has 100 points
        - Second batch has 50 points
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path": "/home/user/docs/test.md",
                    "chunk_index": i,
                    "chunk_text": f"Chunk {i}",
                },
            }
            for i in range(150)
        ]

        await manager.upsert_vectors_batch(vectors_with_metadata)

        # Verify 2 upsert calls
        assert mock_client.upsert.call_count == 2

        # Verify batch sizes
        first_batch = mock_client.upsert.call_args_list[0][1]["points"]
        second_batch = mock_client.upsert.call_args_list[1][1]["points"]
        assert len(first_batch) == 100
        assert len(second_batch) == 50

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_upsert_batch_handles_empty_list(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should handle empty batch gracefully.

        Verifies:
        - No upsert call made for empty list
        - No error raised
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        await manager.upsert_vectors_batch([])

        mock_client.upsert.assert_not_called()

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_upsert_batch_validates_all_vectors(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should validate all vectors before upserting.

        Verifies:
        - ValueError if any vector has wrong dimensions
        - No partial upsert (all-or-nothing)
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection",
            dimensions=1024,
        )

        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path": "/home/user/docs/test.md",
                    "chunk_index": 0,
                    "chunk_text": "Good chunk",
                },
            },
            {
                "vector": [0.1] * 512,  # Wrong dimension
                "metadata": {
                    "file_path": "/home/user/docs/test.md",
                    "chunk_index": 1,
                    "chunk_text": "Bad chunk",
                },
            },
        ]

        with pytest.raises(ValueError, match="dimension"):
            await manager.upsert_vectors_batch(vectors_with_metadata)

        # No partial upsert
        mock_client.upsert.assert_not_called()

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_upsert_batch_validates_all_metadata(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should validate all metadata before upserting.

        Verifies:
        - ValueError if any metadata missing required fields
        - No partial upsert (all-or-nothing)
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path": "/home/user/docs/test.md",
                    "chunk_index": 0,
                    "chunk_text": "Good chunk",
                },
            },
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path": "/home/user/docs/test.md",
                    # Missing chunk_index
                    "chunk_text": "Bad chunk",
                },
            },
        ]

        with pytest.raises(ValueError, match="chunk_index"):
            await manager.upsert_vectors_batch(vectors_with_metadata)

        # No partial upsert
        mock_client.upsert.assert_not_called()


class TestUpsertWithRetry:
    """Test upsert retry logic with exponential backoff."""

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    @patch("crawl4r.storage.qdrant.asyncio.sleep", new_callable=AsyncMock)
    async def test_upsert_retries_on_network_error(
        self, mock_sleep: MagicMock, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should retry on network errors with exponential backoff.

        Verifies:
        - Retries 3 times on network error
        - Uses exponential backoff (1s, 2s)
        - Succeeds on third attempt
        """


        mock_client = _create_async_client()
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
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata = {
            "file_path": "/home/user/docs/test.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }

        await manager.upsert_vector(vector, metadata)

        # Verify 3 upsert attempts
        assert mock_client.upsert.call_count == 3

        # Verify exponential backoff delays (1s, 2s between retries)
        assert mock_sleep.call_count == 2
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert sleep_calls == [1, 2], f"Expected backoff delays [1, 2], got {sleep_calls}"

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    @patch("crawl4r.storage.qdrant.asyncio.sleep", new_callable=AsyncMock)
    async def test_upsert_raises_after_max_retries(
        self, mock_sleep: MagicMock, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should raise error after exhausting retries.

        Verifies:
        - Tries 3 times
        - Raises RuntimeError on final failure
        - Error message includes retry count
        """

        mock_client = _create_async_client()
        mock_client.upsert.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="Server Error",
            content=b"Server Error",
            headers=httpx.Headers(),
        )
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata = {
            "file_path": "/home/user/docs/test.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }

        with pytest.raises(RuntimeError, match="3.*retries"):
            await manager.upsert_vector(vector, metadata)

        # Verify 3 upsert attempts
        assert mock_client.upsert.call_count == 3

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    @patch("crawl4r.storage.qdrant.asyncio.sleep", new_callable=AsyncMock)
    async def test_upsert_batch_retries_per_batch(
        self, mock_sleep: MagicMock, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should retry each batch independently.

        Verifies:
        - First batch fails twice, succeeds on third
        - Second batch succeeds immediately
        - Total 4 upsert calls (3 + 1)
        """

        mock_client = _create_async_client()
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
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        # 150 vectors → 2 batches (100 + 50)
        vectors_with_metadata = [
            {
                "vector": [0.1] * 1024,
                "metadata": {
                    "file_path": "/home/user/docs/test.md",
                    "chunk_index": i,
                    "chunk_text": f"Chunk {i}",
                },
            }
            for i in range(150)
        ]

        await manager.upsert_vectors_batch(vectors_with_metadata)

        # Verify 4 total upsert calls (3 for first batch + 1 for second)
        assert mock_client.upsert.call_count == 4


class TestGeneratePointIdAbsolutePath:
    """Test absolute path point ID generation after migration."""

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_generate_point_id_with_absolute_path(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Verify _generate_point_id accepts absolute paths.

        After the migration, all paths are absolute (from SimpleDirectoryReader).
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        abs_path = "/home/user/docs/guide/install.md"
        chunk_index = 0

        point_id = manager._generate_point_id(abs_path, chunk_index)

        assert point_id is not None
        # Verify it's a valid UUID format
        uuid.UUID(point_id)  # Should not raise

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_generate_point_id_deterministic_for_absolute_path(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Verify _generate_point_id is deterministic for absolute paths.

        Same absolute path + chunk index must always produce the same ID.
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        abs_path = "/home/user/docs/test.md"
        chunk_index = 5

        id1 = manager._generate_point_id(abs_path, chunk_index)
        id2 = manager._generate_point_id(abs_path, chunk_index)

        assert id1 == id2, "Same path + chunk index must produce same ID"

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_generate_point_id_different_for_different_paths(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Verify _generate_point_id produces different IDs for different paths.

        Different absolute paths must produce different IDs.
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        path1 = "/home/user/docs/file1.md"
        path2 = "/home/user/docs/file2.md"
        chunk_index = 0

        id1 = manager._generate_point_id(path1, chunk_index)
        id2 = manager._generate_point_id(path2, chunk_index)

        assert id1 != id2, "Different paths must produce different IDs"


class TestGeneratePointId:
    """Test deterministic point ID generation from content hash."""

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_generate_id_is_deterministic(self, mock_async_qdrant_client: MagicMock) -> None:
        """Should generate same UUID for same file_path + chunk_index.

        Verifies:
        - Same inputs → same UUID every time
        - Uses SHA256 hash
        - Converts to UUID format
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector = [0.1] * 1024
        metadata = {
            "file_path": "/home/user/docs/test.md",
            "chunk_index": 5,
            "chunk_text": "Test",
        }

        # Call twice with same metadata
        await manager.upsert_vector(vector, metadata)
        id1 = mock_client.upsert.call_args[1]["points"][0].id

        mock_client.reset_mock()

        await manager.upsert_vector(vector, metadata)
        id2 = mock_client.upsert.call_args[1]["points"][0].id

        # Should be identical
        assert id1 == id2

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_generate_id_differs_by_chunk_index(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should generate different UUID for different chunk_index.

        Verifies:
        - Same file, different chunk → different UUID
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector = [0.1] * 1024

        metadata1: VectorMetadata = {
            "file_path": "/home/user/docs/test.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }
        metadata2: VectorMetadata = {
            "file_path": "/home/user/docs/test.md",
            "chunk_index": 1,
            "chunk_text": "Test",
        }

        await manager.upsert_vector(vector, metadata1)
        id1 = mock_client.upsert.call_args[1]["points"][0].id

        mock_client.reset_mock()

        await manager.upsert_vector(vector, metadata2)
        id2 = mock_client.upsert.call_args[1]["points"][0].id

        # Should be different
        assert id1 != id2

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_generate_id_differs_by_file_path(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should generate different UUID for different file_path.

        Verifies:
        - Different file, same chunk → different UUID
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        vector = [0.1] * 1024

        metadata1: VectorMetadata = {
            "file_path": "/home/user/docs/test1.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }
        metadata2: VectorMetadata = {
            "file_path": "/home/user/docs/test2.md",
            "chunk_index": 0,
            "chunk_text": "Test",
        }

        await manager.upsert_vector(vector, metadata1)
        id1 = mock_client.upsert.call_args[1]["points"][0].id

        mock_client.reset_mock()

        await manager.upsert_vector(vector, metadata2)
        id2 = mock_client.upsert.call_args[1]["points"][0].id

        # Should be different
        assert id1 != id2


class TestSearchSimilar:
    """Test semantic similarity search operations."""

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_search_similar_returns_results_with_scores(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should return search results with scores and metadata.

        Verifies:
        - Calls client.query_points() with query vector and top_k limit
        - Returns list of results with id, score, and metadata
        - Each result includes file_path, chunk_index, chunk_text
        - Results are ordered by score (highest first)
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        # Mock query_points response with 3 results
        mock_query_response = MagicMock()
        mock_query_response.points = [
            MagicMock(
                id="uuid-1",
                score=0.95,
                payload={
                    "file_path": "/home/user/docs/test1.md",
                    "chunk_index": 0,
                    "chunk_text": "Most similar chunk",
                },
            ),
            MagicMock(
                id="uuid-2",
                score=0.85,
                payload={
                    "file_path": "/home/user/docs/test2.md",
                    "chunk_index": 1,
                    "chunk_text": "Second similar chunk",
                },
            ),
            MagicMock(
                id="uuid-3",
                score=0.75,
                payload={
                    "file_path": "/home/user/docs/test3.md",
                    "chunk_index": 2,
                    "chunk_text": "Third similar chunk",
                },
            ),
        ]
        mock_client.query_points = AsyncMock(return_value=mock_query_response)

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024
        results = await manager.search_similar(query_vector, top_k=5)

        # Verify query_points was called with correct parameters
        mock_client.query_points.assert_called_once_with(
            collection_name="test_collection",
            query=query_vector,
            limit=5,
        )

        # Verify results structure
        assert len(results) == 3
        assert results[0]["id"] == "uuid-1"
        assert results[0]["score"] == 0.95
        assert results[0]["file_path"] == "/home/user/docs/test1.md"
        assert results[0]["chunk_index"] == 0
        assert results[0]["chunk_text"] == "Most similar chunk"

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_search_similar_validates_query_vector_dimensions(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should validate query vector dimensions match collection config.

        Verifies:
        - ValueError if query vector dimensions != configured dimensions
        - No search call made with invalid dimensions
        - Error message includes expected vs actual dimensions
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333",
            collection_name="test_collection",
            dimensions=1024,
        )

        query_vector = [0.1] * 512  # Wrong dimension

        with pytest.raises(ValueError, match="1024.*512"):
            await manager.search_similar(query_vector, top_k=5)

        mock_client.query_points.assert_not_called()

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_search_similar_validates_empty_query_vector(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should reject empty query vector.

        Verifies:
        - ValueError when query vector is empty list
        - No search call made
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector: list[float] = []

        with pytest.raises(ValueError, match="empty"):
            await manager.search_similar(query_vector, top_k=5)

        mock_client.query_points.assert_not_called()

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_search_similar_validates_positive_top_k(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should validate top_k is a positive integer.

        Verifies:
        - ValueError if top_k <= 0
        - No search call made with invalid top_k
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024

        with pytest.raises(ValueError, match="top_k.*positive"):
            await manager.search_similar(query_vector, top_k=0)

        with pytest.raises(ValueError, match="top_k.*positive"):
            await manager.search_similar(query_vector, top_k=-5)

        mock_client.query_points.assert_not_called()

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_search_similar_handles_empty_collection(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should handle empty collection gracefully.

        Verifies:
        - Returns empty list when no results found
        - No error raised for empty collection
        """
        mock_client = _create_async_client()
        # Mock query_points returning response with empty points
        mock_query_response = MagicMock()
        mock_query_response.points = []
        mock_client.query_points = AsyncMock(return_value=mock_query_response)
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024
        results = await manager.search_similar(query_vector, top_k=5)

        assert results == []
        mock_client.query_points.assert_called_once()

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_search_similar_limits_results_to_top_k(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should limit results to top_k parameter.

        Verifies:
        - Passes top_k as limit parameter to query_points
        - Returns at most top_k results
        """
        mock_client = _create_async_client()
        # Mock 3 results when asking for top_k=3
        mock_query_response = MagicMock()
        mock_query_response.points = [
            MagicMock(
                id=f"uuid-{i}",
                score=0.9 - i * 0.1,
                payload={
                    "file_path": f"/home/user/docs/test{i}.md",
                    "chunk_index": i,
                    "chunk_text": f"Chunk {i}",
                },
            )
            for i in range(3)
        ]
        mock_client.query_points = AsyncMock(return_value=mock_query_response)
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024
        results = await manager.search_similar(query_vector, top_k=3)

        # Verify limit parameter passed to query_points
        call_args = mock_client.query_points.call_args
        assert call_args[1]["limit"] == 3

        # Verify result count matches top_k
        assert len(results) == 3

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_search_similar_results_sorted_by_score(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should return results sorted by score (highest first).

        Verifies:
        - Results are in descending order by score
        - First result has highest score
        - Last result has lowest score
        """
        mock_client = _create_async_client()
        # Mock query_points response in descending score order
        mock_query_response = MagicMock()
        mock_query_response.points = [
            MagicMock(
                id="uuid-1",
                score=0.95,
                payload={
                    "file_path": "/home/user/docs/test1.md",
                    "chunk_index": 0,
                    "chunk_text": "Best match",
                },
            ),
            MagicMock(
                id="uuid-2",
                score=0.85,
                payload={
                    "file_path": "/home/user/docs/test2.md",
                    "chunk_index": 1,
                    "chunk_text": "Good match",
                },
            ),
            MagicMock(
                id="uuid-3",
                score=0.75,
                payload={
                    "file_path": "/home/user/docs/test3.md",
                    "chunk_index": 2,
                    "chunk_text": "OK match",
                },
            ),
        ]
        mock_client.query_points = AsyncMock(return_value=mock_query_response)
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024
        results = await manager.search_similar(query_vector, top_k=5)

        # Verify descending score order
        assert results[0]["score"] == 0.95
        assert results[1]["score"] == 0.85
        assert results[2]["score"] == 0.75
        assert results[0]["score"] > results[1]["score"]
        assert results[1]["score"] > results[2]["score"]

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_search_similar_includes_all_metadata_fields(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should include all metadata fields in results.

        Verifies:
        - Each result includes all metadata fields from payload
        - file_path, chunk_index, chunk_text are present
        - Additional metadata fields are preserved
        """
        mock_client = _create_async_client()
        mock_query_response = MagicMock()
        mock_query_response.points = [
            MagicMock(
                id="uuid-1",
                score=0.95,
                payload={
                    "file_path": "/home/user/docs/test.md",
                    "file_name": "test.md",
                    "last_modified_date": "2026-01-15T00:00:00Z",
                    "chunk_index": 0,
                    "chunk_text": "Test content",
                    "section_path": "Introduction",
                    "heading_level": 1,
                },
            ),
        ]
        mock_client.query_points = AsyncMock(return_value=mock_query_response)
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024
        results = await manager.search_similar(query_vector, top_k=5)

        # Verify all metadata fields included
        result = results[0]
        assert result["file_path"] == "/home/user/docs/test.md"
        assert result["file_name"] == "test.md"
        assert result["last_modified_date"] == "2026-01-15T00:00:00Z"
        assert result["chunk_index"] == 0
        assert result["chunk_text"] == "Test content"
        assert result["section_path"] == "Introduction"
        assert result["heading_level"] == 1

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    @patch("crawl4r.storage.qdrant.asyncio.sleep", new_callable=AsyncMock)
    async def test_search_similar_retries_on_connection_error(
        self, mock_sleep: MagicMock, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should retry search on network errors with exponential backoff.

        Verifies:
        - Retries on UnexpectedResponse errors
        - Uses exponential backoff (1s, 2s, 4s)
        - Succeeds on final attempt
        """

        mock_client = _create_async_client()
        # Fail twice, succeed on third attempt
        mock_query_response = MagicMock()
        mock_query_response.points = [
            MagicMock(
                id="uuid-1",
                score=0.95,
                payload={
                    "file_path": "/home/user/docs/test.md",
                    "chunk_index": 0,
                    "chunk_text": "Test",
                },
            )
        ]
        mock_client.query_points.side_effect = [
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
            mock_query_response,
        ]
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        query_vector = [0.1] * 1024
        results = await manager.search_similar(query_vector, top_k=5)

        # Verify 3 query_points attempts
        assert mock_client.query_points.call_count == 3
        # Verify results returned on success
        assert len(results) == 1
        assert results[0]["score"] == 0.95


class TestDeleteById:
    """Test deletion of single points by UUID."""

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_delete_by_id_deletes_single_point(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should delete a single point by its UUID.

        Verifies:
        - Calls client.delete() with point ID
        - Collection name passed correctly
        - No return value (void function)
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        point_id = "550e8400-e29b-41d4-a716-446655440000"
        await manager.delete_by_id(point_id)

        # Verify delete called with correct parameters
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert call_args[1]["points_selector"].points == [point_id]

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_delete_by_id_validates_uuid_format(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should validate UUID format before deletion.

        Verifies:
        - ValueError when point_id is not valid UUID format
        - No delete call made with invalid UUID
        - Error message mentions UUID format
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        invalid_id = "not-a-valid-uuid"

        with pytest.raises(ValueError, match="UUID"):
            await manager.delete_by_id(invalid_id)

        mock_client.delete.assert_not_called()

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_delete_by_id_handles_nonexistent_id_gracefully(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should handle deletion of non-existent ID without error.

        Verifies:
        - No exception raised when ID doesn't exist
        - Delete call still made (Qdrant handles gracefully)
        - Returns without error
        """
        mock_client = _create_async_client()
        # Qdrant delete doesn't error on non-existent IDs
        mock_client.delete = AsyncMock(return_value=None)
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        # Non-existent but valid UUID
        point_id = "550e8400-e29b-41d4-a716-446655440000"
        await manager.delete_by_id(point_id)

        # Verify delete was attempted
        mock_client.delete.assert_called_once()

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    @patch("crawl4r.storage.qdrant.asyncio.sleep", new_callable=AsyncMock)
    async def test_delete_by_id_retries_on_connection_error(
        self, mock_sleep: MagicMock, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should retry deletion on network errors.

        Verifies:
        - Retries on UnexpectedResponse errors
        - Uses exponential backoff (1s, 2s, 4s)
        - Succeeds on final attempt
        """

        mock_client = _create_async_client()
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
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        point_id = "550e8400-e29b-41d4-a716-446655440000"
        await manager.delete_by_id(point_id)

        # Verify 3 delete attempts
        assert mock_client.delete.call_count == 3


class TestDeleteByFile:
    """Test deletion of all chunks from a file."""

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_delete_by_file_deletes_all_chunks(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should delete all chunks for a given file path.

        Verifies:
        - Uses scroll to find all points with matching file_path
        - Calls delete for each found point
        - Returns count of deleted points
        """
        mock_client = _create_async_client()
        # Mock count for _delete_by_filter
        mock_client.count = AsyncMock(return_value=MagicMock(count=3))
        # Mock delete for _delete_by_filter
        mock_client.delete = AsyncMock()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        file_path = "/home/user/docs/test.md"
        count = await manager.delete_by_file(file_path)

        # Verify count called with filter
        mock_client.count.assert_called_once()
        count_args = mock_client.count.call_args
        assert count_args[1]["collection_name"] == "test_collection"
        # Filter should match file_path field
        count_filter = count_args[1]["count_filter"]
        assert count_filter.must[0].key == "file_path"
        assert count_filter.must[0].match.value == file_path

        # Verify delete called with filter
        mock_client.delete.assert_called_once()
        delete_args = mock_client.delete.call_args
        assert delete_args[1]["collection_name"] == "test_collection"
        # Should have filter-based delete
        delete_filter = delete_args[1]["points_selector"]
        assert delete_filter.filter.must[0].key == "file_path"

        # Verify count returned
        assert count == 3

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_delete_by_file_returns_count(self, mock_async_qdrant_client: MagicMock) -> None:
        """Should return count of deleted points.

        Verifies:
        - Return value is integer count
        - Count matches number of points found
        """
        mock_client = _create_async_client()
        # Mock count for _delete_by_filter
        mock_client.count = AsyncMock(return_value=MagicMock(count=5))
        # Mock delete for _delete_by_filter
        mock_client.delete = AsyncMock()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        count = await manager.delete_by_file("/home/user/docs/test.md")

        assert isinstance(count, int)
        assert count == 5

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_delete_by_file_handles_empty_results_gracefully(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should handle file with no chunks gracefully.

        Verifies:
        - Returns 0 when no matching chunks found
        - No delete call made for empty results
        - No error raised
        """
        mock_client = _create_async_client()
        # Mock count returning 0 (no matching points)
        mock_client.count = AsyncMock(return_value=MagicMock(count=0))
        # Mock delete should not be called
        mock_client.delete = AsyncMock()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        count = await manager.delete_by_file("docs/nonexistent.md")

        # Verify count was called
        mock_client.count.assert_called_once()
        # Verify no delete call made (early return when count is 0)
        mock_client.delete.assert_not_called()
        # Verify 0 count returned
        assert count == 0

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_delete_by_file_uses_filter_based_delete(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should use filter-based delete (not scroll-based).

        Verifies:
        - Uses count + filter delete (single operation)
        - No scroll/pagination needed
        - Returns count from count() call
        """
        mock_client = _create_async_client()
        # Mock count returning 150 points
        mock_client.count = AsyncMock(return_value=MagicMock(count=150))
        # Mock delete operation
        mock_client.delete = AsyncMock()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        count = await manager.delete_by_file("docs/large_file.md")

        # Verify count called once
        mock_client.count.assert_called_once()
        # Verify delete called with FilterSelector (not PointIdsList)
        delete_args = mock_client.delete.call_args
        points_selector = delete_args[1]["points_selector"]
        assert hasattr(points_selector, "filter")  # FilterSelector has filter attribute
        # Verify total count
        assert count == 150

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_delete_by_file_count_operation_not_retried(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Count operation is not retried (fails fast).

        Verifies:
        - Count error propagates immediately (no retry)
        - This is expected behavior for count operation
        """

        mock_client = _create_async_client()
        # Count fails immediately
        mock_client.count = AsyncMock(side_effect=UnexpectedResponse(
            status_code=500,
            reason_phrase="Server Error",
            content=b"Server Error",
            headers=httpx.Headers(),
        ))
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        # Should raise the exception (count not retried)
        with pytest.raises(UnexpectedResponse):
            await manager.delete_by_file("docs/test.md")

        # Verify count called only once (no retry)
        assert mock_client.count.call_count == 1

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    @patch("crawl4r.storage.qdrant.asyncio.sleep", new_callable=AsyncMock)
    async def test_delete_by_file_retries_delete_on_error(
        self, mock_sleep: MagicMock, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should retry delete operation on network errors.

        Verifies:
        - Scroll succeeds, delete fails and retries
        - Uses exponential backoff for delete retry
        - Returns correct count on success
        """

        mock_client = _create_async_client()
        # Count succeeds
        mock_client.count = AsyncMock(return_value=MagicMock(count=2))
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
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        count = await manager.delete_by_file("docs/test.md")

        # Verify count called once
        assert mock_client.count.call_count == 1
        # Verify delete retried 3 times
        assert mock_client.delete.call_count == 3
        # Verify count returned
        assert count == 2


class TestDeleteByFilter:
    """Test shared deletion helper using filter criteria."""

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_delete_by_url_deletes_matching_points(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should delete all points matching URL via public delete_by_url method."""
        mock_client = _create_async_client()
        # Mock count for _delete_by_filter
        mock_client.count = AsyncMock(return_value=MagicMock(count=2))
        # Mock delete for _delete_by_filter
        mock_client.delete = AsyncMock()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        count = await manager.delete_by_url("https://example.com")

        # Verify count called with filter
        mock_client.count.assert_called_once()
        count_args = mock_client.count.call_args
        count_filter = count_args[1]["count_filter"]
        assert count_filter.must[0].key == "source_url"
        assert count_filter.must[0].match.value == "https://example.com"

        # Verify delete called with FilterSelector
        mock_client.delete.assert_called_once()
        delete_args = mock_client.delete.call_args
        points_selector = delete_args[1]["points_selector"]
        assert hasattr(points_selector, "filter")  # FilterSelector has filter attribute
        assert count == 2

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_delete_by_url_uses_filter_based_delete(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should use filter-based delete (no pagination needed).

        Verifies:
        - Uses count + filter delete (single operation)
        - No scroll/pagination needed
        - Returns count from count() call
        """
        mock_client = _create_async_client()
        # Mock count returning 150 points
        mock_client.count = AsyncMock(return_value=MagicMock(count=150))
        # Mock delete operation
        mock_client.delete = AsyncMock()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        count = await manager.delete_by_url("https://example.com/large")

        # Verify count called once
        mock_client.count.assert_called_once()
        # Verify delete called with FilterSelector (not PointIdsList)
        delete_args = mock_client.delete.call_args
        points_selector = delete_args[1]["points_selector"]
        assert hasattr(points_selector, "filter")  # FilterSelector has filter attribute
        # Verify total count
        assert count == 150

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    @patch("crawl4r.storage.qdrant.asyncio.sleep", new_callable=AsyncMock)
    async def test_delete_by_filter_count_not_retried(
        self, mock_sleep: MagicMock, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Count operation is not retried (fails fast).

        Verifies:
        - Count error propagates immediately (no retry)
        - This is expected behavior for count operation
        """

        mock_client = _create_async_client()
        # Count fails immediately
        mock_client.count = AsyncMock(side_effect=UnexpectedResponse(
            status_code=500,
            reason_phrase="Server Error",
            content=b"Server Error",
            headers=httpx.Headers(),
        ))
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        # Should raise the exception (count not retried)
        with pytest.raises(UnexpectedResponse):
            await manager.delete_by_url("https://example.com")

        # Verify count called only once (no retry)
        assert mock_client.count.call_count == 1

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    @patch("crawl4r.storage.qdrant.asyncio.sleep", new_callable=AsyncMock)
    async def test_delete_by_filter_retries_delete_on_error(
        self, mock_sleep: MagicMock, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should retry delete operation on network errors.

        Verifies:
        - Count succeeds, delete fails and retries
        - Uses exponential backoff for delete retry
        - Returns correct count on success
        """

        mock_client = _create_async_client()
        # Count succeeds
        mock_client.count = AsyncMock(return_value=MagicMock(count=3))
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
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        count = await manager.delete_by_url("https://example.com")

        # Verify count called once
        assert mock_client.count.call_count == 1
        # Verify delete retried 3 times
        assert mock_client.delete.call_count == 3
        # Verify count returned
        assert count == 3


class TestEnsurePayloadIndexes:
    """Test payload index creation for query performance optimization."""

    @pytest.mark.parametrize(
        "field_name",
        [
            "file_path",
            "file_name",
            "chunk_index",
            "last_modified_date",
            "tags",
        ],
    )
    async def test_create_payload_index_for_field(self, field_name: str) -> None:
        """Should create a payload index for each required field."""
        with patch("crawl4r.storage.qdrant.AsyncQdrantClient") as mock_async_qdrant_client:
            mock_client = _create_async_client()
            mock_async_qdrant_client.return_value = mock_client

            manager = VectorStoreManager(
                qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
            )

            await manager.ensure_payload_indexes()

            calls = mock_client.create_payload_index.call_args_list
            field_calls = [
                c for c in calls if c[1].get("field_name") == field_name
            ]
            assert len(field_calls) == 1
            assert field_calls[0][1]["collection_name"] == "test_collection"

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_ensure_payload_indexes_creates_all_indexes(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should create all required payload indexes in single call.

        Verifies:
        - Creates indexes for all metadata fields
        - file_path, file_name, chunk_index, last_modified_date, tags
        - All indexes created with single method call
        """
        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        await manager.ensure_payload_indexes()

        # Verify create_payload_index called for all expected fields
        calls = mock_client.create_payload_index.call_args_list
        field_names = [c[1]["field_name"] for c in calls]

        assert "file_path" in field_names
        assert "file_name" in field_names
        assert "chunk_index" in field_names
        assert "last_modified_date" in field_names
        assert "tags" in field_names

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_ensure_payload_indexes_is_idempotent(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should handle already-existing indexes gracefully.

        Verifies:
        - Method handles "index already exists" error without raising
        - Method is idempotent and safe to call multiple times
        - Continues creating other indexes even if some already exist
        """
        mock_client = _create_async_client()
        def _conflict_exception() -> ApiException:
            exc = ApiException.__new__(ApiException)
            exc.status = 409
            return exc

        # Simulate "already exists" error for some indexes
        mock_client.create_payload_index.side_effect = [
            _conflict_exception(),  # First field - already exists
            None,  # Second field - created successfully
            _conflict_exception(),  # Third field - already exists
            None,  # Fourth field - created successfully
            None,  # Fifth field - created successfully
            None,  # Sixth field - created successfully
            None,  # Seventh field - created successfully
        ]
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        # Should not raise despite "already exists" errors
        await manager.ensure_payload_indexes()

        # Verify all indexes were attempted (must match PAYLOAD_INDEXES)
        calls = mock_client.create_payload_index.call_args_list
        field_names = [c[1]["field_name"] for c in calls]
        assert "file_path" in field_names
        assert "source_url" in field_names
        assert "source_type" in field_names
        assert "file_name" in field_names
        assert "chunk_index" in field_names
        assert "last_modified_date" in field_names
        assert "tags" in field_names
        assert len(field_names) == 7, f"Expected 7 indexed fields, got {len(field_names)}"

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    @patch("crawl4r.storage.qdrant.asyncio.sleep", new_callable=AsyncMock)
    async def test_ensure_payload_indexes_retries_on_error(
        self, mock_sleep: MagicMock, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should retry index creation on network errors.

        Verifies:
        - Retries on UnexpectedResponse errors
        - Uses exponential backoff (1s, 2s, 4s)
        - Succeeds on final attempt
        """

        mock_client = _create_async_client()
        # Fail twice, succeed on third
        mock_client.create_payload_index.side_effect = itertools.chain(
            [
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
            ],
            itertools.repeat(None),
        )
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        await manager.ensure_payload_indexes()

        # Verify retry happened (3 attempts for first field + 1 each for rest)
        # Should have at least 3 calls for first field due to retries
        assert mock_client.create_payload_index.call_count >= 5

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_ensure_payload_indexes_validates_collection_exists(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """Should validate collection exists before creating indexes.

        Verifies:
        - Checks collection_exists() before creating indexes
        - Raises error if collection doesn't exist
        - Error message explains collection must be created first
        """
        mock_client = _create_async_client()
        mock_client.collection_exists = AsyncMock(return_value=False)
        mock_async_qdrant_client.return_value = mock_client

        manager = VectorStoreManager(
            qdrant_url="http://crawl4r-vectors:6333", collection_name="test_collection"
        )

        with pytest.raises(ValueError, match="Collection.*does not exist"):
            await manager.ensure_payload_indexes()

        # Verify collection_exists was checked
        mock_client.collection_exists.assert_called_once_with("test_collection")
        # Verify no index creation attempted
        mock_client.create_payload_index.assert_not_called()


class TestSearchSimilarQueryPoints:
    """Test search_similar uses query_points() instead of deprecated search()."""

    @patch("crawl4r.storage.qdrant.AsyncQdrantClient")
    async def test_uses_query_points_instead_of_search(
        self, mock_async_qdrant_client: MagicMock
    ) -> None:
        """search_similar should use query_points() not deprecated search().

        Verifies:
        - Calls query_points() method
        - Does NOT call deprecated search() method
        - Passes query parameter (not query_vector)
        - Returns correctly transformed results
        """
        from qdrant_client.models import ScoredPoint

        mock_client = _create_async_client()
        mock_async_qdrant_client.return_value = mock_client

        # Mock query_points response
        mock_query_response = MagicMock()
        mock_query_response.points = [
            ScoredPoint(
                id="test-uuid",
                version=1,
                score=0.95,
                payload={"chunk_text": "test content"},
                vector=None,
            )
        ]
        mock_client.query_points = AsyncMock(return_value=mock_query_response)

        manager = VectorStoreManager(
            qdrant_url="http://localhost:6333",
            collection_name="test",
        )

        query_vector = [0.1] * 1024
        results = await manager.search_similar(query_vector, top_k=5)

        # Verify query_points called (not search)
        mock_client.query_points.assert_called_once()
        mock_client.search.assert_not_called()

        # Verify correct parameters
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs["collection_name"] == "test"
        assert call_kwargs["query"] == query_vector
        assert call_kwargs["limit"] == 5

        # Verify result transformation
        assert len(results) == 1
        assert results[0]["id"] == "test-uuid"
        assert results[0]["score"] == 0.95
        assert results[0]["chunk_text"] == "test content"
