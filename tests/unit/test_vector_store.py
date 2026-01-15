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
