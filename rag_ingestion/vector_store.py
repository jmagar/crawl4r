"""Qdrant vector store manager for RAG ingestion pipeline.

This module provides the VectorStoreManager class for managing Qdrant
collections, including collection creation, configuration, and validation.

Examples:
    >>> manager = VectorStoreManager(
    ...     qdrant_url="http://crawl4r-vectors:6333",
    ...     collection_name="crawl4r"
    ... )
    >>> manager.ensure_collection()
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class VectorStoreManager:
    """Manages Qdrant vector store operations for document embeddings.

    This class provides methods for creating and configuring Qdrant collections
    with appropriate vector dimensions and distance metrics for semantic search.

    Attributes:
        qdrant_url: URL of the Qdrant server
        collection_name: Name of the Qdrant collection to manage
        dimensions: Vector embedding dimensions (default 1024 for Qwen3)
        client: QdrantClient instance for database operations

    Examples:
        >>> manager = VectorStoreManager(
        ...     qdrant_url="http://crawl4r-vectors:6333",
        ...     collection_name="crawl4r",
        ...     dimensions=1024
        ... )
        >>> manager.ensure_collection()
    """

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        dimensions: int = 1024
    ) -> None:
        """Initialize VectorStoreManager with Qdrant connection.

        Args:
            qdrant_url: URL of the Qdrant server (e.g., "http://localhost:6333")
            collection_name: Name of the collection to manage
            dimensions: Vector embedding dimensions (default 1024)

        Examples:
            >>> manager = VectorStoreManager(
            ...     qdrant_url="http://crawl4r-vectors:6333",
            ...     collection_name="crawl4r"
            ... )
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.client = QdrantClient(url=qdrant_url)

    def ensure_collection(self) -> None:
        """Create collection if it does not exist.

        Checks if the collection already exists in Qdrant. If not, creates
        a new collection with the configured vector dimensions and cosine
        distance metric (appropriate for normalized embeddings from Qwen3).

        The method is idempotent - it can be called multiple times safely.

        Examples:
            >>> manager = VectorStoreManager(
            ...     qdrant_url="http://crawl4r-vectors:6333",
            ...     collection_name="crawl4r"
            ... )
            >>> manager.ensure_collection()  # Creates collection
            >>> manager.ensure_collection()  # Skips creation (exists)
        """
        # Check if collection already exists
        if self.client.collection_exists(self.collection_name):
            return

        # Create collection with configured vector dimensions and cosine distance
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimensions,
                distance=Distance.COSINE
            )
        )
