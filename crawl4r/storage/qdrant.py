"""Qdrant vector store manager for RAG ingestion pipeline.

This module provides the VectorStoreManager class for managing Qdrant
collections, including collection creation, configuration, and validation.
Qdrant is used as the vector database for storing document embeddings with
metadata for semantic search and retrieval.

The VectorStoreManager handles:
- Idempotent collection creation with appropriate vector configurations
- Cosine distance metric (optimal for L2-normalized Qwen3 embeddings)
- Connection management to Qdrant server
- Vector upsert operations with retry logic and batch processing

Examples:
    Basic usage with default configuration:
        >>> manager = VectorStoreManager(
        ...     qdrant_url="http://crawl4r-vectors:6333",
        ...     collection_name="crawl4r"
        ... )
        >>> await manager.ensure_collection()

    Custom vector dimensions:
        >>> manager = VectorStoreManager(
        ...     qdrant_url="http://crawl4r-vectors:6333",
        ...     collection_name="crawl4r",
        ...     dimensions=512
        ... )
        >>> await manager.ensure_collection()

    Upserting vectors with metadata:
        >>> vector = [0.1] * 1024
        >>> metadata = {
        ...     "file_path": "/home/user/docs/test.md",
        ...     "chunk_index": 0,
        ...     "chunk_text": "Test content"
        ... }
        >>> await manager.upsert_vector(vector, metadata)

Notes:
    - All embeddings from Qwen3-Embedding-0.6B are L2-normalized (unit vectors)
    - Cosine distance is used for similarity search (appropriate for normalized
      vectors)
    - The ensure_collection() method is idempotent and safe to call multiple
      times
    - Upsert operations include automatic retry with exponential backoff
"""

import asyncio
import hashlib
import time
import uuid
from collections.abc import Callable
from typing import Any, TypedDict, cast

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from crawl4r.core.metadata import MetadataKeys

# Constants for retry and batch operations
MAX_RETRIES = 3
BATCH_SIZE = 100

# Payload index configuration for efficient metadata filtering
# Each tuple defines (field_name, schema_type) for Qdrant payload indexing
# These indexes enable fast filtering queries on metadata fields at scale
PAYLOAD_INDEXES: list[tuple[str, PayloadSchemaType]] = [
    (MetadataKeys.FILE_PATH, PayloadSchemaType.KEYWORD),
    (MetadataKeys.SOURCE_URL, PayloadSchemaType.KEYWORD),
    (MetadataKeys.SOURCE_TYPE, PayloadSchemaType.KEYWORD),
    (MetadataKeys.FILE_NAME, PayloadSchemaType.KEYWORD),
    (MetadataKeys.CHUNK_INDEX, PayloadSchemaType.INTEGER),
    (MetadataKeys.LAST_MODIFIED_DATE, PayloadSchemaType.KEYWORD),
    (MetadataKeys.TAGS, PayloadSchemaType.KEYWORD),
]


class VectorMetadataRequired(TypedDict):
    """Required fields for vector metadata."""

    file_path: str
    chunk_index: int
    chunk_text: str


class VectorMetadata(VectorMetadataRequired, total=False):
    """Type definition for vector metadata payload.

    This TypedDict defines the structure of metadata attached to each vector
    point in Qdrant. It ensures type safety when creating and accessing
    metadata fields.

    Required fields (from VectorMetadataRequired):
        file_path: Absolute path to source file (e.g., "/home/user/docs/test.md")
        chunk_index: Zero-based index of chunk within the file
        chunk_text: Full text content of the chunk

    Optional fields:
        file_name: Name of the file without directory path
        last_modified_date: ISO timestamp of file modification
        section_path: Markdown section hierarchy (e.g., "# Heading / ## Subheading")
        heading_level: Markdown heading level (0-6, 0 for no heading)
        tags: List of tags from frontmatter
        content_hash: SHA256 hash of full file content for integrity verification
        source_url: Source URL for crawled web content
        source_type: Source type identifier (e.g., "web_crawl")

    Examples:
        Minimal required metadata:
            >>> metadata: VectorMetadata = {
            ...     "file_path": "/home/user/docs/test.md",
            ...     "chunk_index": 0,
            ...     "chunk_text": "Test content"
            ... }

        Full metadata with optional fields:
            >>> metadata: VectorMetadata = {
            ...     "file_path": "/home/user/docs/test.md",
            ...     "file_name": "test.md",
            ...     "chunk_index": 0,
            ...     "chunk_text": "Test content",
            ...     "last_modified_date": "2026-01-15T12:00:00Z",
            ...     "section_path": "# Introduction",
            ...     "heading_level": 1,
            ...     "tags": ["documentation", "guide"],
            ...     "content_hash": "a1b2c3d4..."
            ... }
    """

    # Optional fields
    file_name: str
    last_modified_date: str
    section_path: str
    heading_level: int
    tags: list[str]
    content_hash: str
    source_url: str
    source_type: str


class VectorStoreManager:
    """Manages Qdrant vector store operations for document embeddings.

    This class provides methods for creating and configuring Qdrant collections
    with appropriate vector dimensions and distance metrics for semantic search.
    It serves as the primary interface for all Qdrant database operations in the
    RAG ingestion pipeline.

    The manager handles collection lifecycle operations and ensures proper
    configuration for storing embeddings generated by the TEI service using
    the Qwen3-Embedding-0.6B model.

    Attributes:
        qdrant_url: URL of the Qdrant server (e.g., "http://crawl4r-vectors:6333")
        collection_name: Name of the Qdrant collection to manage
        dimensions: Vector embedding dimensions (default 1024 for Qwen3)
    client: AsyncQdrantClient instance for database operations
    sync_client: QdrantClient instance for sync-only integrations

    Examples:
        Create a manager with default settings:
            >>> manager = VectorStoreManager(
            ...     qdrant_url="http://crawl4r-vectors:6333",
            ...     collection_name="crawl4r"
            ... )
            >>> await manager.ensure_collection()

        Use custom dimensions for embeddings:
            >>> manager = VectorStoreManager(
            ...     qdrant_url="http://crawl4r-vectors:6333",
            ...     collection_name="crawl4r",
            ...     dimensions=512
            ... )
            >>> await manager.ensure_collection()

    Notes:
        - The default dimensions (1024) match Qwen3-Embedding-0.6B output
        - Cosine distance is used because Qwen3 produces L2-normalized vectors
        - All operations are designed to be idempotent where possible
    """

    qdrant_url: str
    collection_name: str
    dimensions: int
    client: AsyncQdrantClient
    sync_client: QdrantClient

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        dimensions: int = 1024,
    ) -> None:
        """Initialize VectorStoreManager with Qdrant connection.

        Creates a connection to the Qdrant server and stores configuration
        parameters for collection management. The connection is established
        immediately upon initialization.

        Args:
            qdrant_url: URL of the Qdrant server including protocol and port.
                Examples: "http://localhost:6333", "http://crawl4r-vectors:6333"
            collection_name: Name of the collection to manage. This should match
                the collection name configured in the application settings.
            dimensions: Vector embedding dimensions. Must match the dimension
                size of embeddings produced by your embedding model. Default is
                1024 for Qwen3-Embedding-0.6B at full dimensions.

        Raises:
            ValueError: If qdrant_url is empty or malformed.
            ConnectionError: If unable to connect to Qdrant server.

        Examples:
            Production configuration:
                >>> manager = VectorStoreManager(
                ...     qdrant_url="http://crawl4r-vectors:6333",
                ...     collection_name="crawl4r"
                ... )

            Development with custom dimensions:
                >>> manager = VectorStoreManager(
                ...     qdrant_url="http://localhost:6333",
                ...     collection_name="test-collection",
                ...     dimensions=512
                ... )

        Notes:
            - AsyncQdrantClient and QdrantClient are created on initialization
            - No validation is performed on the URL format (delegated to client)
            - The dimensions value is stored but only used when creating
              collections
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.client = AsyncQdrantClient(url=qdrant_url)
        self.sync_client = QdrantClient(url=qdrant_url)

    async def ensure_collection(self) -> None:
        """Create collection if it does not exist (idempotent operation).

        Checks if the collection already exists in Qdrant. If not, creates
        a new collection with the configured vector dimensions and cosine
        distance metric. This method is idempotent and safe to call multiple
        times without side effects.

        The collection is configured with:
        - Vector size matching self.dimensions (default 1024 for Qwen3)
        - Cosine distance metric (optimal for L2-normalized embeddings)

        Returns:
            None. The method performs the operation silently and does not
            return a value.

        Raises:
            ConnectionError: If unable to communicate with Qdrant server.
            ValueError: If collection creation fails due to invalid parameters.

        Examples:
            First call creates the collection:
                >>> manager = VectorStoreManager(
                ...     qdrant_url="http://crawl4r-vectors:6333",
                ...     collection_name="crawl4r"
                ... )
                >>> await manager.ensure_collection()  # Creates collection

            Subsequent calls are no-ops:
                >>> await manager.ensure_collection()  # Skips (already exists)
                >>> await manager.ensure_collection()  # Skips (already exists)

        Notes:
            - Cosine distance is used because Qwen3-Embedding-0.6B produces
              L2-normalized vectors (unit vectors with norm = 1.0)
            - The method does not verify existing collection configuration.
              If a collection exists with different parameters, it will not
              be modified
            - Collection creation is asynchronous and non-blocking
        """
        # Check if collection already exists to avoid redundant creation
        if await self.client.collection_exists(self.collection_name):
            return

        # Create collection with vector configuration for Qwen3 embeddings
        # - size: Vector dimension count (must match embedding model output)
        # - distance: COSINE for normalized vectors (as opposed to EUCLID or DOT)
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
        )

    def _generate_point_id(
        self,
        file_path: str,
        chunk_index: int,
    ) -> str:
        """Generate deterministic UUID from file path and chunk index.

        Creates a deterministic point ID by hashing the absolute file path and
        chunk index together. This ensures that the same document chunk always
        gets the same ID, enabling idempotent upsert operations.

        Args:
            file_path: Absolute file path (e.g., "/home/user/docs/test.md")
            chunk_index: Position of chunk in document

        Returns:
            Deterministic UUID string derived from SHA256 hash

        Examples:
            Generate point ID for first chunk:
                >>> manager = VectorStoreManager(
                ...     qdrant_url="http://crawl4r-vectors:6333",
                ...     collection_name="crawl4r"
                ... )
                >>> point_id = manager._generate_point_id(
                ...     "/home/user/docs/test.md", 0
                ... )

        Notes:
            - Uses SHA256 for cryptographic-quality hash
            - Converts hash to UUID format for Qdrant compatibility
            - Same inputs always produce same UUID (deterministic)
            - Use absolute paths for consistency across the codebase
        """
        # Create deterministic hash from file path and chunk index
        content = f"{file_path}:{chunk_index}"
        hash_bytes = hashlib.sha256(content.encode()).digest()
        # Convert first 16 bytes to UUID (128-bit collision resistance)
        return str(uuid.UUID(bytes=hash_bytes[:16]))

    def _validate_vector(self, vector: list[float]) -> None:
        """Validate vector dimensions match collection configuration.

        Ensures the provided vector has the correct number of dimensions as
        configured for this collection. This prevents dimension mismatch errors
        when upserting vectors to Qdrant.

        Args:
            vector: Vector embedding to validate. Must be a list of floats
                with length matching self.dimensions.

        Raises:
            ValueError: If vector is empty (length 0) or has dimensions that
                don't match the collection's configured dimensions.

        Examples:
            Validate a 1024-dimensional vector:
                >>> manager = VectorStoreManager(
                ...     qdrant_url="http://crawl4r-vectors:6333",
                ...     collection_name="crawl4r",
                ...     dimensions=1024
                ... )
                >>> vector = [0.1] * 1024
                >>> manager._validate_vector(vector)  # No error

            Invalid vector dimensions:
                >>> bad_vector = [0.1] * 512
                >>> manager._validate_vector(bad_vector)  # Raises ValueError
        """
        if not vector:
            raise ValueError("Vector cannot be empty")
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimensions}, "
                f"got {len(vector)}"
            )

    def _validate_metadata(self, metadata: VectorMetadata) -> None:
        """Validate metadata contains required fields for vector storage.

        Checks that all required metadata fields are present in the provided
        dictionary. These fields are essential for vector identification and
        retrieval operations.

        Args:
            metadata: Metadata dictionary to validate. Must contain all
                required fields: 'file_path', 'chunk_index', and 'chunk_text'.

        Raises:
            ValueError: If any required field is missing from the metadata
                dictionary. The error message specifies which field is missing.

        Examples:
            Valid metadata:
                >>> manager = VectorStoreManager(
                ...     qdrant_url="http://crawl4r-vectors:6333",
                ...     collection_name="crawl4r"
                ... )
                >>> metadata = {
                ...     "file_path": "/home/user/docs/test.md",
                ...     "chunk_index": 0,
                ...     "chunk_text": "Test content"
                ... }
                >>> manager._validate_metadata(metadata)  # No error

            Missing required field:
                >>> bad_metadata = {
                ...     "file_path": "/home/user/docs/test.md",
                ...     "chunk_text": "Test content"
                ... }
                >>> manager._validate_metadata(bad_metadata)  # ValueError
        """
        required_fields = [
            MetadataKeys.FILE_PATH,
            MetadataKeys.CHUNK_INDEX,
            MetadataKeys.CHUNK_TEXT,
        ]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Metadata missing required field: {field}")

    async def _retry_with_backoff(
        self, operation: Callable[[], Any], max_retries: int = MAX_RETRIES
    ) -> None:
        """Retry operation with exponential backoff on network errors.

        Executes the provided operation and retries on UnexpectedResponse
        errors (network/server errors from Qdrant). Uses exponential backoff
        to avoid overwhelming the server: 1s, 2s, 4s delays between attempts.

        Args:
            operation: Callable operation to execute with retry logic. Should
                be a function that performs a Qdrant operation (e.g., upsert).
                Must not take any arguments.
            max_retries: Maximum number of retry attempts before giving up.
                Default is MAX_RETRIES (3 attempts: initial + 2 retries). Must
                be >= 1.

        Raises:
            RuntimeError: If all retry attempts fail. The error message
                includes the number of retries attempted and the underlying
                exception message from the last failed attempt.

        Examples:
            Retry an upsert operation:
                >>> manager = VectorStoreManager(
                ...     qdrant_url="http://crawl4r-vectors:6333",
                ...     collection_name="crawl4r"
                ... )
                >>> def upsert_op():
                ...     manager.client.upsert(
                ...         collection_name="crawl4r",
                ...         points=[...]
                ...     )
                >>> manager._retry_with_backoff(upsert_op, max_retries=3)

        Notes:
            - Only retries on UnexpectedResponse (network/server errors)
            - Other exceptions are raised immediately without retry
            - Backoff delays: 2^0=1s, 2^1=2s, 2^2=4s for attempts 0, 1, 2
            - Uses asyncio.sleep() for delays (non-blocking)
        """
        for attempt in range(max_retries):
            try:
                await operation()
                return
            except UnexpectedResponse as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed after {max_retries} retries: {e}")
                # Exponential backoff: 1s, 2s, 4s
                await asyncio.sleep(2**attempt)

    async def upsert_vector(self, vector: list[float], metadata: VectorMetadata) -> None:
        """Upsert single vector with metadata to Qdrant.

        Validates the vector and metadata, generates a deterministic point ID,
        and upserts the vector to Qdrant with retry logic.

        Args:
            vector: Embedding vector (must match collection dimensions)
            metadata: VectorMetadata dict with required fields:
                - file_path: Absolute path to source file
                - chunk_index: Index of chunk within file
                - chunk_text: Text content of the chunk

        Raises:
            ValueError: If vector dimensions are wrong or metadata is invalid
            RuntimeError: If upsert fails after max retries

        Examples:
            Upsert a single vector:
                >>> manager = VectorStoreManager(
                ...     qdrant_url="http://crawl4r-vectors:6333",
                ...     collection_name="crawl4r"
                ... )
                >>> vector = [0.1] * 1024
                >>> metadata = {
                ...     "file_path": "/home/user/docs/test.md",
                ...     "chunk_index": 0,
                ...     "chunk_text": "Test content"
                ... }
                >>> await manager.upsert_vector(vector, metadata)

        Notes:
            - Point ID is deterministic (SHA256 of file_path:chunk_index)
            - Retries MAX_RETRIES (3) times with exponential backoff (1s, 2s, 4s)
            - Validation happens before attempting upsert
        """
        # Validate inputs before attempting upsert
        self._validate_vector(vector)
        self._validate_metadata(metadata)

        # Generate deterministic point ID
        point_id = self._generate_point_id(
            metadata[MetadataKeys.FILE_PATH],
            metadata[MetadataKeys.CHUNK_INDEX],
        )

        # Create point structure (cast to dict[str, Any] for Qdrant compatibility)
        point = PointStruct(
            id=point_id, vector=vector, payload=cast(dict[str, Any], metadata)
        )

        # Upsert with retry logic
        async def upsert_operation() -> None:
            await self.client.upsert(
                collection_name=self.collection_name, points=[point]
            )

        await self._retry_with_backoff(upsert_operation)

    async def upsert_vectors_batch(self, vectors_with_metadata: list[dict]) -> None:
        """Upsert multiple vectors with metadata in batches.

        Validates all vectors and metadata, then upserts in batches of up to
        100 points. Each batch is retried independently on failure.

        Args:
            vectors_with_metadata: List of dictionaries with keys:
                - vector: Embedding vector (must match collection dimensions)
                - metadata: VectorMetadata dict with required fields

        Raises:
            ValueError: If any vector or metadata is invalid
            RuntimeError: If any batch fails after max retries

        Examples:
            Upsert multiple vectors:
                >>> manager = VectorStoreManager(
                ...     qdrant_url="http://crawl4r-vectors:6333",
                ...     collection_name="crawl4r"
                ... )
                >>> vectors_with_metadata = [
                ...     {
                ...         "vector": [0.1] * 1024,
                ...         "metadata": {
                ...             "file_path": "/home/user/docs/test.md",
                ...             "chunk_index": i,
                ...             "chunk_text": f"Chunk {i}"
                ...         }
                ...     }
                ...     for i in range(5)
                ... ]
                >>> await manager.upsert_vectors_batch(vectors_with_metadata)

        Notes:
            - Validates ALL vectors/metadata before upserting (all-or-nothing)
            - Splits into batches of BATCH_SIZE (100) points
            - Each batch retries independently with exponential backoff
            - Empty list is handled gracefully (no-op)
        """
        # Handle empty list gracefully
        if not vectors_with_metadata:
            return

        # Validate all vectors and metadata before upserting (all-or-nothing)
        for item in vectors_with_metadata:
            # Type assertions for validation
            vec: list[float] = item["vector"]
            meta: VectorMetadata = item["metadata"]
            self._validate_vector(vec)
            self._validate_metadata(meta)

        # Create all point structures with deterministic IDs
        points = []
        for item in vectors_with_metadata:
            meta: VectorMetadata = item["metadata"]
            point_id = self._generate_point_id(
                meta[MetadataKeys.FILE_PATH],
                meta[MetadataKeys.CHUNK_INDEX],
            )
            point = PointStruct(
                id=point_id,
                vector=item["vector"],
                payload=cast(dict[str, Any], meta),
            )
            points.append(point)

        # Split into batches and upsert each batch
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]

            # Upsert batch with retry logic
            # Use default arg to capture batch by value (avoid closure late-binding)
            async def upsert_batch_operation(
                batch: list[PointStruct] = batch,
            ) -> None:
                await self.client.upsert(
                    collection_name=self.collection_name, points=batch
                )

            await self._retry_with_backoff(upsert_batch_operation)

    async def search_similar(
        self, query_vector: list[float], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Search for similar vectors using query_points API.

        Uses the modern query_points() method (qdrant-client 1.16+) instead of
        the deprecated search() method. Returns the top_k most similar vectors
        with their metadata payloads.

        Args:
            query_vector: Query embedding vector (must match collection dimensions)
            top_k: Number of results to return (default: 10)

        Returns:
            List of dicts with 'id', 'score', and all payload fields

        Raises:
            ValueError: If query_vector dimensions don't match or top_k <= 0
            RuntimeError: If search fails after max retries
        """
        # Validate query vector dimensions
        self._validate_vector(query_vector)

        # Validate top_k is positive
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        # Perform search with retry logic
        query_response = None

        async def search_operation() -> None:
            nonlocal query_response
            query_response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
            )

        await self._retry_with_backoff(search_operation)

        # Transform QueryResponse.points to list of dicts
        results = []
        if query_response and query_response.points:
            for point in query_response.points:
                result = {
                    "id": str(point.id),
                    "score": point.score,
                    **(point.payload or {}),
                }
                results.append(result)

        return results

    async def delete_by_id(self, point_id: str) -> None:
        """Delete a single point by its UUID.

        Validates the UUID format and deletes the specified point from the
        collection. This operation is idempotent - deleting a non-existent ID
        will not raise an error.

        Args:
            point_id: UUID string of the point to delete. Must be a valid UUID
                format (e.g., "550e8400-e29b-41d4-a716-446655440000").

        Raises:
            ValueError: If point_id is not a valid UUID format.
            RuntimeError: If deletion fails after max retries.

        Examples:
            Delete a single point:
                >>> manager = VectorStoreManager(
                ...     qdrant_url="http://crawl4r-vectors:6333",
                ...     collection_name="crawl4r"
                ... )
                >>> point_id = "550e8400-e29b-41d4-a716-446655440000"
                >>> await manager.delete_by_id(point_id)

        Notes:
            - UUID format is validated before deletion attempt
            - Retries MAX_RETRIES (3) times with exponential backoff on errors
            - Deleting non-existent ID is safe (no error raised)
        """
        # Validate UUID format
        try:
            uuid.UUID(point_id)
        except ValueError:
            raise ValueError(f"Invalid UUID format: {point_id}")

        # Delete with retry logic
        async def delete_operation() -> None:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[point_id]),
            )

        await self._retry_with_backoff(delete_operation)

    async def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks for a given file.

        Uses scroll API to find all points with matching file_path
        filter, then deletes them in a batch. Returns the count of deleted
        points.

        Args:
            file_path: Absolute path to the file whose chunks should be deleted
                (e.g., "/home/user/docs/test.md").

        Returns:
            Count of deleted points (number of chunks removed).

        Raises:
            RuntimeError: If scroll or delete operations fail after max retries.

        Examples:
            Delete all chunks from a file:
                >>> manager = VectorStoreManager(
                ...     qdrant_url="http://crawl4r-vectors:6333",
                ...     collection_name="crawl4r"
                ... )
                >>> count = await manager.delete_by_file("/home/user/docs/test.md")
                >>> print(f"Deleted {count} chunks")

        Notes:
            - Uses scroll API with file_path filter
            - Handles pagination automatically
            - Returns 0 if no matching chunks found (no error)
            - Retries both scroll and delete operations independently
        """
        return await self._delete_by_filter(MetadataKeys.FILE_PATH, file_path)

    async def delete_by_url(self, source_url: str) -> int:
        """Delete all chunks for a given source URL.

        Uses scroll API to find all points with matching source_url
        filter, then deletes them in a batch. Returns the count of deleted
        points.

        Args:
            source_url: URL of the crawled page whose chunks should be deleted
                (e.g., "https://example.com/docs/page.html").

        Returns:
            Count of deleted points (number of chunks removed).

        Raises:
            RuntimeError: If scroll or delete operations fail after max retries.

        Examples:
            Delete all chunks from a URL:
                >>> manager = VectorStoreManager(
                ...     qdrant_url="http://crawl4r-vectors:6333",
                ...     collection_name="crawl4r"
                ... )
                >>> count = await manager.delete_by_url("https://example.com/docs/page.html")
                >>> print(f"Deleted {count} chunks")

        Notes:
            - Uses scroll API with source_url filter
            - Handles pagination automatically
            - Returns 0 if no matching chunks found (no error)
            - Retries both scroll and delete operations independently
        """
        return await self._delete_by_filter(MetadataKeys.SOURCE_URL, source_url)

    async def _delete_by_filter(self, field_name: str, value: str) -> int:
        """Delete all points matching a metadata filter.

        Args:
            field_name: Payload key to filter on (e.g., "file_path")
            value: Value to match for deletion

        Returns:
            Count of deleted points.
        """
        all_point_ids: list[int | str | uuid.UUID] = []
        next_page_offset = None

        async def scroll_operation() -> None:
            nonlocal all_point_ids, next_page_offset
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key=field_name,
                        match=MatchValue(value=value),
                    )
                ]
            )

            current_offset = next_page_offset
            while True:
                points, next_offset = await self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    offset=current_offset,
                )
                point_ids = cast(
                    list[int | str | uuid.UUID],
                    [point.id for point in points],
                )
                all_point_ids.extend(point_ids)
                if next_offset is None:
                    break
                current_offset = next_offset

        await self._retry_with_backoff(scroll_operation)

        if not all_point_ids:
            return 0

        async def delete_operation() -> None:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=all_point_ids),
            )

        await self._retry_with_backoff(delete_operation)

        return len(all_point_ids)

    async def ensure_payload_indexes(self) -> None:
        """Create payload indexes for efficient metadata filtering.

        Creates indexes on metadata fields to enable fast filtering by file path,
        filename, chunk index, modification date, and tags. This is critical for
        query performance at scale (1M+ vectors).

        Indexes are defined in the PAYLOAD_INDEXES module constant and include:
        - file_path (keyword): Exact match filtering by absolute file path
        - file_name (keyword): Exact match filtering by filename
        - chunk_index (integer): Range queries on chunk position
        - last_modified_date (keyword): Temporal queries on file modification
        - tags (keyword): Multi-value tag filtering

        Raises:
            ValueError: If collection does not exist (must call ensure_collection
                first).
            RuntimeError: If index creation fails after max retries.

        Examples:
            Create indexes after collection setup:
                >>> manager = VectorStoreManager(
                ...     qdrant_url="http://crawl4r-vectors:6333",
                ...     collection_name="crawl4r"
                ... )
                >>> await manager.ensure_collection()
                >>> await manager.ensure_payload_indexes()

            Safe to call multiple times (idempotent):
                >>> await manager.ensure_payload_indexes()  # First call creates
                >>> await manager.ensure_payload_indexes()  # Second call no-op

        Notes:
            - Collection must exist before creating indexes
            - Method is idempotent - handles "already exists" gracefully
            - Each index creation is retried independently with backoff
            - Indexes improve query performance but increase memory usage
            - Index configuration is centralized in PAYLOAD_INDEXES constant
        """
        # Validate collection exists before creating indexes
        if not await self.client.collection_exists(self.collection_name):
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist. "
                "Call ensure_collection() first."
            )

        # Create each index from PAYLOAD_INDEXES configuration
        for field_name, schema_type in PAYLOAD_INDEXES:

            # Use default args to capture values by value (avoid closure late-binding)
            async def create_index_operation(
                field_name: str = field_name,
                schema_type: PayloadSchemaType = schema_type,
            ) -> None:
                try:
                    await self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema_type,
                    )
                except Exception as e:
                    # Handle "already exists" gracefully (idempotent)
                    if "already exists" in str(e).lower():
                        return
                    # Re-raise other exceptions for retry logic
                    raise

            await self._retry_with_backoff(create_index_operation)

    async def scroll(self, limit: int = 100) -> list[dict]:
        """Scroll through all points in the collection.

        Retrieves all vectors and their payloads from the collection using the
        scroll API. Used for state recovery to determine which files have been
        processed.

        Args:
            limit: Number of points to fetch per scroll request (default: 100)

        Returns:
            List of point dictionaries with id and payload

        Raises:
            RuntimeError: If scroll operation fails after max retries

        Example:
            >>> manager = VectorStoreManager("http://localhost:6333", "crawl4r")
            >>> points = await manager.scroll()
            >>> for point in points:
            ...     print(point["payload"]["file_path"])
        """

        all_points = []
        offset = None

        while True:
            result, offset = await self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # We only need metadata
            )

            all_points.extend(
                [{"id": point.id, "payload": point.payload} for point in result]
            )

            # If no offset returned, we've scrolled through all points
            if offset is None:
                break

        return all_points
