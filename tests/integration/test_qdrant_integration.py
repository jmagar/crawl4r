"""Integration tests for Qdrant vector database service.

Tests real Qdrant service endpoints to verify:
- Collection lifecycle (create, verify, delete)
- Vector upsert and retrieval operations
- Deletion by file path metadata
- Payload filtering by metadata fields

These tests require the Qdrant service to be running. The endpoint can be configured
via the QDRANT_URL environment variable. If not set, defaults to
http://localhost:52001. If the service is not available, tests will be skipped.

Example:
    Run only Qdrant integration tests:
    $ pytest tests/integration/test_qdrant_integration.py -v -m integration

    Run with custom endpoint:
    $ QDRANT_URL=http://crawl4r-vectors:6333 pytest tests/integration/test_qdrant_integration.py -v -m integration

    Run with service availability check:
    $ docker compose up -d crawl4r-vectors
    $ pytest tests/integration/test_qdrant_integration.py -v -m integration
"""

import os
import uuid
from typing import List

import httpx
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# Get Qdrant endpoint from environment or use default
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:52001")


@pytest.fixture
async def qdrant_client() -> AsyncQdrantClient:
    """Create Qdrant client configured for testing.

    Uses QDRANT_URL environment variable if set, otherwise defaults to
    http://localhost:52001.

    Returns:
        AsyncQdrantClient instance configured for integration testing

    Example:
        >>> async def test_example(qdrant_client):
        ...     collections = await qdrant_client.get_collections()
        ...     assert collections is not None
    """
    client = AsyncQdrantClient(url=QDRANT_URL, timeout=30.0)
    yield client
    await client.close()


@pytest.fixture
def test_collection_name() -> str:
    """Generate unique collection name for test isolation.

    Returns:
        Unique collection name with 'test_' prefix

    Example:
        >>> def test_example(test_collection_name):
        ...     assert test_collection_name.startswith("test_")
    """
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
async def check_qdrant_service() -> None:
    """Check if Qdrant service is available before running tests.

    Automatically runs before each test to verify the Qdrant service is
    reachable. Uses the QDRANT_URL environment variable or defaults to
    http://localhost:52001. If the service is not available, the test will
    be skipped with an informative message.

    Raises:
        pytest.skip: If Qdrant service is not available at configured endpoint

    Example:
        This fixture runs automatically for all tests in this module.
        No explicit usage required.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Qdrant health endpoint is at /readyz
            response = await client.get(f"{QDRANT_URL}/readyz")
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip(f"Qdrant service not available at {QDRANT_URL}")


@pytest.mark.integration
async def test_qdrant_collection_lifecycle(
    qdrant_client: AsyncQdrantClient, test_collection_name: str
) -> None:
    """Test Qdrant collection creation, verification, and deletion.

    Verifies:
    - Collections can be created with vector configuration
    - Created collections exist and are retrievable
    - Collections can be deleted
    - Deleted collections no longer exist

    Args:
        qdrant_client: AsyncQdrantClient fixture configured for localhost
        test_collection_name: Unique collection name for test isolation

    Raises:
        AssertionError: If collection lifecycle operations fail
    """
    # Create collection with 1024-dimensional vectors (Qwen3-Embedding-0.6B)
    await qdrant_client.create_collection(
        collection_name=test_collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    # Verify collection exists
    collection_exists = await qdrant_client.collection_exists(test_collection_name)
    assert collection_exists, f"Collection {test_collection_name} should exist"

    # Get collection info to verify configuration
    collection_info = await qdrant_client.get_collection(test_collection_name)
    assert (
        collection_info.config.params.vectors.size == 1024
    ), "Collection should have 1024 dimensions"
    assert (
        collection_info.config.params.vectors.distance == Distance.COSINE
    ), "Collection should use cosine distance"

    # Delete collection
    await qdrant_client.delete_collection(test_collection_name)

    # Verify collection no longer exists
    collection_exists = await qdrant_client.collection_exists(test_collection_name)
    assert not collection_exists, f"Collection {test_collection_name} should not exist"


@pytest.mark.integration
async def test_qdrant_upsert_and_retrieve(
    qdrant_client: AsyncQdrantClient, test_collection_name: str
) -> None:
    """Test Qdrant vector upsert and retrieval operations.

    Verifies:
    - Vectors can be upserted with metadata payloads
    - Vectors can be queried by similar vector
    - Retrieved results contain correct metadata
    - Query returns expected number of results

    Args:
        qdrant_client: AsyncQdrantClient fixture configured for localhost
        test_collection_name: Unique collection name for test isolation

    Raises:
        AssertionError: If upsert or retrieval operations fail
    """
    try:
        # Create collection
        await qdrant_client.create_collection(
            collection_name=test_collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

        # Create 5 test vectors with metadata
        points: List[PointStruct] = []
        for i in range(5):
            # Generate dummy 1024-dimensional vector (all zeros except index position)
            vector = [0.0] * 1024
            vector[i] = 1.0  # Make each vector slightly different

            points.append(
                PointStruct(
                    id=f"test_{i}",
                    vector=vector,
                    payload={
                        "file_path": f"/test/file_{i}.md",
                        "chunk_index": i,
                        "content": f"Test content {i}",
                    },
                )
            )

        # Upsert vectors
        await qdrant_client.upsert(collection_name=test_collection_name, points=points)

        # Query for similar vectors (using first vector as query)
        query_vector = [0.0] * 1024
        query_vector[0] = 1.0

        results = await qdrant_client.search(
            collection_name=test_collection_name, query_vector=query_vector, limit=3
        )

        # Verify results
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        # First result should be exact match (score = 1.0 for cosine similarity)
        assert results[0].id == "test_0", "First result should be test_0"
        assert results[0].payload["file_path"] == "/test/file_0.md"
        assert results[0].payload["chunk_index"] == 0

    finally:
        # Cleanup: delete test collection
        await qdrant_client.delete_collection(test_collection_name)


@pytest.mark.integration
async def test_qdrant_delete_by_file_path(
    qdrant_client: AsyncQdrantClient, test_collection_name: str
) -> None:
    """Test Qdrant deletion by file path metadata filter.

    Verifies:
    - Vectors for multiple files can be upserted
    - Vectors can be deleted by file_path metadata filter
    - Only vectors matching filter are deleted
    - Remaining vectors are still retrievable

    Args:
        qdrant_client: AsyncQdrantClient fixture configured for localhost
        test_collection_name: Unique collection name for test isolation

    Raises:
        AssertionError: If delete by filter operations fail
    """
    try:
        # Create collection
        await qdrant_client.create_collection(
            collection_name=test_collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

        # Create vectors for 2 different files
        points: List[PointStruct] = []
        for file_idx in range(2):
            for chunk_idx in range(3):
                vector = [0.0] * 1024
                vector[file_idx * 3 + chunk_idx] = 1.0

                points.append(
                    PointStruct(
                        id=f"file{file_idx}_chunk{chunk_idx}",
                        vector=vector,
                        payload={
                            "file_path": f"/test/file_{file_idx}.md",
                            "chunk_index": chunk_idx,
                        },
                    )
                )

        # Upsert all vectors (6 total: 3 per file)
        await qdrant_client.upsert(collection_name=test_collection_name, points=points)

        # Verify all vectors exist
        collection_info = await qdrant_client.get_collection(test_collection_name)
        assert (
            collection_info.points_count == 6
        ), f"Expected 6 points, got {collection_info.points_count}"

        # Delete vectors for file_0.md using payload filter
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        await qdrant_client.delete(
            collection_name=test_collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="file_path", match=MatchValue(value="/test/file_0.md")
                    )
                ]
            ),
        )

        # Verify only 3 vectors remain (from file_1.md)
        collection_info = await qdrant_client.get_collection(test_collection_name)
        assert (
            collection_info.points_count == 3
        ), f"Expected 3 points after deletion, got {collection_info.points_count}"

        # Verify remaining vectors are from file_1.md
        results = await qdrant_client.scroll(
            collection_name=test_collection_name, limit=10
        )
        remaining_points = results[0]  # scroll returns (points, next_offset)

        for point in remaining_points:
            assert (
                point.payload["file_path"] == "/test/file_1.md"
            ), f"Remaining point should be from file_1.md, got {point.payload['file_path']}"

    finally:
        # Cleanup: delete test collection
        await qdrant_client.delete_collection(test_collection_name)


@pytest.mark.integration
async def test_qdrant_payload_filtering(
    qdrant_client: AsyncQdrantClient, test_collection_name: str
) -> None:
    """Test Qdrant payload filtering during search operations.

    Verifies:
    - Vectors with different metadata can be upserted
    - Search results can be filtered by metadata fields
    - Filtered results only contain matching metadata
    - Filters work correctly with vector similarity search

    Args:
        qdrant_client: AsyncQdrantClient fixture configured for localhost
        test_collection_name: Unique collection name for test isolation

    Raises:
        AssertionError: If payload filtering operations fail
    """
    try:
        # Create collection
        await qdrant_client.create_collection(
            collection_name=test_collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

        # Create vectors with different filenames
        points: List[PointStruct] = []
        filenames = ["doc1.md", "doc2.md", "doc3.md"]

        for i, filename in enumerate(filenames):
            for chunk_idx in range(2):
                vector = [0.0] * 1024
                vector[i * 2 + chunk_idx] = 1.0

                points.append(
                    PointStruct(
                        id=f"{filename}_chunk{chunk_idx}",
                        vector=vector,
                        payload={"filename": filename, "chunk_index": chunk_idx},
                    )
                )

        # Upsert all vectors (6 total: 2 chunks per file)
        await qdrant_client.upsert(collection_name=test_collection_name, points=points)

        # Search with filter for specific filename
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_vector = [0.0] * 1024
        query_vector[0] = 1.0

        results = await qdrant_client.search(
            collection_name=test_collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value="doc1.md"))]
            ),
            limit=10,
        )

        # Verify only doc1.md results returned
        assert len(results) == 2, f"Expected 2 results for doc1.md, got {len(results)}"

        for result in results:
            assert (
                result.payload["filename"] == "doc1.md"
            ), f"Expected filename=doc1.md, got {result.payload['filename']}"

    finally:
        # Cleanup: delete test collection
        await qdrant_client.delete_collection(test_collection_name)
