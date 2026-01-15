"""End-to-end integration tests for the full RAG ingestion pipeline.

Tests the complete document processing pipeline from markdown files through
chunking, embedding generation, and vector storage in Qdrant. These tests
verify that all components work together correctly in realistic scenarios.

Test scenarios covered:
- Complete document ingestion pipeline (file → chunks → embeddings → Qdrant)
- File modification and re-ingestion with vector replacement
- File deletion and vector cleanup

These tests require all services (TEI, Qdrant) to be running and accessible.
Tests use temporary directories and unique collections for isolation.

Example:
    Run end-to-end tests:
    $ pytest tests/integration/test_e2e_pipeline.py -v -m integration

    Run with service availability check:
    $ docker compose up -d crawl4r-embeddings crawl4r-vectors
    $ pytest tests/integration/test_e2e_pipeline.py -v -m integration
"""

import os
from pathlib import Path

import httpx
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from rag_ingestion.chunker import MarkdownChunker
from rag_ingestion.config import Settings
from rag_ingestion.processor import DocumentProcessor
from rag_ingestion.tei_client import TEIClient
from rag_ingestion.vector_store import VectorStoreManager

# Get service endpoints from environment or use defaults
TEI_ENDPOINT = os.getenv("TEI_ENDPOINT", "http://localhost:52000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:52001")


@pytest.fixture(autouse=True)
async def check_services() -> None:
    """Check if TEI and Qdrant services are available before running tests.

    Automatically runs before each test to verify both services are reachable.
    If either service is unavailable, the test will be skipped with an
    informative message.

    Raises:
        pytest.skip: If TEI or Qdrant service is not available
    """
    # Check TEI service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{TEI_ENDPOINT}/health")
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip(f"TEI service not available at {TEI_ENDPOINT}")

    # Check Qdrant service
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{QDRANT_URL}/readyz")
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip(f"Qdrant service not available at {QDRANT_URL}")


@pytest.fixture
async def qdrant_client() -> AsyncQdrantClient:
    """Create Qdrant client for test verification.

    Returns:
        AsyncQdrantClient instance configured for integration testing
    """
    client = AsyncQdrantClient(url=QDRANT_URL, timeout=30.0)
    yield client
    await client.close()


@pytest.mark.integration
async def test_e2e_document_ingestion(
    tmp_path: Path, test_collection: str, cleanup_fixture: None, qdrant_client: AsyncQdrantClient
) -> None:
    """Test complete end-to-end document ingestion pipeline.

    Verifies the full pipeline from markdown files to Qdrant storage:
    1. Create temporary directory with 3 markdown files
    2. Initialize all pipeline components (config, TEI client, vector store, chunker)
    3. Process files using DocumentProcessor.process_batch
    4. Query Qdrant to verify vectors are stored
    5. Validate: All files processed, correct chunk count, metadata present, embeddings 1024 dims

    Args:
        tmp_path: pytest temporary directory fixture
        test_collection: Unique collection name for test isolation
        cleanup_fixture: Ensures collection cleanup after test
        qdrant_client: AsyncQdrantClient for verification queries

    Raises:
        AssertionError: If pipeline does not complete successfully or validation fails
    """
    # Create temp directory with 3 markdown files
    doc1 = tmp_path / "doc1.md"
    doc1.write_text(
        """# Document 1

This is the first test document for end-to-end testing.

## Section 1

Some content in section 1.

## Section 2

More content in section 2.
"""
    )

    doc2 = tmp_path / "doc2.md"
    doc2.write_text(
        """# Document 2

This is the second test document.

## Features

- Feature 1
- Feature 2
- Feature 3

## Conclusion

End of document 2.
"""
    )

    doc3 = tmp_path / "doc3.md"
    doc3.write_text(
        """# Document 3

Short document for testing.
"""
    )

    # Initialize all components with test config
    config = Settings(
        watch_folder=tmp_path,
        tei_endpoint=TEI_ENDPOINT,
        qdrant_url=QDRANT_URL,
        collection_name=test_collection,
        chunk_size_tokens=512,
        chunk_overlap_percent=15,
    )

    # Initialize TEI client
    tei_client = TEIClient(
        endpoint_url=config.tei_endpoint,
        dimensions=1024,
        timeout=30.0,
        max_retries=3,
    )

    # Initialize vector store manager
    vector_store = VectorStoreManager(
        qdrant_url=config.qdrant_url,
        collection_name=config.collection_name,
        dimensions=1024,
    )

    # Create collection in Qdrant
    await qdrant_client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    # Initialize chunker
    chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)

    # Initialize processor
    processor = DocumentProcessor(config, tei_client, vector_store, chunker)

    # Process files using processor.process_batch
    files = [doc1, doc2, doc3]
    results = await processor.process_batch(files)

    # Verify all files processed successfully
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all(r.success for r in results), "All documents should process successfully"

    # Verify chunks were processed
    total_chunks = sum(r.chunks_processed for r in results)
    assert total_chunks > 0, "Should process at least some chunks"

    # Query Qdrant for stored vectors
    collection_info = await qdrant_client.get_collection(test_collection)
    assert (
        collection_info.points_count > 0
    ), "Collection should contain vectors after processing"
    assert (
        collection_info.points_count == total_chunks
    ), f"Expected {total_chunks} vectors, got {collection_info.points_count}"

    # Verify vector dimensions are correct (1024)
    assert (
        collection_info.config.params.vectors.size == 1024
    ), "Vectors should have 1024 dimensions"

    # Verify metadata is present by scrolling through points
    points, _ = await qdrant_client.scroll(collection_name=test_collection, limit=10)

    assert len(points) > 0, "Should retrieve at least some points"

    # Check first point has all required metadata fields
    first_point = points[0]
    assert first_point.payload is not None, "Point should have metadata payload"

    required_fields = [
        "file_path_relative",
        "file_path_absolute",
        "filename",
        "modification_date",
        "chunk_index",
        "chunk_text",
        "section_path",
        "heading_level",
        "content_hash",
    ]

    for field in required_fields:
        assert (
            field in first_point.payload
        ), f"Metadata missing required field: {field}"

    # Verify chunk_text is not empty
    assert (
        len(first_point.payload["chunk_text"]) > 0
    ), "Chunk text should not be empty"


@pytest.mark.integration
async def test_e2e_file_modification(
    tmp_path: Path, test_collection: str, cleanup_fixture: None, qdrant_client: AsyncQdrantClient
) -> None:
    """Test end-to-end file modification and re-ingestion.

    Verifies that modified files are correctly re-processed with vector replacement:
    1. Process file once and verify initial storage
    2. Modify file content
    3. Delete old vectors by file path
    4. Re-process modified file
    5. Query Qdrant to verify old vectors removed and new vectors stored
    6. Verify modification_date is updated

    Args:
        tmp_path: pytest temporary directory fixture
        test_collection: Unique collection name for test isolation
        cleanup_fixture: Ensures collection cleanup after test
        qdrant_client: AsyncQdrantClient for verification queries

    Raises:
        AssertionError: If modification workflow does not work correctly
    """
    # Create initial markdown file
    doc = tmp_path / "test.md"
    original_content = """# Original Document

This is the original content.

## Original Section

Some original text here.
"""
    doc.write_text(original_content)

    # Initialize components
    config = Settings(
        watch_folder=tmp_path,
        tei_endpoint=TEI_ENDPOINT,
        qdrant_url=QDRANT_URL,
        collection_name=test_collection,
        chunk_size_tokens=512,
        chunk_overlap_percent=15,
    )

    tei_client = TEIClient(
        endpoint_url=config.tei_endpoint,
        dimensions=1024,
        timeout=30.0,
        max_retries=3,
    )

    vector_store = VectorStoreManager(
        qdrant_url=config.qdrant_url,
        collection_name=config.collection_name,
        dimensions=1024,
    )

    # Create collection
    await qdrant_client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
    processor = DocumentProcessor(config, tei_client, vector_store, chunker)

    # Process file once
    result = await processor.process_document(doc)
    assert result.success, "Initial processing should succeed"

    original_chunks = result.chunks_processed
    assert original_chunks > 0, "Should process at least one chunk"

    # Verify initial storage
    collection_info = await qdrant_client.get_collection(test_collection)
    assert (
        collection_info.points_count == original_chunks
    ), f"Expected {original_chunks} vectors"

    # Get original modification date
    original_points, _ = await qdrant_client.scroll(
        collection_name=test_collection, limit=10
    )
    original_mod_date = original_points[0].payload["modification_date"]

    # Modify file content
    import time

    time.sleep(0.1)  # Ensure modification time changes
    modified_content = """# Modified Document

This is the MODIFIED content with different text.

## New Section

Completely different text in the new section.

## Another Section

Even more new content.
"""
    doc.write_text(modified_content)

    # Delete old vectors by file path (simulate re-ingestion cleanup)
    file_path_relative = str(doc.relative_to(tmp_path))
    deleted_count = vector_store.delete_by_file(file_path_relative)
    assert (
        deleted_count == original_chunks
    ), f"Should delete {original_chunks} vectors, deleted {deleted_count}"

    # Verify deletion
    collection_info = await qdrant_client.get_collection(test_collection)
    assert collection_info.points_count == 0, "All vectors should be deleted"

    # Re-process modified file
    result = await processor.process_document(doc)
    assert result.success, "Re-processing should succeed"

    new_chunks = result.chunks_processed
    assert new_chunks > 0, "Should process at least one chunk"

    # Query Qdrant to verify new vectors stored
    collection_info = await qdrant_client.get_collection(test_collection)
    assert (
        collection_info.points_count == new_chunks
    ), f"Expected {new_chunks} new vectors"

    # Verify modification_date is updated
    new_points, _ = await qdrant_client.scroll(
        collection_name=test_collection, limit=10
    )
    new_mod_date = new_points[0].payload["modification_date"]
    assert (
        new_mod_date != original_mod_date
    ), "Modification date should be updated"

    # Verify new content is stored
    assert (
        "MODIFIED" in new_points[0].payload["chunk_text"]
        or "different" in new_points[0].payload["chunk_text"]
    ), "New content should be stored in vectors"


@pytest.mark.integration
async def test_e2e_file_deletion(
    tmp_path: Path, test_collection: str, cleanup_fixture: None, qdrant_client: AsyncQdrantClient
) -> None:
    """Test end-to-end file deletion and vector cleanup.

    Verifies that file deletion correctly removes all associated vectors:
    1. Process file and verify storage
    2. Delete vectors by file path
    3. Query Qdrant to verify no vectors remain for the file

    Args:
        tmp_path: pytest temporary directory fixture
        test_collection: Unique collection name for test isolation
        cleanup_fixture: Ensures collection cleanup after test
        qdrant_client: AsyncQdrantClient for verification queries

    Raises:
        AssertionError: If deletion workflow does not work correctly
    """
    # Create markdown file
    doc = tmp_path / "delete_test.md"
    doc.write_text(
        """# Document to Delete

This document will be deleted.

## Section 1

Content in section 1.

## Section 2

Content in section 2.

## Section 3

Content in section 3.
"""
    )

    # Initialize components
    config = Settings(
        watch_folder=tmp_path,
        tei_endpoint=TEI_ENDPOINT,
        qdrant_url=QDRANT_URL,
        collection_name=test_collection,
        chunk_size_tokens=512,
        chunk_overlap_percent=15,
    )

    tei_client = TEIClient(
        endpoint_url=config.tei_endpoint,
        dimensions=1024,
        timeout=30.0,
        max_retries=3,
    )

    vector_store = VectorStoreManager(
        qdrant_url=config.qdrant_url,
        collection_name=config.collection_name,
        dimensions=1024,
    )

    # Create collection
    await qdrant_client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
    processor = DocumentProcessor(config, tei_client, vector_store, chunker)

    # Process file
    result = await processor.process_document(doc)
    assert result.success, "Processing should succeed"

    chunks_processed = result.chunks_processed
    assert chunks_processed > 0, "Should process at least one chunk"

    # Verify storage
    collection_info = await qdrant_client.get_collection(test_collection)
    assert (
        collection_info.points_count == chunks_processed
    ), f"Expected {chunks_processed} vectors"

    # Delete vectors by file path
    file_path_relative = str(doc.relative_to(tmp_path))
    deleted_count = vector_store.delete_by_file(file_path_relative)
    assert (
        deleted_count == chunks_processed
    ), f"Should delete {chunks_processed} vectors, deleted {deleted_count}"

    # Query Qdrant to verify no vectors remain
    collection_info = await qdrant_client.get_collection(test_collection)
    assert (
        collection_info.points_count == 0
    ), "No vectors should remain after deletion"

    # Verify scroll returns empty results
    points, _ = await qdrant_client.scroll(collection_name=test_collection, limit=10)
    assert len(points) == 0, "Should retrieve no points after deletion"
