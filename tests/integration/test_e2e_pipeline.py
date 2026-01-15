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
async def qdrant_client() -> AsyncQdrantClient:  # type: ignore[misc]
    """Create Qdrant client for test verification.

    Returns:
        AsyncQdrantClient instance configured for integration testing
    """
    client = AsyncQdrantClient(url=QDRANT_URL, timeout=30)
    yield client
    await client.close()


# Test data fixtures for reusable markdown samples
@pytest.fixture
def sample_doc1_content() -> str:
    """Sample markdown document with multiple sections.

    Returns:
        Markdown text with heading hierarchy and content for testing chunking
    """
    return """# Document 1

This is the first test document for end-to-end testing.

## Section 1

Some content in section 1.

## Section 2

More content in section 2.
"""


@pytest.fixture
def sample_doc2_content() -> str:
    """Sample markdown document with lists.

    Returns:
        Markdown text containing bullet lists and sections
    """
    return """# Document 2

This is the second test document.

## Features

- Feature 1
- Feature 2
- Feature 3

## Conclusion

End of document 2.
"""


@pytest.fixture
def sample_doc3_content() -> str:
    """Sample short markdown document.

    Returns:
        Minimal markdown text for testing single-chunk processing
    """
    return """# Document 3

Short document for testing.
"""


@pytest.fixture
def sample_original_content() -> str:
    """Sample markdown content for modification testing (original version).

    Returns:
        Markdown text representing the initial state before modification
    """
    return """# Original Document

This is the original content.

## Original Section

Some original text here.
"""


@pytest.fixture
def sample_modified_content() -> str:
    """Sample markdown content for modification testing (modified version).

    Returns:
        Markdown text representing the modified state with different content
    """
    return """# Modified Document

This is the MODIFIED content with different text.

## New Section

Completely different text in the new section.

## Another Section

Even more new content.
"""


@pytest.fixture
def sample_deletion_content() -> str:
    """Sample markdown content for deletion testing.

    Returns:
        Markdown text with multiple sections for testing deletion workflow
    """
    return """# Document to Delete

This document will be deleted.

## Section 1

Content in section 1.

## Section 2

Content in section 2.

## Section 3

Content in section 3.
"""


@pytest.mark.integration
async def test_e2e_document_ingestion(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
    qdrant_client: AsyncQdrantClient,
    sample_doc1_content: str,
    sample_doc2_content: str,
    sample_doc3_content: str,
) -> None:
    """Test complete end-to-end document ingestion pipeline.

    Verifies the full pipeline from markdown files to Qdrant storage:
    1. Create temporary directory with 3 markdown files
    2. Initialize all pipeline components (config, TEI client, vector store, chunker)
    3. Process files using DocumentProcessor.process_batch
    4. Query Qdrant to verify vectors are stored
    5. Validate: All files processed, correct chunk count, metadata present, embeddings 1024 dims

    This test validates the happy path where all components work together correctly.
    It ensures that files are chunked, embeddings are generated via TEI, and vectors
    are stored in Qdrant with proper metadata.

    Args:
        tmp_path: pytest temporary directory fixture
        test_collection: Unique collection name for test isolation
        cleanup_fixture: Ensures collection cleanup after test
        qdrant_client: AsyncQdrantClient for verification queries
        sample_doc1_content: Fixture providing markdown with multiple sections
        sample_doc2_content: Fixture providing markdown with lists
        sample_doc3_content: Fixture providing minimal markdown

    Raises:
        AssertionError: If pipeline does not complete successfully or validation fails
    """
    # Step 1: Create temporary directory with 3 markdown files
    # These files have varying complexity to test different chunking scenarios
    doc1 = tmp_path / "doc1.md"
    doc1.write_text(sample_doc1_content)

    doc2 = tmp_path / "doc2.md"
    doc2.write_text(sample_doc2_content)

    doc3 = tmp_path / "doc3.md"
    doc3.write_text(sample_doc3_content)

    # Step 2: Initialize all components with test-specific configuration
    # Using temporary directory as watch folder for test isolation
    config = Settings(
        watch_folder=tmp_path,
        tei_endpoint=TEI_ENDPOINT,
        qdrant_url=QDRANT_URL,
        collection_name=test_collection,
        chunk_size_tokens=512,
        chunk_overlap_percent=15,
    )

    # Initialize TEI client for embedding generation
    # Dimensions=1024 matches Qwen3-Embedding-0.6B model output
    tei_client = TEIClient(
        endpoint_url=config.tei_endpoint,
        dimensions=1024,
        timeout=30.0,
        max_retries=3,
    )

    # Initialize vector store manager for Qdrant operations
    vector_store = VectorStoreManager(
        qdrant_url=config.qdrant_url,
        collection_name=config.collection_name,
        dimensions=1024,
    )

    # Create collection in Qdrant with cosine similarity distance metric
    # This must be done before vectors can be stored
    await qdrant_client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    # Initialize markdown chunker with heading-based splitting
    chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)

    # Initialize document processor that orchestrates the pipeline
    processor = DocumentProcessor(config, tei_client, vector_store, chunker)

    # Step 3: Process all files in batch using the processor
    # This triggers: file read → chunking → embedding → Qdrant storage
    files = [doc1, doc2, doc3]
    results = await processor.process_batch(files)

    # Step 4: Verify all files processed successfully
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all(r.success for r in results), "All documents should process successfully"

    # Verify that at least some chunks were extracted and processed
    total_chunks = sum(r.chunks_processed for r in results)
    assert total_chunks > 0, "Should process at least some chunks"

    # Step 5: Query Qdrant to verify vectors are stored correctly
    collection_info = await qdrant_client.get_collection(test_collection)
    assert (
        collection_info.points_count > 0  # type: ignore[operator]
    ), "Collection should contain vectors after processing"

    # Verify point count matches total chunks processed
    assert (
        collection_info.points_count == total_chunks
    ), f"Expected {total_chunks} vectors, got {collection_info.points_count}"

    # Verify vector dimensions match TEI model output (1024)
    assert (
        collection_info.config.params.vectors.size == 1024  # type: ignore[union-attr]
    ), "Vectors should have 1024 dimensions"

    # Retrieve sample points to verify metadata is properly attached
    points, _ = await qdrant_client.scroll(collection_name=test_collection, limit=10)
    assert len(points) > 0, "Should retrieve at least some points"

    # Validate that the first point has all required metadata fields
    # Metadata is critical for deletion, re-ingestion, and search functionality
    first_point = points[0]
    assert first_point.payload is not None, "Point should have metadata payload"

    required_fields = [
        "file_path_relative",  # For deletion queries
        "file_path_absolute",  # For file access
        "filename",  # For filtering
        "modification_date",  # For state recovery
        "chunk_index",  # For ordering
        "chunk_text",  # For retrieval
        "section_path",  # For context
        "heading_level",  # For hierarchy
        "content_hash",  # For change detection
    ]

    for field in required_fields:
        assert (
            field in first_point.payload
        ), f"Metadata missing required field: {field}"

    # Verify chunk_text is not empty (validates chunking worked)
    assert (
        len(first_point.payload["chunk_text"]) > 0
    ), "Chunk text should not be empty"


@pytest.mark.integration
async def test_e2e_file_modification(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
    qdrant_client: AsyncQdrantClient,
    sample_original_content: str,
    sample_modified_content: str,
) -> None:
    """Test end-to-end file modification and re-ingestion.

    Verifies that modified files are correctly re-processed with vector replacement:
    1. Process file once and verify initial storage
    2. Modify file content
    3. Delete old vectors by file path
    4. Re-process modified file
    5. Query Qdrant to verify old vectors removed and new vectors stored
    6. Verify modification_date is updated

    This test validates the idempotent re-ingestion pattern used by the file watcher
    when it detects file modifications. The pattern is: delete old vectors → re-process file.

    Args:
        tmp_path: pytest temporary directory fixture
        test_collection: Unique collection name for test isolation
        cleanup_fixture: Ensures collection cleanup after test
        qdrant_client: AsyncQdrantClient for verification queries
        sample_original_content: Fixture providing original markdown content
        sample_modified_content: Fixture providing modified markdown content

    Raises:
        AssertionError: If modification workflow does not work correctly
    """
    # Step 1: Create initial markdown file and process it
    doc = tmp_path / "test.md"
    doc.write_text(sample_original_content)

    # Initialize components for testing
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

    # Create collection before processing
    await qdrant_client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
    processor = DocumentProcessor(config, tei_client, vector_store, chunker)

    # Process the original file and verify it succeeds
    result = await processor.process_document(doc)
    assert result.success, "Initial processing should succeed"

    original_chunks = result.chunks_processed
    assert original_chunks > 0, "Should process at least one chunk"

    # Verify initial vectors are stored in Qdrant
    collection_info = await qdrant_client.get_collection(test_collection)
    assert (
        collection_info.points_count == original_chunks
    ), f"Expected {original_chunks} vectors"

    # Capture the original modification date for comparison later
    # This validates that modification tracking works correctly
    original_points, _ = await qdrant_client.scroll(
        collection_name=test_collection, limit=10
    )
    original_mod_date = original_points[0].payload["modification_date"]  # type: ignore[index]

    # Step 2: Modify file content
    # Add small delay to ensure filesystem modification time changes
    import time

    time.sleep(0.1)
    doc.write_text(sample_modified_content)

    # Step 3: Delete old vectors by file path (simulates watcher cleanup)
    # This is the first step of the re-ingestion pattern
    file_path_relative = str(doc.relative_to(tmp_path))
    deleted_count = vector_store.delete_by_file(file_path_relative)
    assert (
        deleted_count == original_chunks
    ), f"Should delete {original_chunks} vectors, deleted {deleted_count}"

    # Verify all old vectors are removed before re-processing
    collection_info = await qdrant_client.get_collection(test_collection)
    assert collection_info.points_count == 0, "All vectors should be deleted"

    # Step 4: Re-process the modified file
    # This generates new chunks and embeddings from the updated content
    result = await processor.process_document(doc)
    assert result.success, "Re-processing should succeed"

    new_chunks = result.chunks_processed
    assert new_chunks > 0, "Should process at least one chunk"

    # Step 5: Verify new vectors are stored in Qdrant
    collection_info = await qdrant_client.get_collection(test_collection)
    assert (
        collection_info.points_count == new_chunks
    ), f"Expected {new_chunks} new vectors"

    # Step 6: Verify modification_date metadata is updated
    # This is critical for state recovery and change detection
    new_points, _ = await qdrant_client.scroll(
        collection_name=test_collection, limit=10
    )
    new_mod_date = new_points[0].payload["modification_date"]  # type: ignore[index]
    assert (
        new_mod_date != original_mod_date
    ), "Modification date should be updated"

    # Verify new content is actually stored in the vectors
    # Check for keywords that only appear in the modified version
    assert (
        "MODIFIED" in new_points[0].payload["chunk_text"]  # type: ignore[index]
        or "different" in new_points[0].payload["chunk_text"]  # type: ignore[index]
    ), "New content should be stored in vectors"


@pytest.mark.integration
async def test_e2e_file_deletion(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
    qdrant_client: AsyncQdrantClient,
    sample_deletion_content: str,
) -> None:
    """Test end-to-end file deletion and vector cleanup.

    Verifies that file deletion correctly removes all associated vectors:
    1. Process file and verify storage
    2. Delete vectors by file path
    3. Query Qdrant to verify no vectors remain for the file

    This test validates the cleanup pattern used by the file watcher when it detects
    file deletions. It ensures that all vectors associated with a file are removed
    from Qdrant using the file_path_relative metadata field.

    Args:
        tmp_path: pytest temporary directory fixture
        test_collection: Unique collection name for test isolation
        cleanup_fixture: Ensures collection cleanup after test
        qdrant_client: AsyncQdrantClient for verification queries
        sample_deletion_content: Fixture providing markdown content with multiple sections

    Raises:
        AssertionError: If deletion workflow does not work correctly
    """
    # Step 1: Create markdown file with multiple sections
    # Using multiple sections ensures we test deletion of multiple vectors
    doc = tmp_path / "delete_test.md"
    doc.write_text(sample_deletion_content)

    # Initialize components for testing
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

    # Create collection before processing
    await qdrant_client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
    processor = DocumentProcessor(config, tei_client, vector_store, chunker)

    # Process the file and verify it succeeds
    result = await processor.process_document(doc)
    assert result.success, "Processing should succeed"

    chunks_processed = result.chunks_processed
    assert chunks_processed > 0, "Should process at least one chunk"

    # Verify vectors are stored in Qdrant
    collection_info = await qdrant_client.get_collection(test_collection)
    assert (
        collection_info.points_count == chunks_processed
    ), f"Expected {chunks_processed} vectors"

    # Step 2: Delete vectors by file path
    # This simulates the file watcher's deletion handler
    # Uses file_path_relative metadata field to identify all vectors for the file
    file_path_relative = str(doc.relative_to(tmp_path))
    deleted_count = vector_store.delete_by_file(file_path_relative)
    assert (
        deleted_count == chunks_processed
    ), f"Should delete {chunks_processed} vectors, deleted {deleted_count}"

    # Step 3: Query Qdrant to verify complete removal
    # Both collection info and scroll should confirm no vectors remain
    collection_info = await qdrant_client.get_collection(test_collection)
    assert (
        collection_info.points_count == 0
    ), "No vectors should remain after deletion"

    # Double-check by scrolling through the collection
    # This ensures the delete operation was atomic and complete
    points, _ = await qdrant_client.scroll(collection_name=test_collection, limit=10)
    assert len(points) == 0, "Should retrieve no points after deletion"
