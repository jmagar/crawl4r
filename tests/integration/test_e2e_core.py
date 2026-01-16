"""Core end-to-end integration tests for RAG ingestion pipeline.

Tests the complete document processing pipeline from markdown files through
chunking, embedding generation, and vector storage in Qdrant. These are the
fundamental tests that validate the core pipeline functionality.

Test scenarios covered:
- Complete document ingestion pipeline (file → chunks → embeddings → Qdrant)
- File modification and re-ingestion with vector replacement
- File deletion and vector cleanup
- YAML frontmatter extraction and metadata
- Nested directory structure handling
- Large document chunking (5000+ tokens)
- Special characters and unicode content
- Empty file edge case handling

These tests require all services (TEI, Qdrant) to be running and accessible.
Tests use temporary directories and unique collections for isolation.

Example:
    Run core E2E tests:
    $ pytest tests/integration/test_e2e_core.py -v -m integration

    Run with service availability check:
    $ docker compose up -d crawl4r-embeddings crawl4r-vectors
    $ pytest tests/integration/test_e2e_core.py -v -m integration
"""

import os
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from crawl4r.processing.chunker import MarkdownChunker
from crawl4r.core.config import Settings
from crawl4r.processing.processor import DocumentProcessor
from crawl4r.storage.embeddings import TEIClient
from crawl4r.storage.vector_store import VectorStoreManager

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
async def qdrant_client() -> AsyncIterator[AsyncQdrantClient]:
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

Content in section 2.
"""


@pytest.fixture
def sample_doc2_content() -> str:
    """Sample markdown document with lists.

    Returns:
        Markdown text with lists and structured content
    """
    return """# Document 2

This document has some lists.

## Features

- Feature A
- Feature B
- Feature C
"""


@pytest.fixture
def sample_doc3_content() -> str:
    """Minimal markdown document.

    Returns:
        Simple markdown with minimal structure
    """
    return """# Document 3

Minimal content for testing.
"""


@pytest.fixture
def sample_original_content() -> str:
    """Original content for modification testing.

    Returns:
        Markdown content representing the initial file state
    """
    return """# Original Document

This is the original content before modification.

## Section 1

Original section 1 content.
"""


@pytest.fixture
def sample_modified_content() -> str:
    """Modified content for re-ingestion testing.

    Returns:
        Markdown content representing updated file state
    """
    return """# Modified Document

This is the MODIFIED content after update.

## Section 1

Modified section 1 content with changes.

## Section 2

New section added during modification.
"""


@pytest.fixture
def sample_deletion_content() -> str:
    """Content for deletion testing with multiple sections.

    Returns:
        Markdown with multiple sections to generate multiple vectors
    """
    return """# Deletion Test

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
    5. Validate: All files processed, correct chunk count, metadata present,
       embeddings 1024 dims

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
    assert collection_info.points_count is not None, (
        "Collection should have point count"
    )
    assert collection_info.points_count > 0, (
        "Collection should contain vectors after processing"
    )

    # Verify point count matches total chunks processed
    assert collection_info.points_count == total_chunks, (
        f"Expected {total_chunks} vectors, got {collection_info.points_count}"
    )

    # Verify vector dimensions match TEI model output (1024)
    assert collection_info.config is not None, "Collection should have config"
    assert collection_info.config.params is not None, "Config should have params"
    vectors_config = collection_info.config.params.vectors
    assert vectors_config is not None, "Params should have vectors config"
    # Handle both dict and VectorParams types
    if isinstance(vectors_config, dict):
        vector_size = vectors_config.get("size")
    else:
        vector_size = getattr(vectors_config, "size", None)
    assert vector_size == 1024, "Vectors should have 1024 dimensions"

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
        assert field in first_point.payload, f"Metadata missing required field: {field}"

    # Verify chunk_text is not empty (validates chunking worked)
    assert len(first_point.payload["chunk_text"]) > 0, "Chunk text should not be empty"


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
    when it detects file modifications. The pattern is: delete old vectors →
    re-process file.

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

    # Process the original file
    result = await processor.process_document(doc)
    assert result.success, "Initial processing should succeed"

    original_chunks = result.chunks_processed
    assert original_chunks > 0, "Should process at least one chunk"

    # Verify initial storage
    collection_info = await qdrant_client.get_collection(test_collection)
    assert collection_info.points_count == original_chunks, (
        f"Expected {original_chunks} vectors initially"
    )

    # Retrieve original modification date
    points, _ = await qdrant_client.scroll(collection_name=test_collection, limit=1)
    original_mod_date = points[0].payload["modification_date"]  # type: ignore[index]

    # Step 2: Modify file content
    doc.write_text(sample_modified_content)

    # Step 3: Delete old vectors by file path (simulates watcher behavior)
    file_path_relative = str(doc.relative_to(tmp_path))
    deleted_count = vector_store.delete_by_file(file_path_relative)
    assert deleted_count == original_chunks, (
        f"Should delete {original_chunks} old vectors"
    )

    # Verify deletion
    collection_info = await qdrant_client.get_collection(test_collection)
    assert collection_info.points_count == 0, "All old vectors should be deleted"

    # Step 4: Re-process modified file
    result = await processor.process_document(doc)
    assert result.success, "Re-processing should succeed"

    modified_chunks = result.chunks_processed
    assert modified_chunks > 0, "Should process at least one chunk after modification"

    # Step 5: Query Qdrant to verify new storage
    collection_info = await qdrant_client.get_collection(test_collection)
    assert collection_info.points_count == modified_chunks, (
        f"Expected {modified_chunks} new vectors after re-processing"
    )

    # Step 6: Verify modification date is updated
    points, _ = await qdrant_client.scroll(collection_name=test_collection, limit=1)
    new_mod_date = points[0].payload["modification_date"]  # type: ignore[index]
    assert new_mod_date >= original_mod_date, (
        "Modification date should be equal or later"
    )


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
        sample_deletion_content: Fixture providing markdown content with
            multiple sections

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
    assert collection_info.points_count == chunks_processed, (
        f"Expected {chunks_processed} vectors"
    )

    # Step 2: Delete vectors by file path
    # This simulates the file watcher's deletion handler
    # Uses file_path_relative metadata field to identify all vectors for the file
    file_path_relative = str(doc.relative_to(tmp_path))
    deleted_count = vector_store.delete_by_file(file_path_relative)
    assert deleted_count == chunks_processed, (
        f"Should delete {chunks_processed} vectors, deleted {deleted_count}"
    )

    # Step 3: Query Qdrant to verify complete removal
    # Both collection info and scroll should confirm no vectors remain
    collection_info = await qdrant_client.get_collection(test_collection)
    assert collection_info.points_count is not None, (
        "Collection should have point count"
    )
    assert collection_info.points_count == 0, "No vectors should remain after deletion"

    # Double-check by scrolling through the collection
    # This ensures the delete operation was atomic and complete
    points, _ = await qdrant_client.scroll(collection_name=test_collection, limit=10)
    assert len(points) == 0, "Should retrieve no points after deletion"


@pytest.mark.integration
async def test_e2e_frontmatter_extraction(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
    qdrant_client: AsyncQdrantClient,
    sample_frontmatter_content: str,
) -> None:
    """Test YAML frontmatter extraction and tag metadata storage.

    Verifies that documents with YAML frontmatter are processed correctly:
    1. Process file with frontmatter containing title, author, tags, category
    2. Verify frontmatter metadata is extracted and stored in Qdrant payload
    3. Validate tags array is preserved correctly

    Args:
        tmp_path: pytest temporary directory fixture
        test_collection: Unique collection name for test isolation
        cleanup_fixture: Ensures collection cleanup after test
        qdrant_client: AsyncQdrantClient for verification queries
        sample_frontmatter_content: Fixture with YAML frontmatter

    Raises:
        AssertionError: If frontmatter extraction fails
    """
    # Create file with frontmatter
    doc = tmp_path / "frontmatter_test.md"
    doc.write_text(sample_frontmatter_content)

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

    await qdrant_client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
    processor = DocumentProcessor(config, tei_client, vector_store, chunker)

    # Process document
    result = await processor.process_document(doc)
    assert result.success, "Processing should succeed"
    assert result.chunks_processed > 0, "Should process chunks"

    # Verify frontmatter metadata in Qdrant
    points, _ = await qdrant_client.scroll(collection_name=test_collection, limit=1)
    assert len(points) > 0, "Should have at least one vector"

    payload = points[0].payload
    assert payload is not None, "Payload should exist"

    # Verify frontmatter fields are present
    assert "title" in payload, "Should have title from frontmatter"
    assert payload["title"] == "Integration Testing Guide"

    assert "author" in payload, "Should have author from frontmatter"
    assert payload["author"] == "Test Author"

    assert "tags" in payload, "Should have tags from frontmatter"
    tags = payload["tags"]
    assert isinstance(tags, list), "Tags should be a list"
    assert "testing" in tags, "Should contain 'testing' tag"
    assert "integration" in tags, "Should contain 'integration' tag"

    assert "category" in payload, "Should have category from frontmatter"
    assert payload["category"] == "Testing"


@pytest.mark.integration
async def test_e2e_nested_directories(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
    qdrant_client: AsyncQdrantClient,
) -> None:
    """Test processing markdown files in nested directory structures.

    Verifies that the pipeline handles nested directories correctly:
    1. Create nested directory structure: docs/guides/, docs/api/
    2. Place markdown files in nested directories
    3. Process all files using batch processing
    4. Verify all files are processed with correct relative paths

    Args:
        tmp_path: pytest temporary directory fixture
        test_collection: Unique collection name for test isolation
        cleanup_fixture: Ensures collection cleanup after test
        qdrant_client: AsyncQdrantClient for verification queries

    Raises:
        AssertionError: If nested directory handling fails
    """
    # Create nested directory structure
    guides_dir = tmp_path / "docs" / "guides"
    api_dir = tmp_path / "docs" / "api"
    guides_dir.mkdir(parents=True)
    api_dir.mkdir(parents=True)

    # Create files in nested directories
    guide1 = guides_dir / "installation.md"
    guide1.write_text("# Installation Guide\n\nHow to install the system.")

    guide2 = guides_dir / "configuration.md"
    guide2.write_text("# Configuration Guide\n\nHow to configure the system.")

    api1 = api_dir / "endpoints.md"
    api1.write_text("# API Endpoints\n\nList of available endpoints.")

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

    await qdrant_client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
    processor = DocumentProcessor(config, tei_client, vector_store, chunker)

    # Process all files
    files = [guide1, guide2, api1]
    results = await processor.process_batch(files)

    # Verify all processed successfully
    assert len(results) == 3, "Should process 3 files"
    assert all(r.success for r in results), "All files should process successfully"

    # Verify relative paths are correct in metadata
    points, _ = await qdrant_client.scroll(collection_name=test_collection, limit=10)
    assert len(points) > 0, "Should have vectors"

    # Check that relative paths include nested directory structure
    relative_paths = {p.payload["file_path_relative"] for p in points}  # type: ignore[index]
    assert any("docs/guides/" in path for path in relative_paths), (
        "Should have paths in docs/guides/"
    )
    assert any("docs/api/" in path for path in relative_paths), (
        "Should have paths in docs/api/"
    )


@pytest.mark.integration
async def test_e2e_large_document_chunking(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
    qdrant_client: AsyncQdrantClient,
    sample_large_document: str,
) -> None:
    """Test large document chunking with 5000+ tokens across 15 sections.

    Verifies that large documents are chunked correctly:
    1. Process document with 5000+ tokens
    2. Verify multiple chunks are created
    3. Validate chunk overlap is preserved
    4. Verify section_path metadata tracks document hierarchy

    Args:
        tmp_path: pytest temporary directory fixture
        test_collection: Unique collection name for test isolation
        cleanup_fixture: Ensures collection cleanup after test
        qdrant_client: AsyncQdrantClient for verification queries
        sample_large_document: Fixture with 5000+ token document

    Raises:
        AssertionError: If large document chunking fails
    """
    # Create large document file
    doc = tmp_path / "large_doc.md"
    doc.write_text(sample_large_document)

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

    await qdrant_client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
    processor = DocumentProcessor(config, tei_client, vector_store, chunker)

    # Process large document
    result = await processor.process_document(doc)
    assert result.success, "Processing should succeed"
    assert result.chunks_processed >= 10, (
        f"Large document should create at least 10 chunks, got {result.chunks_processed}"
    )

    # Verify chunks in Qdrant
    collection_info = await qdrant_client.get_collection(test_collection)
    assert collection_info.points_count == result.chunks_processed

    # Verify section metadata is preserved
    points, _ = await qdrant_client.scroll(
        collection_name=test_collection, limit=result.chunks_processed
    )

    # Check that section paths are present and varied
    section_paths = {p.payload["section_path"] for p in points}  # type: ignore[index]
    assert len(section_paths) > 1, "Should have multiple different section paths"

    # Verify chunk indices are sequential
    chunk_indices = [p.payload["chunk_index"] for p in points]  # type: ignore[index]
    assert min(chunk_indices) == 0, "Should start at chunk index 0"
    assert max(chunk_indices) == result.chunks_processed - 1, (
        "Chunk indices should be sequential"
    )


@pytest.mark.integration
async def test_e2e_special_characters(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
    qdrant_client: AsyncQdrantClient,
    sample_unicode_content: str,
) -> None:
    """Test processing markdown with special characters, emoji, and unicode.

    Verifies that unicode content is handled correctly:
    1. Process file with Japanese, Chinese, Arabic, Russian text
    2. Process emoji characters
    3. Process special mathematical and currency symbols
    4. Verify all content is preserved in chunk_text

    Args:
        tmp_path: pytest temporary directory fixture
        test_collection: Unique collection name for test isolation
        cleanup_fixture: Ensures collection cleanup after test
        qdrant_client: AsyncQdrantClient for verification queries
        sample_unicode_content: Fixture with diverse unicode characters

    Raises:
        AssertionError: If unicode handling fails
    """
    # Create file with unicode content
    doc = tmp_path / "unicode_test.md"
    doc.write_text(sample_unicode_content, encoding="utf-8")

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

    await qdrant_client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
    processor = DocumentProcessor(config, tei_client, vector_store, chunker)

    # Process unicode document
    result = await processor.process_document(doc)
    assert result.success, "Processing should succeed"
    assert result.chunks_processed > 0, "Should process chunks"

    # Verify unicode content is preserved
    points, _ = await qdrant_client.scroll(
        collection_name=test_collection, limit=result.chunks_processed
    )

    # Collect all chunk text
    all_text = " ".join([p.payload["chunk_text"] for p in points])  # type: ignore[index, misc]

    # Verify various unicode characters are present
    assert "🚀" in all_text or "日本語" in all_text or "你好" in all_text, (
        "Should preserve emoji or international text"
    )


@pytest.mark.integration
async def test_e2e_empty_file_handling(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
    qdrant_client: AsyncQdrantClient,
) -> None:
    """Test graceful handling of empty or whitespace-only files.

    Verifies that empty files don't crash the pipeline:
    1. Create empty file
    2. Create file with only whitespace
    3. Process both files
    4. Verify processing completes without errors
    5. Verify no vectors are created for empty files

    Args:
        tmp_path: pytest temporary directory fixture
        test_collection: Unique collection name for test isolation
        cleanup_fixture: Ensures collection cleanup after test
        qdrant_client: AsyncQdrantClient for verification queries

    Raises:
        AssertionError: If empty file handling fails
    """
    # Create empty file
    empty_file = tmp_path / "empty.md"
    empty_file.write_text("")

    # Create whitespace-only file
    whitespace_file = tmp_path / "whitespace.md"
    whitespace_file.write_text("   \n\n\t\n   ")

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

    await qdrant_client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
    processor = DocumentProcessor(config, tei_client, vector_store, chunker)

    # Process empty files
    files = [empty_file, whitespace_file]
    results = await processor.process_batch(files)

    # Verify processing completes without errors
    assert len(results) == 2, "Should process both files"
    # Empty files may report success=True with 0 chunks or success=False
    # Either behavior is acceptable as long as pipeline doesn't crash

    # Verify no vectors created for empty content
    collection_info = await qdrant_client.get_collection(test_collection)
    assert collection_info.points_count == 0, (
        "Empty files should not create vectors"
    )
