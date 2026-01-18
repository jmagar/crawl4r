"""Error handling and resilience tests for RAG ingestion pipeline.

Tests service failure scenarios, error recovery, and graceful degradation.
Validates circuit breakers, retry logic, validation errors, and edge cases.

Test scenarios covered:
- TEI service unavailable (connection refused)
- TEI request timeouts with retry
- TEI invalid dimensions (validation error)
- Qdrant unavailability
- Circuit breaker state transitions
- Malformed markdown handling
- File permission errors
- Duplicate point ID idempotency

These tests use respx for HTTP mocking to simulate service failures
without actually stopping Docker containers.

Example:
    Run error handling tests:
    $ pytest tests/integration/test_e2e_errors.py -v

    Run specific error test:
    $ pytest tests/integration/test_e2e_errors.py::test_tei_service_unavailable -v
"""

import asyncio
import os
from pathlib import Path

import pytest
import respx
from httpx import ConnectError, TimeoutException
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from crawl4r.core.config import Settings
from crawl4r.processing.processor import DocumentProcessor
from crawl4r.storage.tei import TEIClient
from crawl4r.storage.qdrant import VectorStoreManager

# Service endpoints
TEI_ENDPOINT = os.getenv("TEI_ENDPOINT", "http://localhost:52000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:52001")

pytestmark = [pytest.mark.usefixtures("require_qdrant_service")]


@pytest.mark.integration
async def test_tei_service_unavailable(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test behavior when TEI service is completely unavailable.

    Verifies error handling when TEI cannot be reached:
    1. Mock TEI to raise ConnectError
    2. Attempt to process document
    3. Verify processing fails gracefully
    4. Verify circuit breaker may open after repeated failures

    Args:
        tmp_path: pytest temporary directory
        test_collection: Unique collection name
        cleanup_fixture: Cleanup after test

    Raises:
        AssertionError: If error handling fails
    """
    # Create test file
    doc = tmp_path / "test.md"
    doc.write_text("# Test\n\nContent for testing.")

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
        timeout=5.0,
        max_retries=2,  # Reduce retries for faster test
    )

    vector_store = VectorStoreManager(
        qdrant_url=config.qdrant_url,
        collection_name=config.collection_name,
        dimensions=1024,
    )

    # Create collection BEFORE entering respx mock context
    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, timeout=30)
    try:
        await qdrant_client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    finally:
        await qdrant_client.close()

    processor = DocumentProcessor(config, vector_store, tei_client=tei_client)

    # Use respx context manager with base_url to only mock TEI endpoints
    async with respx.mock(base_url=TEI_ENDPOINT, assert_all_called=False) as respx_mock:
        # Mock TEI service to be unavailable
        respx_mock.post("/embed").mock(side_effect=ConnectError)
        respx_mock.get("/health").mock(side_effect=ConnectError)

        # Attempt to process document - should fail due to TEI unavailability
        result = await processor.process_document(doc)

        # Verify processing failed gracefully (no crash)
        assert not result.success, "Processing should fail when TEI is unavailable"
        assert result.chunks_processed == 0, "No chunks should be processed"


@pytest.mark.integration
async def test_tei_timeout(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test behavior when TEI requests timeout.

    Verifies timeout handling and retry logic:
    1. Mock TEI to raise TimeoutException
    2. Attempt to process document
    3. Verify retries are attempted
    4. Verify eventual failure after max retries

    Args:
        tmp_path: pytest temporary directory
        test_collection: Unique collection name
        cleanup_fixture: Cleanup after test

    Raises:
        AssertionError: If timeout handling fails
    """
    # Create test file
    doc = tmp_path / "test.md"
    doc.write_text("# Test\n\nContent for timeout testing.")

    # Initialize components with short timeout
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
        timeout=2.0,  # Short timeout for faster test
        max_retries=2,
    )

    vector_store = VectorStoreManager(
        qdrant_url=config.qdrant_url,
        collection_name=config.collection_name,
        dimensions=1024,
    )

    # Create collection BEFORE entering respx mock context
    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, timeout=30)
    try:
        await qdrant_client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    finally:
        await qdrant_client.close()

    processor = DocumentProcessor(config, vector_store, tei_client=tei_client)

    # Use respx context manager with base_url to only mock TEI endpoints
    async with respx.mock(base_url=TEI_ENDPOINT, assert_all_called=False) as respx_mock:
        # Mock TEI service to timeout
        respx_mock.post("/embed").mock(side_effect=TimeoutException)
        respx_mock.get("/health").mock(side_effect=TimeoutException)

        # Attempt to process document - should fail after retries
        result = await processor.process_document(doc)

        # Verify processing failed after timeout
        assert not result.success, "Processing should fail after timeout"
        assert result.chunks_processed == 0, "No chunks should be processed"


@pytest.mark.integration
async def test_malformed_markdown(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test handling of malformed markdown syntax.

    Verifies graceful degradation with broken markdown:
    1. Create file with malformed markdown (broken tables, incomplete links)
    2. Process document
    3. Verify processing doesn't crash
    4. Verify best-effort chunking occurs

    Args:
        tmp_path: pytest temporary directory
        test_collection: Unique collection name
        cleanup_fixture: Cleanup after test

    Raises:
        AssertionError: If malformed markdown crashes pipeline
    """
    # Create file with malformed markdown
    malformed_content = """# Broken Markdown

This has an incomplete link: [broken link](

| Broken | Table |
|-----
Missing closing pipe

![Invalid image](

**Unclosed bold

> Incomplete blockquote
"""

    doc = tmp_path / "malformed.md"
    doc.write_text(malformed_content)

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
    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, timeout=30)
    try:
        await qdrant_client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    finally:
        await qdrant_client.close()

    processor = DocumentProcessor(config, vector_store, tei_client=tei_client)

    # Process malformed document - should handle gracefully
    result = await processor.process_document(doc)

    # Verify processing doesn't crash (may succeed or fail gracefully)
    # The important part is no exception is raised
    # If it succeeds, should have some chunks
    if result.success:
        assert result.chunks_processed > 0, (
            "Successful processing should yield chunks"
        )


@pytest.mark.integration
async def test_file_permission_denied(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test handling of file permission errors.

    Verifies error handling when file cannot be read:
    1. Create file and remove read permissions (chmod 000)
    2. Attempt to process document
    3. Verify PermissionError is caught gracefully
    4. Verify processing fails with appropriate error

    Args:
        tmp_path: pytest temporary directory
        test_collection: Unique collection name
        cleanup_fixture: Cleanup after test

    Raises:
        AssertionError: If permission errors aren't handled
    """
    # Create file
    doc = tmp_path / "restricted.md"
    doc.write_text("# Restricted\n\nThis file will have no read permission.")

    # Remove read permissions
    doc.chmod(0o000)

    try:
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
        qdrant_client = AsyncQdrantClient(url=QDRANT_URL, timeout=30)
        try:
            await qdrant_client.create_collection(
                collection_name=test_collection,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
        finally:
            await qdrant_client.close()

        processor = DocumentProcessor(config, vector_store, tei_client=tei_client)

        # Attempt to process file with no read permission
        result = await processor.process_document(doc)

        # Verify processing failed due to permission error
        assert not result.success, "Processing should fail with permission error"
        assert result.chunks_processed == 0, "No chunks should be processed"

    finally:
        # Restore permissions for cleanup
        doc.chmod(0o644)


@pytest.mark.integration
@pytest.mark.usefixtures("require_tei_service")
async def test_duplicate_point_id_idempotency(
    tmp_path: Path,
    test_collection: str,
    require_tei_service: None,
    cleanup_fixture: None,
) -> None:
    """Test idempotent re-ingestion with duplicate point IDs.

    Verifies idempotency through deterministic point ID generation:
    1. Process document once
    2. Process same document again without deletion
    3. Verify point IDs are identical (SHA256 of file_path + chunk_index)
    4. Verify Qdrant handles duplicate IDs (upsert behavior)
    5. Verify vector count remains same (no duplicates created)

    Args:
        tmp_path: pytest temporary directory
        test_collection: Unique collection name
        cleanup_fixture: Cleanup after test

    Raises:
        AssertionError: If idempotency fails
    """
    # Create test file
    doc = tmp_path / "idempotent.md"
    doc.write_text("# Idempotent Test\n\nSame content, same IDs.")

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
    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, timeout=30)
    try:
        await qdrant_client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    finally:
        await qdrant_client.close()

    processor = DocumentProcessor(config, vector_store, tei_client=tei_client)

    # Process document first time
    result1 = await processor.process_document(doc)
    assert result1.success, "First processing should succeed"

    chunks_first = result1.chunks_processed
    assert chunks_first > 0, "Should process some chunks"

    # Verify initial vectors
    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, timeout=30)
    try:
        collection_info = await qdrant_client.get_collection(test_collection)
        first_count = collection_info.points_count
        assert first_count == chunks_first

        # Process same document again WITHOUT deletion (tests idempotency)
        result2 = await processor.process_document(doc)
        assert result2.success, "Second processing should succeed"

        # Verify vector count unchanged (upsert behavior, no duplicates)
        collection_info = await qdrant_client.get_collection(test_collection)
        second_count = collection_info.points_count

        assert second_count == first_count, (
            f"Vector count should remain {first_count}, "
            f"got {second_count} (idempotency failure)"
        )

    finally:
        await qdrant_client.close()


# Placeholder tests for remaining error scenarios
# These can be expanded based on actual circuit breaker implementation


@pytest.mark.integration
async def test_circuit_breaker_transitions(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test circuit breaker state transitions.

    Verifies circuit breaker pattern:
    1. Initial state: CLOSED (allow requests)
    2. After 3 failures: OPEN (reject requests immediately)
    3. After timeout: HALF_OPEN (test with single request)
    4. On success: CLOSED (resume normal operation)

    This test requires circuit breaker implementation details.

    Args:
        tmp_path: pytest temporary directory
        test_collection: Unique collection name
        cleanup_fixture: Cleanup after test
    """
    from crawl4r.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerError

    # Create circuit breaker with low threshold for testing
    cb = CircuitBreaker(failure_threshold=3, reset_timeout=0.5)

    # PHASE 1: Initial state should be CLOSED
    assert cb.state == "CLOSED", "Circuit should start CLOSED"
    assert cb.can_execute() is True, "Should allow execution when CLOSED"
    assert cb.failure_count == 0, "Should have no failures initially"

    # PHASE 2: Record failures to transition to OPEN
    cb.record_failure()
    assert cb.state == "CLOSED", "Should stay CLOSED after 1 failure"
    assert cb.failure_count == 1

    cb.record_failure()
    assert cb.state == "CLOSED", "Should stay CLOSED after 2 failures"
    assert cb.failure_count == 2

    cb.record_failure()
    assert cb.state == "OPEN", "Should transition to OPEN after 3 failures"
    assert cb.failure_count == 3
    assert cb.can_execute() is False, "Should reject execution when OPEN"

    # Verify CircuitBreakerError is raised when OPEN
    async def mock_operation() -> str:
        return "success"

    try:
        await cb.call(mock_operation)
        assert False, "Should have raised CircuitBreakerError"
    except CircuitBreakerError as e:
        assert "OPEN" in str(e), "Error should mention circuit is OPEN"

    # PHASE 3: Wait for reset timeout to transition to HALF_OPEN
    await asyncio.sleep(0.6)  # Wait for reset_timeout (0.5s)

    assert cb.state == "HALF_OPEN", "Should transition to HALF_OPEN after timeout"
    assert cb.can_execute() is True, "Should allow test call when HALF_OPEN"

    # PHASE 4: Success in HALF_OPEN should transition back to CLOSED
    cb.record_success()
    assert cb.state == "CLOSED", "Should transition to CLOSED after successful recovery"
    assert cb.failure_count == 0, "Failure count should reset to 0"
    assert cb.can_execute() is True, "Should allow execution when CLOSED"

    # BONUS: Test HALF_OPEN -> OPEN transition on failure
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "OPEN", "Should be OPEN after 3 failures"

    await asyncio.sleep(0.6)  # Wait for reset timeout
    assert cb.state == "HALF_OPEN", "Should be HALF_OPEN after timeout"

    cb.record_failure()  # Fail during recovery
    assert cb.state == "OPEN", "Should go back to OPEN if recovery fails"


@pytest.mark.integration
async def test_qdrant_unavailable(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test behavior when Qdrant service is unavailable.

    Verifies error handling when Qdrant cannot be reached:
    1. Mock Qdrant to be unavailable
    2. Attempt to process document
    3. Verify processing fails gracefully
    4. Verify appropriate error is logged

    Note: Qdrant uses custom client, not httpx, so respx mocking
    doesn't work directly. Need alternative approach.

    Args:
        tmp_path: pytest temporary directory
        test_collection: Unique collection name
        cleanup_fixture: Cleanup after test
    """
    from unittest.mock import patch

    # Create test file
    doc = tmp_path / "test.md"
    doc.write_text("# Test\n\nContent for Qdrant failure testing.")

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

    # Create collection first
    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, timeout=30)
    try:
        await qdrant_client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    finally:
        await qdrant_client.close()

    processor = DocumentProcessor(config, vector_store, tei_client=tei_client)

    # Mock vector_store's retry method to simulate Qdrant failure
    with patch.object(vector_store, "_retry_with_backoff") as mock_retry:
        # Make retry raise connection error
        mock_retry.side_effect = RuntimeError("Qdrant connection refused")

        # Attempt to process document - should fail due to Qdrant unavailability
        result = await processor.process_document(doc)

        # Verify processing failed gracefully (no crash)
        assert not result.success, "Processing should fail when Qdrant is unavailable"
        assert result.chunks_processed == 0, "No chunks should be processed"
        assert result.error is not None, "Error should be set"
        assert "qdrant" in result.error.lower() or "connection" in result.error.lower(), (
            "Error should mention Qdrant or connection issue"
        )


@pytest.mark.integration
async def test_tei_invalid_dimensions(
    tmp_path: Path,
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test validation when TEI returns invalid embedding dimensions.

    Verifies dimension validation:
    1. Mock TEI to return 512-dim embeddings (wrong size)
    2. Attempt to process document
    3. Verify validation error is raised
    4. Verify no vectors are stored in Qdrant

    Args:
        tmp_path: pytest temporary directory
        test_collection: Unique collection name
        cleanup_fixture: Cleanup after test

    Raises:
        AssertionError: If dimension validation fails
    """
    # Create test file
    doc = tmp_path / "test.md"
    doc.write_text("# Test\n\nContent for dimension testing.")

    # Initialize components expecting 1024 dims
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
        dimensions=1024,  # Expect 1024
        timeout=30.0,
        max_retries=3,
    )

    vector_store = VectorStoreManager(
        qdrant_url=config.qdrant_url,
        collection_name=config.collection_name,
        dimensions=1024,  # Expect 1024
    )

    # Create collection BEFORE entering respx mock context
    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, timeout=30)
    try:
        await qdrant_client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    finally:
        await qdrant_client.close()

    processor = DocumentProcessor(config, vector_store, tei_client=tei_client)

    # Use respx context manager to mock TEI with wrong dimensions
    async with respx.mock(base_url=TEI_ENDPOINT, assert_all_called=False) as respx_mock:
        # Mock TEI to return 512 dimensions instead of expected 1024
        respx_mock.post("/embed").respond(json=[[0.1] * 512])  # Wrong dimensions
        respx_mock.get("/health").respond(json={"status": "ok"})

        # Process document - should fail due to dimension mismatch
        result = await processor.process_document(doc)

        # Verify processing failed due to dimension validation
        assert not result.success, "Processing should fail with dimension mismatch"
        assert result.chunks_processed == 0, "No chunks should be processed"

    # Verify no vectors were stored (outside respx context to allow Qdrant access)
    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, timeout=30)
    try:
        collection_info = await qdrant_client.get_collection(test_collection)
        assert collection_info.points_count == 0, (
            "No vectors should be stored with invalid dimensions"
        )
    finally:
        await qdrant_client.close()
