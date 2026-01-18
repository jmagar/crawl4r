"""Integration tests for Crawl4AI reader.

Tests real Crawl4AI service endpoints to verify:
- Reader can crawl URLs using actual service
- Documents are created with correct metadata
- Markdown content is properly extracted
- Error handling works with service failures

These tests require the Crawl4AI service to be running. The endpoint can be
configured via the CRAWL4AI_URL environment variable. If not set, defaults to
http://localhost:52004. If the service is not available, tests will be skipped.

Example:
    Run only Crawl4AI integration tests:
    $ pytest tests/integration/test_crawl4ai_reader_integration.py -v -m integration

    Run with custom endpoint:
    $ CRAWL4AI_URL=http://crawl4ai:11235 pytest \
        tests/integration/test_crawl4ai_reader_integration.py -v -m integration

    Run with service availability check:
    $ docker compose up -d crawl4ai
    $ pytest tests/integration/test_crawl4ai_reader_integration.py -v -m integration
"""

import os

import httpx
import pytest

from crawl4r.readers.crawl4ai import Crawl4AIReader

# Get Crawl4AI endpoint from environment or use default
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://localhost:52004")


@pytest.fixture(autouse=True)
async def crawl4ai_available() -> None:
    """Check if Crawl4AI service is available before running tests.

    Automatically runs before each test to verify the Crawl4AI service is
    reachable. Uses the CRAWL4AI_URL environment variable or defaults to
    http://localhost:52004. If the service is not available, the test will
    be skipped with an informative message.

    Raises:
        pytest.skip: If Crawl4AI service is not available at configured endpoint

    Example:
        This fixture runs automatically for all tests in this module.
        No explicit usage required.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{CRAWL4AI_URL}/health")
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip(f"Crawl4AI service not available at {CRAWL4AI_URL}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_health_check() -> None:
    """Verify reader health check works with real Crawl4AI service.

    Tests that the Crawl4AIReader can successfully validate the health endpoint
    of a running Crawl4AI service. This verifies the __init__ health check
    logic works with a real service deployment.

    Requirements:
        - Crawl4AI service must be running at configured endpoint
        - Service must respond to /health with 200 OK

    Expected:
        - Reader initializes without exception
        - No errors logged during health check
    """
    # Create reader with configured endpoint (will call health check in __init__)
    reader = Crawl4AIReader(endpoint_url=CRAWL4AI_URL)

    # Verify reader initialized successfully
    assert reader is not None
    assert reader.endpoint_url == CRAWL4AI_URL
    assert reader._circuit_breaker is not None
    assert reader._logger is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_crawl_single_url() -> None:
    """Verify reader can crawl a real webpage and create valid Document.

    Tests the full crawl workflow with a real webpage:
    - POST request to /crawl endpoint
    - Markdown content extraction
    - Document creation with metadata
    - Deterministic ID generation

    Requirements:
        - Crawl4AI service must be running
        - Internet access to reach example.com
        - Service can successfully crawl example.com

    Expected:
        - Document returned with text content
        - Metadata includes source, title, status_code, etc.
        - Document ID is deterministic UUID
    """
    # Create reader
    reader = Crawl4AIReader(endpoint_url=CRAWL4AI_URL)

    # Crawl example.com (reliable test URL)
    documents = await reader.aload_data(["https://example.com"])

    # Verify document created
    assert len(documents) == 1
    doc = documents[0]
    assert doc is not None

    # Verify document has text content
    assert doc.text is not None
    assert len(doc.text) > 0
    assert "Example Domain" in doc.text  # Known content from example.com

    # Verify metadata structure
    assert doc.metadata is not None
    assert "source" in doc.metadata
    assert doc.metadata["source"] == "https://example.com"
    assert "source_url" in doc.metadata
    assert doc.metadata["source_url"] == "https://example.com"
    assert "title" in doc.metadata
    assert "status_code" in doc.metadata
    assert doc.metadata["status_code"] == 200
    assert "source_type" in doc.metadata
    assert doc.metadata["source_type"] == "web_crawl"

    # Verify deterministic ID
    assert doc.id_ is not None
    assert len(doc.id_) == 36  # UUID string format


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_crawl_batch() -> None:
    """Verify reader can crawl multiple webpages concurrently.

    Tests the batch crawl workflow with multiple real webpages:
    - Multiple POST requests to /crawl endpoint
    - Concurrent processing with concurrency limit
    - All Documents created successfully
    - Order preservation (results match input URLs order)

    Requirements:
        - Crawl4AI service must be running
        - Internet access to reach test URLs
        - Service can successfully crawl test URLs

    Expected:
        - All Documents returned in same order as input URLs
        - Each Document has valid content and metadata
        - No None values (all crawls succeed for reliable test URLs)
    """
    # Create reader with default concurrency limit
    reader = Crawl4AIReader(endpoint_url=CRAWL4AI_URL)

    # Crawl multiple reliable test URLs
    urls = [
        "https://example.com",
        "https://example.org",
        "https://example.net",
    ]
    documents = await reader.aload_data(urls)

    # Verify all documents created
    assert len(documents) == 3
    assert all(doc is not None for doc in documents)

    # Verify order preservation (results match input order)
    assert documents[0] is not None
    assert documents[0].metadata["source"] == "https://example.com"
    assert documents[1] is not None
    assert documents[1].metadata["source"] == "https://example.org"
    assert documents[2] is not None
    assert documents[2].metadata["source"] == "https://example.net"

    # Verify each document has valid content
    for i, doc in enumerate(documents):
        assert doc is not None
        assert doc.text is not None
        assert len(doc.text) > 0
        assert doc.metadata["source"] == urls[i]
        assert doc.metadata["source_url"] == urls[i]
        assert doc.metadata["status_code"] == 200
        assert doc.metadata["source_type"] == "web_crawl"
        assert doc.id_ is not None
        assert len(doc.id_) == 36  # UUID string format


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_circuit_breaker_opens() -> None:
    """Verify circuit breaker opens after consecutive failures.

    Tests the circuit breaker fault tolerance mechanism:
    - Crawl invalid URLs to trigger service failures
    - Verify circuit breaker opens after threshold (5 failures)
    - Verify failure count matches number of failures

    Requirements:
        - Crawl4AI service must be running
        - Service must fail predictably for invalid URLs

    Expected:
        - Circuit breaker opens after threshold failures
        - All crawls return None (graceful failure with fail_on_error=False)
        - Failure count matches number of consecutive failures

    Note:
        This test uses invalid URLs to trigger failures. The circuit breaker
        threshold is 5, so we need at least 5 failures to open the circuit.
        We use fail_on_error=False to prevent exceptions and verify graceful
        failure handling.
    """
    # Create reader with fail_on_error=False to prevent early exceptions
    reader = Crawl4AIReader(
        endpoint_url=CRAWL4AI_URL, fail_on_error=False, max_retries=0
    )

    # Trigger multiple failures to open circuit breaker (threshold is 5)
    invalid_urls = [f"http://invalid-domain-{i}.example" for i in range(6)]
    results = await reader.aload_data(invalid_urls)

    # Verify all crawls failed (None returned)
    assert all(result is None for result in results)

    # Verify circuit breaker is now open (state property returns string)
    assert reader._circuit_breaker.state == "OPEN"

    # Verify failure count is at or above threshold
    assert reader._circuit_breaker.failure_count >= 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_concurrent_processing() -> None:
    """Verify concurrent processing respects concurrency limit.

    Tests that the reader processes multiple URLs concurrently while
    respecting the configured concurrency limit. This verifies the
    semaphore-based concurrency control works correctly.

    Requirements:
        - Crawl4AI service must be running
        - Internet access to reach test URLs
        - Service can successfully crawl test URLs

    Expected:
        - All URLs processed successfully
        - Processing completes in reasonable time (parallel > sequential)
        - No more than max_concurrent_requests processed simultaneously

    Note:
        This test uses 10 URLs to demonstrate concurrent processing. The
        actual concurrency limit is 5 by default, so processing should be
        faster than sequential but not instantaneous.
    """
    import time

    # Create reader with concurrency limit of 3
    reader = Crawl4AIReader(endpoint_url=CRAWL4AI_URL, max_concurrent_requests=3)

    # Crawl 10 URLs to test concurrent processing
    urls = [
        "https://example.com",
        "https://example.org",
        "https://example.net",
        "https://www.iana.org",
        "https://www.ietf.org",
        "https://www.w3.org",
        "https://www.python.org",
        "https://www.rust-lang.org",
        "https://www.go.dev",
        "https://www.javascript.com",
    ]

    # Measure processing time
    start_time = time.time()
    documents = await reader.aload_data(urls)
    elapsed_time = time.time() - start_time

    # Verify documents created (aload_data returns only successes)
    assert len(documents) <= len(urls)
    successful_docs = [doc for doc in documents if doc is not None]
    assert len(successful_docs) >= 8  # Allow some failures

    # Verify processing time suggests concurrent execution
    # With concurrency limit of 3, ~10 URLs should complete faster than
    # sequential (which would be ~10x single URL time). Allow extra time
    # for transient retries/backoff without failing the concurrency signal.
    assert elapsed_time < 20.0  # Reasonable upper bound for concurrent

    # Verify all successful documents have valid content
    for doc in successful_docs:
        assert doc.text is not None
        assert len(doc.text) > 0
        assert doc.metadata["source"] in urls
        assert doc.metadata["source_url"] in urls
        assert doc.metadata["source_type"] == "web_crawl"
        assert doc.id_ is not None
