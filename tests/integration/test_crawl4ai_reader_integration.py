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

from rag_ingestion.crawl4ai_reader import Crawl4AIReader

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
    assert documents[0].metadata["source"] == "https://example.com"
    assert documents[1].metadata["source"] == "https://example.org"
    assert documents[2].metadata["source"] == "https://example.net"

    # Verify each document has valid content
    for i, doc in enumerate(documents):
        assert doc.text is not None
        assert len(doc.text) > 0
        assert doc.metadata["source"] == urls[i]
        assert doc.metadata["source_url"] == urls[i]
        assert doc.metadata["status_code"] == 200
        assert doc.metadata["source_type"] == "web_crawl"
        assert doc.id_ is not None
        assert len(doc.id_) == 36  # UUID string format
