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
