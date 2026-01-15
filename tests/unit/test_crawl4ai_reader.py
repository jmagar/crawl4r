"""Unit tests for Crawl4AI reader module.

TDD RED Phase: All tests should FAIL initially (no implementation exists).

This test suite covers:
- Crawl4AIReader initialization and configuration
- Default field values and custom configuration
- Pydantic validation (timeout, max_concurrent ranges)
- LlamaIndex properties (is_remote, class_name)
- Health check validation (success and failure)
- Circuit breaker and logger initialization
- Document ID generation (deterministic UUID from URL)
- Metadata extraction (9 fields including source_url)
- Single URL crawling with circuit breaker
- Markdown extraction with fallback to raw_markdown
- Retry logic with exponential backoff
- Error handling (timeouts, network errors, HTTP status codes)
- Async batch loading with concurrency control
- Order preservation in batch results
- Synchronous load_data wrapper
- Deduplication integration (Issue #16)
"""

import pytest
import respx
import httpx

# This import will fail initially - that's expected in RED phase
# from rag_ingestion.crawl4ai_reader import Crawl4AIReader


@pytest.fixture
def reader_config() -> dict[str, str | int | bool]:
    """Fixture providing default Crawl4AIReader configuration.

    Returns:
        Dictionary with standard configuration values for testing.
    """
    return {
        "endpoint_url": "http://localhost:52004",
        "timeout_seconds": 60,
        "fail_on_error": False,
        "max_concurrent_requests": 5,
        "max_retries": 3,
    }
