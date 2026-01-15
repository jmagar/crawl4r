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


def test_config_class_has_required_fields():
    """Test that Crawl4AIReaderConfig class has all required fields.

    Verifies FR-1: Configuration class structure.

    This test ensures the Crawl4AIReaderConfig class exists with all
    required configuration fields: base_url, timeout, max_retries,
    retry_delays, circuit_breaker_threshold, circuit_breaker_timeout,
    and concurrency_limit.

    RED Phase: This test will FAIL because:
    - Crawl4AIReaderConfig class doesn't exist yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReaderConfig

    # Create config instance
    config = Crawl4AIReaderConfig()

    # Verify all 7 required fields exist with correct types
    assert hasattr(config, "base_url")
    assert isinstance(config.base_url, str)

    assert hasattr(config, "timeout")
    assert isinstance(config.timeout, (int, float))

    assert hasattr(config, "max_retries")
    assert isinstance(config.max_retries, int)

    assert hasattr(config, "retry_delays")
    assert isinstance(config.retry_delays, list)

    assert hasattr(config, "circuit_breaker_threshold")
    assert isinstance(config.circuit_breaker_threshold, int)

    assert hasattr(config, "circuit_breaker_timeout")
    assert isinstance(config.circuit_breaker_timeout, (int, float))

    assert hasattr(config, "concurrency_limit")
    assert isinstance(config.concurrency_limit, int)


def test_reader_respects_crawl4ai_base_url_from_settings():
    """Test that Crawl4AIReader uses CRAWL4AI_BASE_URL from Settings.

    Verifies FR-1.1: Reader respects Settings configuration.

    This test ensures that when a Settings object with a custom
    CRAWL4AI_BASE_URL is passed to the reader constructor, the reader
    uses that URL instead of the default endpoint.

    RED Phase: This test will FAIL because:
    - Settings class doesn't have CRAWL4AI_BASE_URL field yet
    - Crawl4AIReader class doesn't exist yet
    """
    from rag_ingestion.config import Settings
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Create Settings with custom Crawl4AI base URL
    custom_url = "http://custom-crawl4ai.example.com:9999"
    settings = Settings(
        watch_folder="/tmp/test",
        CRAWL4AI_BASE_URL=custom_url,
    )

    # Create reader with Settings
    reader = Crawl4AIReader(settings=settings)

    # Verify reader uses the custom URL from Settings
    assert reader.endpoint_url == custom_url
