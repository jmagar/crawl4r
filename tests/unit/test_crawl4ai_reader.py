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


def test_config_rejects_invalid_timeout():
    """Test that Crawl4AIReaderConfig rejects invalid timeout values.

    Verifies NFR-1: Configuration validation for timeout field.

    This test ensures Pydantic validation catches negative timeout values.
    Valid range is 10-300 seconds (ge=10, le=300 in Field()).

    REFACTOR Phase: This test should PASS immediately since validation
    is already implemented via Pydantic Field() constraints.
    """
    from pydantic import ValidationError

    from rag_ingestion.crawl4ai_reader import Crawl4AIReaderConfig

    # Attempt to create config with negative timeout
    with pytest.raises(ValidationError) as exc_info:
        Crawl4AIReaderConfig(timeout=-5)

    # Verify error mentions timeout field
    assert "timeout" in str(exc_info.value).lower()


def test_config_rejects_invalid_max_retries():
    """Test that Crawl4AIReaderConfig rejects invalid max_retries values.

    Verifies NFR-1: Configuration validation for max_retries field.

    This test ensures Pydantic validation catches max_retries values
    exceeding upper limit. Valid range is 0-10 (ge=0, le=10 in Field()).

    REFACTOR Phase: This test should PASS immediately since validation
    is already implemented via Pydantic Field() constraints.
    """
    from pydantic import ValidationError

    from rag_ingestion.crawl4ai_reader import Crawl4AIReaderConfig

    # Attempt to create config with max_retries > 10
    with pytest.raises(ValidationError) as exc_info:
        Crawl4AIReaderConfig(max_retries=15)

    # Verify error mentions max_retries field
    assert "max_retries" in str(exc_info.value).lower()


def test_config_rejects_extra_fields():
    """Test that Crawl4AIReaderConfig rejects extra fields.

    Verifies NFR-1: Configuration validation for extra fields.

    This test ensures Pydantic validation catches unexpected fields due
    to extra="forbid" in ConfigDict. This prevents typos and config errors.

    REFACTOR Phase: This test should PASS immediately since validation
    is already implemented via Pydantic ConfigDict(extra="forbid").
    """
    from pydantic import ValidationError

    from rag_ingestion.crawl4ai_reader import Crawl4AIReaderConfig

    # Attempt to create config with unexpected field
    with pytest.raises(ValidationError) as exc_info:
        Crawl4AIReaderConfig(invalid_field="should_fail")

    # Verify error mentions extra field not permitted
    error_msg = str(exc_info.value).lower()
    assert "extra" in error_msg or "permitted" in error_msg


@respx.mock
def test_health_check_success():
    """Test that reader initialization succeeds with healthy service.

    Verifies AC-1.5, FR-13: Health check validation on initialization.

    This test ensures that when the Crawl4AI /health endpoint returns 200,
    the reader initializes successfully without raising exceptions.

    RED Phase: This test will FAIL because:
    - Crawl4AIReader.__init__ doesn't call health check yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock /health endpoint returning success
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader - should not raise exception
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Verify reader was created successfully
    assert reader is not None
    assert reader.endpoint_url == "http://localhost:52004"


@respx.mock
def test_health_check_failure():
    """Test that reader initialization fails with unhealthy service.

    Verifies AC-1.6: Health check failure handling.

    This test ensures that when the Crawl4AI /health endpoint fails
    (timeout or 500 error), the reader raises ValueError with clear
    error message indicating service is unreachable.

    RED Phase: This test will FAIL because:
    - Health check validation not implemented yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock /health endpoint failing with 503 Service Unavailable
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(503, json={"error": "Service unavailable"})
    )

    # Attempt to create reader - should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Verify error message mentions service unreachable
    error_msg = str(exc_info.value).lower()
    assert "unreachable" in error_msg or "health" in error_msg


def test_circuit_breaker_initialized():
    """Test that circuit breaker is initialized in __init__.

    Verifies FR-9: Circuit breaker integration.

    This test ensures that the reader initializes a CircuitBreaker
    instance with project standard settings (threshold=5, timeout=60).

    RED Phase: This test will FAIL because:
    - _circuit_breaker attribute doesn't exist yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    with respx.mock:
        respx.get("http://localhost:52004/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Verify circuit breaker was initialized
    assert hasattr(reader, "_circuit_breaker")
    assert reader._circuit_breaker is not None


def test_logger_initialized():
    """Test that logger is initialized in __init__.

    Verifies FR-11: Structured logging integration.

    This test ensures that the reader initializes a logger instance
    via get_logger() for structured logging throughout the lifecycle.

    RED Phase: This test will FAIL because:
    - _logger attribute doesn't exist yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    with respx.mock:
        respx.get("http://localhost:52004/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Verify logger was initialized
    assert hasattr(reader, "_logger")
    assert reader._logger is not None


def test_document_id_deterministic():
    """Test that _generate_document_id produces deterministic UUIDs.

    Verifies FR-4, Issue #15: Deterministic UUID generation from URL.

    This test ensures that calling _generate_document_id() twice with
    the same URL produces identical UUID values, enabling idempotent
    re-ingestion and deduplication.

    RED Phase: This test will FAIL because:
    - _generate_document_id method doesn't exist yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    with respx.mock:
        respx.get("http://localhost:52004/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Call _generate_document_id twice with same URL
    test_url = "https://example.com/test-page"
    uuid1 = reader._generate_document_id(test_url)
    uuid2 = reader._generate_document_id(test_url)

    # Verify both UUIDs are identical (deterministic)
    assert uuid1 == uuid2
