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


def test_document_id_different_urls():
    """Test that _generate_document_id produces different UUIDs for different URLs.

    Verifies FR-4, Issue #15: Unique UUID generation per URL.

    This test ensures that calling _generate_document_id() with two
    different URLs produces different UUID values, preventing collisions
    and ensuring proper isolation between documents.

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

    # Call _generate_document_id with two different URLs
    url1 = "https://example.com/page-one"
    url2 = "https://example.com/page-two"
    uuid1 = reader._generate_document_id(url1)
    uuid2 = reader._generate_document_id(url2)

    # Verify UUIDs are different (unique per URL)
    assert uuid1 != uuid2


def test_document_id_uuid_format():
    """Test that _generate_document_id returns valid UUID format.

    Verifies FR-4, Issue #15: UUID format validation.

    This test ensures that the returned value from _generate_document_id()
    is a properly formatted UUID string that can be parsed by uuid.UUID().
    This guarantees compatibility with systems expecting standard UUID format.

    RED Phase: This test will FAIL because:
    - _generate_document_id method doesn't exist yet
    """
    from uuid import UUID

    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    with respx.mock:
        respx.get("http://localhost:52004/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Generate document ID from URL
    test_url = "https://example.com/test-page"
    doc_id = reader._generate_document_id(test_url)

    # Verify it's a valid UUID format by parsing it
    # This will raise ValueError if format is invalid
    parsed_uuid = UUID(doc_id)

    # Verify the parsed UUID string matches original
    assert str(parsed_uuid) == doc_id


@pytest.fixture
def mock_crawl_result_success() -> dict:
    """Fixture providing successful CrawlResult mock data.

    Returns:
        Dictionary with complete CrawlResult structure for testing.
    """
    return {
        "url": "https://example.com",
        "success": True,
        "status_code": 200,
        "markdown": {
            "fit_markdown": "# Example\n\nThis is example content.",
            "raw_markdown": "# Example\n\nThis is example content.\nFooter.",
        },
        "metadata": {"title": "Example Domain", "description": "Example desc"},
        "links": {
            "internal": [{"href": "/page1"}, {"href": "/page2"}],
            "external": [{"href": "https://other.com"}],
        },
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }


def test_metadata_complete(mock_crawl_result_success):
    """Test that _build_metadata extracts all 9 required metadata fields.

    Verifies AC-5.1-5.10, FR-7, Issue #17: Complete metadata structure.

    This test ensures that _build_metadata() correctly extracts all 9
    metadata fields from a valid CrawlResult:
    - source (URL)
    - source_url (URL, indexed for deduplication)
    - title (page title)
    - description (page description)
    - status_code (HTTP status)
    - crawl_timestamp (ISO8601 timestamp)
    - internal_links_count (count of internal links)
    - external_links_count (count of external links)
    - source_type (always "web_crawl")

    RED Phase: This test will FAIL because:
    - _build_metadata method doesn't exist yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    with respx.mock:
        respx.get("http://localhost:52004/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Call _build_metadata with mock CrawlResult
    test_url = "https://example.com"
    metadata = reader._build_metadata(mock_crawl_result_success, test_url)

    # Verify all 9 required fields are present
    assert "source" in metadata
    assert "source_url" in metadata
    assert "title" in metadata
    assert "description" in metadata
    assert "status_code" in metadata
    assert "crawl_timestamp" in metadata
    assert "internal_links_count" in metadata
    assert "external_links_count" in metadata
    assert "source_type" in metadata

    # Verify field values are correct
    assert metadata["source"] == test_url
    assert metadata["source_url"] == test_url
    assert metadata["title"] == "Example Domain"
    assert metadata["description"] == "Example desc"
    assert metadata["status_code"] == 200
    assert metadata["crawl_timestamp"] == "2026-01-15T12:00:00Z"
    assert metadata["internal_links_count"] == 2
    assert metadata["external_links_count"] == 1
    assert metadata["source_type"] == "web_crawl"

    # Verify all values are flat types (Qdrant compatible)
    assert isinstance(metadata["source"], str)
    assert isinstance(metadata["source_url"], str)
    assert isinstance(metadata["title"], str)
    assert isinstance(metadata["description"], str)
    assert isinstance(metadata["status_code"], int)
    assert isinstance(metadata["crawl_timestamp"], str)
    assert isinstance(metadata["internal_links_count"], int)
    assert isinstance(metadata["external_links_count"], int)
    assert isinstance(metadata["source_type"], str)


def test_metadata_missing_title():
    """Test that _build_metadata defaults to empty string when title is missing.

    Verifies AC-5.2, AC-5.8: Default handling for missing title.

    This test ensures that when the metadata field in CrawlResult is missing
    or has no title field, _build_metadata() defaults to an empty string
    instead of raising an error.

    RED Phase: This test will FAIL because:
    - _build_metadata method doesn't exist yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    with respx.mock:
        respx.get("http://localhost:52004/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Create CrawlResult with missing title field
    crawl_result = {
        "url": "https://example.com",
        "success": True,
        "status_code": 200,
        "markdown": {"fit_markdown": "# Example\n\nContent."},
        "metadata": {"description": "Has description but no title"},
        "links": {"internal": [], "external": []},
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    # Call _build_metadata with missing title
    test_url = "https://example.com"
    metadata = reader._build_metadata(crawl_result, test_url)

    # Verify title defaults to empty string
    assert metadata["title"] == ""
    assert isinstance(metadata["title"], str)


def test_metadata_missing_description():
    """Test that _build_metadata defaults to empty string when description is missing.

    Verifies AC-5.3, AC-5.8: Default handling for missing description.

    This test ensures that when the metadata field in CrawlResult is missing
    or has no description field, _build_metadata() defaults to an empty string
    instead of raising an error.

    RED Phase: This test will FAIL because:
    - _build_metadata method doesn't exist yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    with respx.mock:
        respx.get("http://localhost:52004/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Create CrawlResult with missing description field
    crawl_result = {
        "url": "https://example.com",
        "success": True,
        "status_code": 200,
        "markdown": {"fit_markdown": "# Example\n\nContent."},
        "metadata": {"title": "Has title but no description"},
        "links": {"internal": [], "external": []},
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    # Call _build_metadata with missing description
    test_url = "https://example.com"
    metadata = reader._build_metadata(crawl_result, test_url)

    # Verify description defaults to empty string
    assert metadata["description"] == ""
    assert isinstance(metadata["description"], str)


def test_metadata_missing_links():
    """Test that _build_metadata defaults to 0 when links are missing.

    Verifies AC-5.8: Default handling for missing links.

    This test ensures that when the links field in CrawlResult is missing
    or is empty, _build_metadata() defaults internal_links_count and
    external_links_count to 0 instead of raising an error.

    RED Phase: This test will FAIL because:
    - _build_metadata method doesn't exist yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    with respx.mock:
        respx.get("http://localhost:52004/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Create CrawlResult with empty links field
    crawl_result = {
        "url": "https://example.com",
        "success": True,
        "status_code": 200,
        "markdown": {"fit_markdown": "# Example\n\nContent."},
        "metadata": {"title": "Example", "description": "Description"},
        "links": {},  # Empty links object
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    # Call _build_metadata with missing links
    test_url = "https://example.com"
    metadata = reader._build_metadata(crawl_result, test_url)

    # Verify link counts default to 0
    assert metadata["internal_links_count"] == 0
    assert metadata["external_links_count"] == 0
    assert isinstance(metadata["internal_links_count"], int)
    assert isinstance(metadata["external_links_count"], int)


def test_metadata_flat_types():
    """Test that _build_metadata returns only flat types (Qdrant compatible).

    Verifies AC-5.9: Qdrant payload compatibility with flat types only.

    This test ensures that all metadata values are primitive types (str, int,
    float) with no None values, no nested dicts, and no nested lists. This
    guarantees compatibility with Qdrant's payload schema and enables efficient
    filtering without payload index overhead.

    Qdrant requirement: Only simple types (str, int, float, bool) are allowed
    in metadata payloads for optimal query performance and indexing.

    RED Phase: This test will FAIL because:
    - _build_metadata method doesn't exist yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    with respx.mock:
        respx.get("http://localhost:52004/health").mock(
            return_value=httpx.Response(200, json={"status": "healthy"})
        )

        reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Create CrawlResult with various fields
    crawl_result = {
        "url": "https://example.com",
        "success": True,
        "status_code": 200,
        "markdown": {"fit_markdown": "# Example\n\nContent."},
        "metadata": {"title": "Example", "description": "Description"},
        "links": {"internal": [{"href": "/page1"}], "external": []},
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    # Call _build_metadata
    test_url = "https://example.com"
    metadata = reader._build_metadata(crawl_result, test_url)

    # Verify all values are flat types (str, int, float)
    allowed_types = (str, int, float)
    for key, value in metadata.items():
        assert isinstance(
            value, allowed_types
        ), f"Field '{key}' has invalid type {type(value).__name__}: {value}"

        # Explicitly reject None values
        assert value is not None, f"Field '{key}' must not be None"

        # Explicitly reject nested structures
        assert not isinstance(
            value, (dict, list, tuple, set)
        ), f"Field '{key}' must not be a nested structure: {type(value).__name__}"
