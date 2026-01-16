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

import httpx
import pytest
import respx

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


def test_metadata_links_counting():
    """Test that _build_metadata accurately counts internal and external links.

    Verifies AC-5.6, AC-5.7: Accurate link counting logic.

    This test ensures that _build_metadata() correctly counts the number of
    internal and external links from the CrawlResult links structure. It
    passes a known set of links and validates the counts match exactly.

    Test cases:
    - 3 internal links → internal_links_count = 3
    - 2 external links → external_links_count = 2

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

    # Create CrawlResult with known number of links
    crawl_result = {
        "url": "https://example.com",
        "success": True,
        "status_code": 200,
        "markdown": {"fit_markdown": "# Example\n\nContent with links."},
        "metadata": {"title": "Example", "description": "Description"},
        "links": {
            "internal": [
                {"href": "/page1"},
                {"href": "/page2"},
                {"href": "/about"},
            ],
            "external": [
                {"href": "https://other.com"},
                {"href": "https://external.org"},
            ],
        },
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    # Call _build_metadata
    test_url = "https://example.com"
    metadata = reader._build_metadata(crawl_result, test_url)

    # Verify counts match exactly
    assert metadata["internal_links_count"] == 3
    assert metadata["external_links_count"] == 2
    assert isinstance(metadata["internal_links_count"], int)
    assert isinstance(metadata["external_links_count"], int)


def test_metadata_source_url_present():
    """Test that _build_metadata includes source_url field equal to source.

    Verifies AC-5.10, Issue #17: source_url field presence and value.

    This test ensures that _build_metadata() includes a source_url field
    in the metadata dict and that its value equals the source field (both
    should be the URL). This field is indexed in Qdrant for efficient
    deduplication queries.

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

    # Create CrawlResult
    crawl_result = {
        "url": "https://example.com/test",
        "success": True,
        "status_code": 200,
        "markdown": {"fit_markdown": "# Example\n\nContent."},
        "metadata": {"title": "Example", "description": "Description"},
        "links": {"internal": [], "external": []},
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    # Call _build_metadata
    test_url = "https://example.com/test"
    metadata = reader._build_metadata(crawl_result, test_url)

    # Verify source_url field exists
    assert "source_url" in metadata

    # Verify source_url equals source field (both are the URL)
    assert metadata["source_url"] == metadata["source"]
    assert metadata["source_url"] == test_url

    # Verify it's a string type
    assert isinstance(metadata["source_url"], str)


@pytest.mark.asyncio
@respx.mock
async def test_crawl_single_url_success():
    """Test that _crawl_single_url successfully crawls URL and returns Document.

    Verifies AC-2.1, FR-5: Single URL crawling with fit_markdown extraction.

    This test ensures that _crawl_single_url() correctly:
    1. Makes POST request to /crawl endpoint with proper payload
    2. Parses successful CrawlResult response
    3. Extracts fit_markdown content as Document text
    4. Includes complete metadata in Document
    5. Returns Document with deterministic ID

    RED Phase: This test will FAIL because:
    - _crawl_single_url method doesn't exist yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Mock successful crawl response
    test_url = "https://example.com/test-page"
    mock_response = {
        "url": test_url,
        "success": True,
        "status_code": 200,
        "markdown": {
            "fit_markdown": "# Test Page\n\nThis is test content.",
            "raw_markdown": "# Test Page\n\nThis is test content.\nFooter.",
        },
        "metadata": {
            "title": "Test Page Title",
            "description": "Test page description",
        },
        "links": {
            "internal": [{"href": "/page1"}, {"href": "/page2"}],
            "external": [{"href": "https://other.com"}],
        },
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    # Create httpx client and call _crawl_single_url
    async with httpx.AsyncClient() as client:
        document = await reader._crawl_single_url(client, test_url)

    # Verify Document was returned
    assert document is not None
    from llama_index.core.schema import Document

    assert isinstance(document, Document)

    # Verify text content is fit_markdown
    assert document.text == "# Test Page\n\nThis is test content."

    # Verify metadata fields are present
    assert document.metadata["source"] == test_url
    assert document.metadata["source_url"] == test_url
    assert document.metadata["title"] == "Test Page Title"
    assert document.metadata["description"] == "Test page description"
    assert document.metadata["status_code"] == 200
    assert document.metadata["crawl_timestamp"] == "2026-01-15T12:00:00Z"
    assert document.metadata["internal_links_count"] == 2
    assert document.metadata["external_links_count"] == 1
    assert document.metadata["source_type"] == "web_crawl"

    # Verify deterministic ID was set
    assert document.id_ is not None
    assert len(document.id_) > 0


@pytest.mark.asyncio
@respx.mock
async def test_crawl_single_url_fallback_raw_markdown():
    """Test _crawl_single_url fallback to raw_markdown when fit_markdown missing.

    Verifies AC-2.2, FR-6: Fallback to raw_markdown when fit_markdown is None/missing.

    This test ensures that _crawl_single_url() correctly:
    1. Handles responses where fit_markdown is None or missing
    2. Falls back to raw_markdown for Document text content
    3. Still includes complete metadata in Document
    4. Returns valid Document with raw_markdown as text

    RED Phase: This test will FAIL because:
    - _crawl_single_url method doesn't implement fallback logic yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Mock crawl response with fit_markdown missing but raw_markdown present
    test_url = "https://example.com/test-page"
    mock_response = {
        "url": test_url,
        "success": True,
        "status_code": 200,
        "markdown": {
            "fit_markdown": None,  # Missing fit_markdown
            "raw_markdown": "# Raw Content\n\nThis is raw markdown with footer.",
        },
        "metadata": {
            "title": "Test Page Title",
            "description": "Test page description",
        },
        "links": {
            "internal": [{"href": "/page1"}],
            "external": [{"href": "https://other.com"}],
        },
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    # Create httpx client and call _crawl_single_url
    async with httpx.AsyncClient() as client:
        document = await reader._crawl_single_url(client, test_url)

    # Verify Document was returned
    assert document is not None
    from llama_index.core.schema import Document

    assert isinstance(document, Document)

    # Verify text content is raw_markdown (fallback)
    assert document.text == "# Raw Content\n\nThis is raw markdown with footer."

    # Verify metadata fields are still present
    assert document.metadata["source"] == test_url
    assert document.metadata["source_url"] == test_url
    assert document.metadata["title"] == "Test Page Title"
    assert document.metadata["description"] == "Test page description"
    assert document.metadata["status_code"] == 200
    assert document.metadata["crawl_timestamp"] == "2026-01-15T12:00:00Z"
    assert document.metadata["internal_links_count"] == 1
    assert document.metadata["external_links_count"] == 1
    assert document.metadata["source_type"] == "web_crawl"

    # Verify deterministic ID was set
    assert document.id_ is not None
    assert len(document.id_) > 0


@pytest.mark.asyncio
@respx.mock
async def test_crawl_single_url_no_markdown():
    """Test that _crawl_single_url raises ValueError when no markdown content available.

    Verifies AC-2.3, FR-8: Error handling for missing markdown content.

    This test ensures that _crawl_single_url() correctly:
    1. Detects when both fit_markdown and raw_markdown are None/missing
    2. Raises ValueError with clear error message
    3. Does not return a Document with empty text content

    Edge case: Crawl4AI may return success=True but no extractable content.
    This should be treated as an error condition.

    RED Phase: This test will FAIL because:
    - _crawl_single_url method doesn't implement error handling yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    reader = Crawl4AIReader(endpoint_url="http://localhost:52004", fail_on_error=True)

    # Mock crawl response with both markdown fields missing/None
    test_url = "https://example.com/test-page"
    mock_response = {
        "url": test_url,
        "success": True,
        "status_code": 200,
        "markdown": {
            "fit_markdown": None,  # Missing fit_markdown
            "raw_markdown": None,  # Missing raw_markdown
        },
        "metadata": {
            "title": "Test Page Title",
            "description": "Test page description",
        },
        "links": {
            "internal": [{"href": "/page1"}],
            "external": [{"href": "https://other.com"}],
        },
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    # Create httpx client and call _crawl_single_url
    # Should raise ValueError due to missing markdown content
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError) as exc_info:
            await reader._crawl_single_url(client, test_url)

    # Verify error message mentions markdown or content
    error_msg = str(exc_info.value).lower()
    assert "markdown" in error_msg or "content" in error_msg


@pytest.mark.asyncio
@respx.mock
async def test_crawl_single_url_success_false():
    """Test that _crawl_single_url raises RuntimeError when CrawlResult success=False.

    Verifies FR-8, US-6: Error handling for failed crawl operations.

    This test ensures that _crawl_single_url() correctly:
    1. Detects when CrawlResult has success=False
    2. Raises RuntimeError with error_message from response
    3. Does not attempt to extract markdown or create Document
    4. Provides clear error context including URL

    Edge case: Crawl4AI may return 200 OK but success=False with error_message
    indicating crawl failure (network timeout, DNS error, blocked, etc).

    RED Phase: This test will FAIL because:
    - _crawl_single_url method doesn't check success field yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    reader = Crawl4AIReader(endpoint_url="http://localhost:52004", fail_on_error=True)

    # Mock crawl response with success=False and error_message
    test_url = "https://example.com/blocked-page"
    mock_response = {
        "url": test_url,
        "success": False,
        "status_code": 0,
        "error_message": "Connection timeout after 30 seconds",
        "markdown": None,
        "metadata": None,
        "links": None,
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    # Create httpx client and call _crawl_single_url
    # Should raise RuntimeError with error_message from response
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError) as exc_info:
            await reader._crawl_single_url(client, test_url)

    # Verify error message includes the error_message from response
    error_msg = str(exc_info.value)
    assert "Connection timeout after 30 seconds" in error_msg
    assert test_url in error_msg


@pytest.mark.asyncio
@respx.mock
async def test_crawl_single_url_circuit_breaker_open():
    """Test that _crawl_single_url raises CircuitBreakerError when circuit is OPEN.

    Verifies AC-4.7, FR-9: Circuit breaker protection for failing services.

    This test ensures that _crawl_single_url() correctly:
    1. Checks circuit breaker state before making HTTP request
    2. Raises CircuitBreakerError when circuit is OPEN
    3. Does not attempt HTTP request when circuit is OPEN (fail-fast)
    4. Provides clear error message indicating circuit state

    Edge case: When Crawl4AI service is experiencing outages, circuit breaker
    should prevent cascading failures by rejecting calls immediately without
    waiting for timeout or making HTTP requests.

    RED Phase: This test will FAIL because:
    - _crawl_single_url method doesn't integrate circuit breaker yet
    """
    from rag_ingestion.circuit_breaker import CircuitBreakerError, CircuitState
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    reader = Crawl4AIReader(endpoint_url="http://localhost:52004", fail_on_error=True)

    # Manually set circuit breaker to OPEN state
    # This simulates the circuit breaker opening after repeated failures
    import time

    reader._circuit_breaker._state = CircuitState.OPEN
    # Set to current time (won't auto-recover immediately)
    reader._circuit_breaker.opened_at = time.time()

    # Create test URL
    test_url = "https://example.com/test-page"

    # Create httpx client and call _crawl_single_url
    # Should raise CircuitBreakerError without making HTTP request
    async with httpx.AsyncClient() as client:
        with pytest.raises(CircuitBreakerError) as exc_info:
            await reader._crawl_single_url(client, test_url)

    # Verify error message indicates circuit breaker is OPEN
    error_msg = str(exc_info.value).lower()
    assert "circuit" in error_msg or "open" in error_msg


@pytest.mark.asyncio
@respx.mock
async def test_crawl_single_url_fail_on_error_false():
    """Test that _crawl_single_url returns None when fail_on_error=False.

    Verifies AC-2.7, AC-6.5, FR-8: Graceful error handling with fail_on_error.

    This test ensures that _crawl_single_url() correctly:
    1. Returns None instead of raising exception when fail_on_error=False
    2. Logs error details for observability (check via logger calls)
    3. Allows batch operations to continue processing remaining URLs
    4. Handles both success=False and error_message scenarios gracefully

    Edge case: When fail_on_error=False, failed crawls should return None
    to enable partial success in batch operations rather than failing fast.
    This is critical for bulk crawling where some URLs may be unreachable.

    RED Phase: This test will FAIL because:
    - _crawl_single_url method doesn't check fail_on_error parameter yet
    - Method raises RuntimeError instead of returning None
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with fail_on_error=False
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004", fail_on_error=False
    )

    # Mock crawl response with success=False (failed crawl)
    test_url = "https://example.com/unreachable-page"
    mock_response = {
        "url": test_url,
        "success": False,
        "status_code": 0,
        "error_message": "DNS resolution failed",
        "markdown": None,
        "metadata": None,
        "links": None,
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    # Create httpx client and call _crawl_single_url
    # Should return None instead of raising RuntimeError
    async with httpx.AsyncClient() as client:
        document = await reader._crawl_single_url(client, test_url)

    # Verify None was returned (graceful failure)
    assert document is None


@pytest.mark.asyncio
@respx.mock
async def test_crawl_single_url_timeout_retry():
    """Test successful retry after timeout.

    Verifies AC-7.2: Retry on transient errors (timeout → success)
    Verifies FR-10: Exponential backoff retry strategy
    Verifies US-7: Crawl robustness with error recovery

    Tests that _crawl_single_url correctly:
    1. Catches httpx.TimeoutException on first attempt
    2. Waits for exponential backoff delay
    3. Retries request on second attempt
    4. Returns Document on successful retry

    Expected behavior: First request times out, second request succeeds,
    Document is returned after retry.
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check BEFORE reader initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Initialize reader with retry enabled (max_retries=3)
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004", fail_on_error=True, max_retries=3
    )

    # Test URL
    test_url = "https://example.com/timeout-then-success"

    # Mock successful crawl response (for second attempt)
    mock_response = {
        "url": test_url,
        "success": True,
        "status_code": 200,
        "markdown": {
            "raw_markdown": "# Test Content\n\nSuccessful after retry.",
            "fit_markdown": "# Test Content\n\nSuccessful after retry.",
        },
        "metadata": {
            "title": "Test Page",
            "description": "Test description",
            "language": "en",
            "keywords": "",
            "author": "",
            "og_title": "Test Page",
            "og_description": "Test description",
            "og_image": "",
        },
        "links": {"internal": [], "external": []},
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    # Track call count to simulate timeout on first call, success on second
    call_count = 0

    def crawl_side_effect(request):
        """Side effect that raises timeout on first call, returns success on second."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt - raise timeout
            raise httpx.TimeoutException("Request timeout")
        else:
            # Second attempt - return success
            return httpx.Response(200, json=mock_response)

    # Mock crawl endpoint with side_effect callback
    respx.post("http://localhost:52004/crawl").mock(side_effect=crawl_side_effect)

    # Create httpx client and call _crawl_single_url
    # Should retry after timeout and return Document
    async with httpx.AsyncClient() as client:
        document = await reader._crawl_single_url(client, test_url)

    # Verify Document was returned (not None)
    assert document is not None
    assert document.text == "# Test Content\n\nSuccessful after retry."
    assert document.metadata["source_url"] == test_url
    assert document.metadata["title"] == "Test Page"


@pytest.mark.asyncio
@respx.mock
async def test_crawl_single_url_max_retries_exhausted():
    """Test that exception is raised when all retry attempts fail.

    Verifies AC-7.1, AC-7.7: Max retries exhausted handling
    Verifies FR-10: Exponential backoff retry strategy with failure
    Verifies US-7: Error handling when recovery fails

    Tests that _crawl_single_url correctly:
    1. Catches httpx.TimeoutException on all attempts (initial + 3 retries = 4 total)
    2. Waits for exponential backoff delay between retries
    3. Exhausts all retry attempts
    4. Raises exception after max retries exceeded

    Expected behavior: All 4 attempts (initial + 3 retries) time out,
    then TimeoutException is raised to caller.
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check BEFORE reader initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Initialize reader with retry enabled (max_retries=3 → 4 total attempts)
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004", fail_on_error=True, max_retries=3
    )

    # Test URL
    test_url = "https://example.com/always-timeout"

    # Track call count to verify all retries were attempted
    call_count = 0

    def crawl_side_effect(request):
        """Side effect that always raises timeout on every attempt."""
        nonlocal call_count
        call_count += 1
        # Always raise timeout (all attempts fail)
        raise httpx.TimeoutException("Request timeout")

    # Mock crawl endpoint with side_effect callback that always times out
    respx.post("http://localhost:52004/crawl").mock(side_effect=crawl_side_effect)

    # Create httpx client and call _crawl_single_url
    # Should raise TimeoutException after exhausting all retries
    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.TimeoutException) as exc_info:
            await reader._crawl_single_url(client, test_url)

    # Verify exception message mentions timeout
    error_msg = str(exc_info.value).lower()
    assert "timeout" in error_msg

    # Verify all retry attempts were made (initial + 3 retries = 4 total)
    assert call_count == 4


@pytest.mark.asyncio
@respx.mock
async def test_crawl_single_url_http_404_no_retry():
    """Test that 4xx errors do not trigger retry attempts.

    Verifies AC-7.3: No retry on permanent errors (4xx)
    Verifies FR-10: Retry strategy only applies to transient errors
    Verifies US-7: Efficient error handling by failing fast on client errors

    Tests that _crawl_single_url correctly:
    1. Makes HTTP request to /crawl endpoint
    2. Receives 404 Not Found response (or any 4xx error)
    3. Does NOT retry the request (only 1 attempt made)
    4. Raises HTTPStatusError immediately after first failure

    Expected behavior: Client errors (4xx) are permanent and should not
    be retried. Only 1 request should be made before raising exception.
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check BEFORE reader initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Initialize reader with retry enabled (max_retries=3)
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004", fail_on_error=True, max_retries=3
    )

    # Test URL
    test_url = "https://example.com/not-found"

    # Track call count to verify NO retry is attempted
    call_count = 0

    def crawl_side_effect(request):
        """Side effect that returns 404 on every call (should only be called once)."""
        nonlocal call_count
        call_count += 1
        # Return 404 Not Found (permanent client error)
        return httpx.Response(404, json={"error": "Page not found"})

    # Mock crawl endpoint with side_effect callback that returns 404
    respx.post("http://localhost:52004/crawl").mock(side_effect=crawl_side_effect)

    # Create httpx client and call _crawl_single_url
    # Should raise HTTPStatusError immediately without retry
    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await reader._crawl_single_url(client, test_url)

    # Verify exception is HTTPStatusError with 404 status
    assert exc_info.value.response.status_code == 404

    # Verify only 1 request was made (no retry on 4xx)
    assert call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_crawl_single_url_http_500_retry():
    """Test that 5xx errors trigger retry attempts.

    Verifies AC-7.3: Retry on transient errors (5xx)
    Verifies FR-10: Exponential backoff retry strategy for server errors
    Verifies US-7: Error recovery for transient server failures

    Tests that _crawl_single_url correctly:
    1. Makes HTTP request to /crawl endpoint
    2. Receives 500 Internal Server Error response (transient error)
    3. Retries the request after exponential backoff delay
    4. Succeeds on second attempt
    5. Returns Document after successful retry

    Expected behavior: Server errors (5xx) are transient and should be
    retried. First request returns 500, second request succeeds, Document
    is returned after retry.
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check BEFORE reader initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Initialize reader with retry enabled (max_retries=3)
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004", fail_on_error=True, max_retries=3
    )

    # Test URL
    test_url = "https://example.com/server-error"

    # Mock successful crawl response (for second attempt)
    mock_response = {
        "url": test_url,
        "success": True,
        "status_code": 200,
        "markdown": {
            "fit_markdown": "# Test Content\n\nSuccessful after 500 retry.",
            "raw_markdown": "# Test Content\n\nSuccessful after 500 retry.",
        },
        "metadata": {
            "title": "Test Page",
            "description": "Test description",
        },
        "links": {"internal": [], "external": []},
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    # Track call count to verify retry was attempted
    call_count = 0

    def crawl_side_effect(request):
        """Side effect that returns 500 on first call, success on second."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt - return 500 Internal Server Error
            return httpx.Response(500, json={"error": "Internal server error"})
        else:
            # Second attempt - return success
            return httpx.Response(200, json=mock_response)

    # Mock crawl endpoint with side_effect callback
    respx.post("http://localhost:52004/crawl").mock(side_effect=crawl_side_effect)

    # Create httpx client and call _crawl_single_url
    # Should retry after 500 error and return Document
    async with httpx.AsyncClient() as client:
        document = await reader._crawl_single_url(client, test_url)

    # Verify Document was returned (not None)
    assert document is not None
    assert document.text == "# Test Content\n\nSuccessful after 500 retry."
    assert document.metadata["source_url"] == test_url
    assert document.metadata["title"] == "Test Page"

    # Verify retry was attempted (call_count == 2)
    assert call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_retry_exponential_backoff():
    """Test that retry logic uses exponential backoff delays [1.0, 2.0, 4.0].

    Verifies AC-7.2, NFR-8: Exponential backoff delay pattern
    Verifies FR-10: Retry delays follow [1.0, 2.0, 4.0] seconds pattern

    Tests that _crawl_single_url correctly:
    1. Makes initial HTTP request to /crawl endpoint
    2. Receives transient error (timeout) on first 3 attempts
    3. Waits for exponential backoff delays: 1.0s, 2.0s, 4.0s
    4. Succeeds on 4th attempt (initial + 3 retries)
    5. Returns Document after successful retry

    This test uses unittest.mock to patch asyncio.sleep and capture
    the actual delay values passed to sleep() calls. It verifies that
    the delays match the configured retry_delays=[1.0, 2.0, 4.0] pattern.

    Expected behavior: Delays should be exactly [1.0, 2.0, 4.0] seconds,
    matching the default retry_delays configuration.
    """
    from unittest.mock import patch

    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check BEFORE reader initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Initialize reader with default retry_delays=[1.0, 2.0, 4.0]
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004", fail_on_error=True, max_retries=3
    )

    # Test URL
    test_url = "https://example.com/exponential-backoff-test"

    # Mock successful crawl response (for 4th attempt)
    mock_response = {
        "url": test_url,
        "success": True,
        "status_code": 200,
        "markdown": {
            "fit_markdown": "# Test Content\n\nSuccessful after exponential backoff.",
            "raw_markdown": "# Test Content\n\nSuccessful after exponential backoff.",
        },
        "metadata": {
            "title": "Test Page",
            "description": "Test description",
        },
        "links": {"internal": [], "external": []},
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    # Track call count and sleep delays
    call_count = 0
    sleep_delays = []

    def crawl_side_effect(request):
        """Side effect that raises timeout on first 3 calls, success on 4th."""
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            # First 3 attempts - raise timeout
            raise httpx.TimeoutException("Request timeout")
        else:
            # 4th attempt - return success
            return httpx.Response(200, json=mock_response)

    # Mock crawl endpoint with side_effect callback
    respx.post("http://localhost:52004/crawl").mock(side_effect=crawl_side_effect)

    # Mock asyncio.sleep to capture delay values
    async def mock_sleep(delay):
        """Mock sleep that records delay value."""
        sleep_delays.append(delay)

    # Create httpx client and call _crawl_single_url with mocked sleep
    async with httpx.AsyncClient() as client:
        with patch("asyncio.sleep", side_effect=mock_sleep):
            document = await reader._crawl_single_url(client, test_url)

    # Verify Document was returned (success after retries)
    assert document is not None
    assert document.text == "# Test Content\n\nSuccessful after exponential backoff."
    assert document.metadata["source_url"] == test_url

    # Verify exponential backoff delays match [1.0, 2.0, 4.0] pattern
    expected = [1.0, 2.0, 4.0]
    assert sleep_delays == expected, f"Expected {expected}, got {sleep_delays}"

    # Verify all 4 attempts were made (initial + 3 retries)
    assert call_count == 4


@pytest.mark.asyncio
@respx.mock
async def test_deduplicate_url_called():
    """Test that delete_by_url is called for each URL when deduplication enabled.

    Verifies Issue #16: Automatic deduplication on re-crawl
    Verifies AC-8.1: Deduplication enabled by default
    Verifies AC-8.2: delete_by_url called before crawling

    This test ensures that aload_data() correctly:
    1. Accepts enable_deduplication parameter (default True)
    2. Accepts vector_store parameter (VectorStoreManager instance)
    3. Calls vector_store.delete_by_url() for each URL before crawling
    4. Proceeds with normal crawl after deduplication

    Expected behavior: For each URL in the batch, delete_by_url() is called
    BEFORE the URL is crawled. This matches file watcher pattern: delete old
    versions before processing new content.

    RED Phase: This test will FAIL because:
    - aload_data method doesn't exist yet
    """
    from unittest.mock import AsyncMock, MagicMock

    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create mock VectorStoreManager with delete_by_url method
    mock_vector_store = MagicMock()
    mock_vector_store.delete_by_url = AsyncMock(return_value=5)

    # Create reader with deduplication enabled and vector_store
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        enable_deduplication=True,
        vector_store=mock_vector_store,
    )

    # Test URLs
    test_urls = [
        "https://example.com/page1",
        "https://example.com/page2",
    ]

    # Mock successful crawl responses for both URLs
    for test_url in test_urls:
        mock_response = {
            "url": test_url,
            "success": True,
            "status_code": 200,
            "markdown": {
                "fit_markdown": f"# Content from {test_url}",
                "raw_markdown": f"# Content from {test_url}",
            },
            "metadata": {
                "title": "Test Page",
                "description": "Test description",
            },
            "links": {"internal": [], "external": []},
            "crawl_timestamp": "2026-01-15T12:00:00Z",
        }
        respx.post("http://localhost:52004/crawl").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

    # Call aload_data with URLs
    documents = await reader.aload_data(test_urls)

    # Verify delete_by_url was called for each URL before crawling
    assert mock_vector_store.delete_by_url.call_count == 2
    mock_vector_store.delete_by_url.assert_any_call(test_urls[0])
    mock_vector_store.delete_by_url.assert_any_call(test_urls[1])

    # Verify documents were returned (crawling happened after deduplication)
    assert len(documents) == 2
    assert all(doc is not None for doc in documents)


@pytest.mark.asyncio
@respx.mock
async def test_deduplicate_url_skipped():
    """Test that delete_by_url is NOT called when deduplication disabled.

    Verifies Issue #16: Deduplication can be disabled
    Verifies AC-8.3: enable_deduplication=False skips deletion

    This test ensures that aload_data() correctly:
    1. Accepts enable_deduplication parameter set to False
    2. Does NOT call vector_store.delete_by_url() when disabled
    3. Proceeds directly with crawling without deletion
    4. Returns documents normally

    Expected behavior: When enable_deduplication=False, no calls to
    delete_by_url() should be made, allowing re-crawls to create
    duplicate entries in the vector store.

    RED Phase: This test will FAIL because:
    - aload_data method doesn't exist yet
    """
    from unittest.mock import AsyncMock, MagicMock

    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create mock VectorStoreManager with delete_by_url method
    mock_vector_store = MagicMock()
    mock_vector_store.delete_by_url = AsyncMock(return_value=5)

    # Create reader with deduplication DISABLED and vector_store
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        enable_deduplication=False,
        vector_store=mock_vector_store,
    )

    # Test URLs
    test_urls = [
        "https://example.com/page1",
        "https://example.com/page2",
    ]

    # Mock successful crawl responses for both URLs
    for test_url in test_urls:
        mock_response = {
            "url": test_url,
            "success": True,
            "status_code": 200,
            "markdown": {
                "fit_markdown": f"# Content from {test_url}",
                "raw_markdown": f"# Content from {test_url}",
            },
            "metadata": {
                "title": "Test Page",
                "description": "Test description",
            },
            "links": {"internal": [], "external": []},
            "crawl_timestamp": "2026-01-15T12:00:00Z",
        }
        respx.post("http://localhost:52004/crawl").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

    # Call aload_data with URLs
    documents = await reader.aload_data(test_urls)

    # Verify delete_by_url was NOT called (deduplication disabled)
    assert mock_vector_store.delete_by_url.call_count == 0

    # Verify documents were returned (crawling happened without deduplication)
    assert len(documents) == 2
    assert all(doc is not None for doc in documents)


@pytest.mark.asyncio
@respx.mock
async def test_deduplicate_url_no_vector_store():
    """Test that deduplication is skipped when vector_store is None.

    Verifies Issue #16: Deduplication skipped gracefully when no vector_store
    Verifies AC-8.4: No errors when vector_store=None (even if enabled=True)

    This test ensures that aload_data() correctly:
    1. Accepts vector_store=None (no vector store available)
    2. Does NOT attempt deduplication operations
    3. Proceeds directly with crawling without errors
    4. Returns documents normally despite deduplication being "enabled"

    Expected behavior: When vector_store=None, deduplication should be
    skipped silently (no delete operations, no errors) because there's
    no vector store to deduplicate against. enable_deduplication flag
    is meaningless without a vector_store instance.

    Edge case: User may set enable_deduplication=True but forget to
    provide vector_store instance. This should gracefully skip
    deduplication rather than crashing.

    RED Phase: This test will FAIL because:
    - aload_data method doesn't exist yet
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with deduplication enabled BUT no vector_store
    # This should skip deduplication gracefully without errors
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        enable_deduplication=True,
        vector_store=None,  # No vector store available
    )

    # Test URLs
    test_urls = [
        "https://example.com/page1",
        "https://example.com/page2",
    ]

    # Mock successful crawl responses for both URLs
    for test_url in test_urls:
        mock_response = {
            "url": test_url,
            "success": True,
            "status_code": 200,
            "markdown": {
                "fit_markdown": f"# Content from {test_url}",
                "raw_markdown": f"# Content from {test_url}",
            },
            "metadata": {
                "title": "Test Page",
                "description": "Test description",
            },
            "links": {"internal": [], "external": []},
            "crawl_timestamp": "2026-01-15T12:00:00Z",
        }
        respx.post("http://localhost:52004/crawl").mock(
            return_value=httpx.Response(200, json=mock_response)
        )

    # Call aload_data with URLs
    # Should NOT raise error despite enable_deduplication=True and None store
    documents = await reader.aload_data(test_urls)

    # Verify documents were returned (crawling succeeded without deduplication)
    assert len(documents) == 2
    assert all(doc is not None for doc in documents)

    # Verify documents have expected content (assert not None for type safety)
    assert documents[0] is not None
    assert documents[0].text.startswith("# Content from")
    assert documents[1] is not None
    assert documents[1].text.startswith("# Content from")


@pytest.mark.asyncio
@respx.mock
async def test_aload_data_empty_list():
    """Test that aload_data returns empty list when given empty URL list.

    Verifies AC-3.1: Edge case handling for empty URL list
    Verifies FR-14: Batch loading with no URLs returns immediately

    This test ensures that aload_data() correctly:
    1. Accepts empty URL list []
    2. Returns empty list immediately without making HTTP requests
    3. Does not validate service health (optimization for empty input)
    4. Does not attempt deduplication operations

    Expected behavior: When given empty list, return empty list without
    any network operations or service validation. This is an optimization
    for the common case of empty input.

    GREEN Phase: This test should PASS immediately because aload_data was
    already implemented in task 2.7.2c with empty list handling (line 664-665).
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Call aload_data with empty list
    documents = await reader.aload_data([])

    # Verify empty list is returned
    assert documents == []
    assert isinstance(documents, list)
    assert len(documents) == 0


@pytest.mark.asyncio
@respx.mock
async def test_aload_data_single_url():
    """Test that aload_data successfully crawls single URL.

    Verifies AC-2.1, AC-2.2: Single URL batch processing
    Verifies FR-14: Async batch loading with one URL

    This test ensures that aload_data() correctly:
    1. Validates service health before processing
    2. Crawls single URL in batch mode
    3. Returns list with one Document
    4. Document contains correct markdown content and metadata

    Expected behavior: Single URL in list should be crawled successfully,
    return list with one Document with proper content and metadata.

    RED Phase: This test will FAIL because:
    - aload_data method doesn't exist yet (was partially implemented in 2.7.2c)
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Test URL
    test_url = "https://example.com/test-single"

    # Mock successful crawl response
    mock_response = {
        "url": test_url,
        "success": True,
        "status_code": 200,
        "markdown": {
            "fit_markdown": "# Single URL Test\n\nThis is content from single URL.",
            "raw_markdown": "# Single URL Test\n\nThis is content from single URL.",
        },
        "metadata": {
            "title": "Single URL Test Page",
            "description": "Test page for single URL batch",
        },
        "links": {
            "internal": [{"href": "/page1"}],
            "external": [{"href": "https://other.com"}],
        },
        "crawl_timestamp": "2026-01-15T12:00:00Z",
    }

    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(200, json=mock_response)
    )

    # Call aload_data with single URL
    documents = await reader.aload_data([test_url])

    # Verify list with one Document returned
    assert isinstance(documents, list)
    assert len(documents) == 1
    assert documents[0] is not None

    # Verify Document content and metadata
    from llama_index.core.schema import Document

    assert isinstance(documents[0], Document)
    assert documents[0].text == "# Single URL Test\n\nThis is content from single URL."
    assert documents[0].metadata["source"] == test_url
    assert documents[0].metadata["source_url"] == test_url
    assert documents[0].metadata["title"] == "Single URL Test Page"
    assert documents[0].metadata["description"] == "Test page for single URL batch"
    assert documents[0].metadata["status_code"] == 200
    assert documents[0].metadata["internal_links_count"] == 1
    assert documents[0].metadata["external_links_count"] == 1


@pytest.mark.asyncio
@respx.mock
async def test_aload_data_multiple_urls():
    """Test that aload_data crawls multiple URLs concurrently.

    Verifies AC-3.1, AC-3.2, AC-3.3: Multiple URLs concurrent processing
    Verifies US-3: Batch crawling with concurrency control

    This test ensures that aload_data() correctly:
    1. Validates service health before processing batch
    2. Crawls multiple URLs concurrently with semaphore control
    3. Returns list with all Documents in same order as input URLs
    4. All Documents contain correct markdown content and metadata
    5. Uses shared httpx.AsyncClient for connection pooling

    Expected behavior: Multiple URLs in list should be crawled concurrently
    (respecting max_concurrent_requests limit), returning list with all
    Documents in the same order as input URLs.

    GREEN Phase: This test should PASS immediately because aload_data was
    already fully implemented in task 2.7.2c with concurrent processing.
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Test URLs
    test_urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ]

    # Mock successful crawl responses with side_effect to match URL from request
    def crawl_side_effect(request):
        """Return appropriate response based on request URL."""
        request_data = request.read()
        import json

        request_json = json.loads(request_data)
        url = request_json["url"]

        # Find which page this URL corresponds to
        if "page1" in url:
            page_num = 1
        elif "page2" in url:
            page_num = 2
        elif "page3" in url:
            page_num = 3
        else:
            page_num = 0

        mock_response = {
            "url": url,
            "success": True,
            "status_code": 200,
            "markdown": {
                "fit_markdown": f"# Page {page_num}\n\nContent from page {page_num}.",
                "raw_markdown": f"# Page {page_num}\n\nContent from page {page_num}.",
            },
            "metadata": {
                "title": f"Page {page_num} Title",
                "description": f"Description for page {page_num}",
            },
            "links": {
                "internal": [{"href": f"/page{page_num}-link"}],
                "external": [{"href": f"https://external{page_num}.com"}],
            },
            "crawl_timestamp": "2026-01-15T12:00:00Z",
        }

        return httpx.Response(200, json=mock_response)

    respx.post("http://localhost:52004/crawl").mock(side_effect=crawl_side_effect)

    # Call aload_data with multiple URLs
    documents = await reader.aload_data(test_urls)

    # Verify list with all Documents returned in same order
    assert isinstance(documents, list)
    assert len(documents) == 3
    assert all(doc is not None for doc in documents)

    # Verify all Documents are proper Document instances
    from llama_index.core.schema import Document

    assert all(isinstance(doc, Document) for doc in documents)

    # Verify Documents content matches URL order (assert not None for type safety)
    assert documents[0] is not None
    assert documents[0].text == "# Page 1\n\nContent from page 1."
    assert documents[1] is not None
    assert documents[1].text == "# Page 2\n\nContent from page 2."
    assert documents[2] is not None
    assert documents[2].text == "# Page 3\n\nContent from page 3."

    # Verify metadata for all Documents
    assert documents[0].metadata["source"] == test_urls[0]
    assert documents[0].metadata["source_url"] == test_urls[0]
    assert documents[0].metadata["title"] == "Page 1 Title"

    assert documents[1].metadata["source"] == test_urls[1]
    assert documents[1].metadata["source_url"] == test_urls[1]
    assert documents[1].metadata["title"] == "Page 2 Title"

    assert documents[2].metadata["source"] == test_urls[2]
    assert documents[2].metadata["source_url"] == test_urls[2]
    assert documents[2].metadata["title"] == "Page 3 Title"


@pytest.mark.asyncio
@respx.mock
async def test_aload_data_order_preservation():
    """Test that aload_data preserves order with failures (None for failed URLs).

    Verifies AC-3.4: Order preservation with failures
    Verifies Issue #1: Results list maintains input order even with failures

    This test ensures that aload_data() correctly:
    1. Maintains input URL order in results list
    2. Returns None for failed URLs at their original position
    3. Returns Document for successful URLs at their original position
    4. Does not filter out failures or reorder results

    Expected behavior: Given URLs [success, failure, success], return
    list [Document, None, Document] preserving the original order.
    This enables callers to correlate results with input URLs.

    Pattern: success → failure → success
    Result:  Document → None → Document (order preserved)

    RED Phase: This test will FAIL because:
    - aload_data method doesn't preserve order with failures yet
    - Method may filter None values or reorder results
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with fail_on_error=False to enable graceful failures
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004", fail_on_error=False
    )

    # Test URLs: success, failure, success pattern
    test_urls = [
        "https://example.com/success1",
        "https://example.com/failure",
        "https://example.com/success2",
    ]

    # Mock crawl responses with side_effect for different outcomes
    def crawl_side_effect(request):
        """Return success/failure based on request URL."""
        request_data = request.read()
        import json

        request_json = json.loads(request_data)
        url = request_json["url"]

        # Second URL (failure) returns success=False
        if "failure" in url:
            return httpx.Response(
                200,
                json={
                    "url": url,
                    "success": False,
                    "status_code": 0,
                    "error_message": "Crawl failed intentionally",
                    "markdown": None,
                    "metadata": None,
                    "links": None,
                    "crawl_timestamp": "2026-01-15T12:00:00Z",
                },
            )

        # First and third URLs (success) return valid documents
        page_num = 1 if "success1" in url else 2
        return httpx.Response(
            200,
            json={
                "url": url,
                "success": True,
                "status_code": 200,
                "markdown": {
                    "fit_markdown": f"# Success {page_num}\n\nContent.",
                    "raw_markdown": f"# Success {page_num}\n\nContent.",
                },
                "metadata": {
                    "title": f"Success {page_num}",
                    "description": f"Description {page_num}",
                },
                "links": {"internal": [], "external": []},
                "crawl_timestamp": "2026-01-15T12:00:00Z",
            },
        )

    respx.post("http://localhost:52004/crawl").mock(side_effect=crawl_side_effect)

    # Call aload_data with success-failure-success pattern
    documents = await reader.aload_data(test_urls)

    # Verify results list preserves order with None for failure
    assert isinstance(documents, list)
    assert len(documents) == 3, "Results list should match input length"

    # Verify order: Document, None, Document (preserves input order)
    assert documents[0] is not None, "First URL should succeed (Document)"
    assert documents[1] is None, "Second URL should fail (None)"
    assert documents[2] is not None, "Third URL should succeed (Document)"

    # Verify successful Documents have correct content
    from llama_index.core.schema import Document

    assert isinstance(documents[0], Document)
    assert documents[0].text == "# Success 1\n\nContent."
    assert documents[0].metadata["source"] == test_urls[0]

    assert isinstance(documents[2], Document)
    assert documents[2].text == "# Success 2\n\nContent."
    assert documents[2].metadata["source"] == test_urls[2]


@pytest.mark.asyncio
@respx.mock
async def test_aload_data_concurrent_limit():
    """Test that aload_data enforces max_concurrent_requests limit.

    Verifies AC-3.3, NFR-4: Concurrency limit enforcement
    Verifies FR-14: Semaphore-based concurrency control
    Verifies US-3: Batch crawling with resource constraints

    This test ensures that aload_data() correctly:
    1. Creates asyncio.Semaphore with max_concurrent_requests limit
    2. Enforces that no more than max_concurrent requests run at same time
    3. Processes all URLs eventually (queuing excess beyond limit)
    4. Uses semaphore wrapper around _crawl_single_url calls

    Expected behavior: With 10 URLs and max_concurrent=3, at most 3 requests
    should be active concurrently. Remaining 7 URLs should wait in queue.
    This prevents resource exhaustion and respects service rate limits.

    Test strategy: Track concurrent request count using side_effect callback
    that increments counter on entry, decrements on exit. Assert max count
    never exceeds configured limit.

    RED Phase: This test will FAIL because:
    - aload_data method doesn't enforce concurrency limit yet
    - Method may process all URLs concurrently without semaphore control
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with max_concurrent=3 (low limit to test enforcement)
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004", max_concurrent_requests=3
    )

    # Test URLs: 10 URLs to test concurrency limit
    test_urls = [f"https://example.com/page{i}" for i in range(1, 11)]

    # Track concurrent requests and max concurrent
    concurrent_count = 0
    max_concurrent_reached = 0

    import asyncio

    # Lock to protect concurrent_count updates
    count_lock = asyncio.Lock()

    async def crawl_side_effect(request):
        """Track concurrent requests and enforce delay."""
        nonlocal concurrent_count, max_concurrent_reached

        # Increment concurrent count
        async with count_lock:
            concurrent_count += 1
            max_concurrent_reached = max(max_concurrent_reached, concurrent_count)

        # Simulate processing delay (50ms)
        await asyncio.sleep(0.05)

        # Parse request to get URL
        request_data = request.read()
        import json

        request_json = json.loads(request_data)
        url = request_json["url"]

        # Extract page number from URL
        page_num = int(url.split("page")[1])

        # Create mock response
        mock_response = {
            "url": url,
            "success": True,
            "status_code": 200,
            "markdown": {
                "fit_markdown": f"# Page {page_num}\n\nContent.",
                "raw_markdown": f"# Page {page_num}\n\nContent.",
            },
            "metadata": {
                "title": f"Page {page_num}",
                "description": f"Description {page_num}",
            },
            "links": {"internal": [], "external": []},
            "crawl_timestamp": "2026-01-15T12:00:00Z",
        }

        # Decrement concurrent count
        async with count_lock:
            concurrent_count -= 1

        return httpx.Response(200, json=mock_response)

    # Mock crawl endpoint with async side_effect
    respx.post("http://localhost:52004/crawl").mock(side_effect=crawl_side_effect)

    # Call aload_data with 10 URLs
    documents = await reader.aload_data(test_urls)

    # Verify all documents returned
    assert len(documents) == 10
    assert all(doc is not None for doc in documents)

    # Verify concurrency limit was enforced (max 3 concurrent)
    assert (
        max_concurrent_reached <= 3
    ), f"Max concurrent requests was {max_concurrent_reached}, expected <= 3"

    # Verify final concurrent count is 0 (all requests completed)
    assert concurrent_count == 0, "Not all requests completed"


@pytest.mark.asyncio
@respx.mock
async def test_aload_data_logging(caplog):
    """Test that aload_data logs batch statistics.

    Verifies AC-2.8, AC-3.8, FR-11: Structured logging for batch operations
    Verifies US-3: Observability for batch crawling

    This test ensures that aload_data() correctly:
    1. Logs batch start message with URL count and max_concurrent
    2. Logs batch completion message with success/failure counts
    3. Uses structured logging with extra dict fields for filtering
    4. Includes URL count, succeeded count, failed count in logs

    Expected behavior: aload_data should log at INFO level:
    - "Starting batch crawl of N URLs" with url_count and max_concurrent
    - "Batch crawl complete: X succeeded, Y failed" with total/succeeded/failed

    Test strategy: Use pytest caplog fixture to capture log messages,
    assert required messages present with correct values.

    RED Phase: This test will FAIL because:
    - aload_data method doesn't include batch statistics logging yet
    - Method may log crawl events but not batch-level statistics
    """
    import logging

    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with fail_on_error=False to enable partial success
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004", fail_on_error=False
    )

    # Test URLs: 3 URLs (2 success, 1 failure for batch statistics)
    test_urls = [
        "https://example.com/success1",
        "https://example.com/failure",
        "https://example.com/success2",
    ]

    # Mock crawl responses with side_effect for different outcomes
    def crawl_side_effect(request):
        """Return success/failure based on request URL."""
        request_data = request.read()
        import json

        request_json = json.loads(request_data)
        url = request_json["url"]

        # Second URL (failure) returns success=False
        if "failure" in url:
            return httpx.Response(
                200,
                json={
                    "url": url,
                    "success": False,
                    "status_code": 0,
                    "error_message": "Crawl failed",
                    "markdown": None,
                    "metadata": None,
                    "links": None,
                    "crawl_timestamp": "2026-01-15T12:00:00Z",
                },
            )

        # First and third URLs (success) return valid documents
        page_num = 1 if "success1" in url else 2
        return httpx.Response(
            200,
            json={
                "url": url,
                "success": True,
                "status_code": 200,
                "markdown": {
                    "fit_markdown": f"# Success {page_num}\n\nContent.",
                    "raw_markdown": f"# Success {page_num}\n\nContent.",
                },
                "metadata": {
                    "title": f"Success {page_num}",
                    "description": f"Description {page_num}",
                },
                "links": {"internal": [], "external": []},
                "crawl_timestamp": "2026-01-15T12:00:00Z",
            },
        )

    respx.post("http://localhost:52004/crawl").mock(side_effect=crawl_side_effect)

    # Capture logs at INFO level
    with caplog.at_level(logging.INFO):
        # Call aload_data with 3 URLs (2 success, 1 failure)
        documents = await reader.aload_data(test_urls)

    # Verify batch start log message
    start_messages = [
        record.message
        for record in caplog.records
        if "Starting batch crawl" in record.message
    ]
    assert len(start_messages) >= 1, "Missing batch start log message"

    # Verify batch start message includes URL count
    start_message = start_messages[0]
    assert "3 URLs" in start_message or "3" in start_message

    # Verify batch completion log message
    completion_messages = [
        record.message
        for record in caplog.records
        if "Batch crawl complete" in record.message
    ]
    assert len(completion_messages) >= 1, "Missing batch completion log message"

    # Verify completion message includes success/failure counts
    completion_message = completion_messages[0]
    assert "2 succeeded" in completion_message or "succeeded: 2" in completion_message
    assert "1 failed" in completion_message or "failed: 1" in completion_message

    # Verify documents returned with expected pattern (2 success, 1 failure)
    assert len(documents) == 3
    assert documents[0] is not None  # Success
    assert documents[1] is None  # Failure
    assert documents[2] is not None  # Success


# ==============================================================================
# Synchronous load_data Tests (Task 2.9.1)
# ==============================================================================


@respx.mock
def test_load_data_delegates_to_aload_data(respx_mock: respx.MockRouter) -> None:
    """Test that load_data properly delegates to aload_data using asyncio.run."""
    from unittest.mock import AsyncMock, patch

    from llama_index.core.schema import Document

    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check
    respx_mock.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    # Create reader
    reader = Crawl4AIReader()

    # Mock aload_data to verify it's called
    mock_aload_data = AsyncMock(
        return_value=[Document(text="Test", id_="test-id")]
    )

    # Patch aload_data on the class, not the instance (Pydantic compatibility)
    with patch.object(Crawl4AIReader, "aload_data", mock_aload_data):
        # Call synchronous load_data
        result = reader.load_data(["https://example.com"])

    # Verify aload_data was called with correct arguments
    # Note: When patching at class level, self is not included in call args
    mock_aload_data.assert_called_once_with(["https://example.com"])

    # Verify result is returned from aload_data
    assert len(result) == 1
    assert result[0] is not None
    assert result[0].text == "Test"
    assert result[0].id_ == "test-id"


@respx.mock
def test_load_data_single_url(respx_mock: respx.MockRouter) -> None:
    """Test load_data with single URL returns single Document."""
    from llama_index.core.schema import Document

    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check
    respx_mock.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    # Mock crawl response
    respx_mock.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(
            200,
            json={
                "url": "https://example.com",
                "success": True,
                "status_code": 200,
                "markdown": {
                    "fit_markdown": "# Example Page\n\nThis is test markdown.",
                    "raw_markdown": "# Example Page\n\nThis is test markdown.",
                },
                "metadata": {"title": "Example", "description": "Test"},
                "links": {"internal": [], "external": []},
                "crawl_timestamp": "2026-01-15T12:00:00Z",
            },
        )
    )

    # Create reader
    reader = Crawl4AIReader()

    # Call load_data (synchronous) with single URL
    result = reader.load_data(["https://example.com"])

    # Verify single Document returned
    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].text == "# Example Page\n\nThis is test markdown."
    assert result[0].metadata["source"] == "https://example.com"
    assert result[0].metadata["source_url"] == "https://example.com"


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
@respx.mock
async def test_error_timeout_exception():
    """Test HTTP timeout exception handling with retry and fail_on_error.

    Verifies FR-8: Error handling and resilience
    Verifies US-6: Error messages and debug context

    This test ensures that _crawl_single_url() correctly:
    1. Catches httpx.TimeoutException during crawl requests
    2. Retries with exponential backoff for transient timeout errors
    3. Respects fail_on_error flag: True raises exception, False returns None
    4. Logs timeout errors with URL context for debugging

    Expected behavior: Timeout exceptions are transient errors that trigger
    retry logic. After exhausting retries, behavior depends on fail_on_error:
    - fail_on_error=True: Raise TimeoutException with context
    - fail_on_error=False: Log error, return None gracefully

    This test verifies the fail_on_error=True path (exception propagation).
    """
    from rag_ingestion.crawl4ai_reader import Crawl4AIReader

    # Mock health check
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with fail_on_error=True (explicit, default is False)
    reader = Crawl4AIReader(fail_on_error=True)

    # Mock crawl endpoint to always timeout
    call_count = 0

    def timeout_side_effect(request):
        nonlocal call_count
        call_count += 1
        raise httpx.TimeoutException(
            "Request timeout", request=request
        )

    respx.post("http://localhost:52004/crawl").mock(
        side_effect=timeout_side_effect
    )

    # Call _crawl_single_url and expect TimeoutException
    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.TimeoutException):
            await reader._crawl_single_url(
                client, "https://example.com"
            )

    # Verify retries attempted (initial + max_retries)
    assert call_count == 4, "Should retry 3 times after initial timeout"
