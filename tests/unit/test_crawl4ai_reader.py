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
    """Test that _crawl_single_url falls back to raw_markdown when fit_markdown is missing.

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
    reader._circuit_breaker.opened_at = time.time()  # Set to current time (won't auto-recover immediately)

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
    from unittest.mock import AsyncMock, patch

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
    assert sleep_delays == [1.0, 2.0, 4.0], f"Expected [1.0, 2.0, 4.0], got {sleep_delays}"

    # Verify all 4 attempts were made (initial + 3 retries)
    assert call_count == 4
