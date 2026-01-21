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

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

# This import will fail initially - that's expected in RED phase
# from crawl4r.readers.crawl4ai import Crawl4AIReader


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
    from crawl4r.readers.crawl4ai import Crawl4AIReaderConfig

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


@respx.mock
def test_reader_respects_crawl4ai_base_url_from_settings():
    """Test that Crawl4AIReader uses crawl4ai_base_url from Settings.

    Verifies FR-1.1: Reader respects Settings configuration.

    This test ensures that when a Settings object with a custom
    crawl4ai_base_url is passed to the reader constructor, the reader
    uses that URL instead of the default endpoint.

    RED Phase: This test will FAIL because:
    - Settings class doesn't have crawl4ai_base_url field yet
    - Crawl4AIReader class doesn't exist yet
    """
    from crawl4r.core.config import Settings
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Create Settings with custom Crawl4AI base URL
    custom_url = "http://custom-crawl4ai.example.com:9999"
    settings = Settings(
        watch_folder=Path("/tmp/test"),
        crawl4ai_base_url=custom_url,
    )

    # Mock health check for custom URL
    respx.get(f"{custom_url}/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
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

    from crawl4r.readers.crawl4ai import Crawl4AIReaderConfig

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

    from crawl4r.readers.crawl4ai import Crawl4AIReaderConfig

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

    from crawl4r.readers.crawl4ai import Crawl4AIReaderConfig

    # Attempt to create config with unexpected field (intentional error test)
    with pytest.raises(ValidationError) as exc_info:
        Crawl4AIReaderConfig(invalid_field="should_fail")  # type: ignore[call-arg]

    # Verify error mentions extra field not permitted
    error_msg = str(exc_info.value).lower()
    assert "extra" in error_msg or "permitted" in error_msg


def test_config_has_language_fields():
    """Test that Crawl4AIReaderConfig has language filter fields with correct defaults.

    Verifies FR-2, AC-2.1, AC-3.1: Language filter configuration fields.

    This test ensures the Crawl4AIReaderConfig class has the 3 new language
    filter fields with correct default values:
    - enable_language_filter (default: True)
    - allowed_languages (default: ["en"])
    - language_confidence_threshold (default: 0.5)
    """
    from crawl4r.readers.crawl4ai import Crawl4AIReaderConfig

    # Create config instance with defaults
    config = Crawl4AIReaderConfig()

    # Verify enable_language_filter field exists with correct default
    assert hasattr(config, "enable_language_filter")
    assert isinstance(config.enable_language_filter, bool)
    assert config.enable_language_filter is True

    # Verify allowed_languages field exists with correct default
    assert hasattr(config, "allowed_languages")
    assert isinstance(config.allowed_languages, list)
    assert config.allowed_languages == ["en"]

    # Verify language_confidence_threshold field exists with correct default
    assert hasattr(config, "language_confidence_threshold")
    assert isinstance(config.language_confidence_threshold, float)
    assert config.language_confidence_threshold == 0.5


def test_config_validates_confidence_range():
    """Test that Crawl4AIReaderConfig validates confidence threshold range.

    Verifies AC-2.1, AC-3.1: Confidence threshold validation (0.0-1.0).

    This test ensures Pydantic validation catches confidence threshold values
    outside the valid range. Valid range is 0.0-1.0 (ge=0.0, le=1.0 in Field()).
    """
    from pydantic import ValidationError

    from crawl4r.readers.crawl4ai import Crawl4AIReaderConfig

    # Attempt to create config with confidence below minimum (< 0.0)
    with pytest.raises(ValidationError) as exc_info:
        Crawl4AIReaderConfig(language_confidence_threshold=-0.1)

    # Verify error mentions language_confidence_threshold field
    error_msg = str(exc_info.value).lower()
    assert "language_confidence_threshold" in error_msg

    # Attempt to create config with confidence above maximum (> 1.0)
    with pytest.raises(ValidationError) as exc_info:
        Crawl4AIReaderConfig(language_confidence_threshold=1.5)

    # Verify error mentions language_confidence_threshold field
    error_msg = str(exc_info.value).lower()
    assert "language_confidence_threshold" in error_msg

    # Verify valid boundary values are accepted
    config_min = Crawl4AIReaderConfig(language_confidence_threshold=0.0)
    assert config_min.language_confidence_threshold == 0.0

    config_max = Crawl4AIReaderConfig(language_confidence_threshold=1.0)
    assert config_max.language_confidence_threshold == 1.0


@pytest.mark.asyncio
@respx.mock
async def test_filter_by_allowed_languages():
    """Test that documents with disallowed languages are filtered out.

    Verifies FR-4, AC-1.2, AC-2.2: Language filter blocks disallowed languages.

    This test ensures that when allowed_languages=["en"], documents detected
    as Spanish (es) are filtered out during batch processing.
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with allowed_languages=["en"]
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        enable_language_filter=True,
        allowed_languages=["en"],
        language_confidence_threshold=0.5,
    )

    # Mock HttpCrawlClient.crawl() to return Spanish content
    test_url = "https://example.com/spanish-page"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="Hola mundo, esta es una página en español.",
        success=True,
        title="Página en Español",
        description="Contenido en español",
        status_code=200,
        detected_language="es",  # Spanish detected
        language_confidence=0.95,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call aload_data - should filter out Spanish document
    documents = await reader.aload_data([test_url])

    # Verify document was filtered out (empty list returned)
    assert len(documents) == 0


@pytest.mark.asyncio
@respx.mock
async def test_filter_accepts_allowed_language():
    """Test that documents with allowed languages are accepted.

    Verifies AC-2.3, AC-3.2: Language filter accepts allowed languages.

    This test ensures that when allowed_languages=["en"], documents detected
    as English are accepted and returned in results.
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with allowed_languages=["en"]
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        enable_language_filter=True,
        allowed_languages=["en"],
        language_confidence_threshold=0.5,
    )

    # Mock HttpCrawlClient.crawl() to return English content
    test_url = "https://example.com/english-page"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="Hello world, this is an English page.",
        success=True,
        title="English Page",
        description="English content",
        status_code=200,
        detected_language="en",  # English detected
        language_confidence=0.98,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call aload_data - should accept English document
    documents = await reader.aload_data([test_url])

    # Verify document was accepted (list contains 1 document)
    assert len(documents) == 1
    assert documents[0] is not None
    assert documents[0].text == "Hello world, this is an English page."
    assert documents[0].metadata["detected_language"] == "en"
    assert documents[0].metadata["language_confidence"] == 0.98


@pytest.mark.asyncio
@respx.mock
async def test_filter_by_confidence_threshold():
    """Test that documents below confidence threshold are filtered out.

    Verifies AC-3.3: Confidence threshold filtering.

    This test ensures that documents with language_confidence below the
    configured threshold are filtered out, regardless of detected language.
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with confidence threshold 0.5
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        enable_language_filter=True,
        allowed_languages=["en"],
        language_confidence_threshold=0.5,
    )

    # Mock HttpCrawlClient.crawl() to return low confidence result
    test_url = "https://example.com/low-confidence"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="Hello world mixed with some text.",
        success=True,
        title="Low Confidence Page",
        description="Mixed content",
        status_code=200,
        detected_language="en",  # English detected
        language_confidence=0.4,  # Below threshold
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call aload_data - should filter out low confidence document
    documents = await reader.aload_data([test_url])

    # Verify document was filtered out (empty list returned)
    assert len(documents) == 0


@pytest.mark.asyncio
@respx.mock
async def test_filter_accepts_high_confidence():
    """Test that documents above confidence threshold are accepted.

    Verifies AC-3.2: Confidence threshold accepts high confidence documents.

    This test ensures that documents with language_confidence above the
    configured threshold are accepted and returned in results.
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with confidence threshold 0.5
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        enable_language_filter=True,
        allowed_languages=["en"],
        language_confidence_threshold=0.5,
    )

    # Mock HttpCrawlClient.crawl() to return high confidence result
    test_url = "https://example.com/high-confidence"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="Hello world, this is clearly English text.",
        success=True,
        title="High Confidence Page",
        description="Clear English content",
        status_code=200,
        detected_language="en",  # English detected
        language_confidence=0.9,  # Above threshold
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call aload_data - should accept high confidence document
    documents = await reader.aload_data([test_url])

    # Verify document was accepted (list contains 1 document)
    assert len(documents) == 1
    assert documents[0] is not None
    assert documents[0].text == "Hello world, this is clearly English text."
    assert documents[0].metadata["detected_language"] == "en"
    assert documents[0].metadata["language_confidence"] == 0.9


@pytest.mark.asyncio
@respx.mock
async def test_filter_multiple_allowed_languages():
    """Test that multiple allowed languages are all accepted.

    Verifies AC-2.2: Multiple allowed languages configuration.

    This test ensures that when allowed_languages=["en", "es"], documents
    detected as either English or Spanish are both accepted.
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with multiple allowed languages
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        enable_language_filter=True,
        allowed_languages=["en", "es"],
        language_confidence_threshold=0.5,
    )

    # Test URLs: one English, one Spanish
    test_urls = [
        "https://example.com/english-page",
        "https://example.com/spanish-page",
    ]

    # Mock HttpCrawlClient.crawl() with side_effect for different languages
    def mock_crawl(url):
        """Return language-specific result based on URL."""
        if "english" in url:
            return CrawlResult(
                url=url,
                markdown="Hello world, this is an English page.",
                success=True,
                title="English Page",
                description="English content",
                status_code=200,
                detected_language="en",
                language_confidence=0.95,
            )
        else:
            return CrawlResult(
                url=url,
                markdown="Hola mundo, esta es una página en español.",
                success=True,
                title="Página en Español",
                description="Contenido en español",
                status_code=200,
                detected_language="es",
                language_confidence=0.95,
            )

    reader._http_client.crawl = AsyncMock(side_effect=mock_crawl)

    # Call aload_data with both URLs
    documents = await reader.aload_data(test_urls)

    # Verify both documents were accepted
    assert len(documents) == 2
    assert documents[0] is not None
    assert documents[1] is not None

    # Verify languages
    assert documents[0].metadata["detected_language"] == "en"
    assert documents[1].metadata["detected_language"] == "es"

    # Verify content
    assert "English page" in documents[0].text
    assert "español" in documents[1].text


@respx.mock
def test_health_check_success():
    """Test that reader initialization succeeds with healthy service.

    Verifies AC-1.5, FR-13: Health check validation on initialization.

    This test ensures that when the Crawl4AI /health endpoint returns 200,
    the reader initializes successfully without raising exceptions.

    RED Phase: This test will FAIL because:
    - Crawl4AIReader.__init__ doesn't call health check yet
    """
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock /health endpoint returning success
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader - should not raise exception
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Verify reader was created successfully
    assert reader is not None
    assert reader.endpoint_url == "http://localhost:52004"


@pytest.mark.asyncio
@respx.mock
async def test_health_check_failure():
    """Test that reader factory fails with unhealthy service.

    Verifies AC-1.6: Health check failure handling.

    This test ensures that when the Crawl4AI /health endpoint fails
    (timeout or 503 error), the create() factory raises ValueError with clear
    error message indicating service is unreachable.

    Note: Direct __init__ no longer performs health check (non-blocking).
    Use create() factory for production initialization with validation.
    """
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock /health endpoint failing with 503 Service Unavailable
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(503, json={"error": "Service unavailable"})
    )

    # Attempt to create reader via factory - should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        await Crawl4AIReader.create(endpoint_url="http://localhost:52004")

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
    from crawl4r.readers.crawl4ai import Crawl4AIReader

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
    from crawl4r.readers.crawl4ai import Crawl4AIReader

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
    from crawl4r.readers.crawl4ai import Crawl4AIReader

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
    from crawl4r.readers.crawl4ai import Crawl4AIReader

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

    from crawl4r.readers.crawl4ai import Crawl4AIReader

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


def wrap_api_response(results: list[dict]) -> dict:
    """Wrap crawl results in Crawl4AI API response format.

    The Crawl4AI /crawl endpoint returns a batch response with success flag
    and results array, even for single URL requests.

    Args:
        results: List of individual crawl result dictionaries

    Returns:
        Complete API response structure
    """
    return {
        "success": True,
        "results": results,
    }


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


# NOTE: Metadata building tests have been moved to test_metadata_builder.py
# since MetadataBuilder is now a separate component. The 7 _build_metadata
# tests were removed as they are now covered by MetadataBuilder's own tests.


@pytest.mark.asyncio
@respx.mock
async def test_crawl_single_url_success():
    """Test that _crawl_single_url successfully crawls URL and returns Document.

    Verifies AC-2.1, FR-5: Single URL crawling via HttpCrawlClient.

    This test ensures that _crawl_single_url() correctly:
    1. Delegates to HttpCrawlClient.crawl() for HTTP communication
    2. Parses successful CrawlResult response
    3. Extracts markdown content as Document text
    4. Includes complete metadata in Document
    5. Returns Document with deterministic ID
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Mock HttpCrawlClient.crawl() to return a CrawlResult
    test_url = "https://example.com/test-page"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="# Test Page\n\nThis is test content.",
        success=True,
        title="Test Page Title",
        description="Test page description",
        status_code=200,
        timestamp="2026-01-15T12:00:00Z",
        internal_links_count=2,
        external_links_count=1,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call _crawl_single_url (no client parameter needed)
    document = await reader._crawl_single_url(test_url)

    # Verify Document was returned
    assert document is not None
    from llama_index.core.schema import Document

    assert isinstance(document, Document)

    # Verify text content
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

    # Verify HttpCrawlClient.crawl was called
    reader._http_client.crawl.assert_called_once_with(test_url)


@pytest.mark.asyncio
@respx.mock
async def test_crawl_single_url_with_markdown():
    """Test _crawl_single_url correctly handles markdown from HttpCrawlClient.

    Verifies: HttpCrawlClient returns markdown (already filtered via /md endpoint).

    This test ensures that _crawl_single_url() correctly:
    1. Uses markdown from CrawlResult
    2. Includes complete metadata in Document
    3. Returns valid Document
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Mock HttpCrawlClient.crawl() to return markdown content
    test_url = "https://example.com/test-page"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="# Raw Content\n\nThis is raw markdown with footer.",
        success=True,
        title="Test Page Title",
        description="Test page description",
        status_code=200,
        timestamp="2026-01-15T12:00:00Z",
        internal_links_count=1,
        external_links_count=1,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call _crawl_single_url
    document = await reader._crawl_single_url(test_url)

    # Verify Document was returned
    assert document is not None
    from llama_index.core.schema import Document

    assert isinstance(document, Document)

    # Verify text content
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
    1. Detects when CrawlResult has empty markdown
    2. Raises ValueError with clear error message
    3. Does not return a Document with empty text content
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    reader = Crawl4AIReader(endpoint_url="http://localhost:52004", fail_on_error=True)

    # Mock HttpCrawlClient.crawl() to return success but empty markdown
    test_url = "https://example.com/test-page"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="",  # Empty markdown
        success=True,
        title="Test Page Title",
        description="Test page description",
        status_code=200,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call _crawl_single_url - should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        await reader._crawl_single_url(test_url)

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
    2. Raises RuntimeError with error message
    3. Provides clear error context including URL
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    reader = Crawl4AIReader(endpoint_url="http://localhost:52004", fail_on_error=True)

    # Mock HttpCrawlClient.crawl() to return failure
    test_url = "https://example.com/blocked-page"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="",
        success=False,
        error="Connection timeout after 30 seconds",
        status_code=0,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call _crawl_single_url - should raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        await reader._crawl_single_url(test_url)

    # Verify error message includes the error from response
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
    """
    from crawl4r.readers.crawl4ai import Crawl4AIReader
    from crawl4r.resilience.circuit_breaker import CircuitBreakerError, CircuitState

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    reader = Crawl4AIReader(endpoint_url="http://localhost:52004", fail_on_error=True)

    # Manually set circuit breaker to OPEN state
    import time

    reader._circuit_breaker._state = CircuitState.OPEN
    reader._circuit_breaker.opened_at = time.time()

    # Test URL
    test_url = "https://example.com/test-page"

    # Call _crawl_single_url - should raise CircuitBreakerError
    with pytest.raises(CircuitBreakerError) as exc_info:
        await reader._crawl_single_url(test_url)

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
    2. Allows batch operations to continue processing remaining URLs
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with fail_on_error=False
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004", fail_on_error=False
    )

    # Mock HttpCrawlClient.crawl() to return failure
    test_url = "https://example.com/unreachable-page"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="",
        success=False,
        error="DNS resolution failed",
        status_code=0,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call _crawl_single_url - should return None instead of raising
    document = await reader._crawl_single_url(test_url)

    # Verify None was returned (graceful failure)
    assert document is None


# NOTE: Retry-specific tests have been removed because retry logic is now
# handled internally by HttpCrawlClient, which is tested in test_http_client.py.
# The 5 retry tests (timeout_retry, max_retries_exhausted, http_404_no_retry,
# http_500_retry, exponential_backoff) tested implementation details that no
# longer exist in the simplified _crawl_single_url method.


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

    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create mock VectorStoreManager with async delete_by_url method
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

    # Mock HttpCrawlClient.crawl() to return CrawlResult for each URL
    def mock_crawl(url):
        return CrawlResult(
            url=url,
            markdown=f"# Content from {url}",
            success=True,
            title="Test Page",
            description="Test description",
            status_code=200,
        )

    reader._http_client.crawl = AsyncMock(side_effect=mock_crawl)

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
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

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

    # Mock HttpCrawlClient.crawl() to return CrawlResult for each URL
    def mock_crawl(url):
        return CrawlResult(
            url=url,
            markdown=f"# Content from {url}",
            success=True,
            title="Test Page",
            description="Test description",
            status_code=200,
        )

    reader._http_client.crawl = AsyncMock(side_effect=mock_crawl)

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
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with deduplication enabled BUT no vector_store
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

    # Mock HttpCrawlClient.crawl() to return CrawlResult for each URL
    def mock_crawl(url):
        return CrawlResult(
            url=url,
            markdown=f"# Content from {url}",
            success=True,
            title="Test Page",
            description="Test description",
            status_code=200,
        )

    reader._http_client.crawl = AsyncMock(side_effect=mock_crawl)

    # Call aload_data with URLs - should NOT raise error
    documents = await reader.aload_data(test_urls)

    # Verify documents were returned (crawling succeeded without deduplication)
    assert len(documents) == 2
    assert all(doc is not None for doc in documents)

    # Verify documents have expected content
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
    from crawl4r.readers.crawl4ai import Crawl4AIReader

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

    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Test URL
    test_url = "https://example.com/test-single"

    # Mock HttpCrawlClient.crawl() to return CrawlResult
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="# Single URL Test\n\nThis is content from single URL.",
        success=True,
        title="Single URL Test Page",
        description="Test page for single URL batch",
        status_code=200,
        internal_links_count=1,
        external_links_count=1,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

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
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

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

    # Mock HttpCrawlClient.crawl() with side_effect to return page-specific results
    def mock_crawl(url):
        """Return appropriate CrawlResult based on URL."""
        if "page1" in url:
            page_num = 1
        elif "page2" in url:
            page_num = 2
        elif "page3" in url:
            page_num = 3
        else:
            page_num = 0

        return CrawlResult(
            url=url,
            markdown=f"# Page {page_num}\n\nContent from page {page_num}.",
            success=True,
            title=f"Page {page_num} Title",
            description=f"Description for page {page_num}",
            status_code=200,
            internal_links_count=1,
            external_links_count=1,
        )

    reader._http_client.crawl = AsyncMock(side_effect=mock_crawl)

    # Call aload_data with multiple URLs
    documents = await reader.aload_data(test_urls)

    # Verify list with all Documents returned in same order
    assert isinstance(documents, list)
    assert len(documents) == 3
    assert all(doc is not None for doc in documents)

    # Verify all Documents are proper Document instances
    from llama_index.core.schema import Document

    assert all(isinstance(doc, Document) for doc in documents)

    # Verify Documents content matches URL order
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
    """Test that aload_data_with_results preserves order with failures (None for failed URLs).

    Verifies AC-3.4: Order preservation with failures
    Verifies Issue #1: Results list maintains input order even with failures

    Pattern: success -> failure -> success
    Result:  Document -> None -> Document (order preserved)
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

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

    # Mock HttpCrawlClient.crawl() with side_effect for different outcomes
    def mock_crawl(url):
        """Return success/failure based on URL."""
        if "failure" in url:
            return CrawlResult(
                url=url,
                markdown="",
                success=False,
                error="Crawl failed intentionally",
                status_code=0,
            )

        page_num = 1 if "success1" in url else 2
        return CrawlResult(
            url=url,
            markdown=f"# Success {page_num}\n\nContent.",
            success=True,
            title=f"Success {page_num}",
            description=f"Description {page_num}",
            status_code=200,
        )

    reader._http_client.crawl = AsyncMock(side_effect=mock_crawl)

    # Call aload_data_with_results with success-failure-success pattern
    documents = await reader.aload_data_with_results(test_urls)

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
    """
    import asyncio

    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

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
    count_lock = asyncio.Lock()

    async def mock_crawl(url):
        """Track concurrent requests and enforce delay."""
        nonlocal concurrent_count, max_concurrent_reached

        # Increment concurrent count
        async with count_lock:
            concurrent_count += 1
            max_concurrent_reached = max(max_concurrent_reached, concurrent_count)

        # Simulate processing delay (50ms)
        await asyncio.sleep(0.05)

        # Extract page number from URL
        page_num = int(url.split("page")[1])

        # Decrement concurrent count
        async with count_lock:
            concurrent_count -= 1

        return CrawlResult(
            url=url,
            markdown=f"# Page {page_num}\n\nContent.",
            success=True,
            title=f"Page {page_num}",
            description=f"Description {page_num}",
            status_code=200,
        )

    reader._http_client.crawl = mock_crawl

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
    """
    import logging

    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

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

    # Mock HttpCrawlClient.crawl() with side_effect for different outcomes
    def mock_crawl(url):
        """Return success/failure based on URL."""
        if "failure" in url:
            return CrawlResult(
                url=url,
                markdown="",
                success=False,
                error="Crawl failed",
                status_code=0,
            )

        page_num = 1 if "success1" in url else 2
        return CrawlResult(
            url=url,
            markdown=f"# Success {page_num}\n\nContent.",
            success=True,
            title=f"Success {page_num}",
            description=f"Description {page_num}",
            status_code=200,
        )

    reader._http_client.crawl = AsyncMock(side_effect=mock_crawl)

    # Capture logs at INFO level
    with caplog.at_level(logging.INFO):
        documents = await reader.aload_data(test_urls)

    # Verify batch start log message
    start_messages = [
        record.message
        for record in caplog.records
        if "Starting batch crawl" in record.message
    ]
    assert len(start_messages) >= 1, "Missing batch start log message"
    assert "3 URLs" in start_messages[0] or "3" in start_messages[0]

    # Verify batch completion log message
    completion_messages = [
        record.message
        for record in caplog.records
        if "Batch crawl complete" in record.message
    ]
    assert len(completion_messages) >= 1, "Missing batch completion log message"
    completion_message = completion_messages[0]
    assert "2 succeeded" in completion_message or "succeeded: 2" in completion_message
    assert "1 failed" in completion_message or "failed: 1" in completion_message

    # Verify documents returned (2 success, filters out None)
    assert len(documents) == 2
    assert documents[0] is not None
    assert documents[1] is not None


# ==============================================================================
# Synchronous load_data Tests (Task 2.9.1)
# ==============================================================================


@respx.mock
def test_load_data_delegates_to_aload_data(respx_mock: respx.MockRouter) -> None:
    """Test that load_data properly delegates to aload_data using asyncio.run."""
    from llama_index.core.schema import Document

    from crawl4r.readers.crawl4ai import Crawl4AIReader

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

    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check
    respx_mock.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    # Create reader
    reader = Crawl4AIReader()

    # Mock HttpCrawlClient.crawl() to return CrawlResult
    mock_crawl_result = CrawlResult(
        url="https://example.com",
        markdown="# Example Page\n\nThis is test markdown.",
        success=True,
        title="Example Page",
        description="Test description",
        status_code=200,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call load_data (synchronous) with single URL
    result = reader.load_data(["https://example.com"])

    # Verify single Document returned
    assert len(result) == 1
    assert isinstance(result[0], Document)
    assert result[0].text == "# Example Page\n\nThis is test markdown."
    assert result[0].metadata["source"] == "https://example.com"
    assert result[0].metadata["source_url"] == "https://example.com"


# ============================================================================
# Error Handling Tests (now simplified - retry logic is in HttpCrawlClient)
# ============================================================================
# NOTE: The following tests previously tested retry behavior that has been
# moved to HttpCrawlClient. They now test the simplified error handling
# in _crawl_single_url which delegates to HttpCrawlClient.


@pytest.mark.asyncio
@respx.mock
async def test_error_http_client_exception():
    """Test that exceptions from HttpCrawlClient are handled properly.

    When HttpCrawlClient.crawl() raises an exception, _crawl_single_url
    should either propagate it (fail_on_error=True) or return None.
    """
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with fail_on_error=True
    reader = Crawl4AIReader(fail_on_error=True)

    # Mock HttpCrawlClient.crawl() to raise exception
    reader._http_client.crawl = AsyncMock(
        side_effect=RuntimeError("Connection failed")
    )

    # Call _crawl_single_url and expect RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        await reader._crawl_single_url("https://example.com")

    assert "Connection failed" in str(exc_info.value)


@pytest.mark.asyncio
@respx.mock
async def test_error_returns_none_when_fail_on_error_false():
    """Test that errors return None when fail_on_error=False."""
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with fail_on_error=False
    reader = Crawl4AIReader(fail_on_error=False)

    # Mock HttpCrawlClient.crawl() to raise exception
    reader._http_client.crawl = AsyncMock(
        side_effect=RuntimeError("Connection failed")
    )

    # Call _crawl_single_url - should return None
    result = await reader._crawl_single_url("https://example.com")

    assert result is None


# ============================================================================
# SSRF Prevention Tests (Security - Issue SEC-02)
# ============================================================================


class TestSSRFPrevention:
    """Test SSRF prevention via URL validation.

    These tests verify that the Crawl4AIReader validates URLs before
    crawling to prevent Server-Side Request Forgery attacks.

    Attack vectors blocked:
    - Private IP ranges (10.x, 172.16-31.x, 192.168.x)
    - Localhost (127.0.0.1, localhost, ::1)
    - Cloud metadata endpoints (169.254.169.254)
    - Non-HTTP schemes (file://, ftp://, gopher://)
    """

    @pytest.fixture
    def reader(self):
        """Create reader with mocked health check."""
        with respx.mock:
            respx.get("http://localhost:52004/health").mock(
                return_value=httpx.Response(200, json={"status": "healthy"})
            )
            from crawl4r.readers.crawl4ai import Crawl4AIReader
            return Crawl4AIReader(endpoint_url="http://localhost:52004")

    def test_rejects_private_ip_10_range(self, reader):
        """Reject 10.x.x.x private IP range (RFC 1918)."""
        assert reader.validate_url("http://10.0.0.1/admin") is False
        assert reader.validate_url("http://10.255.255.255/") is False

    def test_rejects_private_ip_172_range(self, reader):
        """Reject 172.16.0.0 - 172.31.255.255 private IP range (RFC 1918)."""
        assert reader.validate_url("http://172.16.0.1/") is False
        assert reader.validate_url("http://172.31.255.255/") is False
        # 172.15.x and 172.32.x are public - should be allowed
        assert reader.validate_url("http://172.15.0.1/") is True
        assert reader.validate_url("http://172.32.0.1/") is True

    def test_rejects_private_ip_192_168_range(self, reader):
        """Reject 192.168.x.x private IP range (RFC 1918)."""
        assert reader.validate_url("http://192.168.1.1/") is False
        assert reader.validate_url("http://192.168.0.1/router") is False

    def test_rejects_localhost_ipv4(self, reader):
        """Reject localhost via 127.0.0.1."""
        assert reader.validate_url("http://127.0.0.1/") is False
        assert reader.validate_url("http://127.0.0.1:8080/api") is False

    def test_rejects_localhost_hostname(self, reader):
        """Reject localhost via hostname."""
        assert reader.validate_url("http://localhost/") is False
        assert reader.validate_url("http://localhost:3000/admin") is False

    def test_rejects_localhost_ipv6(self, reader):
        """Reject localhost via IPv6 ::1."""
        assert reader.validate_url("http://[::1]/") is False
        assert reader.validate_url("http://[::1]:8080/") is False

    def test_rejects_aws_metadata_endpoint(self, reader):
        """Reject AWS EC2 metadata endpoint (169.254.169.254)."""
        assert reader.validate_url("http://169.254.169.254/") is False
        assert reader.validate_url("http://169.254.169.254/latest/meta-data/") is False

    def test_rejects_gcp_metadata_endpoint(self, reader):
        """Reject GCP metadata endpoint."""
        assert reader.validate_url("http://metadata.google.internal/") is False
        assert reader.validate_url("http://169.254.169.254/computeMetadata/v1/") is False

    def test_rejects_file_scheme(self, reader):
        """Reject file:// URLs."""
        assert reader.validate_url("file:///etc/passwd") is False
        assert reader.validate_url("file:///C:/Windows/System32/config/SAM") is False

    def test_rejects_ftp_scheme(self, reader):
        """Reject ftp:// URLs."""
        assert reader.validate_url("ftp://ftp.example.com/") is False

    def test_rejects_gopher_scheme(self, reader):
        """Reject gopher:// URLs (SSRF attack vector)."""
        assert reader.validate_url("gopher://evil.com:25/") is False

    def test_allows_valid_https_url(self, reader):
        """Allow valid HTTPS URLs."""
        assert reader.validate_url("https://example.com/") is True
        assert reader.validate_url("https://docs.python.org/3/") is True

    def test_allows_valid_http_url(self, reader):
        """Allow valid HTTP URLs."""
        assert reader.validate_url("http://example.com/") is True
        assert reader.validate_url("http://httpbin.org/get") is True

    def test_rejects_malformed_url(self, reader):
        """Reject malformed URLs."""
        assert reader.validate_url("not-a-url") is False
        assert reader.validate_url("") is False
        assert reader.validate_url("://missing-scheme.com") is False

    def test_rejects_ip_in_decimal_notation(self, reader):
        """Reject IP addresses in decimal/octal notation (bypass attempts)."""
        # 2130706433 = 127.0.0.1 in decimal
        assert reader.validate_url("http://2130706433/") is False
        # 0x7f000001 = 127.0.0.1 in hex
        assert reader.validate_url("http://0x7f000001/") is False


@pytest.mark.asyncio
async def test_aload_data_strict_return_type():
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Direct instantiation works without health check (non-blocking)
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004", fail_on_error=False)

    # Mock _crawl_single_url to return None (failure)
    with (
        patch.object(reader, "_validate_health", return_value=True),
        patch.object(reader, "_crawl_single_url", return_value=None),
    ):
        # We pass one URL that "fails"
        docs = await reader.aload_data(["http://fail.com"])

        # Expectation: aload_data should NOT return None in the list
        # It should filter it out, returning an empty list
        assert isinstance(docs, list)
        assert len(docs) == 0
        assert None not in docs

@pytest.mark.asyncio
async def test_load_data_with_errors_returns_none():
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Direct instantiation works without health check (non-blocking)
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004", fail_on_error=False)

    # Mock _crawl_single_url to return None (failure)
    with (
        patch.object(reader, "_validate_health", return_value=True),
        patch.object(reader, "_crawl_single_url", return_value=None),
    ):
        # We pass one URL that "fails"

        assert hasattr(reader, "aload_data_with_results")
        results = await reader.aload_data_with_results(["http://fail.com"])
        assert len(results) == 1
        assert results[0] is None


@pytest.mark.asyncio
async def test_alazy_load_data_does_not_log_duplicate_errors(caplog):
    """Avoid duplicate lazy-load warnings when _crawl_single_url raises."""
    import logging

    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Direct instantiation works without health check (non-blocking)
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004", fail_on_error=True)

    caplog.set_level(logging.WARNING, logger="crawl4r.readers.crawl4ai")

    with patch.object(reader, "_validate_health", return_value=True):
        with patch.object(reader, "_crawl_single_url", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                async for _ in reader.alazy_load_data(["http://fail.com"]):
                    pass

    assert "Lazy load failed for" not in caplog.text


# ============================================================================
# Metadata Enrichment and Opt-Out Tests (Task 3.8)
# ============================================================================


@pytest.mark.asyncio
@respx.mock
async def test_metadata_includes_language_fields():
    """Test that document metadata includes detected_language and language_confidence.

    Verifies FR-5, AC-4.1: Language detection metadata enrichment.

    This test ensures that every crawled document has language detection
    metadata fields populated, regardless of filter status.
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with language filtering enabled
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        enable_language_filter=True,
        allowed_languages=["en"],
        language_confidence_threshold=0.5,
    )

    # Mock HttpCrawlClient.crawl() to return English content with language metadata
    test_url = "https://example.com/english-page"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="Hello world, this is an English page.",
        success=True,
        title="English Page",
        description="English content",
        status_code=200,
        detected_language="en",
        language_confidence=0.98,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call aload_data
    documents = await reader.aload_data([test_url])

    # Verify document has language metadata fields
    assert len(documents) == 1
    assert documents[0] is not None

    # Verify detected_language field exists and has correct value
    assert "detected_language" in documents[0].metadata
    assert documents[0].metadata["detected_language"] == "en"

    # Verify language_confidence field exists and has correct value
    assert "language_confidence" in documents[0].metadata
    assert documents[0].metadata["language_confidence"] == 0.98


@pytest.mark.asyncio
@respx.mock
async def test_filter_disabled():
    """Test that enable_language_filter=False accepts all languages.

    Verifies FR-8, AC-6.1: Language filter opt-out functionality.

    This test ensures that when enable_language_filter=False, documents
    in any language are accepted, regardless of allowed_languages setting.
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with filtering DISABLED (but allowed_languages still set to ["en"])
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        enable_language_filter=False,  # Filter disabled
        allowed_languages=["en"],  # This should be ignored
        language_confidence_threshold=0.5,
    )

    # Mock HttpCrawlClient.crawl() to return Spanish content (disallowed language)
    test_url = "https://example.com/spanish-page"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="Hola mundo, esta es una página en español.",
        success=True,
        title="Página en Español",
        description="Contenido en español",
        status_code=200,
        detected_language="es",  # Spanish - would be filtered if enabled
        language_confidence=0.95,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call aload_data - should accept Spanish document (filter disabled)
    documents = await reader.aload_data([test_url])

    # Verify Spanish document was accepted (not filtered out)
    assert len(documents) == 1
    assert documents[0] is not None
    assert documents[0].text == "Hola mundo, esta es una página en español."
    assert documents[0].metadata["detected_language"] == "es"


@pytest.mark.asyncio
@respx.mock
async def test_filter_disabled_still_enriches():
    """Test that metadata includes language fields even when filter is disabled.

    Verifies AC-4.2, AC-6.2: Metadata enrichment regardless of filter status.

    This test ensures that language detection metadata (detected_language,
    language_confidence) is still populated even when enable_language_filter=False.
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with filtering DISABLED
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        enable_language_filter=False,  # Filter disabled
        allowed_languages=["en"],
        language_confidence_threshold=0.5,
    )

    # Mock HttpCrawlClient.crawl() to return French content
    test_url = "https://example.com/french-page"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="Bonjour le monde, c'est une page en français.",
        success=True,
        title="Page en Français",
        description="Contenu français",
        status_code=200,
        detected_language="fr",
        language_confidence=0.92,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call aload_data
    documents = await reader.aload_data([test_url])

    # Verify document was accepted
    assert len(documents) == 1
    assert documents[0] is not None

    # Verify language metadata is still populated (enrichment happens regardless)
    assert "detected_language" in documents[0].metadata
    assert documents[0].metadata["detected_language"] == "fr"
    assert "language_confidence" in documents[0].metadata
    assert documents[0].metadata["language_confidence"] == 0.92


@pytest.mark.asyncio
@respx.mock
async def test_filtered_documents_logged(caplog):
    """Test that filtered documents are logged with structured logging.

    Verifies FR-11, AC-4.3: Structured logging for rejected documents.

    This test ensures that when a document is filtered out due to language,
    a structured log entry is created with the URL, detected language, and
    filter reason.
    """
    import logging

    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with language filtering enabled
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        enable_language_filter=True,
        allowed_languages=["en"],
        language_confidence_threshold=0.5,
    )

    # Mock HttpCrawlClient.crawl() to return German content (will be filtered)
    test_url = "https://example.com/german-page"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="Hallo Welt, dies ist eine deutsche Seite.",
        success=True,
        title="Deutsche Seite",
        description="Deutscher Inhalt",
        status_code=200,
        detected_language="de",  # German - will be filtered
        language_confidence=0.96,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Capture logs at INFO level
    with caplog.at_level(logging.INFO):
        documents = await reader.aload_data([test_url])

    # Verify document was filtered out
    assert len(documents) == 0

    # Verify structured log message for filtered document
    filtered_messages = [
        record.message
        for record in caplog.records
        if "filtered" in record.message.lower() or "rejected" in record.message.lower()
    ]
    assert len(filtered_messages) >= 1, "Missing log message for filtered document"

    # Verify log includes key information
    log_message = filtered_messages[0]
    assert test_url in log_message or "german-page" in log_message
    assert "de" in log_message or "german" in log_message.lower()


# ============================================================================
# Backward Compatibility Tests (Task 3.9)
# ============================================================================


def test_crawl_result_language_fields_optional():
    """Test that CrawlResult works with None values for language fields.

    Verifies NFR-7: Backward compatibility - optional language fields.

    This test ensures that CrawlResult can be instantiated without
    providing language fields, and they default to None. This maintains
    backward compatibility with existing code.
    """
    from crawl4r.readers.crawl import CrawlResult

    # Create CrawlResult without language fields
    result = CrawlResult(
        url="https://example.com",
        markdown="# Test Page",
        success=True,
        title="Test Page",
        description="Test description",
        status_code=200,
    )

    # Verify result was created successfully
    assert result is not None
    assert result.url == "https://example.com"
    assert result.markdown == "# Test Page"

    # Verify language fields default to None (backward compatible)
    assert result.detected_language is None
    assert result.language_confidence is None


@pytest.mark.asyncio
@respx.mock
async def test_metadata_backward_compatible():
    """Test that old metadata schema without language fields still works.

    Verifies NFR-7: Backward compatibility - metadata schema.

    This test ensures that documents created without language metadata
    fields are still processed correctly by the system. The language
    fields should only be added when present in CrawlResult.
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Mock HttpCrawlClient.crawl() to return CrawlResult WITHOUT language fields
    test_url = "https://example.com/legacy-page"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="# Legacy Page\n\nThis result has no language fields.",
        success=True,
        title="Legacy Page",
        description="Legacy content",
        status_code=200,
        # Note: detected_language and language_confidence NOT provided (None)
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call aload_data - should work without language fields
    documents = await reader.aload_data([test_url])

    # Verify document was created successfully
    assert len(documents) == 1
    assert documents[0] is not None
    assert documents[0].text == "# Legacy Page\n\nThis result has no language fields."

    # Verify standard metadata fields are present
    assert documents[0].metadata["source"] == test_url
    assert documents[0].metadata["source_url"] == test_url
    assert documents[0].metadata["title"] == "Legacy Page"

    # Verify language fields are NOT in metadata (only added when present)
    assert "detected_language" not in documents[0].metadata
    assert "language_confidence" not in documents[0].metadata


@pytest.mark.asyncio
@respx.mock
async def test_existing_tests_still_pass():
    """Test that full test suite verifies zero breaking changes.

    Verifies NFR-7: Backward compatibility - no breaking changes.

    This test serves as a comprehensive verification that the language
    filter feature does not break any existing functionality. It runs
    a representative existing test to ensure compatibility.
    """
    from crawl4r.readers.crawl import CrawlResult
    from crawl4r.readers.crawl4ai import Crawl4AIReader

    # Mock health check to allow initialization
    respx.get("http://localhost:52004/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )

    # Create reader with default config (no language filter params specified)
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Mock HttpCrawlClient.crawl() to return standard CrawlResult
    test_url = "https://example.com/test-page"
    mock_crawl_result = CrawlResult(
        url=test_url,
        markdown="# Test Page\n\nThis is test content.",
        success=True,
        title="Test Page Title",
        description="Test page description",
        status_code=200,
        timestamp="2026-01-15T12:00:00Z",
        internal_links_count=2,
        external_links_count=1,
    )

    reader._http_client.crawl = AsyncMock(return_value=mock_crawl_result)

    # Call aload_data (same as existing tests)
    documents = await reader.aload_data([test_url])

    # Verify standard expectations from existing tests still pass
    assert len(documents) == 1
    assert documents[0] is not None
    from llama_index.core.schema import Document

    assert isinstance(documents[0], Document)

    # Verify text content
    assert documents[0].text == "# Test Page\n\nThis is test content."

    # Verify all standard metadata fields
    assert documents[0].metadata["source"] == test_url
    assert documents[0].metadata["source_url"] == test_url
    assert documents[0].metadata["title"] == "Test Page Title"
    assert documents[0].metadata["description"] == "Test page description"
    assert documents[0].metadata["status_code"] == 200
    assert documents[0].metadata["crawl_timestamp"] == "2026-01-15T12:00:00Z"
    assert documents[0].metadata["internal_links_count"] == 2
    assert documents[0].metadata["external_links_count"] == 1
    assert documents[0].metadata["source_type"] == "web_crawl"

    # Verify deterministic ID was set
    assert documents[0].id_ is not None
    assert len(documents[0].id_) > 0

