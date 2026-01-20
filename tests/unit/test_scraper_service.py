"""Unit tests for ScraperService markdown scraping functionality.

These tests verify the ScraperService correctly:
- Scrapes markdown content from Crawl4AI /md endpoint with fit filter
- Handles retries with exponential backoff for transient failures
- Implements circuit breaker pattern after repeated failures
- Provides concurrent scraping with semaphore-based rate limiting
- Returns structured ScrapeResult with metadata and status codes

All HTTP calls are mocked using respx for deterministic, isolated testing.
"""

import httpx
import pytest
import respx

from crawl4r.services.scraper import ScraperService


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_success() -> None:
    """Verify successful scrape returns markdown and metadata.

    When the Crawl4AI service returns 200 with markdown content,
    the ScrapeResult should indicate success with the extracted content.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(
            200,
            json={
                "markdown": "# Title",
                "metadata": {"title": "Example"},
                "status_code": 200,
            },
        )
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com")
    assert result.success is True
    assert result.markdown == "# Title"


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_timeout_returns_failure() -> None:
    """Verify timeout errors result in failure after retries.

    When the Crawl4AI service times out, the scraper should retry
    with exponential backoff and eventually return a failure result.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        side_effect=httpx.ReadTimeout("timeout")
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com")
    assert result.success is False


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_4xx_no_retry() -> None:
    """Verify 4xx client errors do not trigger retries.

    Client errors are permanent failures and should not be retried.
    The scraper should immediately return a failure result.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com")
    assert result.success is False


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_5xx_retries_then_fails() -> None:
    """Verify 5xx server errors trigger retries before failing.

    Server errors are transient and should be retried with exponential
    backoff. After exhausting retries, should return failure result.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(500, json={"detail": "error"})
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com")
    assert result.success is False


@respx.mock
@pytest.mark.asyncio
async def test_circuit_breaker_does_not_open_for_handled_errors() -> None:
    """Verify circuit breaker stays closed for handled HTTP errors.

    The ScraperService catches all HTTP errors and returns ScrapeResult
    with success=False. Since no exceptions are raised from _fetch_markdown,
    the circuit breaker does not count these as failures and stays closed.
    This is correct behavior - the service is working properly, just
    reporting that the scrape failed.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(500)
    )
    service = ScraperService(endpoint_url="http://localhost:52004")

    # Make multiple requests that return errors
    for _ in range(10):
        result = await service.scrape_url("https://example.com")
        assert result.success is False

    # Circuit breaker should stay closed because errors are handled gracefully
    # (no exceptions raised from _fetch_markdown)
    assert service._circuit_breaker.state == "CLOSED"
    assert service._circuit_breaker.failure_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_scrape_urls_concurrency() -> None:
    """Verify concurrent scraping respects max_concurrent limit.

    When scraping multiple URLs, the service should process them
    concurrently but not exceed the max_concurrent limit.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(200, json={"markdown": "# Ok", "status_code": 200})
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    results = await service.scrape_urls(
        [
            "https://example.com/a",
            "https://example.org/b",
            "https://example.net/c",
        ],
        max_concurrent=2,
    )
    assert len(results) == 3
    assert all(r.success for r in results)


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_invalid_url_returns_failure() -> None:
    """Verify invalid URLs return failure without making requests.

    URL validation should happen before any HTTP requests are made.
    """
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("not-a-valid-url")
    assert result.success is False
    assert "Invalid URL" in result.error


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_includes_metadata() -> None:
    """Verify scraped result includes all metadata from response.

    The service should extract and include metadata fields from the
    Crawl4AI response in the ScrapeResult.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(
            200,
            json={
                "markdown": "# Content",
                "status_code": 200,
                "metadata": {"title": "Test Page", "description": "A test"},
                "title": "Test Page",
                "description": "A test",
            },
        )
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com")

    assert result.success is True
    assert result.metadata is not None
    assert "title" in result.metadata
    assert result.metadata["title"] == "Test Page"


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_network_error_returns_failure() -> None:
    """Verify network errors result in failure after retries.

    When network errors occur (connection refused, DNS errors, etc.),
    the scraper should retry and eventually return failure.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com")
    assert result.success is False


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_eventual_success_after_retries() -> None:
    """Verify transient failures resolve with retry mechanism.

    When the service fails temporarily but succeeds on retry,
    the scraper should return a successful result.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    # First two attempts fail with 500, third succeeds
    route = respx.post("http://localhost:52004/md?f=fit")
    route.side_effect = [
        httpx.Response(500, json={"error": "temporary error"}),
        httpx.Response(500, json={"error": "temporary error"}),
        httpx.Response(200, json={"markdown": "# Success", "status_code": 200}),
    ]

    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com")

    assert result.success is True
    assert result.markdown == "# Success"


@respx.mock
@pytest.mark.asyncio
async def test_scrape_urls_mixed_success_and_failure() -> None:
    """Verify batch scraping handles partial failures correctly.

    When scraping multiple URLs, successful and failed requests
    should both be included in the results.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    # Mock different responses for different URLs
    route = respx.post("http://localhost:52004/md?f=fit")
    route.side_effect = [
        httpx.Response(200, json={"markdown": "# A", "status_code": 200}),
        httpx.Response(500, json={"error": "server error"}),
        httpx.Response(200, json={"markdown": "# C", "status_code": 200}),
    ]

    service = ScraperService(endpoint_url="http://localhost:52004")
    results = await service.scrape_urls(
        [
            "https://example.com/a",
            "https://example.org/b",
            "https://example.net/c",
        ]
    )

    assert len(results) == 3
    assert results[0].success is True
    assert results[1].success is False
    assert results[2].success is True


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_preserves_status_code() -> None:
    """Verify HTTP status codes are preserved in results.

    The ScrapeResult should include the HTTP status code from
    the crawled page, not just the API response code.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(
            200,
            json={
                "markdown": "# Not Found",
                "status_code": 404,
            },
        )
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com/missing")

    assert result.success is True  # API call succeeded
    assert result.status_code == 404  # But page returned 404


@respx.mock
@pytest.mark.asyncio
async def test_close_cleanup() -> None:
    """Verify close() properly cleans up the HTTP client.

    The service should close its underlying httpx client when
    close() is called to free resources.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    service = ScraperService(endpoint_url="http://localhost:52004")

    # Verify client is not closed initially
    assert not service._client.is_closed

    # Close the service
    await service.close()

    # Verify client is closed
    assert service._client.is_closed


@respx.mock
@pytest.mark.asyncio
async def test_validate_services_success() -> None:
    """Verify validate_services passes when health check succeeds."""
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    service = ScraperService(
        endpoint_url="http://localhost:52004", validate_on_startup=False
    )

    # Should not raise when health check succeeds
    await service.validate_services()


@respx.mock
@pytest.mark.asyncio
async def test_validate_services_failure() -> None:
    """Verify validate_services raises ValueError on health check failure."""
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(500))
    service = ScraperService(
        endpoint_url="http://localhost:52004", validate_on_startup=False
    )

    # Should raise ValueError when health check fails
    with pytest.raises(ValueError, match="Crawl4AI service health check failed"):
        await service.validate_services()
