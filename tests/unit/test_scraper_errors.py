"""Unit tests for ScraperService error handling."""

from __future__ import annotations

import httpx
import pytest

from crawl4r.services.scraper import ScraperService


@pytest.mark.asyncio
async def test_scraper_returns_clear_error_on_network_failure() -> None:
    """ScraperService should return clear network error message on connection failure."""
    service = ScraperService(endpoint_url="http://localhost:52004")

    # Mock the client to raise a network error
    async def _raise_network_error(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("Connection refused")

    service._client.post = _raise_network_error  # type: ignore[assignment]

    result = await service.scrape_url("https://example.com")

    assert result.success is False
    assert result.error is not None
    assert "network" in result.error.lower() or "connection" in result.error.lower()


@pytest.mark.asyncio
async def test_scraper_returns_clear_error_on_timeout() -> None:
    """ScraperService should return clear timeout error message."""
    service = ScraperService(endpoint_url="http://localhost:52004", timeout=0.001)

    # Mock the client to raise a timeout error
    async def _raise_timeout(*args: object, **kwargs: object) -> None:
        raise httpx.TimeoutException("Request timed out")

    service._client.post = _raise_timeout  # type: ignore[assignment]

    result = await service.scrape_url("https://example.com")

    assert result.success is False
    assert result.error is not None
    assert "timeout" in result.error.lower() or "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_scraper_returns_clear_error_on_dns_failure() -> None:
    """ScraperService should return clear DNS error message."""
    service = ScraperService(endpoint_url="http://localhost:52004")

    # Mock the client to raise a DNS error
    async def _raise_dns_error(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("Name or service not known")

    service._client.post = _raise_dns_error  # type: ignore[assignment]

    result = await service.scrape_url("https://example.com")

    assert result.success is False
    assert result.error is not None
    # Should have actionable error message
    assert len(result.error) > 10  # More than just raw exception


@pytest.mark.asyncio
async def test_scraper_includes_url_in_error() -> None:
    """ScraperService error messages should include the failing URL for debugging."""
    service = ScraperService(endpoint_url="http://localhost:52004")

    # Mock the client to raise an error
    async def _raise_error(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("Connection refused")

    service._client.post = _raise_error  # type: ignore[assignment]

    result = await service.scrape_url("https://example.com")

    assert result.success is False
    assert result.url == "https://example.com"
    # URL should be accessible in result for logging
