"""Tests for crawl4r.readers.crawl.http_client module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crawl4r.readers.crawl.http_client import HttpCrawlClient


def _create_mock_response(
    status_code: int = 200, json_data: dict | None = None
) -> MagicMock:
    """Create a mock HTTP response with synchronous json() method."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data or {}
    return mock_resp


@pytest.mark.asyncio
async def test_crawl_returns_success_result() -> None:
    """Verify client returns CrawlResult on successful crawl."""
    with patch("crawl4r.readers.crawl.http_client.httpx.AsyncClient") as mock_client:
        mock_resp = _create_mock_response(
            status_code=200,
            json_data={
                "markdown": "# Test",
                "title": "Test Page",
                "description": "A test",
            },
        )

        mock_ctx = AsyncMock()
        mock_ctx.post = AsyncMock(return_value=mock_resp)
        mock_client.return_value.__aenter__.return_value = mock_ctx

        client = HttpCrawlClient(endpoint_url="http://localhost:52004")
        result = await client.crawl("https://example.com")

        assert result.success is True
        assert result.url == "https://example.com"
        assert result.markdown == "# Test"
        assert result.title == "Test Page"


@pytest.mark.asyncio
async def test_crawl_returns_failure_on_timeout() -> None:
    """Verify client returns failure CrawlResult on timeout."""
    with patch("crawl4r.readers.crawl.http_client.httpx.AsyncClient") as mock_client:
        mock_ctx = AsyncMock()
        mock_ctx.post = AsyncMock(side_effect=Exception("Timeout"))
        mock_client.return_value.__aenter__.return_value = mock_ctx

        client = HttpCrawlClient(endpoint_url="http://localhost:52004")
        result = await client.crawl("https://example.com")

        assert result.success is False
        assert "Timeout" in result.error


@pytest.mark.asyncio
async def test_crawl_returns_failure_on_http_error() -> None:
    """Verify client returns failure CrawlResult on HTTP error status."""
    with patch("crawl4r.readers.crawl.http_client.httpx.AsyncClient") as mock_client:
        mock_resp = _create_mock_response(status_code=500)

        mock_ctx = AsyncMock()
        mock_ctx.post = AsyncMock(return_value=mock_resp)
        mock_client.return_value.__aenter__.return_value = mock_ctx

        client = HttpCrawlClient(endpoint_url="http://localhost:52004")
        result = await client.crawl("https://example.com")

        assert result.success is False
        assert result.status_code == 500
        assert "HTTP 500" in result.error


@pytest.mark.asyncio
async def test_crawl_handles_empty_markdown() -> None:
    """Verify client handles response with empty markdown."""
    with patch("crawl4r.readers.crawl.http_client.httpx.AsyncClient") as mock_client:
        mock_resp = _create_mock_response(
            status_code=200,
            json_data={"markdown": "", "title": "Empty Page"},
        )

        mock_ctx = AsyncMock()
        mock_ctx.post = AsyncMock(return_value=mock_resp)
        mock_client.return_value.__aenter__.return_value = mock_ctx

        client = HttpCrawlClient(endpoint_url="http://localhost:52004")
        result = await client.crawl("https://example.com")

        assert result.success is True
        assert result.markdown == ""
        assert result.title == "Empty Page"


@pytest.mark.asyncio
async def test_crawl_uses_correct_endpoint() -> None:
    """Verify client makes request to /md endpoint with fit filter."""
    with patch("crawl4r.readers.crawl.http_client.httpx.AsyncClient") as mock_client:
        mock_resp = _create_mock_response(
            status_code=200,
            json_data={"markdown": "# Test"},
        )

        mock_post = AsyncMock(return_value=mock_resp)
        mock_ctx = AsyncMock()
        mock_ctx.post = mock_post
        mock_client.return_value.__aenter__.return_value = mock_ctx

        client = HttpCrawlClient(endpoint_url="http://localhost:52004")
        await client.crawl("https://example.com")

        # Verify correct endpoint and payload
        mock_post.assert_called_once_with(
            "http://localhost:52004/md",
            json={"url": "https://example.com", "f": "fit"},
        )


@pytest.mark.asyncio
async def test_crawl_strips_trailing_slash_from_endpoint() -> None:
    """Verify client strips trailing slash from endpoint URL."""
    with patch("crawl4r.readers.crawl.http_client.httpx.AsyncClient") as mock_client:
        mock_resp = _create_mock_response(
            status_code=200,
            json_data={"markdown": "# Test"},
        )

        mock_post = AsyncMock(return_value=mock_resp)
        mock_ctx = AsyncMock()
        mock_ctx.post = mock_post
        mock_client.return_value.__aenter__.return_value = mock_ctx

        client = HttpCrawlClient(endpoint_url="http://localhost:52004/")
        await client.crawl("https://example.com")

        # Verify no double slash
        mock_post.assert_called_once_with(
            "http://localhost:52004/md",
            json={"url": "https://example.com", "f": "fit"},
        )


def test_client_initialization() -> None:
    """Verify client initializes with correct defaults."""
    client = HttpCrawlClient(endpoint_url="http://localhost:52004")

    assert client.endpoint_url == "http://localhost:52004"
    assert client.timeout == 60.0
    assert client.max_retries == 3


def test_client_custom_configuration() -> None:
    """Verify client accepts custom configuration."""
    client = HttpCrawlClient(
        endpoint_url="http://custom:8080",
        timeout=120.0,
        max_retries=5,
    )

    assert client.endpoint_url == "http://custom:8080"
    assert client.timeout == 120.0
    assert client.max_retries == 5
