"""Tests for Crawl4AIReader async initialization pattern."""

import pytest
from unittest.mock import AsyncMock, patch

from crawl4r.readers.crawl4ai import Crawl4AIReader


@pytest.mark.asyncio
async def test_create_async_validates_health_without_blocking() -> None:
    """Verify create() factory performs async health check.

    Ensures:
    - Health check uses AsyncClient (non-blocking)
    - Raises ValueError if service unreachable
    """
    with patch("crawl4r.readers.crawl4ai.httpx.AsyncClient") as mock_async:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = AsyncMock(
            return_value=AsyncMock(status_code=200)
        )
        mock_async.return_value = mock_client

        reader = await Crawl4AIReader.create(endpoint_url="http://localhost:52004")

        assert reader is not None
        assert reader.endpoint_url == "http://localhost:52004"


@pytest.mark.asyncio
async def test_create_async_raises_on_unhealthy_service() -> None:
    """Verify create() raises if service unreachable.

    Ensures:
    - Timeout errors raise ValueError
    - Error message includes endpoint URL
    """
    with patch("crawl4r.readers.crawl4ai.httpx.AsyncClient") as mock_async:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        mock_async.return_value = mock_client

        with pytest.raises(ValueError, match="Crawl4AI service unreachable"):
            await Crawl4AIReader.create(endpoint_url="http://localhost:52004")


def test_init_skips_health_check() -> None:
    """Verify __init__ no longer performs blocking health check.

    Ensures:
    - Direct instantiation works without service running
    - Health validation deferred to create() factory
    """
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")
    assert reader.endpoint_url == "http://localhost:52004"
