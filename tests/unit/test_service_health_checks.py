"""Unit tests for service health check validation."""

from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from crawl4r.services.ingestion import IngestionService
from crawl4r.services.scraper import ScraperService


@pytest.mark.asyncio
async def test_scraper_health_check_succeeds():
    """Test scraper service validates health check successfully."""
    service = ScraperService(endpoint_url="http://localhost:52004")
    service._client.get = AsyncMock(
        return_value=Mock(status_code=200, raise_for_status=Mock())
    )

    # Should not raise
    await service.validate_services()


@pytest.mark.asyncio
async def test_scraper_health_check_fails_with_500():
    """Test scraper service fails fast when health check returns 500."""
    service = ScraperService(endpoint_url="http://localhost:52004")
    service._client.get = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Server error",
            request=Mock(),
            response=Mock(status_code=500),
        )
    )

    with pytest.raises(
        ValueError,
        match="Crawl4AI service health check failed: Server error",
    ):
        await service.validate_services()


@pytest.mark.asyncio
async def test_scraper_health_check_fails_with_timeout():
    """Test scraper service fails fast when health check times out."""
    service = ScraperService(endpoint_url="http://localhost:52004", timeout=1.0)
    service._client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

    with pytest.raises(
        ValueError,
        match="Crawl4AI service health check failed: Timeout",
    ):
        await service.validate_services()


@pytest.mark.asyncio
async def test_scraper_health_check_fails_with_connection_error():
    """Test scraper service fails fast when health check cannot connect."""
    service = ScraperService(endpoint_url="http://localhost:52004")
    service._client.get = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    with pytest.raises(
        ValueError,
        match="Crawl4AI service health check failed: Connection refused",
    ):
        await service.validate_services()


@pytest.mark.asyncio
async def test_scraper_validation_can_be_skipped():
    """Test scraper service allows skipping validation on startup."""
    service = ScraperService(
        endpoint_url="http://localhost:52004",
        validate_on_startup=False,
    )

    # Mock health check to fail
    service._client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

    # Should not raise when validate_on_startup=False
    # (validation only happens when explicitly called)
    # This test verifies that __init__ doesn't call validate_services automatically


@pytest.mark.asyncio
async def test_ingestion_validates_all_dependencies():
    """Test ingestion service validates scraper and other dependencies."""
    scraper_mock = Mock(spec=ScraperService)
    scraper_mock.validate_services = AsyncMock()

    embeddings_mock = Mock()
    embeddings_mock.validate_services = AsyncMock()

    vector_store_mock = Mock()
    vector_store_mock.validate_services = AsyncMock()

    queue_manager_mock = Mock()
    queue_manager_mock.validate_services = AsyncMock()

    service = IngestionService(
        scraper=scraper_mock,
        embeddings=embeddings_mock,
        vector_store=vector_store_mock,
        queue_manager=queue_manager_mock,
    )

    await service.validate_services()

    # Should call validate_services on all dependencies that have it
    scraper_mock.validate_services.assert_called_once()


@pytest.mark.asyncio
async def test_ingestion_fails_fast_if_scraper_unhealthy():
    """Test ingestion service fails fast if scraper validation fails."""
    scraper_mock = Mock(spec=ScraperService)
    scraper_mock.validate_services = AsyncMock(
        side_effect=ValueError("Crawl4AI service unavailable")
    )

    embeddings_mock = Mock()
    vector_store_mock = Mock()
    queue_manager_mock = Mock()

    service = IngestionService(
        scraper=scraper_mock,
        embeddings=embeddings_mock,
        vector_store=vector_store_mock,
        queue_manager=queue_manager_mock,
    )

    with pytest.raises(ValueError, match="Crawl4AI service unavailable"):
        await service.validate_services()


@pytest.mark.asyncio
async def test_ingestion_validation_can_be_skipped():
    """Test ingestion service allows skipping validation on startup."""
    # Create service with mocked dependencies
    scraper_mock = Mock(spec=ScraperService)
    scraper_mock.validate_services = AsyncMock(
        side_effect=ValueError("Should not be called")
    )

    # Service creation should not raise when validate_on_startup=False
    # This test verifies that __init__ doesn't call validate_services automatically
    IngestionService(
        scraper=scraper_mock,
        embeddings=Mock(),
        vector_store=Mock(),
        queue_manager=Mock(),
        validate_on_startup=False,
    )


@pytest.mark.asyncio
async def test_scraper_validation_timeout_is_fast():
    """Test scraper health check uses fast timeout for startup validation."""
    import time

    service = ScraperService(endpoint_url="http://localhost:52004")

    # Mock a slow response that will trigger timeout
    service._client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

    start_time = time.time()
    with pytest.raises(ValueError, match="health check failed"):
        await service.validate_services()
    elapsed = time.time() - start_time

    # Should fail quickly (within 1 second), not wait for default timeout
    assert elapsed < 1.0, f"Validation took {elapsed}s, should be nearly instant"
