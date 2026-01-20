import httpx
import pytest
import respx

from crawl4r.services.scraper import ScraperService


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_hits_md_endpoint() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(200, json={"markdown": "# Title", "status_code": 200})
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com")
    assert result.url == "https://example.com"
    assert result.markdown == "# Title"
    assert result.success is True


@respx.mock
@pytest.mark.asyncio
async def test_scrape_urls_batch_returns_results() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(200, json={"markdown": "# Ok", "status_code": 200})
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    results = await service.scrape_urls(["https://a.com", "https://b.com"], max_concurrent=2)
    assert len(results) == 2
