"""Unit tests for MapperService URL discovery functionality.

These tests verify the MapperService correctly:
- Discovers internal and external links from Crawl4AI /crawl endpoint
- Filters links by same-domain policy
- Handles recursive depth crawling with deduplication
- Reports accurate link counts and depth reached

This is the RED phase of TDD - tests should fail because
crawl4r.services.mapper module does not yet exist.
"""

import httpx
import pytest
import respx

from crawl4r.services.mapper import MapperService
from tests.fixtures.crawl4ai_responses import (
    MOCK_MAP_RESULT_NESTED,
    MOCK_MAP_RESULT_SUCCESS,
    MOCK_MAP_RESULT_WITH_DUPLICATES,
)


@respx.mock
@pytest.mark.asyncio
async def test_map_url_same_domain_filters_and_counts() -> None:
    """Verify same_domain=True filters to internal links only.

    When same_domain is True, only links matching the seed URL's domain
    should be returned. External links should be counted but not included.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(200, json=MOCK_MAP_RESULT_SUCCESS)
    )

    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("https://example.com", depth=0, same_domain=True)

    assert result.success is True
    assert result.internal_count == 3
    assert result.external_count == 2
    assert all(link.startswith("https://example.com") for link in result.links)
    assert result.depth_reached == 0


@respx.mock
@pytest.mark.asyncio
async def test_map_url_includes_external_when_requested() -> None:
    """Verify same_domain=False includes external links in results.

    When same_domain is False, both internal and external links should
    be included in the results list.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(200, json=MOCK_MAP_RESULT_SUCCESS)
    )

    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("https://example.com", depth=0, same_domain=False)

    assert result.success is True
    assert result.internal_count == 3
    assert result.external_count == 2
    assert "https://www.iana.org/domains/example" in result.links


@respx.mock
@pytest.mark.asyncio
async def test_map_url_depth_recurses_and_dedupes() -> None:
    """Verify depth crawling follows internal links and deduplicates.

    When depth > 0, the service should recursively crawl discovered
    internal links. Duplicate links should appear only once in results.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    route = respx.post("http://localhost:52004/crawl")
    route.side_effect = [
        httpx.Response(200, json=MOCK_MAP_RESULT_WITH_DUPLICATES),
        httpx.Response(200, json=MOCK_MAP_RESULT_NESTED),
    ]

    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("https://example.com", depth=1, same_domain=True)

    assert result.success is True
    assert "https://example.com/about" in result.links
    assert "https://example.com/team" in result.links
    assert result.links.count("https://example.com/about") == 1
    assert result.depth_reached == 1


@respx.mock
@pytest.mark.asyncio
async def test_map_url_returns_failure_on_service_error() -> None:
    """Verify MapResult indicates failure when service returns error."""
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(500, json={"error": "Internal server error"})
    )

    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("https://example.com", depth=0, same_domain=True)

    assert result.success is False
    assert result.error is not None


@respx.mock
@pytest.mark.asyncio
async def test_map_url_handles_timeout() -> None:
    """Verify MapResult indicates failure on request timeout."""
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/crawl").mock(
        side_effect=httpx.TimeoutException("Connection timeout")
    )

    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("https://example.com", depth=0, same_domain=True)

    assert result.success is False
    assert result.error is not None


@respx.mock
@pytest.mark.asyncio
async def test_map_url_resolves_relative_paths() -> None:
    """Verify relative paths are resolved to absolute URLs.

    Internal links like "/about" should be converted to full URLs
    like "https://example.com/about".
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(200, json=MOCK_MAP_RESULT_SUCCESS)
    )

    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("https://example.com", depth=0, same_domain=True)

    assert result.success is True
    # Relative paths /a, /b, /c should become absolute URLs
    assert "https://example.com/a" in result.links
    assert "https://example.com/b" in result.links
    assert "https://example.com/c" in result.links


@respx.mock
@pytest.mark.asyncio
async def test_map_url_excludes_seed_url_from_results() -> None:
    """Verify the seed URL is not included in discovered links.

    The seed URL should be crawled but not listed in the results
    to avoid redundancy.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "results": [
                    {
                        "url": "https://example.com/",
                        "links": {
                            "internal": [{"href": "/"}, {"href": "/about"}],
                            "external": [],
                        },
                    }
                ],
            },
        )
    )

    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("https://example.com/", depth=0, same_domain=True)

    assert result.success is True
    # Root "/" resolves to the seed URL and should be excluded
    assert "https://example.com/" not in result.links
    assert "https://example.com/about" in result.links


@respx.mock
@pytest.mark.asyncio
async def test_map_url_validates_invalid_url() -> None:
    """Verify invalid URLs return failure without making requests."""
    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("not-a-valid-url", depth=0, same_domain=True)

    assert result.success is False
    assert "Invalid URL" in result.error


@respx.mock
@pytest.mark.asyncio
async def test_map_url_respects_max_depth_limit() -> None:
    """Verify crawling stops at specified depth.

    Even if discovered links have more nested pages, crawling should
    not exceed the configured depth parameter.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    # First call returns link to /about
    # Second call would return more links, but should not be called
    route = respx.post("http://localhost:52004/crawl")
    route.side_effect = [
        httpx.Response(
            200,
            json={
                "success": True,
                "results": [
                    {
                        "url": "https://example.com",
                        "links": {"internal": [{"href": "/about"}], "external": []},
                    }
                ],
            },
        ),
        # This should not be called with depth=0
        httpx.Response(
            200,
            json={
                "success": True,
                "results": [
                    {
                        "url": "https://example.com",
                        "links": {"internal": [{"href": "/team"}], "external": []},
                    }
                ],
            },
        ),
    ]

    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("https://example.com", depth=0, same_domain=True)

    assert result.success is True
    assert result.depth_reached == 0
    # With depth=0, should only crawl seed URL, not follow links
    assert "https://example.com/about" in result.links
    assert "https://example.com/team" not in result.links


# =============================================================================
# Spec-required test names
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_map_url_success() -> None:
    """Test successful URL mapping with link discovery (spec-required name)."""
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "results": [
                    {
                        "url": "https://example.com",
                        "links": {
                            "internal": [{"href": "https://example.com/a"}],
                            "external": [],
                        },
                    }
                ],
            },
        )
    )

    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("https://example.com", depth=1, same_domain=True)

    assert result.success is True
    assert "https://example.com/a" in result.links
