"""Mapper service for URL discovery using Crawl4AI.

This module provides URL discovery functionality using the Crawl4AI /crawl
endpoint. It supports BFS-style depth crawling with link deduplication and
same-domain filtering.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from urllib.parse import urljoin, urlparse

import httpx

from crawl4r.core.url_validation import validate_url
from crawl4r.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerError
from crawl4r.services.models import MapResult


class MapperService:
    """Service for discovering URLs from a seed URL using Crawl4AI.

    The MapperService crawls web pages to discover internal and external links.
    It supports recursive depth crawling with BFS traversal, link deduplication,
    and same-domain filtering.

    Attributes:
        endpoint_url: Base URL for the Crawl4AI service.

    Example:
        >>> service = MapperService(endpoint_url="http://localhost:52004")
        >>> result = await service.map_url("https://example.com", depth=1)
        >>> print(result.links)
        ['https://example.com/about', 'https://example.com/contact']
    """

    def __init__(
        self,
        endpoint_url: str,
        timeout: float = 60.0,
        health_endpoint: str = "/health",
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
    ) -> None:
        """Initialize the mapper service.

        Args:
            endpoint_url: Base URL for the Crawl4AI service.
            timeout: Request timeout in seconds.
            health_endpoint: Path to health check endpoint.
            circuit_breaker_threshold: Failures before opening circuit.
            circuit_breaker_timeout: Seconds before allowing recovery.
        """
        self._endpoint_url = endpoint_url.rstrip("/")
        self._health_endpoint = health_endpoint
        self._client = httpx.AsyncClient(base_url=self._endpoint_url, timeout=timeout)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_timeout,
        )

    async def map_url(
        self,
        url: str,
        depth: int = 0,
        same_domain: bool = True,
        progress_callback: Callable[[str, int, int], Awaitable[None]] | None = None,
    ) -> MapResult:
        """Discover URLs from a seed URL.

        Crawls the seed URL and discovers internal and external links. When
        depth > 0, recursively crawls discovered internal links using BFS
        traversal. Duplicate links are removed from results.

        Args:
            url: Seed URL to start crawling from.
            depth: Maximum crawl depth (0 = only seed URL).
            same_domain: If True, only return internal links matching seed domain.
            progress_callback: Optional async callback(url, depth, total_visited)
                called after each URL is crawled for progress reporting.

        Returns:
            MapResult with discovered links, counts, and depth reached.

        Example:
            >>> result = await service.map_url(
            ...     "https://example.com",
            ...     depth=1,
            ...     same_domain=True
            ... )
            >>> print(f"Found {result.internal_count} internal links")
        """
        # Validate URL before processing
        if not validate_url(url):
            return MapResult(url=url, success=False, error="Invalid URL")

        # Normalize seed URL
        seed_url = url.rstrip("/")
        seed_parsed = urlparse(seed_url)
        seed_domain = seed_parsed.netloc

        # Track discovered links and visited URLs
        all_internal: set[str] = set()
        all_external: set[str] = set()
        visited: set[str] = set()

        # BFS queue: (url_to_crawl, current_depth)
        queue: deque[tuple[str, int]] = deque([(seed_url, 0)])
        max_depth_reached = 0

        while queue:
            current_url, current_depth = queue.popleft()

            # Skip if already visited
            normalized_current = current_url.rstrip("/")
            if normalized_current in visited:
                continue
            visited.add(normalized_current)

            # Fetch links from current URL
            try:
                result = await self._circuit_breaker.call(
                    lambda u=current_url: self._fetch_links(u)
                )
            except CircuitBreakerError as exc:
                return MapResult(url=url, success=False, error=str(exc))
            except Exception as exc:  # noqa: BLE001
                return MapResult(url=url, success=False, error=str(exc))

            if result is None:
                # Request failed after retries
                return MapResult(
                    url=url,
                    success=False,
                    error="Failed to fetch links after retries",
                )

            internal_links, external_links = result
            max_depth_reached = max(max_depth_reached, current_depth)

            # Resolve relative URLs to absolute
            for href in internal_links:
                absolute_url = urljoin(current_url + "/", href).rstrip("/")
                absolute_parsed = urlparse(absolute_url)

                # Verify it's actually internal (same domain)
                if absolute_parsed.netloc == seed_domain:
                    all_internal.add(absolute_url)
                else:
                    all_external.add(absolute_url)

            for href in external_links:
                if href.startswith(("http://", "https://")):
                    all_external.add(href.rstrip("/"))

            # Queue internal links for next depth level if not at max depth
            if current_depth < depth:
                for href in internal_links:
                    absolute_url = urljoin(current_url + "/", href).rstrip("/")
                    absolute_parsed = urlparse(absolute_url)

                    # Only queue internal links
                    if absolute_parsed.netloc == seed_domain:
                        if absolute_url not in visited:
                            queue.append((absolute_url, current_depth + 1))

            # Report progress if callback provided
            if progress_callback is not None:
                await progress_callback(current_url, current_depth, len(visited))

        # Remove seed URL from internal links
        seed_normalized = seed_url.rstrip("/")
        all_internal.discard(seed_normalized)
        # Also check with trailing slash variant
        all_internal.discard(seed_normalized + "/")

        # Build result links based on same_domain filter
        if same_domain:
            result_links = sorted(all_internal)
        else:
            result_links = sorted(all_internal | all_external)

        return MapResult(
            url=url,
            success=True,
            links=result_links,
            internal_count=len(all_internal),
            external_count=len(all_external),
            depth_reached=max_depth_reached,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> MapperService:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager and cleanup resources."""
        await self.close()

    async def _fetch_links(self, url: str) -> tuple[list[str], list[str]] | None:
        """Fetch links from a URL using Crawl4AI /crawl endpoint.

        Args:
            url: URL to crawl for links.

        Returns:
            Tuple of (internal_links, external_links) or None on failure.
        """
        # Crawl4AI v0.5.1 API expects {"urls": [...]} (plural, array format)
        payload = {"urls": [url]}
        backoff_seconds = [1.0, 2.0, 4.0]

        for attempt in range(len(backoff_seconds) + 1):
            try:
                response = await self._client.post("/crawl", json=payload)

                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        "Server error from Crawl4AI",
                        request=response.request,
                        response=response,
                    )

                if response.status_code >= 400:
                    # Client error, no retry
                    return None

                return self._parse_links_response(response)

            except (
                httpx.TimeoutException,
                httpx.RequestError,
                httpx.HTTPStatusError,
            ):
                if attempt >= len(backoff_seconds):
                    return None
                await asyncio.sleep(backoff_seconds[attempt])

        return None

    def _parse_links_response(
        self, response: httpx.Response
    ) -> tuple[list[str], list[str]]:
        """Parse links from Crawl4AI response.

        Args:
            response: HTTP response from Crawl4AI /crawl endpoint.

        Returns:
            Tuple of (internal_links, external_links).
        """
        data = response.json()
        # Crawl4AI v0.5.1 returns {"results": [{...}]} (array format)
        results = data.get("results", [])
        if not results:
            return [], []
        links = results[0].get("links", {})

        internal_raw = links.get("internal", [])
        external_raw = links.get("external", [])

        internal_links = [
            item.get("href", "") for item in internal_raw if item.get("href")
        ]
        external_links = [
            item.get("href", "") for item in external_raw if item.get("href")
        ]

        return internal_links, external_links
