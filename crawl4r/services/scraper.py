"""Scraper service for retrieving markdown from Crawl4AI."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from crawl4r.core.url_validation import validate_url
from crawl4r.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerError
from crawl4r.services.models import ScrapeResult


class ScraperService:
    """Service for scraping markdown from the Crawl4AI API."""

    def __init__(
        self,
        endpoint_url: str,
        timeout: float = 60.0,
        health_endpoint: str = "/health",
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        validate_on_startup: bool = True,
    ) -> None:
        """Initialize the scraper service.

        Args:
            endpoint_url: Base URL for the Crawl4AI service
            timeout: Request timeout in seconds
            health_endpoint: Path to health check endpoint
            circuit_breaker_threshold: Failures before opening circuit
            circuit_breaker_timeout: Seconds before allowing recovery
            validate_on_startup: Whether to validate service availability on init
        """
        self._endpoint_url = endpoint_url.rstrip("/")
        self._health_endpoint = health_endpoint
        self._client = httpx.AsyncClient(base_url=self._endpoint_url, timeout=timeout)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_timeout,
        )
        self._validate_on_startup = validate_on_startup

    async def scrape_url(self, url: str) -> ScrapeResult:
        """Scrape a single URL for markdown content.

        Args:
            url: URL to scrape

        Returns:
            ScrapeResult with markdown or error details
        """
        # Validate URL before processing
        if not validate_url(url):
            return ScrapeResult(url=url, success=False, error="Invalid URL")

        try:
            # Circuit breaker handles availability - no need for separate health check
            result = await self._circuit_breaker.call(
                lambda: self._fetch_markdown(url)
            )
            return result
        except CircuitBreakerError as exc:
            return ScrapeResult(url=url, success=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            return ScrapeResult(url=url, success=False, error=str(exc))

    async def scrape_urls(
        self, urls: list[str], max_concurrent: int = 5
    ) -> list[ScrapeResult]:
        """Scrape multiple URLs with concurrency limits.

        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum concurrent requests

        Returns:
            List of ScrapeResult values
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded_scrape(target_url: str) -> ScrapeResult:
            async with semaphore:
                return await self.scrape_url(target_url)

        tasks = [
            asyncio.create_task(_bounded_scrape(target_url)) for target_url in urls
        ]
        return await asyncio.gather(*tasks)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def validate_services(self) -> None:
        """Validate that the Crawl4AI service is available.

        Raises:
            ValueError: If the service health check fails
        """
        try:
            # Use shorter timeout for startup validation
            timeout = httpx.Timeout(5.0)
            response = await self._client.get(self._health_endpoint, timeout=timeout)
            response.raise_for_status()
        except (
            httpx.TimeoutException,
            httpx.HTTPStatusError,
            httpx.ConnectError,
            httpx.RequestError,
        ) as exc:
            msg = f"Crawl4AI service health check failed: {exc}"
            raise ValueError(msg) from exc

    async def _fetch_markdown(self, url: str) -> ScrapeResult:
        payload = {"url": url, "f": "fit"}
        backoff_seconds = [1.0, 2.0, 4.0]

        for attempt in range(len(backoff_seconds) + 1):
            try:
                response = await self._client.post("/md?f=fit", json=payload)
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        "Server error from Crawl4AI",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 400:
                    return self._error_result(
                        url,
                        f"Request failed with status {response.status_code}",
                        response.status_code,
                    )
                return self._parse_response(url, response)
            except (
                httpx.TimeoutException,
                httpx.RequestError,
                httpx.HTTPStatusError,
            ) as exc:
                if attempt >= len(backoff_seconds):
                    return ScrapeResult(url=url, success=False, error=str(exc))
                await asyncio.sleep(backoff_seconds[attempt])

        return ScrapeResult(url=url, success=False, error="Unexpected scrape failure")

    def _parse_response(self, url: str, response: httpx.Response) -> ScrapeResult:
        data = response.json()
        markdown = data.get("markdown")
        status_code = data.get("status_code", response.status_code)
        metadata = self._extract_metadata(data)
        return ScrapeResult(
            url=url,
            success=True,
            markdown=markdown,
            status_code=status_code,
            metadata=metadata,
        )

    def _extract_metadata(self, data: dict[str, Any]) -> dict[str, Any] | None:
        metadata = {
            key: value for key, value in data.items() if key not in {"markdown"}
        }
        return metadata or None

    def _error_result(
        self, url: str, error: str, status_code: int | None
    ) -> ScrapeResult:
        return ScrapeResult(
            url=url,
            success=False,
            error=error,
            status_code=status_code,
        )
