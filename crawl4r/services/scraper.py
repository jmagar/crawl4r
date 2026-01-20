"""Scraper service for retrieving markdown from Crawl4AI."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from crawl4r.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerError
from crawl4r.services.models import ScrapeResult


class ScraperService:
    """Service for scraping markdown from the Crawl4AI API."""

    def __init__(
        self,
        endpoint_url: str,
        timeout: float = 60.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
    ) -> None:
        """Initialize the scraper service.

        Args:
            endpoint_url: Base URL for the Crawl4AI service
            timeout: Request timeout in seconds
            circuit_breaker_threshold: Failures before opening circuit
            circuit_breaker_timeout: Seconds before allowing recovery
        """
        self._endpoint_url = endpoint_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._endpoint_url, timeout=timeout)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_timeout,
        )

    async def scrape_url(self, url: str) -> ScrapeResult:
        """Scrape a single URL for markdown content.

        Args:
            url: URL to scrape

        Returns:
            ScrapeResult with markdown or error details
        """
        try:
            await self._check_health()
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

    async def _check_health(self) -> None:
        response = await self._client.get("/health")
        response.raise_for_status()

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
