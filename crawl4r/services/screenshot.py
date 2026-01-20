"""Screenshot service for capturing web page screenshots using Crawl4AI.

This module provides screenshot capture functionality using the Crawl4AI
/screenshot endpoint. It supports full page and viewport captures with
configurable wait times and viewport dimensions.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import re
from pathlib import Path
from typing import Any

import httpx

from crawl4r.core.url_validation import validate_url
from crawl4r.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerError
from crawl4r.services.models import ScreenshotResult


class ScreenshotService:
    """Service for capturing screenshots of web pages.

    The ScreenshotService leverages Crawl4AI's /screenshot endpoint to capture
    screenshots of web pages. It supports various capture options including:

    1. Full page capture: Capture the entire scrollable page.
    2. Viewport capture: Capture only the visible viewport.
    3. Wait options: Wait for page load or specific CSS selectors.
    4. Custom viewport: Configure viewport dimensions for responsive testing.

    Attributes:
        endpoint_url: Base URL for the Crawl4AI service.

    Example:
        >>> service = ScreenshotService(endpoint_url="http://localhost:52004")
        >>> result = await service.capture(
        ...     "https://example.com",
        ...     output_path=Path("/tmp/screenshot.png"),
        ...     full_page=True,
        ... )
        >>> print(result.file_path)
        /tmp/screenshot.png
    """

    def __init__(
        self,
        endpoint_url: str,
        timeout: float = 60.0,
        health_endpoint: str = "/health",
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
    ) -> None:
        """Initialize the screenshot service.

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

    async def capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        """Capture a screenshot of a web page.

        Sends the URL and options to the Crawl4AI /screenshot endpoint,
        decodes the base64 response, and saves the image to the specified path.

        Args:
            url: URL to capture screenshot from.
            output_path: Path where the screenshot should be saved.
            full_page: If True, capture the full scrollable page.
            wait: Seconds to wait after page load before capture.
            wait_for_selector: CSS selector to wait for before capture.
            viewport_width: Viewport width in pixels.
            viewport_height: Viewport height in pixels.

        Returns:
            ScreenshotResult with file path and size, or error details.

        Example:
            >>> result = await service.capture(
            ...     "https://example.com",
            ...     output_path=Path("/tmp/screenshot.png"),
            ...     full_page=True,
            ...     viewport_width=1920,
            ...     viewport_height=1080,
            ... )
            >>> print(result.file_path)
            /tmp/screenshot.png
        """
        # Validate URL before processing
        if not url or not validate_url(url):
            return ScreenshotResult(
                url=url,
                success=False,
                error="Invalid URL",
            )

        try:
            result = await self._circuit_breaker.call(
                lambda: self._fetch_screenshot(
                    url=url,
                    output_path=output_path,
                    full_page=full_page,
                    wait=wait,
                    wait_for_selector=wait_for_selector,
                    viewport_width=viewport_width,
                    viewport_height=viewport_height,
                )
            )
            return result
        except CircuitBreakerError as exc:
            return ScreenshotResult(
                url=url,
                success=False,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return ScreenshotResult(
                url=url,
                success=False,
                error=str(exc),
            )

    async def capture_batch(
        self,
        urls: list[str],
        output_dir: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
        max_concurrent: int = 5,
    ) -> list[ScreenshotResult]:
        """Capture screenshots from multiple URLs.

        Processes multiple URLs concurrently with a configurable concurrency limit.
        Each URL is captured independently, and partial failures do not affect
        other URLs in the batch.

        Args:
            urls: List of URLs to capture screenshots from.
            output_dir: Directory where screenshots should be saved.
            full_page: If True, capture the full scrollable page.
            wait: Seconds to wait after page load before capture.
            wait_for_selector: CSS selector to wait for before capture.
            viewport_width: Viewport width in pixels.
            viewport_height: Viewport height in pixels.
            max_concurrent: Maximum concurrent screenshot requests.

        Returns:
            List of ScreenshotResult for each URL, in the same order as input.

        Example:
            >>> results = await service.capture_batch(
            ...     urls=["https://example.com/1", "https://example.com/2"],
            ...     output_dir=Path("/tmp/screenshots"),
            ... )
            >>> print(len(results))
            2
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded_capture(target_url: str) -> ScreenshotResult:
            async with semaphore:
                output_path = self._generate_output_path(target_url, output_dir)
                return await self.capture(
                    url=target_url,
                    output_path=output_path,
                    full_page=full_page,
                    wait=wait,
                    wait_for_selector=wait_for_selector,
                    viewport_width=viewport_width,
                    viewport_height=viewport_height,
                )

        tasks = [asyncio.create_task(_bounded_capture(url)) for url in urls]
        return await asyncio.gather(*tasks)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def validate_services(self) -> None:
        """Validate that the Crawl4AI service is available.

        Raises:
            ValueError: If the service health check fails.
        """
        try:
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

    async def _fetch_screenshot(
        self,
        url: str,
        output_path: Path,
        full_page: bool,
        wait: float | None,
        wait_for_selector: str | None,
        viewport_width: int | None,
        viewport_height: int | None,
    ) -> ScreenshotResult:
        """Fetch screenshot from Crawl4AI /screenshot endpoint.

        Args:
            url: URL to capture screenshot from.
            output_path: Path where the screenshot should be saved.
            full_page: If True, capture the full scrollable page.
            wait: Seconds to wait after page load before capture.
            wait_for_selector: CSS selector to wait for before capture.
            viewport_width: Viewport width in pixels.
            viewport_height: Viewport height in pixels.

        Returns:
            ScreenshotResult with file path and size, or error details.
        """
        # Build request payload
        payload: dict[str, Any] = {"url": url}
        if full_page:
            payload["full_page"] = True
        if wait is not None:
            payload["wait"] = wait
        if wait_for_selector is not None:
            payload["wait_for_selector"] = wait_for_selector
        if viewport_width is not None or viewport_height is not None:
            payload["viewport"] = {}
            if viewport_width is not None:
                payload["viewport"]["width"] = viewport_width
            if viewport_height is not None:
                payload["viewport"]["height"] = viewport_height

        backoff_seconds = [1.0, 2.0, 4.0]

        for attempt in range(len(backoff_seconds) + 1):
            try:
                response = await self._client.post("/screenshot", json=payload)

                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        "Server error from Crawl4AI",
                        request=response.request,
                        response=response,
                    )

                if response.status_code >= 400:
                    # Client error, no retry
                    return ScreenshotResult(
                        url=url,
                        success=False,
                        error=f"Request failed with status {response.status_code}",
                    )

                return self._parse_and_save_screenshot(url, response, output_path)

            except httpx.TimeoutException:
                if attempt >= len(backoff_seconds):
                    return ScreenshotResult(
                        url=url,
                        success=False,
                        error="Request timeout during screenshot capture",
                    )
                await asyncio.sleep(backoff_seconds[attempt])

            except httpx.ConnectError:
                if attempt >= len(backoff_seconds):
                    return ScreenshotResult(
                        url=url,
                        success=False,
                        error="Connection error during screenshot capture",
                    )
                await asyncio.sleep(backoff_seconds[attempt])

            except httpx.RequestError:
                if attempt >= len(backoff_seconds):
                    return ScreenshotResult(
                        url=url,
                        success=False,
                        error="Request error during screenshot capture",
                    )
                await asyncio.sleep(backoff_seconds[attempt])

            except httpx.HTTPStatusError:
                if attempt >= len(backoff_seconds):
                    return ScreenshotResult(
                        url=url,
                        success=False,
                        error="Server error during screenshot capture",
                    )
                await asyncio.sleep(backoff_seconds[attempt])

        return ScreenshotResult(
            url=url,
            success=False,
            error="Unexpected screenshot capture failure",
        )

    def _parse_and_save_screenshot(
        self,
        url: str,
        response: httpx.Response,
        output_path: Path,
    ) -> ScreenshotResult:
        """Parse screenshot response and save to file.

        Args:
            url: Source URL for the screenshot.
            response: HTTP response from Crawl4AI.
            output_path: Path where the screenshot should be saved.

        Returns:
            ScreenshotResult with file path and size, or error details.
        """
        try:
            data = response.json()
        except Exception:
            return ScreenshotResult(
                url=url,
                success=False,
                error="Invalid JSON response from screenshot endpoint",
            )

        # Check for error in response body
        if data.get("success") is False:
            return ScreenshotResult(
                url=url,
                success=False,
                error=data.get("error", "Screenshot capture failed"),
            )

        # Extract the screenshot field
        screenshot_b64 = data.get("screenshot")
        if screenshot_b64 is None:
            error_msg = data.get("error", "No screenshot in response")
            return ScreenshotResult(
                url=url,
                success=False,
                error=error_msg,
            )

        # Decode base64 screenshot
        try:
            image_bytes = base64.b64decode(screenshot_b64)
        except Exception:
            return ScreenshotResult(
                url=url,
                success=False,
                error="Invalid base64 encoding in screenshot response",
            )

        # Save to file
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_bytes)
        except Exception as exc:
            return ScreenshotResult(
                url=url,
                success=False,
                error=f"Failed to write screenshot file: {exc}",
            )

        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=len(image_bytes),
        )

    def _generate_output_path(self, url: str, output_dir: Path) -> Path:
        """Generate a unique output path for a URL.

        Creates a filename based on the URL's domain and path, using a hash
        to ensure uniqueness and valid filename characters.

        Args:
            url: URL to generate filename from.
            output_dir: Directory where the file should be saved.

        Returns:
            Path object for the output file.
        """
        # Create a sanitized filename from URL
        # Remove scheme and extract domain + path
        clean_url = re.sub(r"^https?://", "", url)
        # Replace non-alphanumeric chars with underscore
        sanitized = re.sub(r"[^a-zA-Z0-9]", "_", clean_url)
        # Truncate to reasonable length
        sanitized = sanitized[:50]
        # Add hash for uniqueness
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]  # noqa: S324

        filename = f"{sanitized}_{url_hash}.png"
        return output_dir / filename
