"""Screenshot command for capturing web pages.

This module provides a CLI command to capture screenshots of web pages using
the Crawl4AI service. It supports full-page capture, viewport customization,
and wait options.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import typer
from rich.console import Console

from crawl4r.services.screenshot import ScreenshotService

console = Console()


def _default_output_path(url: str) -> Path:
    """Generate default screenshot output path from URL.

    Extracts the netloc (domain) from the URL and uses it as the base
    filename with a .png extension. Replaces invalid filename characters
    (like colons in ports) with underscores.

    Args:
        url: Source URL to extract domain from.

    Returns:
        Path with domain-based filename (e.g., "example.com.png",
        "localhost_8080.png").
    """
    parsed = urlparse(url)
    netloc = parsed.netloc or "screenshot"
    # Replace colons with underscores for port numbers (e.g., localhost:8080)
    safe_netloc = netloc.replace(":", "_")
    return Path(f"{safe_netloc}.png")


def screenshot_command(
    url: Annotated[str, typer.Argument(help="URL to capture")],
    output: Annotated[
        Path | None, typer.Option("-o", "--output", help="Output file path")
    ] = None,
    full_page: Annotated[
        bool, typer.Option("-f", "--full-page", help="Capture full page")
    ] = False,
    wait: Annotated[
        float, typer.Option("-w", "--wait", help="Wait seconds before capture")
    ] = 0.0,
    selector: Annotated[
        str | None, typer.Option("-s", "--selector", help="Wait for CSS selector")
    ] = None,
    width: Annotated[
        int | None, typer.Option("--width", help="Viewport width in pixels")
    ] = None,
    height: Annotated[
        int | None, typer.Option("--height", help="Viewport height in pixels")
    ] = None,
) -> None:
    """Capture screenshot of a web page.

    Takes a screenshot of the specified URL using the Crawl4AI service.
    Supports full-page capture, viewport customization, and wait options.

    Args:
        url: URL to capture.
        output: Optional output file path. Defaults to domain-based filename.
        full_page: Whether to capture the full scrollable page.
        wait: Seconds to wait after page load before capture.
        selector: Optional CSS selector to wait for before capture.
        width: Optional viewport width in pixels.
        height: Optional viewport height in pixels.
    """
    if output is None:
        output = _default_output_path(url)

    async def _run() -> None:
        """Execute screenshot capture and report result."""
        endpoint_url = os.getenv("CRAWL4AI_BASE_URL", "http://localhost:52004")

        async with ScreenshotService(endpoint_url=endpoint_url) as service:
            # Only pass wait if non-zero
            wait_value: float | None = wait if wait > 0 else None

            result = await service.capture(
                url,
                output_path=output,
                full_page=full_page,
                wait=wait_value,
                wait_for_selector=selector,
                viewport_width=width,
                viewport_height=height,
            )

            if not result.success:
                console.print(f"[red]Failed: {result.error}[/red]")
                raise typer.Exit(code=1)

            console.print(f"Saved {result.file_size} bytes to {result.file_path}")

    asyncio.run(_run())
