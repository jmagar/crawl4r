"""Map command for URL discovery.

This module provides a CLI command to discover URLs from a seed webpage using
the Crawl4AI service. It supports configurable crawl depth, same-domain filtering,
and output to file or stdout.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from crawl4r.services.mapper import MapperService

console = Console()


def map_command(
    url: str = typer.Argument(..., help="URL to map"),
    depth: int = typer.Option(0, "-d", "--depth", help="Max crawl depth"),
    same_domain: bool = typer.Option(
        True,
        "--same-domain/--external",
        help="Restrict mapping to same-domain links",
    ),
    output: Path | None = typer.Option(None, "-o", "--output"),
) -> None:
    """Discover URLs from a seed URL.

    Maps a web page to discover internal and optionally external links. When depth
    is greater than 0, recursively crawls discovered internal links using BFS
    traversal.

    Args:
        url: URL to map.
        depth: Maximum recursion depth (0 = only seed URL).
        same_domain: When True, only return same-domain links.
        output: Optional output file for URLs.
    """

    async def _run() -> None:
        """Execute map request and write output."""
        endpoint_url = os.getenv("CRAWL4AI_BASE_URL", "http://localhost:52004")

        async with MapperService(endpoint_url=endpoint_url) as service:
            # Use progress indicators for depth crawls
            if depth > 0:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Crawling...", total=None)

                    async def update_progress(
                        current_url: str, current_depth: int, total_visited: int
                    ) -> None:
                        """Update progress bar with current crawl status."""
                        progress.update(
                            task,
                            description=f"[cyan]Depth {current_depth}[/cyan] | "
                            f"[green]{total_visited} visited[/green] | "
                            f"{current_url[:60]}...",
                        )

                    result = await service.map_url(
                        url,
                        depth=depth,
                        same_domain=same_domain,
                        progress_callback=update_progress,
                    )
            else:
                # No progress indicator for single URL
                result = await service.map_url(
                    url, depth=depth, same_domain=same_domain
                )

            if not result.success:
                console.print(f"[red]Failed: {result.error}[/red]")
                raise typer.Exit(code=1)

            lines = result.links or []

            if output is None:
                for link in lines:
                    console.print(link)
                console.print(f"Unique URLs: {len(lines)}")
            else:
                output.write_text("\n".join(lines) + "\n" if lines else "")
                console.print(f"Wrote {len(lines)} URLs to {output}")

    asyncio.run(_run())
