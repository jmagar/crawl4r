"""Crawl command for ingestion workflows."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from crawl4r.services.ingestion import IngestionService
from crawl4r.services.mapper import MapperService
from crawl4r.services.models import CrawlStatus, CrawlStatusInfo, IngestResult

def crawl_command(
    urls: list[str] = typer.Argument(..., help="URLs to crawl"),
    file: Path | None = typer.Option(None, "-f", "--file", help="File containing URLs (one per line)"),
    depth: int = typer.Option(1, "-d", "--depth", help="Crawl depth (0=no discovery, 1+=recursive)"),
    external: bool = typer.Option(False, "--external", help="Include external links in discovery"),
) -> None:
    """Crawl URLs and ingest results into the vector store."""
    resolved_urls = _merge_urls(urls or [], file)
    if not resolved_urls:
        typer.echo("No URLs provided")
        raise typer.Exit(code=1)

    service = IngestionService()
    result, queue_position = asyncio.run(_run_crawl(service, resolved_urls, depth, external))

    console = Console()
    console.print(f"Crawl ID: {result.crawl_id}")
    if result.queued:
        queue_note = (
            f"Queue position: {queue_position}"
            if queue_position is not None
            else "Queued for processing"
        )
        console.print(queue_note)
    panel = Panel(
        f"URLs: {result.urls_total}\n"
        f"Failed: {result.urls_failed}\n"
        f"Chunks: {result.chunks_created}",
        title="Crawl Summary",
    )
    console.print(panel)

    if not result.success:
        raise typer.Exit(code=1)


def _merge_urls(urls: list[str], file: Path | None) -> list[str]:
    max_url_file_size = 1024 * 1024  # 1MB

    merged = [url.strip() for url in urls if url.strip()]
    if file is None:
        return merged
    if not file.exists():
        raise typer.BadParameter(f"URL file not found: {file}")

    # Check file size before reading
    if file.stat().st_size > max_url_file_size:
        raise typer.BadParameter(
            f"URL file too large (max {max_url_file_size} bytes): {file}"
        )

    file_urls = [line.strip() for line in file.read_text().splitlines()]
    merged.extend([url for url in file_urls if url])
    return merged


async def _run_crawl(
    service: IngestionService, urls: list[str], depth: int, external: bool
) -> tuple[IngestResult, int | None]:
    console = Console()
    crawl_id = None
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        """Handle SIGINT and SIGTERM by setting stop event.

        Note: On Windows, signal.signal() is used as a fallback since
        add_signal_handler() is not supported. This may not work correctly
        for all signal scenarios due to threading limitations.
        """
        stop_event.set()

    # Register signal handlers
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        signal.signal(signal.SIGINT, lambda *_: _signal_handler())
        signal.signal(signal.SIGTERM, lambda *_: _signal_handler())

    try:
        # URL discovery phase (if depth > 0)
        if depth > 0:
            if len(urls) > 1:
                console.print("[yellow]Warning: Depth crawling only uses first URL as seed[/yellow]")

            seed_url = urls[0]
            endpoint_url = os.getenv("CRAWL4AI_BASE_URL", "http://localhost:52004")

            with console.status(f"Discovering URLs (depth={depth}, external={external})..."):
                async with MapperService(endpoint_url=endpoint_url) as mapper:
                    map_result = await mapper.map_url(
                        seed_url,
                        depth=depth,
                        same_domain=not external
                    )

            if not map_result.success:
                console.print(f"[red]URL discovery failed: {map_result.error}[/red]")
                raise typer.Exit(code=1)

            urls_to_ingest = map_result.links or []
            console.print(f"[green]Discovered {len(urls_to_ingest)} URLs[/green]")
        else:
            urls_to_ingest = urls

        # Ingestion phase
        with console.status("Crawling URLs..."):
            result = await service.ingest_urls(urls_to_ingest)
            crawl_id = result.crawl_id

        queue_position = None
        if result.queued:
            queue_position = await service.queue_manager.get_queue_length()

        return result, queue_position

    except (KeyboardInterrupt, SystemExit):
        # Handle graceful shutdown
        console.print("\n[yellow]Crawl interrupted by user[/yellow]")

        # Only set status if crawl was actually started
        if crawl_id is not None:
            finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            await service.queue_manager.set_status(
                CrawlStatusInfo(
                    crawl_id=crawl_id,
                    status=CrawlStatus.FAILED,
                    error="Interrupted by user",
                    finished_at=finished_at,
                )
            )

        # Note: IngestionService.ingest_urls releases the lock in its finally block,
        # so we don't need to manually release it here

        # Re-raise the exception to let the caller handle it
        raise
    finally:
        # Restore original signal handlers
        try:
            loop.remove_signal_handler(signal.SIGINT)
            loop.remove_signal_handler(signal.SIGTERM)
        except (NotImplementedError, RuntimeError):
            # Windows or loop already closed
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
