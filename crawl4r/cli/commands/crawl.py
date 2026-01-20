"""Crawl command for ingestion workflows."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from crawl4r.services.ingestion import IngestionService
from crawl4r.services.models import IngestResult

app = typer.Typer(no_args_is_help=True, invoke_without_command=True)


@app.callback()
def crawl(
    urls: list[str] = typer.Argument(None),
    file: Path | None = typer.Option(None, "-f", "--file"),
    depth: int = typer.Option(1, "-d", "--depth"),
) -> None:
    """Crawl URLs and ingest results into the vector store."""
    resolved_urls = _merge_urls(urls or [], file)
    if not resolved_urls:
        typer.echo("No URLs provided")
        raise typer.Exit(code=1)

    _ = depth
    service = IngestionService()
    result, queue_position = asyncio.run(_run_crawl(service, resolved_urls))

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
    merged = [url.strip() for url in urls if url.strip()]
    if file is None:
        return merged
    if not file.exists():
        raise typer.BadParameter(f"URL file not found: {file}")
    file_urls = [line.strip() for line in file.read_text().splitlines()]
    merged.extend([url for url in file_urls if url])
    return merged


async def _run_crawl(
    service: IngestionService, urls: list[str]
) -> tuple[IngestResult, int | None]:
    console = Console()
    with console.status("Crawling URLs..."):
        result = await service.ingest_urls(urls)

    queue_position = None
    if result.queued:
        queue_position = await service.queue_manager.get_queue_length()

    return result, queue_position
