"""Scrape command for fetching markdown from URLs."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from crawl4r.core.config import Settings
from crawl4r.services.models import ScrapeResult
from crawl4r.services.scraper import ScraperService

app = typer.Typer(no_args_is_help=True)


@app.callback()
def scrape(
    urls: list[str] = typer.Argument(None),
    file: Path | None = typer.Option(None, "-f", "--file"),
    output: Path | None = typer.Option(None, "-o", "--output"),
    concurrent: int = typer.Option(5, "-c", "--concurrent"),
) -> None:
    """Scrape URLs and output markdown."""
    resolved_urls = _merge_urls(urls or [], file)
    if not resolved_urls:
        typer.echo("No URLs provided")
        raise typer.Exit(code=1)

    settings = Settings(watch_folder=".")
    service = ScraperService(endpoint_url=settings.CRAWL4AI_BASE_URL)

    results = asyncio.run(_scrape_urls(service, resolved_urls, concurrent))
    successes = [result for result in results if result.success]
    failures = [result for result in results if not result.success]

    _write_outputs(results, output)
    _print_summary(len(successes), len(failures))

    if failures:
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


async def _scrape_urls(
    service: ScraperService, urls: list[str], max_concurrent: int
) -> list[ScrapeResult]:
    console = Console()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run(url: str, task_id: int, progress: Progress):
        async with semaphore:
            result = await service.scrape_url(url)
            progress.advance(task_id)
            return result

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task_id = progress.add_task("Scraping", total=len(urls))
        tasks = [
            asyncio.create_task(_run(url, task_id, progress)) for url in urls
        ]
        return await asyncio.gather(*tasks)


def _write_outputs(results: list[ScrapeResult], output: Path | None) -> None:
    console = Console()
    if output is None:
        for result in results:
            if result.success:
                console.print(result.markdown or "")
            else:
                console.print(f"[red]Failed:[/red] {result.url} {result.error}")
        return

    if len(results) == 1:
        _write_file(output, results[0].markdown or "")
        return

    output.mkdir(parents=True, exist_ok=True)
    for result in results:
        if not result.success:
            continue
        filename = _slugify_url(result.url) + ".md"
        _write_file(output / filename, result.markdown or "")


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _slugify_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "index"
    slug = f"{parsed.netloc}{path}"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", slug).strip("_")
    return slug[:80] or "page"


def _print_summary(successes: int, failures: int) -> None:
    console = Console()
    console.print(f"Scrape complete: {successes} succeeded, {failures} failed")
