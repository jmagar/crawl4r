"""Status command for crawl queue visibility."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from crawl4r.core.config import Settings
from crawl4r.services.models import CrawlStatus, CrawlStatusInfo
from crawl4r.services.queue import QueueManager

app = typer.Typer(no_args_is_help=True, invoke_without_command=True)


@app.callback()
def status(
    crawl_id: str | None = typer.Argument(None),
    list_recent: bool = typer.Option(False, "--list"),
    active: bool = typer.Option(False, "--active"),
) -> None:
    """Show crawl status information."""
    settings = Settings(watch_folder=Path("."))
    queue = QueueManager(settings.REDIS_URL)
    results = asyncio.run(_fetch_status(queue, crawl_id, list_recent, active))

    console = Console()
    if isinstance(results, CrawlStatusInfo):
        _print_single(console, results)
        return
    if not results:
        console.print("No crawl status records found")
        return
    _print_table(console, results)


def _status_style(status: CrawlStatus) -> str:
    return {
        CrawlStatus.QUEUED: "yellow",
        CrawlStatus.RUNNING: "blue",
        CrawlStatus.COMPLETED: "green",
        CrawlStatus.FAILED: "red",
    }[status]


async def _fetch_status(
    queue: QueueManager,
    crawl_id: str | None,
    list_recent: bool,
    active: bool,
) -> CrawlStatusInfo | list[CrawlStatusInfo]:
    if crawl_id:
        status = await queue.get_status(crawl_id)
        if status is None:
            return []
        return status
    if active:
        return await queue.get_active()
    if list_recent:
        return await queue.list_recent()
    return await queue.list_recent()


def _print_single(console: Console, status: CrawlStatusInfo) -> None:
    color = _status_style(status.status)
    console.print(f"{status.crawl_id} [{color}]{status.status.value}[/{color}]")


def _print_table(console: Console, statuses: list[CrawlStatusInfo]) -> None:
    table = Table(title="Crawl Status")
    table.add_column("Crawl ID")
    table.add_column("Status")
    table.add_column("Started")
    table.add_column("Finished")
    table.add_column("Error")
    for info in statuses:
        color = _status_style(info.status)
        table.add_row(
            info.crawl_id,
            f"[{color}]{info.status.value}[/{color}]",
            info.started_at or "-",
            info.finished_at or "-",
            info.error or "-",
        )
    console.print(table)
