"""Typer application entry point for crawl4r CLI."""

import typer

from crawl4r.cli.commands import crawl as crawl_command
from crawl4r.cli.commands import scrape as scrape_command
from crawl4r.cli.commands import status as status_command

app = typer.Typer(no_args_is_help=True, name="crawl4r")

app.add_typer(scrape_command.app, name="scrape")
app.add_typer(crawl_command.app, name="crawl")
app.add_typer(status_command.app, name="status")


if __name__ == "__main__":
    app()
