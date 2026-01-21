"""Typer application entry point for crawl4r CLI."""

import typer

from crawl4r.cli.commands import crawl as crawl_command
from crawl4r.cli.commands import extract as extract_command
from crawl4r.cli.commands import map as map_urls
from crawl4r.cli.commands import scrape as scrape_command
from crawl4r.cli.commands import screenshot as screenshot_command
from crawl4r.cli.commands import status as status_command
from crawl4r.cli.commands import watch as watch_command

app = typer.Typer(no_args_is_help=True, name="crawl4r")

app.add_typer(scrape_command.app, name="scrape")
app.add_typer(status_command.app, name="status")
app.add_typer(watch_command.app, name="watch")

# Register these as direct commands (not sub-typers) to avoid argument parsing
# issues
app.command(name="crawl", help="Crawl URLs and ingest into vector store")(
    crawl_command.crawl_command
)
app.command(name="map", help="Discover URLs from a web page")(map_urls.map_command)

# Register extract as a direct command for structured data extraction
app.command(name="extract", help="Extract structured data from a web page")(
    extract_command.extract_command
)

# Register screenshot as a direct command for page capture
app.command(name="screenshot", help="Capture screenshots of web pages")(
    screenshot_command.screenshot_command
)


if __name__ == "__main__":
    app()
