"""Scrape command for fetching markdown from URLs."""

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def scrape() -> None:
    """Scrape URLs and output markdown."""
    return None
