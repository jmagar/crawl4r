"""Crawl command for ingestion workflows."""

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def crawl() -> None:
    """Crawl URLs and ingest results into the vector store."""
    return None
