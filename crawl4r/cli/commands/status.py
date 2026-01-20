"""Status command for crawl queue visibility."""

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def status() -> None:
    """Show crawl status information."""
    return None
