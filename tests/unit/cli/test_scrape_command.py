from typer.testing import CliRunner

from crawl4r.cli.app import app

runner = CliRunner()


def test_scrape_help() -> None:
    result = runner.invoke(app, ["scrape", "--help"])
    assert result.exit_code == 0
    assert "scrape" in result.output
