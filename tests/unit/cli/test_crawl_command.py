from typer.testing import CliRunner

from crawl4r.cli.app import app

runner = CliRunner()


def test_crawl_help() -> None:
    result = runner.invoke(app, ["crawl", "--help"])
    assert result.exit_code == 0
    assert "crawl" in result.output
