from typer.testing import CliRunner

from crawl4r.cli.app import app

runner = CliRunner()


def test_status_help() -> None:
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0
    assert "status" in result.output
