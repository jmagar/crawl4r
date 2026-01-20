"""Unit tests for map CLI command.

These tests verify the map command correctly invokes the MapperService
and outputs discovered URLs to stdout or file. Tests use monkeypatching
to mock the MapperService.map_url async method.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from crawl4r.cli.app import app
from crawl4r.services.models import MapResult

runner = CliRunner()


def test_map_command_writes_stdout(monkeypatch) -> None:
    """Test map command outputs discovered URLs to stdout.

    Args:
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_map_url(
        self, url: str, depth: int = 0, same_domain: bool = True
    ) -> MapResult:
        return MapResult(
            url=url,
            success=True,
            links=["https://example.com/a", "https://example.com/b"],
            internal_count=2,
            external_count=0,
            depth_reached=0,
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.map.MapperService.map_url", _fake_map_url
    )

    result = runner.invoke(app, ["map", "https://example.com"])
    assert result.exit_code == 0
    assert "https://example.com/a" in result.output
    assert "Unique URLs: 2" in result.output


def test_map_command_writes_file(tmp_path: Path, monkeypatch) -> None:
    """Test map command writes discovered URLs to output file.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_map_url(
        self, url: str, depth: int = 0, same_domain: bool = True
    ) -> MapResult:
        return MapResult(
            url=url,
            success=True,
            links=["https://example.com/a"],
            internal_count=1,
            external_count=0,
            depth_reached=0,
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.map.MapperService.map_url", _fake_map_url
    )

    output_path = tmp_path / "urls.txt"
    result = runner.invoke(app, ["map", "https://example.com", "-o", str(output_path)])
    assert result.exit_code == 0
    assert output_path.read_text().strip() == "https://example.com/a"


def test_map_command_help() -> None:
    """Test map command shows help text."""
    result = runner.invoke(app, ["map", "--help"])
    assert result.exit_code == 0
    assert "map" in result.output.lower()


def test_map_command_depth_option(monkeypatch) -> None:
    """Test map command passes depth option to service.

    Args:
        monkeypatch: Pytest fixture for patching.
    """
    captured_depth: int | None = None

    async def _fake_map_url(
        self, url: str, depth: int = 0, same_domain: bool = True
    ) -> MapResult:
        nonlocal captured_depth
        captured_depth = depth
        return MapResult(
            url=url,
            success=True,
            links=["https://example.com/deep"],
            internal_count=1,
            external_count=0,
            depth_reached=depth,
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.map.MapperService.map_url", _fake_map_url
    )

    result = runner.invoke(app, ["map", "https://example.com", "-d", "2"])
    assert result.exit_code == 0
    assert captured_depth == 2


def test_map_command_failure_returns_nonzero(monkeypatch) -> None:
    """Test map command returns nonzero exit code on failure.

    Args:
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_map_url(
        self, url: str, depth: int = 0, same_domain: bool = True
    ) -> MapResult:
        return MapResult(
            url=url,
            success=False,
            error="Connection refused",
            links=None,
            internal_count=None,
            external_count=None,
            depth_reached=None,
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.map.MapperService.map_url", _fake_map_url
    )

    result = runner.invoke(app, ["map", "https://example.com"])
    assert result.exit_code != 0
    assert "Connection refused" in result.output or "Failed" in result.output


def test_map_command_external_links_option(monkeypatch) -> None:
    """Test map command passes same_domain=False with --external flag.

    Args:
        monkeypatch: Pytest fixture for patching.
    """
    captured_same_domain: bool | None = None

    async def _fake_map_url(
        self, url: str, depth: int = 0, same_domain: bool = True
    ) -> MapResult:
        nonlocal captured_same_domain
        captured_same_domain = same_domain
        return MapResult(
            url=url,
            success=True,
            links=["https://example.com/a", "https://other.com/b"],
            internal_count=1,
            external_count=1,
            depth_reached=0,
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.map.MapperService.map_url", _fake_map_url
    )

    result = runner.invoke(app, ["map", "https://example.com", "--external"])
    assert result.exit_code == 0
    assert captured_same_domain is False
