"""Unit tests for screenshot CLI command.

These tests verify the screenshot command correctly invokes the ScreenshotService
and outputs capture results. Tests use monkeypatching to mock the
ScreenshotService.capture async method.

This is the RED phase of TDD - tests should fail because
crawl4r/cli/commands/screenshot.py does not yet exist.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from crawl4r.cli.app import app
from crawl4r.services.models import ScreenshotResult

runner = CliRunner()


# =============================================================================
# Default output path tests
# =============================================================================


def test_screenshot_command_default_name(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command uses domain-based filename by default.

    When no output path is specified, the command should generate a filename
    based on the URL's domain (e.g., example.com.png).

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=5,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(app, ["screenshot", "https://example.com"])
    assert result.exit_code == 0
    assert "example.com" in result.output.lower()


def test_screenshot_command_default_name_with_path(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command handles URLs with paths in default filename.

    When no output path is specified and the URL has a path component,
    the command should still generate a sensible filename.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=10,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(app, ["screenshot", "https://example.com/products/item"])
    assert result.exit_code == 0
    assert ".png" in result.output.lower()


# =============================================================================
# Custom output path tests
# =============================================================================


def test_screenshot_command_custom_output(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command accepts custom output path.

    The -o/--output option should allow specifying the exact file path
    for the screenshot.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=10,
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    output_path = tmp_path / "page.png"
    result = runner.invoke(
        app, ["screenshot", "https://example.com", "-o", str(output_path)]
    )
    assert result.exit_code == 0
    assert str(output_path) in result.output


def test_screenshot_command_custom_output_long_flag(
    tmp_path: Path, monkeypatch
) -> None:
    """Test screenshot command accepts --output long flag.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=15,
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    output_path = tmp_path / "custom.png"
    result = runner.invoke(
        app, ["screenshot", "https://example.com", "--output", str(output_path)]
    )
    assert result.exit_code == 0
    assert str(output_path) in result.output


# =============================================================================
# Full page option tests
# =============================================================================


def test_screenshot_command_full_page_option(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command passes --full-page option to service.

    The --full-page flag should capture the entire scrollable page
    rather than just the viewport.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """
    captured_full_page: bool | None = None

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        nonlocal captured_full_page
        captured_full_page = full_page
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=20,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(app, ["screenshot", "https://example.com", "--full-page"])
    assert result.exit_code == 0
    assert captured_full_page is True


def test_screenshot_command_full_page_short_flag(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command accepts -f short flag for full page.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """
    captured_full_page: bool | None = None

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        nonlocal captured_full_page
        captured_full_page = full_page
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=25,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(app, ["screenshot", "https://example.com", "-f"])
    assert result.exit_code == 0
    assert captured_full_page is True


def test_screenshot_command_default_viewport_only(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command captures viewport only by default.

    When --full-page is not specified, full_page should be False.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """
    captured_full_page: bool | None = None

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        nonlocal captured_full_page
        captured_full_page = full_page
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=30,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(app, ["screenshot", "https://example.com"])
    assert result.exit_code == 0
    assert captured_full_page is False


# =============================================================================
# Wait option tests
# =============================================================================


def test_screenshot_command_wait_option(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command passes --wait option to service.

    The --wait option should specify seconds to wait after page load
    before capturing the screenshot.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """
    captured_wait: float | None = None

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        nonlocal captured_wait
        captured_wait = wait
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=35,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(app, ["screenshot", "https://example.com", "--wait", "2.5"])
    assert result.exit_code == 0
    assert captured_wait == 2.5


def test_screenshot_command_wait_short_flag(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command accepts -w short flag for wait.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """
    captured_wait: float | None = None

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        nonlocal captured_wait
        captured_wait = wait
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=40,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(app, ["screenshot", "https://example.com", "-w", "3"])
    assert result.exit_code == 0
    assert captured_wait == 3.0


def test_screenshot_command_wait_integer(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command accepts integer wait values.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """
    captured_wait: float | None = None

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        nonlocal captured_wait
        captured_wait = wait
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=45,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(app, ["screenshot", "https://example.com", "--wait", "5"])
    assert result.exit_code == 0
    assert captured_wait == 5.0


# =============================================================================
# Wait for selector option tests
# =============================================================================


def test_screenshot_command_selector_option(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command passes --selector option to service.

    The --selector option should wait for a CSS selector to be present
    before capturing the screenshot.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """
    captured_selector: str | None = None

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        nonlocal captured_selector
        captured_selector = wait_for_selector
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=50,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(
        app, ["screenshot", "https://example.com", "--selector", "#main-content"]
    )
    assert result.exit_code == 0
    assert captured_selector == "#main-content"


def test_screenshot_command_selector_short_flag(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command accepts -s short flag for selector.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """
    captured_selector: str | None = None

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        nonlocal captured_selector
        captured_selector = wait_for_selector
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=55,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(
        app, ["screenshot", "https://example.com", "-s", ".loaded-content"]
    )
    assert result.exit_code == 0
    assert captured_selector == ".loaded-content"


# =============================================================================
# Viewport option tests
# =============================================================================


def test_screenshot_command_viewport_width_option(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command passes --width option to service.

    The --width option should set the viewport width in pixels.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """
    captured_width: int | None = None

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        nonlocal captured_width
        captured_width = viewport_width
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=60,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(
        app, ["screenshot", "https://example.com", "--width", "1920"]
    )
    assert result.exit_code == 0
    assert captured_width == 1920


def test_screenshot_command_viewport_height_option(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command passes --height option to service.

    The --height option should set the viewport height in pixels.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """
    captured_height: int | None = None

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        nonlocal captured_height
        captured_height = viewport_height
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=65,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(
        app, ["screenshot", "https://example.com", "--height", "1080"]
    )
    assert result.exit_code == 0
    assert captured_height == 1080


def test_screenshot_command_combined_viewport_options(
    tmp_path: Path, monkeypatch
) -> None:
    """Test screenshot command passes both width and height options.

    When both viewport dimensions are specified, both should be passed
    to the service.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """
    captured_width: int | None = None
    captured_height: int | None = None

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        nonlocal captured_width, captured_height
        captured_width = viewport_width
        captured_height = viewport_height
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=70,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(
        app,
        ["screenshot", "https://example.com", "--width", "375", "--height", "812"],
    )
    assert result.exit_code == 0
    assert captured_width == 375
    assert captured_height == 812


# =============================================================================
# Error handling tests
# =============================================================================


def test_screenshot_command_failure_returns_nonzero(
    tmp_path: Path, monkeypatch
) -> None:
    """Test screenshot command returns nonzero exit code on failure.

    When the ScreenshotService returns success=False, the command
    should exit with a nonzero code.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        return ScreenshotResult(
            url=url,
            success=False,
            error="Screenshot capture timeout",
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(app, ["screenshot", "https://example.com"])
    assert result.exit_code != 0
    assert "timeout" in result.output.lower() or "failed" in result.output.lower()


def test_screenshot_command_displays_error_message(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command displays service error message.

    When the service returns an error, the error message should be
    shown to the user.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        return ScreenshotResult(
            url=url,
            success=False,
            error="Connection refused to Crawl4AI service",
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(app, ["screenshot", "https://example.com"])
    assert result.exit_code != 0
    assert "connection" in result.output.lower() or "refused" in result.output.lower()


def test_screenshot_command_invalid_url_error(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command handles invalid URL errors.

    When the service returns an invalid URL error, the command should
    display an appropriate message.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        return ScreenshotResult(
            url=url,
            success=False,
            error="Invalid URL",
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(app, ["screenshot", "not-a-valid-url"])
    assert result.exit_code != 0
    assert "invalid" in result.output.lower() or "url" in result.output.lower()


# =============================================================================
# Help text tests
# =============================================================================


def test_screenshot_command_help() -> None:
    """Test screenshot command shows help text."""
    result = runner.invoke(app, ["screenshot", "--help"])
    assert result.exit_code == 0
    assert "screenshot" in result.output.lower()


def test_screenshot_command_help_shows_options() -> None:
    """Test screenshot command help shows all available options."""
    result = runner.invoke(app, ["screenshot", "--help"])
    assert result.exit_code == 0
    # Verify key options are documented
    output_lower = result.output.lower()
    assert "output" in output_lower
    assert "full" in output_lower or "page" in output_lower
    assert "wait" in output_lower


def test_screenshot_command_help_shows_url_argument() -> None:
    """Test screenshot command help shows URL argument."""
    result = runner.invoke(app, ["screenshot", "--help"])
    assert result.exit_code == 0
    assert "url" in result.output.lower()


# =============================================================================
# Output display tests
# =============================================================================


def test_screenshot_command_displays_file_path(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command displays saved file path on success.

    When capture succeeds, the command should show where the screenshot
    was saved.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=1024,
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(app, ["screenshot", "https://example.com"])
    assert result.exit_code == 0
    # Output should mention the file path or "saved"
    assert ".png" in result.output.lower() or "saved" in result.output.lower()


def test_screenshot_command_displays_file_size(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command displays file size on success.

    When capture succeeds, the command may optionally show the file size.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=51200,  # 50KB
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    result = runner.invoke(app, ["screenshot", "https://example.com"])
    assert result.exit_code == 0
    # Output should be non-empty (file info displayed)
    assert len(result.output.strip()) > 0


# =============================================================================
# Combined options tests
# =============================================================================


def test_screenshot_command_all_options(tmp_path: Path, monkeypatch) -> None:
    """Test screenshot command handles all options together.

    Verify the command works when all options are specified at once.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """
    captured_params: dict = {}

    async def _fake_capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: float | None = None,
        wait_for_selector: str | None = None,
        viewport_width: int | None = None,
        viewport_height: int | None = None,
    ) -> ScreenshotResult:
        captured_params.update(
            {
                "url": url,
                "output_path": output_path,
                "full_page": full_page,
                "wait": wait,
                "wait_for_selector": wait_for_selector,
                "viewport_width": viewport_width,
                "viewport_height": viewport_height,
            }
        )
        return ScreenshotResult(
            url=url,
            success=True,
            file_path=str(output_path),
            file_size=100,
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.screenshot.ScreenshotService.capture",
        _fake_capture,
    )

    output_path = tmp_path / "full_test.png"
    result = runner.invoke(
        app,
        [
            "screenshot",
            "https://example.com/page",
            "-o",
            str(output_path),
            "--full-page",
            "--wait",
            "2",
            "--selector",
            "#content",
            "--width",
            "1920",
            "--height",
            "1080",
        ],
    )
    assert result.exit_code == 0
    assert captured_params["url"] == "https://example.com/page"
    assert captured_params["full_page"] is True
    assert captured_params["wait"] == 2.0
    assert captured_params["wait_for_selector"] == "#content"
    assert captured_params["viewport_width"] == 1920
    assert captured_params["viewport_height"] == 1080
