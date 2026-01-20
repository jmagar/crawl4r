"""Unit tests for ScreenshotService page screenshot functionality.

These tests verify the ScreenshotService correctly:
- Captures screenshots via Crawl4AI /screenshot endpoint
- Saves screenshots to specified file paths
- Handles full_page and wait options
- Reports file sizes and capture errors

This is the RED phase of TDD - tests should fail because
crawl4r.services.screenshot module does not yet exist.
"""

import base64
from pathlib import Path

import httpx
import pytest
import respx
from crawl4r.services.screenshot import ScreenshotService

# =============================================================================
# Basic screenshot capture tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_saves_png(tmp_path: Path) -> None:
    """Verify screenshot is captured and saved as PNG file.

    The service should POST to /screenshot endpoint, decode the base64
    response, and save the image bytes to the specified output path.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    image_bytes = b"hello"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(200, json={"screenshot": encoded})
    )

    output_path = tmp_path / "page.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture("https://example.com", output_path=output_path)

    assert result.success is True
    assert output_path.exists()
    assert output_path.read_bytes() == image_bytes
    assert result.file_size == len(image_bytes)


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_returns_file_path_in_result(tmp_path: Path) -> None:
    """Verify ScreenshotResult includes the output file path.

    The result should contain the absolute path to the saved screenshot
    for downstream processing.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    image_bytes = b"PNG_DATA"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(200, json={"screenshot": encoded})
    )

    output_path = tmp_path / "test_screenshot.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture("https://example.com", output_path=output_path)

    assert result.success is True
    assert result.file_path == str(output_path)


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_includes_source_url_in_result(tmp_path: Path) -> None:
    """Verify ScreenshotResult includes the source URL.

    The result should track which URL was captured for traceability.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    image_bytes = b"PNG_DATA"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(200, json={"screenshot": encoded})
    )

    output_path = tmp_path / "screenshot.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture(
        "https://example.com/page", output_path=output_path
    )

    assert result.success is True
    assert result.url == "https://example.com/page"


# =============================================================================
# Full page option tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_full_page_option(tmp_path: Path) -> None:
    """Verify full_page=True captures entire scrollable page.

    When full_page is True, the service should pass this option to the
    /screenshot endpoint to capture the full page height.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    image_bytes = b"FULL_PAGE_PNG"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    route = respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(200, json={"screenshot": encoded})
    )

    output_path = tmp_path / "full_page.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture(
        "https://example.com",
        output_path=output_path,
        full_page=True,
    )

    assert result.success is True
    assert route.called
    request_body = route.calls.last.request.content
    assert b"full_page" in request_body or b"fullPage" in request_body


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_viewport_only_by_default(tmp_path: Path) -> None:
    """Verify default behavior captures viewport only, not full page.

    When full_page is not specified, only the visible viewport should
    be captured (full_page=False).
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    image_bytes = b"VIEWPORT_PNG"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    route = respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(200, json={"screenshot": encoded})
    )

    output_path = tmp_path / "viewport.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture("https://example.com", output_path=output_path)

    assert result.success is True
    assert route.called


# =============================================================================
# Wait option tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_wait_option(tmp_path: Path) -> None:
    """Verify wait option delays screenshot capture.

    When wait is specified, the service should pass it to the endpoint
    to allow page rendering before capture.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    image_bytes = b"WAITED_PNG"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    route = respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(200, json={"screenshot": encoded})
    )

    output_path = tmp_path / "waited.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture(
        "https://example.com",
        output_path=output_path,
        wait=2.0,
    )

    assert result.success is True
    assert route.called
    request_body = route.calls.last.request.content
    assert b"wait" in request_body


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_wait_for_selector(tmp_path: Path) -> None:
    """Verify wait_for_selector option waits for element before capture.

    When wait_for_selector is specified, the service should wait for the
    specified CSS selector to be present before capturing.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    image_bytes = b"SELECTOR_PNG"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    route = respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(200, json={"screenshot": encoded})
    )

    output_path = tmp_path / "selector.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture(
        "https://example.com",
        output_path=output_path,
        wait_for_selector="#content",
    )

    assert result.success is True
    assert route.called
    request_body = route.calls.last.request.content
    assert b"selector" in request_body or b"wait" in request_body


# =============================================================================
# Error handling tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_returns_failure_on_service_error(tmp_path: Path) -> None:
    """Verify ScreenshotResult indicates failure when service returns error."""
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(500, json={"error": "Internal server error"})
    )

    output_path = tmp_path / "failed.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture("https://example.com", output_path=output_path)

    assert result.success is False
    assert result.error is not None
    assert not output_path.exists()


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_handles_timeout(tmp_path: Path) -> None:
    """Verify ScreenshotResult indicates failure on request timeout."""
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/screenshot").mock(
        side_effect=httpx.TimeoutException("Screenshot timeout")
    )

    output_path = tmp_path / "timeout.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture("https://example.com", output_path=output_path)

    assert result.success is False
    assert result.error is not None
    assert "timeout" in result.error.lower()


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_handles_connection_error(tmp_path: Path) -> None:
    """Verify ScreenshotResult indicates failure on connection error."""
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/screenshot").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    output_path = tmp_path / "connection_error.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture("https://example.com", output_path=output_path)

    assert result.success is False
    assert result.error is not None


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_handles_invalid_base64_response(tmp_path: Path) -> None:
    """Verify ScreenshotResult handles malformed base64 from endpoint.

    If the /screenshot endpoint returns invalid base64, the service should
    gracefully handle the error and return a failure result.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(200, json={"screenshot": "not_valid_base64!!!"})
    )

    output_path = tmp_path / "invalid_b64.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture("https://example.com", output_path=output_path)

    assert result.success is False
    assert result.error is not None


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_handles_missing_screenshot_field(tmp_path: Path) -> None:
    """Verify ScreenshotResult handles missing screenshot field in response.

    If the /screenshot endpoint returns a response without the expected
    screenshot field, the service should return a failure result.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(200, json={"error": "No screenshot generated"})
    )

    output_path = tmp_path / "missing_field.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture("https://example.com", output_path=output_path)

    assert result.success is False
    assert result.error is not None


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_handles_file_write_error(tmp_path: Path) -> None:
    """Verify ScreenshotResult handles file write permission errors.

    If the output path is not writable, the service should return a
    failure result with an appropriate error message.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    image_bytes = b"PNG_DATA"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(200, json={"screenshot": encoded})
    )

    # Use a path that doesn't exist and can't be created
    output_path = Path("/nonexistent/directory/screenshot.png")
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture("https://example.com", output_path=output_path)

    assert result.success is False
    assert result.error is not None


# =============================================================================
# URL validation tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_validates_invalid_url(tmp_path: Path) -> None:
    """Verify invalid URLs return failure without making requests."""
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture(
        "not-a-valid-url",
        output_path=tmp_path / "invalid.png",
    )

    assert result.success is False
    assert "Invalid URL" in result.error or "url" in result.error.lower()


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_validates_empty_url(tmp_path: Path) -> None:
    """Verify empty URLs return failure without making requests."""
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture("", output_path=tmp_path / "empty.png")

    assert result.success is False
    assert result.error is not None


# =============================================================================
# Viewport configuration tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_custom_viewport_size(tmp_path: Path) -> None:
    """Verify custom viewport dimensions are passed to endpoint.

    The service should allow specifying viewport width and height
    for responsive design testing.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    image_bytes = b"VIEWPORT_PNG"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    route = respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(200, json={"screenshot": encoded})
    )

    output_path = tmp_path / "custom_viewport.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture(
        "https://example.com",
        output_path=output_path,
        viewport_width=1920,
        viewport_height=1080,
    )

    assert result.success is True
    assert route.called
    request_body = route.calls.last.request.content
    assert b"1920" in request_body or b"viewport" in request_body


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_mobile_viewport(tmp_path: Path) -> None:
    """Verify mobile viewport dimensions can be specified.

    The service should support mobile device emulation via viewport size.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    image_bytes = b"MOBILE_PNG"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    route = respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(200, json={"screenshot": encoded})
    )

    output_path = tmp_path / "mobile.png"
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    result = await service.capture(
        "https://example.com",
        output_path=output_path,
        viewport_width=375,
        viewport_height=812,
    )

    assert result.success is True
    assert route.called


# =============================================================================
# Service lifecycle tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_service_close() -> None:
    """Verify service close method cleans up resources.

    The service should have a close method to release HTTP client resources.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    service = ScreenshotService(endpoint_url="http://localhost:52004")
    await service.close()

    # Should not raise any errors


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_service_validate_services() -> None:
    """Verify service health check method validates Crawl4AI availability.

    The validate_services method should check if the Crawl4AI service
    is accessible before attempting operations.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    service = ScreenshotService(endpoint_url="http://localhost:52004")
    await service.validate_services()

    # Should not raise any errors


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_service_health_check_failure() -> None:
    """Verify service raises error when Crawl4AI is unavailable.

    If the health check fails, validate_services should raise ValueError.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(503))

    service = ScreenshotService(endpoint_url="http://localhost:52004")

    with pytest.raises(ValueError, match="health check failed"):
        await service.validate_services()


# =============================================================================
# Batch screenshot tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_batch_captures_multiple_urls(tmp_path: Path) -> None:
    """Verify batch screenshot captures multiple URLs.

    The service should support capturing screenshots from multiple URLs
    in a single call, returning results for each.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    image_bytes_1 = b"PNG_1"
    image_bytes_2 = b"PNG_2"

    route = respx.post("http://localhost:52004/screenshot")
    route.side_effect = [
        httpx.Response(
            200, json={"screenshot": base64.b64encode(image_bytes_1).decode()}
        ),
        httpx.Response(
            200, json={"screenshot": base64.b64encode(image_bytes_2).decode()}
        ),
    ]

    service = ScreenshotService(endpoint_url="http://localhost:52004")
    results = await service.capture_batch(
        urls=["https://example.com/1", "https://example.com/2"],
        output_dir=tmp_path,
    )

    assert len(results) == 2
    assert results[0].success is True
    assert results[1].success is True
    assert (tmp_path / "example.com_1.png").exists() or any(
        f.suffix == ".png" for f in tmp_path.iterdir()
    )


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_batch_handles_partial_failures(tmp_path: Path) -> None:
    """Verify batch screenshot continues on individual URL failures.

    If one URL fails during batch capture, the service should continue
    processing remaining URLs and return partial results.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    image_bytes = b"PNG_DATA"

    route = respx.post("http://localhost:52004/screenshot")
    route.side_effect = [
        httpx.Response(
            200, json={"screenshot": base64.b64encode(image_bytes).decode()}
        ),
        httpx.Response(500, json={"error": "Failed"}),
        httpx.Response(
            200, json={"screenshot": base64.b64encode(image_bytes).decode()}
        ),
    ]

    service = ScreenshotService(endpoint_url="http://localhost:52004")
    results = await service.capture_batch(
        urls=[
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ],
        output_dir=tmp_path,
    )

    assert len(results) == 3
    assert results[0].success is True
    assert results[1].success is False
    assert results[2].success is True
