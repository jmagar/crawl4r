"""Unit tests for extract CLI command.

These tests verify the extract command correctly invokes the ExtractorService
and outputs extracted structured data to stdout or file. Tests use monkeypatching
to mock the ExtractorService async methods.

This is the RED phase of TDD - tests should fail because
crawl4r/cli/commands/extract.py does not yet exist.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from crawl4r.cli.app import app
from crawl4r.services.models import ExtractResult

runner = CliRunner()


# =============================================================================
# Schema-based extraction tests
# =============================================================================


def test_extract_command_schema_file(tmp_path: Path, monkeypatch) -> None:
    """Test extract command reads schema from file and extracts data.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_extract_with_schema(
        self, url: str, schema: dict, provider: str | None = None
    ) -> ExtractResult:
        return ExtractResult(
            url=url,
            success=True,
            data={"title": "Example"},
            extraction_method="schema",
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.extract.ExtractorService.extract_with_schema",
        _fake_extract_with_schema,
    )

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps({"type": "object"}))

    result = runner.invoke(app, ["extract", "https://example.com", "--schema", str(schema_path)])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["title"] == "Example"


def test_extract_command_schema_inline(monkeypatch) -> None:
    """Test extract command accepts inline JSON schema string.

    Args:
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_extract_with_schema(
        self, url: str, schema: dict, provider: str | None = None
    ) -> ExtractResult:
        return ExtractResult(
            url=url,
            success=True,
            data={"name": "Product", "price": 29.99},
            extraction_method="schema",
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.extract.ExtractorService.extract_with_schema",
        _fake_extract_with_schema,
    )

    schema_json = '{"type": "object", "properties": {"name": {"type": "string"}}}'
    result = runner.invoke(
        app,
        ["extract", "https://example.com/product", "--schema", schema_json],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["name"] == "Product"


# =============================================================================
# Prompt-based extraction tests
# =============================================================================


def test_extract_command_prompt(monkeypatch) -> None:
    """Test extract command uses prompt for natural language extraction.

    Args:
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_extract_with_prompt(
        self, url: str, prompt: str, provider: str | None = None
    ) -> ExtractResult:
        return ExtractResult(
            url=url,
            success=True,
            data={"heading": "Hello"},
            extraction_method="prompt",
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.extract.ExtractorService.extract_with_prompt",
        _fake_extract_with_prompt,
    )

    result = runner.invoke(
        app, ["extract", "https://example.com", "--prompt", "extract heading"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["heading"] == "Hello"


def test_extract_command_prompt_long_text(monkeypatch) -> None:
    """Test extract command handles long prompts correctly.

    Args:
        monkeypatch: Pytest fixture for patching.
    """
    captured_prompt: str | None = None

    async def _fake_extract_with_prompt(
        self, url: str, prompt: str, provider: str | None = None
    ) -> ExtractResult:
        nonlocal captured_prompt
        captured_prompt = prompt
        return ExtractResult(
            url=url,
            success=True,
            data={"products": [{"name": "A"}, {"name": "B"}]},
            extraction_method="prompt",
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.extract.ExtractorService.extract_with_prompt",
        _fake_extract_with_prompt,
    )

    long_prompt = "Extract all product names and their prices as a structured list"
    result = runner.invoke(
        app, ["extract", "https://example.com/products", "--prompt", long_prompt]
    )
    assert result.exit_code == 0
    assert captured_prompt == long_prompt


# =============================================================================
# Validation tests
# =============================================================================


def test_extract_command_requires_schema_or_prompt() -> None:
    """Test extract command fails when neither schema nor prompt provided."""
    result = runner.invoke(app, ["extract", "https://example.com"])
    assert result.exit_code != 0
    # Should show error about missing schema or prompt
    assert "schema" in result.output.lower() or "prompt" in result.output.lower()


def test_extract_command_rejects_both_schema_and_prompt(tmp_path: Path) -> None:
    """Test extract command fails when both schema and prompt are provided.

    Args:
        tmp_path: Pytest fixture providing temp directory.
    """
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps({"type": "object"}))

    result = runner.invoke(
        app,
        [
            "extract",
            "https://example.com",
            "--schema",
            str(schema_path),
            "--prompt",
            "extract data",
        ],
    )
    assert result.exit_code != 0


# =============================================================================
# Output file tests
# =============================================================================


def test_extract_command_writes_file(tmp_path: Path, monkeypatch) -> None:
    """Test extract command writes extracted data to output file.

    Args:
        tmp_path: Pytest fixture providing temp directory.
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_extract_with_prompt(
        self, url: str, prompt: str, provider: str | None = None
    ) -> ExtractResult:
        return ExtractResult(
            url=url,
            success=True,
            data={"title": "Output Test"},
            extraction_method="prompt",
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.extract.ExtractorService.extract_with_prompt",
        _fake_extract_with_prompt,
    )

    output_path = tmp_path / "output.json"
    result = runner.invoke(
        app,
        [
            "extract",
            "https://example.com",
            "--prompt",
            "extract title",
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0
    assert output_path.exists()

    data = json.loads(output_path.read_text())
    assert data["title"] == "Output Test"


# =============================================================================
# Provider option tests
# =============================================================================


def test_extract_command_provider_option(monkeypatch) -> None:
    """Test extract command passes provider option to service.

    Args:
        monkeypatch: Pytest fixture for patching.
    """
    captured_provider: str | None = None

    async def _fake_extract_with_prompt(
        self, url: str, prompt: str, provider: str | None = None
    ) -> ExtractResult:
        nonlocal captured_provider
        captured_provider = provider
        return ExtractResult(
            url=url,
            success=True,
            data={"result": "ok"},
            extraction_method="prompt",
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.extract.ExtractorService.extract_with_prompt",
        _fake_extract_with_prompt,
    )

    result = runner.invoke(
        app,
        [
            "extract",
            "https://example.com",
            "--prompt",
            "extract data",
            "--provider",
            "ollama/llama3",
        ],
    )
    assert result.exit_code == 0
    assert captured_provider == "ollama/llama3"


# =============================================================================
# Error handling tests
# =============================================================================


def test_extract_command_failure_returns_nonzero(monkeypatch) -> None:
    """Test extract command returns nonzero exit code on failure.

    Args:
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_extract_with_prompt(
        self, url: str, prompt: str, provider: str | None = None
    ) -> ExtractResult:
        return ExtractResult(
            url=url,
            success=False,
            error="LLM extraction failed",
            extraction_method="prompt",
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.extract.ExtractorService.extract_with_prompt",
        _fake_extract_with_prompt,
    )

    result = runner.invoke(
        app, ["extract", "https://example.com", "--prompt", "extract data"]
    )
    assert result.exit_code != 0
    assert "LLM extraction failed" in result.output or "Failed" in result.output


def test_extract_command_invalid_schema_file(tmp_path: Path) -> None:
    """Test extract command handles invalid schema file path.

    Args:
        tmp_path: Pytest fixture providing temp directory.
    """
    nonexistent_path = tmp_path / "nonexistent.json"
    result = runner.invoke(
        app, ["extract", "https://example.com", "--schema", str(nonexistent_path)]
    )
    assert result.exit_code != 0


def test_extract_command_invalid_json_schema(tmp_path: Path) -> None:
    """Test extract command handles malformed JSON schema file.

    Args:
        tmp_path: Pytest fixture providing temp directory.
    """
    schema_path = tmp_path / "bad_schema.json"
    schema_path.write_text("not valid json {{{")

    result = runner.invoke(
        app, ["extract", "https://example.com", "--schema", str(schema_path)]
    )
    assert result.exit_code != 0


# =============================================================================
# Help text tests
# =============================================================================


def test_extract_command_help() -> None:
    """Test extract command shows help text."""
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    assert "extract" in result.output.lower()
    assert "schema" in result.output.lower()
    assert "prompt" in result.output.lower()


# =============================================================================
# Nested data output tests
# =============================================================================


def test_extract_command_outputs_nested_json(monkeypatch) -> None:
    """Test extract command outputs nested JSON structures correctly.

    Args:
        monkeypatch: Pytest fixture for patching.
    """

    async def _fake_extract_with_schema(
        self, url: str, schema: dict, provider: str | None = None
    ) -> ExtractResult:
        return ExtractResult(
            url=url,
            success=True,
            data={
                "product": {
                    "name": "Widget",
                    "details": {
                        "sku": "WGT-001",
                        "price": 29.99,
                    },
                }
            },
            extraction_method="schema",
        )

    monkeypatch.setattr(
        "crawl4r.cli.commands.extract.ExtractorService.extract_with_schema",
        _fake_extract_with_schema,
    )

    schema_json = '{"type": "object"}'
    result = runner.invoke(
        app, ["extract", "https://example.com", "--schema", schema_json]
    )
    assert result.exit_code == 0

    data = json.loads(result.output)
    assert data["product"]["name"] == "Widget"
    assert data["product"]["details"]["sku"] == "WGT-001"
