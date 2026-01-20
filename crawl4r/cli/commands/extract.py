"""Extract command for structured data extraction.

This module provides a CLI command for extracting structured data from web pages
using the Crawl4AI service. It supports both JSON schema-based extraction and
natural language prompt-based extraction.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import typer
from rich.console import Console

from crawl4r.services.extractor import ExtractorService

console = Console()


def extract_command(
    url: str = typer.Argument(..., help="URL to extract from"),
    schema: str | None = typer.Option(
        None,
        "--schema",
        help="Path to JSON schema file OR inline JSON schema string",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Natural language extraction prompt",
    ),
    output: Path | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file path for extracted JSON data",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="LLM provider (e.g., 'ollama/llama3', 'openai/gpt-4o-mini')",
    ),
) -> None:
    """Extract structured data from a web page.

    Use either --schema for JSON schema-based extraction or --prompt for
    natural language prompt-based extraction. Exactly one must be provided.

    Args:
        url: URL to extract data from.
        schema: Path to JSON schema file OR inline JSON schema string.
        prompt: Natural language extraction prompt.
        output: Optional output file for JSON data.
        provider: Optional LLM provider override.
    """
    # Validate exactly one of schema or prompt is provided
    if schema is None and prompt is None:
        console.print("[red]Error: Provide exactly one of --schema or --prompt[/red]")
        raise typer.Exit(code=1)

    if schema is not None and prompt is not None:
        console.print("[red]Error: Provide exactly one of --schema or --prompt[/red]")
        raise typer.Exit(code=1)

    async def _run() -> None:
        """Execute extraction and write JSON output."""
        endpoint_url = os.getenv("CRAWL4AI_BASE_URL", "http://localhost:52004")

        async with ExtractorService(endpoint_url=endpoint_url) as service:
            if schema is not None:
                # Try to parse as inline JSON first, else treat as file path
                schema_str = schema.strip()
                if schema_str.startswith("{"):
                    try:
                        schema_dict = json.loads(schema_str)
                    except json.JSONDecodeError as e:
                        console.print(
                            f"[red]Error: Invalid inline JSON schema: {e}[/red]"
                        )
                        raise typer.Exit(code=1)
                else:
                    # Treat as file path
                    schema_path = Path(schema_str)
                    try:
                        schema_dict = json.loads(schema_path.read_text())
                    except FileNotFoundError:
                        console.print(
                            f"[red]Error: Schema file not found: {schema_str}[/red]"
                        )
                        raise typer.Exit(code=1)
                    except json.JSONDecodeError as e:
                        console.print(
                            f"[red]Error: Invalid JSON in schema file: {e}[/red]"
                        )
                        raise typer.Exit(code=1)

                result = await service.extract_with_schema(
                    url, schema=schema_dict, provider=provider
                )
            else:
                result = await service.extract_with_prompt(
                    url, prompt=prompt or "", provider=provider
                )

            if not result.success:
                console.print(f"[red]Failed: {result.error}[/red]")
                raise typer.Exit(code=1)

            payload = json.dumps(result.data or {}, indent=2)

            if output is None:
                console.print(payload)
            else:
                output.write_text(payload + "\n")
                console.print(f"Wrote JSON to {output}")

    asyncio.run(_run())
