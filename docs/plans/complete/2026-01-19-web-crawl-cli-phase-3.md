# Web Crawl CLI Phase 3 (P1 Commands) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement MapperService/ExtractorService/ScreenshotService plus map/extract/screenshot CLI commands with TDD coverage for services and CLI.

**Architecture:** Add three service modules that mirror ScraperService patterns (httpx + CircuitBreaker + retries) and return MapResult/ExtractResult/ScreenshotResult. Add thin Typer commands that validate inputs, call services via asyncio, and format outputs. Tests use respx for HTTP mocks and Typer CliRunner for CLI behavior.

**Tech Stack:** Python 3.10+, httpx, respx, pytest-asyncio, Typer, Rich, Pydantic Settings, CircuitBreaker.

**Required Skills:** @superpowers:test-driven-development, @superpowers:testing-anti-patterns

**Docstrings:** Use Google-style docstrings for all new modules, classes, and functions.

---

## Task 0: Verify CLI dependencies and entry point exist

**Files:**
- Modify (if missing): `pyproject.toml`
- Create (if missing): `crawl4r/cli/app.py`

**Step 1: Verify Typer/Rich dependencies**
Check `pyproject.toml` for Typer and Rich. If missing, add to `[project.dependencies]`:
```toml
"typer[all]>=0.12.0",
"rich>=13.7.0",
```

**Step 2: Verify CLI entry point**
If `crawl4r/cli/app.py` does not exist, create it with the Typer app shell from `specs/web-crawl-cli/design.md` (register existing commands only).

**Step 3: Verify (no tests needed)**
Run: `rg -n "typer|rich" pyproject.toml`
Expected: Typer and Rich are listed in `[project.dependencies]`.

**Step 4: Commit (only if changes made)**
```bash
git add pyproject.toml crawl4r/cli/app.py
git commit -m "build(cli): add typer/rich and CLI app entrypoint"
```

---

## Task 1: MapperService tests (RED)

**Files:**
- Create: `tests/fixtures/crawl4ai_responses.py`
- Create: `tests/unit/services/__init__.py`
- Create: `tests/unit/services/test_mapper_service.py`

**Step 1: Add fixture data**
Create `tests/fixtures/crawl4ai_responses.py`:
```python
MOCK_CRAWL_RESULT_SUCCESS = {
    "links": {
        "internal": [{"href": "/a"}, {"href": "/b"}, {"href": "/c"}],
        "external": [{"href": "https://www.iana.org/domains/example"}, {"href": "https://example.net"}],
    }
}
```

**Step 2: Write failing tests**
Create `tests/unit/services/test_mapper_service.py`:
```python
import httpx
import pytest
import respx

from tests.fixtures.crawl4ai_responses import MOCK_CRAWL_RESULT_SUCCESS
from crawl4r.services.mapper import MapperService


@respx.mock
@pytest.mark.asyncio
async def test_map_url_same_domain_filters_and_counts() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(200, json=MOCK_CRAWL_RESULT_SUCCESS)
    )

    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("https://example.com", depth=0, same_domain=True)

    assert result.success is True
    assert result.internal_count == 3
    assert result.external_count == 2
    assert all(link.startswith("https://example.com") for link in result.links)
    assert result.depth_reached == 0


@respx.mock
@pytest.mark.asyncio
async def test_map_url_includes_external_when_requested() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(200, json=MOCK_CRAWL_RESULT_SUCCESS)
    )

    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("https://example.com", depth=0, same_domain=False)

    assert result.success is True
    assert result.internal_count == 3
    assert result.external_count == 2
    assert "https://www.iana.org/domains/example" in result.links


@respx.mock
@pytest.mark.asyncio
async def test_map_url_depth_recurses_and_dedupes() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    route = respx.post("http://localhost:52004/crawl")
    route.side_effect = [
        httpx.Response(
            200,
            json={
                "links": {
                    "internal": [{"href": "/about"}, {"href": "/about"}],
                    "external": [],
                }
            },
        ),
        httpx.Response(
            200,
            json={
                "links": {
                    "internal": [{"href": "/team"}],
                    "external": [],
                }
            },
        ),
    ]

    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("https://example.com", depth=1, same_domain=True)

    assert result.success is True
    assert "https://example.com/about" in result.links
    assert "https://example.com/team" in result.links
    assert result.links.count("https://example.com/about") == 1
    assert result.depth_reached == 1
```

**Step 3: Run tests to verify failure**
Run: `pytest tests/unit/services/test_mapper_service.py -v`
Expected: FAIL (module not found).

**Step 4: Commit**
```bash
git add tests/fixtures/crawl4ai_responses.py tests/unit/services/__init__.py \
  tests/unit/services/test_mapper_service.py
git commit -m "test(services): add MapperService coverage"
```

---

## Task 2: Implement MapperService (GREEN)

**Files:**
- Create: `crawl4r/services/mapper.py`
- Modify: `crawl4r/services/__init__.py`

**Step 1: Implement MapperService**
Create `crawl4r/services/mapper.py`:
```python
"""URL discovery service using Crawl4AI /crawl endpoint."""

from __future__ import annotations

import asyncio
from collections import deque
from urllib.parse import urljoin, urlparse

import httpx

from crawl4r.core.config import Settings
from crawl4r.core.logger import get_logger
from crawl4r.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerError
from crawl4r.services.models import MapResult


class MapperService:
    """Service for URL discovery via Crawl4AI."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delays: list[float] | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize mapper service.

        Args:
            endpoint_url: Crawl4AI base URL.
            timeout: HTTP timeout in seconds.
            max_retries: Maximum retry attempts.
            retry_delays: Backoff delays in seconds.
            settings: Optional Settings instance.
        """
        if settings is None:
            settings = Settings()  # type: ignore[call-arg]

        self.endpoint_url = (endpoint_url or settings.CRAWL4AI_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delays = retry_delays or [1.0, 2.0, 4.0]
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
        self._logger = get_logger("crawl4r.services.mapper")

    async def map_url(
        self, url: str, depth: int = 0, same_domain: bool = True
    ) -> MapResult:
        """Discover links for a URL up to a specified depth.

        Args:
            url: URL to map.
            depth: Maximum recursion depth.
            same_domain: When True, only keep same-domain links.

        Returns:
            MapResult: Mapping result with discovered links.
        """
        async def _impl() -> MapResult:
            """Wrap map implementation for circuit breaker.

            Returns:
                MapResult: Mapping result.
            """
            return await self._map_impl(url, depth=depth, same_domain=same_domain)

        try:
            return await self._circuit_breaker.call(_impl)
        except CircuitBreakerError as e:
            self._logger.error(f"Circuit breaker open for {url}")
            return MapResult(url=url, success=False, error=f"Service unavailable: {e}")
        except Exception as e:
            return MapResult(url=url, success=False, error=str(e))

    async def _map_impl(self, url: str, depth: int, same_domain: bool) -> MapResult:
        """Internal mapping implementation with retries.

        Args:
            url: URL to map.
            depth: Maximum recursion depth.
            same_domain: When True, only keep same-domain links.

        Returns:
            MapResult: Mapping result with discovered links.
        """
        base_domain = urlparse(url).netloc
        if not base_domain:
            raise ValueError("Invalid URL")

        visited: set[str] = set()
        discovered: list[str] = []
        internal_count = 0
        external_count = 0
        max_depth_reached = 0

        queue: deque[tuple[str, int]] = deque([(url, 0)])

        while queue:
            current_url, current_depth = queue.popleft()
            if current_url in visited:
                continue
            visited.add(current_url)

            links_internal: list[str] = []
            links_external: list[str] = []

            for attempt in range(self.max_retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(
                            f"{self.endpoint_url}/crawl",
                            json={"url": current_url},
                        )
                        response.raise_for_status()
                        data = response.json()
                        links = data.get("links", {})
                        links_internal = [
                            urljoin(current_url, item.get("href", ""))
                            for item in links.get("internal", [])
                            if item.get("href")
                        ]
                        links_external = [
                            urljoin(current_url, item.get("href", ""))
                            for item in links.get("external", [])
                            if item.get("href")
                        ]
                        break
                except httpx.HTTPStatusError as e:
                    if 400 <= e.response.status_code < 500:
                        raise
                    if attempt >= self.max_retries:
                        raise
                except (httpx.ConnectError, httpx.TimeoutException):
                    if attempt >= self.max_retries:
                        raise
                if attempt < len(self.retry_delays):
                    await asyncio.sleep(self.retry_delays[attempt])

            internal_count += len(links_internal)
            external_count += len(links_external)

            combined_links = links_internal + links_external
            for link in combined_links:
                if link not in discovered:
                    if same_domain and urlparse(link).netloc != base_domain:
                        continue
                    discovered.append(link)

            if current_depth < depth:
                for link in combined_links:
                    if same_domain and urlparse(link).netloc != base_domain:
                        continue
                    if link not in visited:
                        queue.append((link, current_depth + 1))
                        max_depth_reached = max(max_depth_reached, current_depth + 1)

        return MapResult(
            url=url,
            success=True,
            links=discovered,
            internal_count=internal_count,
            external_count=external_count,
            depth_reached=max_depth_reached,
        )
```

**Step 2: Export service**
Update `crawl4r/services/__init__.py` to export `MapperService` if the file exists (or create it if missing).

**Step 3: Run tests to verify pass**
Run: `pytest tests/unit/services/test_mapper_service.py -v`
Expected: PASS.

**Step 4: Commit**
```bash
git add crawl4r/services/mapper.py crawl4r/services/__init__.py
git commit -m "feat(services): implement MapperService for URL discovery"
```

---

## Task 3: Map command tests (RED)

**Files:**
- Create: `tests/unit/cli/__init__.py`
- Create: `tests/unit/cli/test_map_command.py`

**Step 1: Write failing tests**
Create `tests/unit/cli/test_map_command.py`:
```python
from typer.testing import CliRunner

from crawl4r.cli.app import app
from crawl4r.services.models import MapResult


runner = CliRunner()


def test_map_command_writes_stdout(monkeypatch) -> None:
    async def _fake_map_url(self, url: str, depth: int = 0, same_domain: bool = True):
        return MapResult(
            url=url,
            success=True,
            links=["https://example.com/a", "https://example.com/b"],
            internal_count=2,
            external_count=0,
            depth_reached=0,
        )

    monkeypatch.setattr("crawl4r.cli.commands.map.MapperService.map_url", _fake_map_url)

    result = runner.invoke(app, ["map", "https://example.com"])
    assert result.exit_code == 0
    assert "https://example.com/a" in result.output
    assert "Unique URLs: 2" in result.output


def test_map_command_writes_file(tmp_path, monkeypatch) -> None:
    async def _fake_map_url(self, url: str, depth: int = 0, same_domain: bool = True):
        return MapResult(
            url=url,
            success=True,
            links=["https://example.com/a"],
            internal_count=1,
            external_count=0,
            depth_reached=0,
        )

    monkeypatch.setattr("crawl4r.cli.commands.map.MapperService.map_url", _fake_map_url)

    output_path = tmp_path / "urls.txt"
    result = runner.invoke(app, ["map", "https://example.com", "-o", str(output_path)])
    assert result.exit_code == 0
    assert output_path.read_text().strip() == "https://example.com/a"
```

**Step 2: Run tests to verify failure**
Run: `pytest tests/unit/cli/test_map_command.py -v`
Expected: FAIL (command module missing).

**Step 3: Commit**
```bash
git add tests/unit/cli/__init__.py tests/unit/cli/test_map_command.py
git commit -m "test(cli): add map command coverage"
```

---

## Task 4: Implement map command (GREEN)

**Files:**
- Create: `crawl4r/cli/commands/map.py`
- Modify: `crawl4r/cli/app.py`

**Step 1: Implement command**
Create `crawl4r/cli/commands/map.py`:
```python
"""Map command for URL discovery."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from crawl4r.services.mapper import MapperService


app = typer.Typer(help="Discover URLs from a web page", invoke_without_command=True)
console = Console()


@app.callback()
def map_command(
    url: Annotated[str, typer.Argument(help="URL to map")],
    depth: Annotated[int, typer.Option(0, "--depth", help="Max crawl depth")],
    same_domain: Annotated[
        bool,
        typer.Option(
            True,
            "--same-domain/--include-external",
            help="Restrict mapping to same-domain links",
        ),
    ],
    output: Annotated[Path | None, typer.Option("-o", "--output")] = None,
) -> None:
    """Run map command and output discovered URLs.

    Args:
        url: URL to map.
        depth: Maximum recursion depth.
        same_domain: When True, only keep same-domain links.
        output: Optional output file for URLs.
    """

    async def _run() -> None:
        """Execute map request and write output."""
        service = MapperService()
        result = await service.map_url(url, depth=depth, same_domain=same_domain)
        if not result.success:
            raise typer.Exit(code=1)

        lines = result.links
        if output is None:
            for link in lines:
                console.print(link)
            console.print(f"Unique URLs: {len(lines)}")
        else:
            output.write_text("\n".join(lines) + "\n")
            console.print(f"Wrote {len(lines)} URLs to {output}")

    asyncio.run(_run())
```

**Step 2: Register command**
Update `crawl4r/cli/app.py` to import `map` module as `map_urls` and register it:
```python
from crawl4r.cli.commands import map as map_urls
app.add_typer(map_urls.app, name="map")
```

**Step 3: Run tests to verify pass**
Run: `pytest tests/unit/cli/test_map_command.py -v`
Expected: PASS.

**Step 4: Commit**
```bash
git add crawl4r/cli/commands/map.py crawl4r/cli/app.py
git commit -m "feat(cli): implement map command for URL discovery"
```

---

## Task 5: ExtractorService tests (RED)

**Files:**
- Create: `tests/unit/services/test_extractor_service.py`

**Step 1: Write failing tests**
Create `tests/unit/services/test_extractor_service.py`:
```python
import httpx
import pytest
import respx

from crawl4r.services.extractor import ExtractorService


@respx.mock
@pytest.mark.asyncio
async def test_extract_with_schema_calls_llm_job() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(200, json={"data": {"title": "Example"}})
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "https://example.com", schema={"type": "object"}
    )

    assert result.success is True
    assert result.data == {"title": "Example"}


@respx.mock
@pytest.mark.asyncio
async def test_extract_with_prompt_calls_llm_job() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(200, json={"data": {"heading": "Hello"}})
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_prompt(
        "https://example.com", prompt="extract heading"
    )

    assert result.success is True
    assert result.data == {"heading": "Hello"}
```

**Step 2: Run tests to verify failure**
Run: `pytest tests/unit/services/test_extractor_service.py -v`
Expected: FAIL (module not found).

**Step 3: Commit**
```bash
git add tests/unit/services/test_extractor_service.py
git commit -m "test(services): add ExtractorService coverage"
```

---

## Task 6: Implement ExtractorService (GREEN)

**Files:**
- Create: `crawl4r/services/extractor.py`
- Modify: `crawl4r/services/__init__.py`

**Step 1: Implement ExtractorService**
Create `crawl4r/services/extractor.py`:
```python
"""Structured data extraction service using Crawl4AI /llm/job."""

from __future__ import annotations

import asyncio

import httpx

from crawl4r.core.config import Settings
from crawl4r.core.logger import get_logger
from crawl4r.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerError
from crawl4r.services.models import ExtractResult


class ExtractorService:
    """Service for structured data extraction via Crawl4AI /llm/job."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delays: list[float] | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize extractor service.

        Args:
            endpoint_url: Crawl4AI base URL.
            timeout: HTTP timeout in seconds.
            max_retries: Maximum retry attempts.
            retry_delays: Backoff delays in seconds.
            settings: Optional Settings instance.
        """
        if settings is None:
            settings = Settings()  # type: ignore[call-arg]

        self.endpoint_url = (endpoint_url or settings.CRAWL4AI_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delays = retry_delays or [1.0, 2.0, 4.0]
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
        self._logger = get_logger("crawl4r.services.extractor")

    async def extract_with_schema(self, url: str, schema: dict) -> ExtractResult:
        """Extract structured data using a JSON schema.

        Args:
            url: URL to extract from.
            schema: JSON schema payload.

        Returns:
            ExtractResult: Extraction result.
        """
        async def _impl() -> ExtractResult:
            """Wrap schema extraction for circuit breaker.

            Returns:
                ExtractResult: Extraction result.
            """
            return await self._extract_impl(url, payload={"url": url, "schema": schema})

        try:
            return await self._circuit_breaker.call(_impl)
        except CircuitBreakerError as e:
            self._logger.error(f"Circuit breaker open for {url}")
            return ExtractResult(url=url, success=False, error=f"Service unavailable: {e}")
        except Exception as e:
            return ExtractResult(url=url, success=False, error=str(e))

    async def extract_with_prompt(self, url: str, prompt: str) -> ExtractResult:
        """Extract structured data using an LLM prompt.

        Args:
            url: URL to extract from.
            prompt: Extraction prompt.

        Returns:
            ExtractResult: Extraction result.
        """
        async def _impl() -> ExtractResult:
            """Wrap prompt extraction for circuit breaker.

            Returns:
                ExtractResult: Extraction result.
            """
            return await self._extract_impl(url, payload={"url": url, "prompt": prompt})

        try:
            return await self._circuit_breaker.call(_impl)
        except CircuitBreakerError as e:
            self._logger.error(f"Circuit breaker open for {url}")
            return ExtractResult(url=url, success=False, error=f"Service unavailable: {e}")
        except Exception as e:
            return ExtractResult(url=url, success=False, error=str(e))

    async def _extract_impl(self, url: str, payload: dict) -> ExtractResult:
        """Internal extraction implementation with retries.

        Args:
            url: URL to extract from.
            payload: Request payload for extraction.

        Returns:
            ExtractResult: Extraction result.
        """
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.endpoint_url}/llm/job",
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json().get("data")
                    if data is None:
                        data = response.json().get("result")
                    return ExtractResult(url=url, success=True, data=data)
            except httpx.HTTPStatusError as e:
                if 400 <= e.response.status_code < 500:
                    raise
                if attempt >= self.max_retries:
                    raise
            except (httpx.ConnectError, httpx.TimeoutException):
                if attempt >= self.max_retries:
                    raise
            if attempt < len(self.retry_delays):
                await asyncio.sleep(self.retry_delays[attempt])
        raise RuntimeError("Extraction failed")
```

**Step 2: Export service**
Update `crawl4r/services/__init__.py` to export `ExtractorService`.

**Step 3: Run tests to verify pass**
Run: `pytest tests/unit/services/test_extractor_service.py -v`
Expected: PASS.

**Step 4: Commit**
```bash
git add crawl4r/services/extractor.py crawl4r/services/__init__.py
git commit -m "feat(services): implement ExtractorService for structured extraction"
```

---

## Task 7: Extract command tests (RED)

**Files:**
- Create: `tests/unit/cli/test_extract_command.py`

**Step 1: Write failing tests**
Create `tests/unit/cli/test_extract_command.py`:
```python
import json

from typer.testing import CliRunner

from crawl4r.cli.app import app
from crawl4r.services.models import ExtractResult


runner = CliRunner()


def test_extract_command_schema_file(tmp_path, monkeypatch) -> None:
    async def _fake_extract_with_schema(self, url: str, schema: dict):
        return ExtractResult(url=url, success=True, data={"title": "Example"})

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


def test_extract_command_prompt(tmp_path, monkeypatch) -> None:
    async def _fake_extract_with_prompt(self, url: str, prompt: str):
        return ExtractResult(url=url, success=True, data={"heading": "Hello"})

    monkeypatch.setattr(
        "crawl4r.cli.commands.extract.ExtractorService.extract_with_prompt",
        _fake_extract_with_prompt,
    )

    result = runner.invoke(app, ["extract", "https://example.com", "--prompt", "extract heading"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["heading"] == "Hello"


def test_extract_command_requires_schema_or_prompt() -> None:
    result = runner.invoke(app, ["extract", "https://example.com"])
    assert result.exit_code != 0
```

**Step 2: Run tests to verify failure**
Run: `pytest tests/unit/cli/test_extract_command.py -v`
Expected: FAIL (command module missing).

**Step 3: Commit**
```bash
git add tests/unit/cli/test_extract_command.py
git commit -m "test(cli): add extract command coverage"
```

---

## Task 8: Implement extract command (GREEN)

**Files:**
- Create: `crawl4r/cli/commands/extract.py`
- Modify: `crawl4r/cli/app.py`

**Step 1: Implement command**
Create `crawl4r/cli/commands/extract.py`:
```python
"""Extract command for structured data."""

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from crawl4r.services.extractor import ExtractorService


app = typer.Typer(help="Extract structured data from a web page", invoke_without_command=True)
console = Console()


@app.callback()
def extract_command(
    url: Annotated[str, typer.Argument(help="URL to extract from")],
    schema_path: Annotated[
        Path | None,
        typer.Option("--schema", help="Path to JSON schema"),
    ] = None,
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", help="LLM extraction prompt"),
    ] = None,
    output: Annotated[Path | None, typer.Option("-o", "--output")] = None,
) -> None:
    """Run extraction using schema or prompt and output JSON.

    Args:
        url: URL to extract from.
        schema_path: Optional path to JSON schema file.
        prompt: Optional LLM extraction prompt.
        output: Optional output file for JSON data.
    """
    if (schema_path is None and prompt is None) or (schema_path and prompt):
        raise typer.BadParameter("Provide exactly one of --schema or --prompt")

    async def _run() -> None:
        """Execute extraction and write JSON output."""
        service = ExtractorService()
        if schema_path is not None:
            schema = json.loads(schema_path.read_text())
            result = await service.extract_with_schema(url, schema=schema)
        else:
            result = await service.extract_with_prompt(url, prompt=prompt or "")

        if not result.success:
            raise typer.Exit(code=1)

        payload = json.dumps(result.data or {}, indent=2)
        if output is None:
            console.print(payload)
        else:
            output.write_text(payload + "\n")
            console.print(f"Wrote JSON to {output}")

    asyncio.run(_run())
```

**Step 2: Register command**
Update `crawl4r/cli/app.py` to import `extract` and register it:
```python
from crawl4r.cli.commands import extract
app.add_typer(extract.app, name="extract")
```

**Step 3: Run tests to verify pass**
Run: `pytest tests/unit/cli/test_extract_command.py -v`
Expected: PASS.

**Step 4: Commit**
```bash
git add crawl4r/cli/commands/extract.py crawl4r/cli/app.py
git commit -m "feat(cli): implement extract command for structured data"
```

---

## Task 9: ScreenshotService tests (RED)

**Files:**
- Create: `tests/unit/services/test_screenshot_service.py`

**Step 1: Write failing tests**
Create `tests/unit/services/test_screenshot_service.py`:
```python
import base64

import httpx
import pytest
import respx

from crawl4r.services.screenshot import ScreenshotService


@respx.mock
@pytest.mark.asyncio
async def test_screenshot_saves_png(tmp_path) -> None:
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
```

**Step 2: Run tests to verify failure**
Run: `pytest tests/unit/services/test_screenshot_service.py -v`
Expected: FAIL (module not found).

**Step 3: Commit**
```bash
git add tests/unit/services/test_screenshot_service.py
git commit -m "test(services): add ScreenshotService coverage"
```

---

## Task 10: Implement ScreenshotService (GREEN)

**Files:**
- Create: `crawl4r/services/screenshot.py`
- Modify: `crawl4r/services/__init__.py`

**Step 1: Implement ScreenshotService**
Create `crawl4r/services/screenshot.py`:
```python
"""Screenshot capture service using Crawl4AI /screenshot."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import httpx

from crawl4r.core.config import Settings
from crawl4r.core.logger import get_logger
from crawl4r.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerError
from crawl4r.services.models import ScreenshotResult


class ScreenshotService:
    """Service for screenshot capture via Crawl4AI /screenshot."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delays: list[float] | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize screenshot service.

        Args:
            endpoint_url: Crawl4AI base URL.
            timeout: HTTP timeout in seconds.
            max_retries: Maximum retry attempts.
            retry_delays: Backoff delays in seconds.
            settings: Optional Settings instance.
        """
        if settings is None:
            settings = Settings()  # type: ignore[call-arg]

        self.endpoint_url = (endpoint_url or settings.CRAWL4AI_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delays = retry_delays or [1.0, 2.0, 4.0]
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
        self._logger = get_logger("crawl4r.services.screenshot")

    async def capture(
        self,
        url: str,
        output_path: Path,
        full_page: bool = False,
        wait: int = 0,
    ) -> ScreenshotResult:
        """Capture a screenshot for a URL and write it to disk.

        Args:
            url: URL to capture.
            output_path: Path for saving the screenshot.
            full_page: Whether to capture full page.
            wait: Seconds to wait before capture.

        Returns:
            ScreenshotResult: Capture result with file info.
        """
        async def _impl() -> ScreenshotResult:
            """Wrap capture implementation for circuit breaker.

            Returns:
                ScreenshotResult: Capture result.
            """
            return await self._capture_impl(
                url,
                output_path=output_path,
                full_page=full_page,
                wait=wait,
            )

        try:
            return await self._circuit_breaker.call(_impl)
        except CircuitBreakerError as e:
            self._logger.error(f"Circuit breaker open for {url}")
            return ScreenshotResult(url=url, success=False, error=f"Service unavailable: {e}")
        except Exception as e:
            return ScreenshotResult(url=url, success=False, error=str(e))

    async def _capture_impl(
        self, url: str, output_path: Path, full_page: bool, wait: int
    ) -> ScreenshotResult:
        """Internal screenshot capture with retries.

        Args:
            url: URL to capture.
            output_path: Path for saving the screenshot.
            full_page: Whether to capture full page.
            wait: Seconds to wait before capture.

        Returns:
            ScreenshotResult: Capture result with file info.
        """
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.endpoint_url}/screenshot",
                        json={
                            "url": url,
                            "full_page": full_page,
                            "wait": wait,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    encoded = payload.get("screenshot") or payload.get("image")
                    if not encoded:
                        raise ValueError("Missing screenshot data")

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(base64.b64decode(encoded))
                    file_size = output_path.stat().st_size

                    return ScreenshotResult(
                        url=url,
                        success=True,
                        file_path=str(output_path),
                        file_size=file_size,
                    )
            except httpx.HTTPStatusError as e:
                if 400 <= e.response.status_code < 500:
                    raise
                if attempt >= self.max_retries:
                    raise
            except (httpx.ConnectError, httpx.TimeoutException):
                if attempt >= self.max_retries:
                    raise
            if attempt < len(self.retry_delays):
                await asyncio.sleep(self.retry_delays[attempt])
        raise RuntimeError("Screenshot capture failed")
```

**Step 2: Export service**
Update `crawl4r/services/__init__.py` to export `ScreenshotService`.

**Step 3: Run tests to verify pass**
Run: `pytest tests/unit/services/test_screenshot_service.py -v`
Expected: PASS.

**Step 4: Commit**
```bash
git add crawl4r/services/screenshot.py crawl4r/services/__init__.py
git commit -m "feat(services): implement ScreenshotService for page capture"
```

---

## Task 11: Screenshot command tests (RED)

**Files:**
- Create: `tests/unit/cli/test_screenshot_command.py`

**Step 1: Write failing tests**
Create `tests/unit/cli/test_screenshot_command.py`:
```python
from typer.testing import CliRunner

from crawl4r.cli.app import app
from crawl4r.services.models import ScreenshotResult


runner = CliRunner()


def test_screenshot_command_default_name(tmp_path, monkeypatch) -> None:
    async def _fake_capture(self, url: str, output_path, full_page: bool = False, wait: int = 0):
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
    assert "example.com.png" in result.output


def test_screenshot_command_custom_output(tmp_path, monkeypatch) -> None:
    async def _fake_capture(self, url: str, output_path, full_page: bool = False, wait: int = 0):
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
    result = runner.invoke(app, ["screenshot", "https://example.com", "-o", str(output_path)])
    assert result.exit_code == 0
    assert str(output_path) in result.output
```

**Step 2: Run tests to verify failure**
Run: `pytest tests/unit/cli/test_screenshot_command.py -v`
Expected: FAIL (command module missing).

**Step 3: Commit**
```bash
git add tests/unit/cli/test_screenshot_command.py
git commit -m "test(cli): add screenshot command coverage"
```

---

## Task 12: Implement screenshot command (GREEN)

**Files:**
- Create: `crawl4r/cli/commands/screenshot.py`
- Modify: `crawl4r/cli/app.py`

**Step 1: Implement command**
Create `crawl4r/cli/commands/screenshot.py`:
```python
"""Screenshot command for web pages."""

import asyncio
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import typer
from rich.console import Console

from crawl4r.services.screenshot import ScreenshotService


app = typer.Typer(help="Capture screenshots of web pages", invoke_without_command=True)
console = Console()


def _default_output_path(url: str) -> Path:
    """Generate default screenshot output path from URL.

    Args:
        url: Source URL.

    Returns:
        Path: Default screenshot output path.
    """
    netloc = urlparse(url).netloc or "screenshot"
    return Path(f"{netloc}.png")


@app.callback()
def screenshot_command(
    url: Annotated[str, typer.Argument(help="URL to capture")],
    output: Annotated[Path | None, typer.Option("-o", "--output")] = None,
    full_page: Annotated[
        bool,
        typer.Option("--full-page", help="Capture full page"),
    ] = False,
    wait: Annotated[int, typer.Option("--wait", help="Wait seconds before capture")] = 0,
) -> None:
    """Run screenshot capture and write PNG.

    Args:
        url: URL to capture.
        output: Optional output file path.
        full_page: Whether to capture full page.
        wait: Seconds to wait before capture.
    """
    if output is None:
        output = _default_output_path(url)

    async def _run() -> None:
        """Execute screenshot capture and report result."""
        service = ScreenshotService()
        result = await service.capture(url, output_path=output, full_page=full_page, wait=wait)
        if not result.success:
            raise typer.Exit(code=1)
        console.print(f"Saved {result.file_size} bytes to {result.file_path}")

    asyncio.run(_run())
```

**Step 2: Register command**
Update `crawl4r/cli/app.py` to import `screenshot` and register it:
```python
from crawl4r.cli.commands import screenshot
app.add_typer(screenshot.app, name="screenshot")
```

**Step 3: Run tests to verify pass**
Run: `pytest tests/unit/cli/test_screenshot_command.py -v`
Expected: PASS.

**Step 4: Commit**
```bash
git add crawl4r/cli/commands/screenshot.py crawl4r/cli/app.py
git commit -m "feat(cli): implement screenshot command for page capture"
```

---

## Task 13: Quality checkpoints (lint + type check)

**Files:**
- Modify as needed: newly created service/CLI files and tests

**Step 1: Ruff**
Run: `ruff check crawl4r/services crawl4r/cli/commands tests/unit/services tests/unit/cli`
Expected: No lint errors.

**Step 2: Ty**
Run: `ty check crawl4r/services crawl4r/cli/commands tests/unit/services tests/unit/cli`
Expected: No type errors.

**Step 3: Commit (only if fixes needed)**
```bash
git add crawl4r/services crawl4r/cli/commands tests/unit/services tests/unit/cli
git commit -m "chore(services): pass quality checkpoint"
```

---

## Task 14: CLI verification (manual)

**Files:**
- None

**Step 1: Verify CLI help**
Run: `python -m crawl4r.cli.app map --help`
Expected: Map command help displays.

**Step 2: Verify extract help**
Run: `python -m crawl4r.cli.app extract --help`
Expected: Extract command help displays.

**Step 3: Verify screenshot help**
Run: `python -m crawl4r.cli.app screenshot --help`
Expected: Screenshot command help displays.

**Step 4: Commit**
No commit unless code changes required.

---

## Task 15: P1 command smoke checks (optional with real services)

**Files:**
- None

**Step 1: Map**
Run: `python -m crawl4r.cli.app map https://example.com`
Expected: Lists URLs and total count.

**Step 2: Extract**
Run: `python -m crawl4r.cli.app extract https://example.com --prompt "extract main heading"`
Expected: JSON output printed.

**Step 3: Screenshot**
Run: `python -m crawl4r.cli.app screenshot https://example.com`
Expected: PNG file saved and size reported.
```
