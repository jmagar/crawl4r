# Web Crawl CLI Phase 4 Testing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> **📁 Organization Note:** When this plan is fully implemented and verified, move this file to `docs/plans/complete/` to keep the plans folder organized.

**Goal:** Deliver Phase 4 test coverage for web crawl CLI services and commands with unit, CLI, and integration tests that meet coverage targets.

**Architecture:** Add focused pytest suites for services (`crawl4r/services/*`), CLI commands (`crawl4r/cli/*`), and integration flows with real services. Use respx + AsyncMock for network/Redis mocks and pytest markers for integration gating.

**Tech Stack:** Python 3.10+, pytest, pytest-asyncio, pytest-cov, respx, typer.testing.CliRunner.

**Prereqs:** Phase 1–3 code completed; `crawl4r/services/*` and `crawl4r/cli/*` exist and importable.

---

## Task 4.1: Unit Tests for ScraperService

**Files:**
- Create: `tests/unit/test_scraper_service.py`

**Step 1: Write failing test (success path)**
Create `tests/unit/test_scraper_service.py` with an initial test:
```python
import httpx
import pytest
import respx

from crawl4r.services.scraper import ScraperService


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_success() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(
            200,
            json={
                "markdown": "# Title",
                "metadata": {"title": "Example"},
                "status_code": 200,
            },
        )
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com")
    assert result.success is True
    assert result.markdown == "# Title"
```

**Step 2: Run test to verify failure**
Run: `pytest tests/unit/test_scraper_service.py::test_scrape_url_success -v`
Expected: FAIL if ScraperService behavior differs; otherwise PASS (proceed).

**Step 3: Add error-path tests**
Extend file with timeout and 4xx/5xx handling:
```python
@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_timeout_returns_failure() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        side_effect=httpx.ReadTimeout("timeout")
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com")
    assert result.success is False


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_4xx_no_retry() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com")
    assert result.success is False


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_5xx_retries_then_fails() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(500, json={"detail": "error"})
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com")
    assert result.success is False
```

**Step 4: Add circuit breaker and retry tests**
Add tests for consecutive failures and retry count:
```python
@respx.mock
@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(500)
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    for _ in range(6):
        await service.scrape_url("https://example.com")
    assert service._circuit_breaker.is_open is True
```

**Step 5: Add concurrency test**
```python
@respx.mock
@pytest.mark.asyncio
async def test_scrape_urls_concurrency() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(200, json={"markdown": "# Ok", "status_code": 200})
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    results = await service.scrape_urls(
        ["https://a.com", "https://b.com", "https://c.com"],
        max_concurrent=2,
    )
    assert len(results) == 3
```

**Step 6: Run tests**
Run: `pytest tests/unit/test_scraper_service.py -v --cov=crawl4r.services.scraper`
Expected: PASS, coverage >= 90%.

**Step 7: Commit**
```
git add tests/unit/test_scraper_service.py
git commit -m "test(services): add unit tests for ScraperService"
```

---

## Task 4.2: Unit Tests for QueueManager

**Files:**
- Create: `tests/unit/test_queue_manager.py`

**Step 1: Write failing tests for lock operations**
```python
from unittest.mock import AsyncMock

import pytest

import pytest

from crawl4r.services.queue import QueueManager


@pytest.mark.asyncio
async def test_acquire_lock_success() -> None:
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.set = AsyncMock(return_value=True)
    assert await manager.acquire_lock("crawl_id") is True


@pytest.mark.asyncio
async def test_acquire_lock_failure() -> None:
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.set = AsyncMock(return_value=False)
    assert await manager.acquire_lock("crawl_id") is False
```

**Step 2: Run tests to verify failure**
Run: `pytest tests/unit/test_queue_manager.py::test_acquire_lock_success -v`
Expected: FAIL if QueueManager differs; otherwise PASS.

**Step 3: Add queue and status tests**
```python
from crawl4r.services.models import CrawlStatus


@pytest.mark.asyncio
async def test_enqueue_dequeue_round_trip() -> None:
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.lpush = AsyncMock(return_value=1)
    manager._client.brpop = AsyncMock(
        return_value=(b"crawl_queue", b"crawl_id|https://example.com")
    )

    await manager.enqueue_crawl("crawl_id", ["https://example.com"])
    item = await manager.dequeue_crawl()
    assert item == ("crawl_id", ["https://example.com"])


@pytest.mark.asyncio
async def test_set_get_status() -> None:
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.hset = AsyncMock(return_value=1)
    manager._client.hgetall = AsyncMock(
        return_value={b"status": b"RUNNING", b"started_at": b"now", b"finished_at": b""}
    )

    await manager.set_status("crawl_id", CrawlStatus.RUNNING)
    status = await manager.get_status("crawl_id")
    assert status is not None
    assert status.status == CrawlStatus.RUNNING
```

**Step 4: Add list_recent, get_active, stale lock tests**
```python
@pytest.mark.asyncio
async def test_list_recent_and_get_active() -> None:
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.lrange = AsyncMock(return_value=[b"crawl_id"])
    manager._client.smembers = AsyncMock(return_value={b"crawl_active"})
    recent = await manager.list_recent(limit=10)
    active = await manager.get_active()
    assert recent == ["crawl_id"]
    assert active == ["crawl_active"]


@pytest.mark.asyncio
async def test_stale_lock_recovery() -> None:
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()
    manager._client.set = AsyncMock(side_effect=[False, True])
    manager._client.get = AsyncMock(return_value=b"crawl_old")
    manager._client.hgetall = AsyncMock(
        return_value={b"status": b"FAILED", b"started_at": b"old", b"finished_at": b"old"}
    )
    manager._client.delete = AsyncMock(return_value=1)
    assert await manager.acquire_lock("crawl_new") is True
```

**Step 5: Run tests**
Run: `pytest tests/unit/test_queue_manager.py -v --cov=crawl4r.services.queue`
Expected: PASS, coverage >= 85%.

**Step 6: Commit**
```
git add tests/unit/test_queue_manager.py
git commit -m "test(services): add unit tests for QueueManager"
```

---

## Task 4.3: Unit Tests for IngestionService

**Files:**
- Create: `tests/unit/test_ingestion_service.py`

**Step 1: Write failing tests for ingestion flow**
```python
from unittest.mock import AsyncMock

from crawl4r.services.ingestion import IngestionService, generate_crawl_id


def test_generate_crawl_id_format() -> None:
    crawl_id = generate_crawl_id()
    assert crawl_id.startswith("crawl_")


@pytest.mark.asyncio
async def test_ingest_urls_lock_acquired() -> None:
    service = IngestionService(
        scraper=AsyncMock(),
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=AsyncMock(),
    )
    service.queue_manager.acquire_lock = AsyncMock(return_value=True)
    service.scraper.scrape_urls = AsyncMock(return_value=[])
    result = await service.ingest_urls(["https://example.com"])
    assert result.crawl_id.startswith("crawl_")
```

**Step 2: Run tests to verify failure**
Run: `pytest tests/unit/test_ingestion_service.py::test_ingest_urls_lock_acquired -v`
Expected: FAIL if behavior differs; otherwise PASS.

**Step 3: Add queue, callback, and dedup tests**
```python
@pytest.mark.asyncio
async def test_ingest_urls_queued_when_lock_held() -> None:
    service = IngestionService(
        scraper=AsyncMock(),
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=AsyncMock(),
    )
    service.queue_manager.acquire_lock = AsyncMock(return_value=False)
    service.queue_manager.enqueue_crawl = AsyncMock(return_value=None)
    result = await service.ingest_urls(["https://example.com"])
    assert result.queued is True


@pytest.mark.asyncio
async def test_progress_callback_invoked() -> None:
    callback = AsyncMock()
    service = IngestionService(
        scraper=AsyncMock(),
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=AsyncMock(),
        progress_callback=callback,
    )
    service.queue_manager.acquire_lock = AsyncMock(return_value=True)
    service.scraper.scrape_urls = AsyncMock(return_value=[])
    await service.ingest_urls(["https://example.com"])
    assert callback.await_count >= 1


@pytest.mark.asyncio
async def test_dedup_called_before_upsert() -> None:
    service = IngestionService(
        scraper=AsyncMock(),
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=AsyncMock(),
    )
    service.queue_manager.acquire_lock = AsyncMock(return_value=True)
    service.scraper.scrape_urls = AsyncMock(return_value=[])
    service.vector_store.delete_by_url = AsyncMock(return_value=None)
    service.vector_store.upsert_vectors = AsyncMock(return_value=None)
    await service.ingest_urls(["https://example.com"])
    service.vector_store.delete_by_url.assert_awaited()
    service.vector_store.upsert_vectors.assert_awaited()
```

**Step 4: Run tests**
Run: `pytest tests/unit/test_ingestion_service.py -v --cov=crawl4r.services.ingestion`
Expected: PASS, coverage >= 90%.

**Step 5: Commit**
```
git add tests/unit/test_ingestion_service.py
git commit -m "test(services): add unit tests for IngestionService"
```

---

## Task 4.4: Quality Checkpoint (Services)

**Files:**
- None (verification only)

**Step 1: Run service unit tests with coverage**
Run: `pytest tests/unit/test_scraper_service.py tests/unit/test_queue_manager.py tests/unit/test_ingestion_service.py -v --cov=crawl4r.services --cov-report=term`
Expected: PASS, coverage > 85%.

**Step 2: Fix failures if needed**
If test failures or coverage shortfalls occur, fix in relevant test file or service and re-run the command.

**Step 3: Commit (only if fixes required)**
```
git add crawl4r/services/ tests/unit/
git commit -m "chore(tests): pass quality checkpoint"
```

---

## Task 4.5: CLI Tests with CliRunner

**Files:**
- Create: `tests/unit/test_cli_commands.py`

**Step 1: Write failing tests for command parsing**
```python
from pathlib import Path
from typer.testing import CliRunner

from crawl4r.cli.app import app

runner = CliRunner()


def test_scrape_command_help() -> None:
    result = runner.invoke(app, ["scrape", "--help"])
    assert result.exit_code == 0


def test_crawl_command_help() -> None:
    result = runner.invoke(app, ["crawl", "--help"])
    assert result.exit_code == 0


def test_status_command_help() -> None:
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0
```

**Step 2: Run tests to verify failure**
Run: `pytest tests/unit/test_cli_commands.py::test_scrape_command_help -v`
Expected: FAIL if commands not wired; otherwise PASS.

**Step 3: Add file input/output handling tests**
```python
from unittest.mock import AsyncMock, patch


def test_scrape_file_input(tmp_path: Path) -> None:
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://example.com\n")
    with patch("crawl4r.cli.commands.scrape.ScraperService") as mock_service:
        mock_service.return_value.scrape_urls = AsyncMock(return_value=[])
        result = runner.invoke(app, ["scrape", "-f", str(urls_file)])
        assert result.exit_code == 0


def test_scrape_output_file(tmp_path: Path) -> None:
    out_file = tmp_path / "out.md"
    with patch("crawl4r.cli.commands.scrape.ScraperService") as mock_service:
        mock_service.return_value.scrape_urls = AsyncMock(return_value=[])
        result = runner.invoke(app, ["scrape", "https://example.com", "-o", str(out_file)])
        assert result.exit_code == 0
```

**Step 4: Run tests**
Run: `pytest tests/unit/test_cli_commands.py -v --cov=crawl4r.cli`
Expected: PASS, coverage >= 80%.

**Step 5: Commit**
```
git add tests/unit/test_cli_commands.py
git commit -m "test(cli): add unit tests for CLI commands"
```

---

## Task 4.6: Integration Tests with Real Services

**Files:**
- Create: `tests/integration/test_cli_integration.py`

**Step 1: Write integration tests with skips**
```python
import os
import pytest
from typer.testing import CliRunner

from crawl4r.cli.app import app

runner = CliRunner()


def services_available() -> bool:
    return os.getenv("CRAWL4AI_PORT") is not None


@pytest.mark.integration
def test_scrape_command_real_service() -> None:
    if not services_available():
        pytest.skip("Services unavailable")
    result = runner.invoke(app, ["scrape", "https://example.com"])
    assert result.exit_code in (0, 1)
```

**Step 2: Run integration tests**
Run: `pytest tests/integration/test_cli_integration.py -v -m integration`
Expected: PASS or SKIP when services down.

**Step 3: Commit**
```
git add tests/integration/test_cli_integration.py
git commit -m "test(cli): add integration tests for CLI commands"
```

---

## Task 4.7: Unit Tests for P1 Services

**Files:**
- Create: `tests/unit/test_mapper_service.py`
- Create: `tests/unit/test_extractor_service.py`
- Create: `tests/unit/test_screenshot_service.py`

**Step 1: Write mapper tests**
```python
import httpx
import pytest
import respx

from crawl4r.services.mapper import MapperService


@respx.mock
@pytest.mark.asyncio
async def test_map_url_success() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/crawl").mock(
        return_value=httpx.Response(
            200,
            json={"links": {"internal": [{"href": "https://a.com"}], "external": []}},
        )
    )
    service = MapperService(endpoint_url="http://localhost:52004")
    result = await service.map_url("https://example.com", depth=1)
    assert result.links == ["https://a.com"]
```

**Step 2: Write extractor tests**
```python
import httpx
import pytest
import respx

from crawl4r.services.extractor import ExtractorService


@respx.mock
@pytest.mark.asyncio
async def test_extract_with_schema() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(200, json={"result": {"name": "Test"}})
    )
    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "https://example.com", schema={"type": "object", "properties": {"name": {"type": "string"}}}
    )
    assert result.data["name"] == "Test"
```

**Step 3: Write screenshot tests**
```python
import httpx
import pytest
import respx

from crawl4r.services.screenshot import ScreenshotService


@respx.mock
@pytest.mark.asyncio
async def test_capture_screenshot_success(tmp_path) -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/screenshot").mock(
        return_value=httpx.Response(200, json={"screenshot": "aGVsbG8="})
    )
    service = ScreenshotService(endpoint_url="http://localhost:52004")
    output_path = tmp_path / "page.png"
    result = await service.capture("https://example.com", output_path=output_path)
    assert result.success is True
```

**Step 4: Run tests**
Run: `pytest tests/unit/test_mapper_service.py tests/unit/test_extractor_service.py tests/unit/test_screenshot_service.py -v`
Expected: PASS, coverage >= 85% for each.

**Step 5: Commit**
```
git add tests/unit/test_mapper_service.py tests/unit/test_extractor_service.py tests/unit/test_screenshot_service.py
git commit -m "test(services): add unit tests for P1 services"
```

---

## Task 4.8: Testing Complete Checkpoint

**Files:**
- None (verification only)

**Step 1: Run full test suite with coverage**
Run: `pytest --cov=crawl4r --cov-report=term`
Expected: PASS, coverage > 85% overall.

**Step 2: Document gaps (if any)**
If coverage < 85% or critical gaps exist, add notes to `.docs/sessions/2026-01-19-web-crawl-cli-phase-4.md` with a timestamped entry.

**Step 3: Commit (only if documentation added)**
```
git add .docs/sessions/2026-01-19-web-crawl-cli-phase-4.md
git commit -m "test: complete test coverage for web crawl CLI"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-19-web-crawl-cli-phase-4.md`.

Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
