# Web Crawl CLI Phase 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> **📁 Organization Note:** When this plan is fully implemented and verified, move this file to `docs/plans/complete/` to keep the plans folder organized.

**Goal:** Implement Phase 2 P0 CLI behavior: watch command refactor, graceful crawl shutdown, lock recovery, error handling, URL validation, and service startup validation.

**Architecture:** Keep Typer CLI commands thin; push logic into services. Reuse existing FileWatcher/StateRecovery pipeline for watch; reuse CircuitBreaker and Crawl4AIReader URL validation pattern in services. Redis queue coordination remains optional; when unavailable, fall back to local-only operation.

**Tech Stack:** Python 3.10+, Typer, httpx, asyncio, redis.asyncio, Pydantic settings, pytest, ruff, ty.

---

## Assumptions & Preconditions

- Phase 1 is complete (services and CLI scaffolding exist). Required files:
  - `crawl4r/cli/app.py`, `crawl4r/cli/commands/scrape.py`, `crawl4r/cli/commands/crawl.py`, `crawl4r/cli/commands/status.py`
  - `crawl4r/services/scraper.py`, `crawl4r/services/ingestion.py`, `crawl4r/services/queue.py`, `crawl4r/services/models.py`
- If any are missing, stop and execute the Phase 1 plan in `docs/plans/2026-01-19-web-crawl-cli-phase-1.md` first.

---

### Task 0: Phase 1 Presence Check (Fast Guardrail)

**Files:**
- Read: `crawl4r/cli/app.py`
- Read: `crawl4r/cli/commands/crawl.py`
- Read: `crawl4r/services/queue.py`
- Read: `crawl4r/services/scraper.py`
- Read: `crawl4r/services/ingestion.py`

**Step 1: Verify files exist**

Run:
```bash
ls crawl4r/cli/app.py crawl4r/cli/commands/crawl.py crawl4r/services/queue.py crawl4r/services/scraper.py crawl4r/services/ingestion.py
```
Expected: All files listed.

**Step 2: If any missing, stop**

Note in the log: Phase 1 incomplete. Execute the Phase 1 plan before continuing.

---

### Task 1: Watch Command Refactor (2.1)

**Files:**
- Create: `crawl4r/cli/commands/watch.py`
- Modify: `crawl4r/cli/app.py`
- Read: `crawl4r/cli/main.py`
- Read: `crawl4r/readers/file_watcher.py`
- Test: `tests/unit/test_watch_command.py` (new)

**Step 1: Write failing unit test for watch command**

Create `tests/unit/test_watch_command.py`:
```python
from typer.testing import CliRunner

from crawl4r.cli.app import app


def test_watch_help_shows_options() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["watch", "--help"])
    assert result.exit_code == 0
    assert "--folder" in result.output
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/unit/test_watch_command.py::test_watch_help_shows_options -v
```
Expected: FAIL (watch command not registered).

**Step 3: Implement watch command by refactoring main.py**

Create `crawl4r/cli/commands/watch.py` and copy/refactor logic from `crawl4r/cli/main.py` into a Typer command. Required behavior:
- `--folder` overrides `Settings.watch_folder`.
- Perform startup batch recovery (StateRecovery + processor.process_batch).
- Start FileWatcher/Observer and block until Ctrl+C.
- Log event count and queue depth (reuse `FileWatcher` queue if available).
- Graceful shutdown on Ctrl+C.

Skeleton to adapt:
```python
import asyncio
from pathlib import Path
import typer

from crawl4r.core.config import Settings
from crawl4r.core.llama_settings import configure_llama_settings
from crawl4r.core.quality import QualityVerifier
from crawl4r.processing.processor import DocumentProcessor
from crawl4r.readers.file_watcher import FileWatcher
from crawl4r.resilience.recovery import StateRecovery
from crawl4r.storage.qdrant import VectorStoreManager
from crawl4r.storage.tei import TEIClient

app = typer.Typer(invoke_without_command=True)

@app.callback()
def watch(
    folder: Path | None = typer.Option(None, "--folder", help="Override watch folder"),
) -> None:
    asyncio.run(_watch_async(folder))

async def _watch_async(folder: Path | None) -> None:
    config = Settings()  # type: ignore[call-arg]
    if folder is not None:
        config.watch_folder = folder
    configure_llama_settings(app_settings=config)
    # initialize components, run validations, state recovery, batch, observer
```

**Step 4: Register command in app.py**

In `crawl4r/cli/app.py`, import and add the `watch` command to the Typer app.

**Step 5: Run test to verify it passes**

Run:
```bash
pytest tests/unit/test_watch_command.py::test_watch_help_shows_options -v
```
Expected: PASS.

**Step 6: Commit**

```bash
git add crawl4r/cli/commands/watch.py crawl4r/cli/app.py tests/unit/test_watch_command.py
git commit -m "feat(cli): implement watch command refactored from main.py"
```

---

### Task 2: Crawl Command Signal Handling (2.2)

**Files:**
- Modify: `crawl4r/cli/commands/crawl.py`
- Test: `tests/unit/test_crawl_interrupt.py` (new)

**Step 1: Write failing unit test for Ctrl+C handling**

Create `tests/unit/test_crawl_interrupt.py`:
```python
import signal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from crawl4r.cli.commands.crawl import _run_crawl


@pytest.mark.asyncio
async def test_crawl_releases_lock_on_interrupt() -> None:
    queue = Mock()
    queue.release_lock = AsyncMock()
    queue.set_status = AsyncMock()

    with patch("crawl4r.cli.commands.crawl.QueueManager", return_value=queue):
        with patch("crawl4r.cli.commands.crawl.asyncio.get_running_loop") as loop_patch:
            loop = Mock()
            loop.add_signal_handler = Mock()
            loop_patch.return_value = loop
            with patch("crawl4r.cli.commands.crawl.asyncio.Event") as event_patch:
                event = Mock()
                event.wait = AsyncMock()
                event_patch.return_value = event
                await _run_crawl(["https://example.com"])  # should register handlers
                assert loop.add_signal_handler.called
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/unit/test_crawl_interrupt.py::test_crawl_releases_lock_on_interrupt -v
```
Expected: FAIL (no signal handling).

**Step 3: Implement signal handling**

In `crawl4r/cli/commands/crawl.py`, add SIGINT/SIGTERM handlers that:
- Release the Redis lock.
- Set crawl status to FAILED with error "Interrupted by user".
- Clean up resources.

Use an `asyncio.Event` and register handlers with the running loop to avoid
`asyncio.create_task` during `asyncio.run` shutdown:
```python
import signal

async def _run_crawl(urls: list[str]) -> None:
    queue = QueueManager()
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda *_: _signal_handler())
        signal.signal(signal.SIGTERM, lambda *_: _signal_handler())

    try:
        # normal crawl flow...
        await stop_event.wait()
    finally:
        await queue.release_lock()
        await queue.set_status(
            CrawlStatusInfo(..., status=CrawlStatus.FAILED, error="Interrupted by user")
        )
```

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/unit/test_crawl_interrupt.py::test_crawl_releases_lock_on_interrupt -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add crawl4r/cli/commands/crawl.py tests/unit/test_crawl_interrupt.py
git commit -m "feat(cli): add graceful shutdown handling for crawl command"
```

---

### Task 3: Stale Lock Recovery in QueueManager (2.3)

**Files:**
- Modify: `crawl4r/services/queue.py`
- Test: `tests/unit/test_queue_manager_stale_lock.py` (new)

**Step 1: Write failing test for stale lock recovery**

Create `tests/unit/test_queue_manager_stale_lock.py`:
```python
from unittest.mock import AsyncMock, Mock

import pytest

from crawl4r.services.queue import QueueManager
from crawl4r.services.models import CrawlStatus, CrawlStatusInfo


@pytest.mark.asyncio
async def test_acquire_lock_recovers_failed_holder() -> None:
    queue = QueueManager()
    queue._client = Mock()  # type: ignore[attr-defined]
    queue._client.set = AsyncMock(return_value=False)
    queue._client.get = AsyncMock(return_value=b"crawl_1")

    queue.get_status = AsyncMock(return_value=CrawlStatusInfo(
        crawl_id="crawl_1",
        status=CrawlStatus.FAILED,
        error="boom",
        started_at="",
        finished_at="",
    ))

    queue.release_lock = AsyncMock()
    queue._client.set = AsyncMock(return_value=True)

    acquired = await queue.acquire_lock("crawl_2")
    assert acquired is True
```

**Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/unit/test_queue_manager_stale_lock.py::test_acquire_lock_recovers_failed_holder -v
```
Expected: FAIL (no recovery behavior).

**Step 3: Implement stale lock recovery**

In `QueueManager.acquire_lock()`:
- If initial `SETNX` fails, read lock holder ID.
- If holder status is FAILED, release lock and retry acquire.
- Log recovery action.

Pseudo-code:
```python
if not acquired:
    holder = await self._client.get(LOCK_KEY)
    if holder:
        status = await self.get_status(holder.decode())
        if status and status.status == CrawlStatus.FAILED:
            await self.release_lock()
            acquired = await self._client.set(..., nx=True, ex=ttl)
```

**Step 4: Run test to verify it passes**

Run:
```bash
pytest tests/unit/test_queue_manager_stale_lock.py::test_acquire_lock_recovers_failed_holder -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add crawl4r/services/queue.py tests/unit/test_queue_manager_stale_lock.py
git commit -m "feat(services): add stale lock recovery to QueueManager"
```

---

### Task 4: Service Error Handling + Redis Fallback (2.4)

**Files:**
- Modify: `crawl4r/services/scraper.py`
- Modify: `crawl4r/services/queue.py`
- Modify: `crawl4r/services/ingestion.py`
- Test: `tests/unit/test_scraper_errors.py` (new)
- Test: `tests/unit/test_queue_fallback.py` (new)
- Test: `tests/unit/test_ingestion_partial_failures.py` (new)

**Step 1: Write failing tests**

Scraper network error:
```python
import httpx
import pytest

from crawl4r.services.scraper import ScraperService


@pytest.mark.asyncio
async def test_scraper_returns_clear_error_on_network_failure():
    service = ScraperService()
    service._client = httpx.AsyncClient()  # ensure client exists
    async def _boom(*args, **kwargs):
        raise httpx.ConnectError("down")
    service._client.post = _boom  # type: ignore[assignment]

    result = await service.scrape_url("https://example.com")
    assert result.success is False
    assert "network" in (result.error or "").lower()
```

Queue fallback when Redis unavailable:
```python
import pytest
from unittest.mock import AsyncMock, Mock

from crawl4r.services.queue import QueueManager


@pytest.mark.asyncio
async def test_queue_fallback_when_redis_unavailable():
    queue = QueueManager()
    queue._client = Mock()
    queue._client.ping = AsyncMock(side_effect=ConnectionError("down"))

    ok = await queue.is_available()
    assert ok is False
```

Ingestion continues on partial failures:
```python
import pytest
from unittest.mock import AsyncMock, Mock

from crawl4r.services.ingestion import IngestionService


@pytest.mark.asyncio
async def test_ingestion_continues_on_single_failure():
    service = IngestionService()
    service.scraper = Mock()
    service.scraper.scrape_urls = AsyncMock(return_value=[
        Mock(success=True, markdown="# ok", metadata={}),
        Mock(success=False, error="bad"),
    ])
    result = await service.ingest_urls(["https://ok", "https://bad"])
    assert result.success is False
    assert result.urls_total == 2
    assert result.urls_failed == 1
```

**Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/unit/test_scraper_errors.py::test_scraper_returns_clear_error_on_network_failure -v
pytest tests/unit/test_queue_fallback.py::test_queue_fallback_when_redis_unavailable -v
pytest tests/unit/test_ingestion_partial_failures.py::test_ingestion_continues_on_single_failure -v
```
Expected: FAIL (missing handling and fallback).

**Step 3: Implement error handling and fallback**

- `ScraperService`: catch `httpx.RequestError`, return `ScrapeResult(success=False, error="Network error: ...")`.
- `QueueManager`: add `is_available()` and wrap Redis calls with try/except; when unavailable, log warning and act as no-op.
- `IngestionService`: treat per-URL failures as partial failures, continue batch; ensure returned `IngestResult` reflects counts and errors.

**Step 4: Run tests to verify pass**

Re-run tests from Step 2. Expected: PASS.

**Step 5: Commit**

```bash
git add crawl4r/services/scraper.py crawl4r/services/queue.py crawl4r/services/ingestion.py \
  tests/unit/test_scraper_errors.py tests/unit/test_queue_fallback.py tests/unit/test_ingestion_partial_failures.py

git commit -m "feat(services): add comprehensive error handling and fallbacks"
```

---

### Task 5: URL Validation in Services (2.5)

**Files:**
- Modify: `crawl4r/services/scraper.py`
- Modify: `crawl4r/services/ingestion.py`
- Test: `tests/unit/test_service_url_validation.py` (new)

**Step 1: Write failing tests**

Create `tests/unit/test_service_url_validation.py`:
```python
import pytest

from crawl4r.services.scraper import ScraperService
from crawl4r.services.ingestion import IngestionService


@pytest.mark.asyncio
async def test_scraper_rejects_invalid_url():
    service = ScraperService()
    result = await service.scrape_url("not-a-url")
    assert result.success is False


def test_ingestion_rejects_invalid_url_sync():
    service = IngestionService()
    assert service.validate_url("not-a-url") is False
```

**Step 2: Run tests to verify failure**

Run:
```bash
pytest tests/unit/test_service_url_validation.py -v
```
Expected: FAIL (no validation).

**Step 3: Implement validation**

- Add a `validate_url()` helper in services using `crawl4r.readers.crawl4ai.Crawl4AIReader.validate_url` logic (copy or call).
- `ScraperService.scrape_url()` should return `ScrapeResult(success=False, error="Invalid URL")` if invalid.
- `IngestionService` should validate all URLs up front and return an error if any invalid.
- `IngestionService.__init__` should allow defaults for dependencies when not provided.

**Step 4: Run tests to verify pass**

Re-run tests from Step 2. Expected: PASS.

**Step 5: Commit**

```bash
git add crawl4r/services/scraper.py crawl4r/services/ingestion.py tests/unit/test_service_url_validation.py

git commit -m "feat(services): add URL validation to scraper and ingestion"
```

---

### Task 6: Service Startup Validation (2.6)

**Files:**
- Modify: `crawl4r/services/scraper.py`
- Modify: `crawl4r/services/ingestion.py`
- Test: `tests/unit/test_service_health_checks.py` (new)

**Step 1: Write failing tests**

Create `tests/unit/test_service_health_checks.py`:
```python
from unittest.mock import AsyncMock, Mock

import pytest

from crawl4r.services.scraper import ScraperService
from crawl4r.services.ingestion import IngestionService


@pytest.mark.asyncio
async def test_scraper_health_check_fails_fast():
    service = ScraperService()
    service._health_check = AsyncMock(return_value=False)
    with pytest.raises(ValueError, match="Crawl4AI service health check failed"):
        await service.validate_services()


@pytest.mark.asyncio
async def test_ingestion_health_check_fails_fast():
    service = IngestionService()
    service.scraper = Mock()
    service.scraper.validate_services = AsyncMock(side_effect=ValueError("fail"))
    with pytest.raises(ValueError, match="fail"):
        await service.validate_services()
```

**Step 2: Run tests to verify failure**

Run:
```bash
pytest tests/unit/test_service_health_checks.py -v
```
Expected: FAIL (no health checks).

**Step 3: Implement health checks**

- `ScraperService.__init__` should accept `validate_on_startup: bool = True`.
- Implement `_health_check()` calling `GET /health`.
- Implement `validate_services()` that raises `ValueError` if health check fails.
- `IngestionService.validate_services()` delegates to scraper and other dependencies (if needed).

**Step 4: Run tests to verify pass**

Re-run tests from Step 2. Expected: PASS.

**Step 5: Commit**

```bash
git add crawl4r/services/scraper.py crawl4r/services/ingestion.py tests/unit/test_service_health_checks.py

git commit -m "feat(services): validate service availability on startup"
```

---

### Task 7: Quality Checkpoint (2.7)

**Files:**
- None (verify only)

**Step 1: Run ruff, ty, unit tests**

Run:
```bash
ruff check .
ty check crawl4r/
pytest tests/unit/ -x
```
Expected: PASS. If failure occurs, fix and re-run.

**Step 2: Commit fixes if needed**

```bash
git add -A
git commit -m "chore: pass quality checkpoint"
```

---

### Task 8: P0 Complete Manual Verification (2.8)

**Files:**
- None (manual verification)

**Step 1: Verify P0 commands locally**

Run:
```bash
python -m crawl4r.cli.app scrape https://example.com
python -m crawl4r.cli.app crawl https://example.com
python -m crawl4r.cli.app status --active
python -m crawl4r.cli.app watch --help
```
Expected: commands run, help displays, watch shows --folder.

**Step 2: Verify queue coordination (two terminals)**

Terminal 1:
```bash
crawl4r crawl https://docs.python.org
```

Terminal 2:
```bash
crawl4r crawl https://docs.example.com
```
Expected: Second crawl queues; status shows queued.

**Step 3: Document any remaining issues**

If any failures, log in `docs/issues.md` (create if missing) with steps to reproduce.

**Step 4: Commit P0 completion**

```bash
git commit -am "feat(cli): complete P0 commands"
```

---

## Notes

- Follow existing patterns from `crawl4r/readers/crawl4ai.py` for URL validation and health checks.
- Keep new errors user-facing and actionable. Log internal exceptions with context.
- Avoid breaking Phase 1 commands (`scrape`, `crawl`, `status`).
