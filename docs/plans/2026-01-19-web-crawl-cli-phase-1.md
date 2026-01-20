# Web Crawl CLI Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver Phase 1 POC CLI (scrape/crawl/status) and core services (scraper, queue, ingestion) with minimal but real TDD coverage.

**Architecture:** Add CLI entrypoint with Typer commands that delegate to new service layer. Services use Crawl4AI `/md?f=fit`, TEI embeddings, Qdrant vector storage, and Redis for queue coordination. Minimal error handling per Phase 1, but circuit breaker and retries are required.

**Tech Stack:** Python 3.10+, Typer, Rich, httpx, redis.asyncio, LlamaIndex MarkdownNodeParser, pytest, respx, ruff, ty.

---

## Task 1.1: Add CLI Dependencies and Entry Point

**Files:**
- Modify: `pyproject.toml`

**Step 1: Edit dependencies**
Add `typer[all]>=0.12.0` and `redis>=5.0.0` to `[project.dependencies]`.

**Step 2: Add CLI script**
Add `[project.scripts]` and `crawl4r = "crawl4r.cli.app:app"`.

**Step 3: Verify (no tests needed)**
Run: `grep -E "typer|redis" pyproject.toml && grep -A1 "\[project.scripts\]" pyproject.toml`
Expected: Both dependencies listed and scripts section contains `crawl4r`.

**Step 4: Sync dependencies**
Run: `uv sync`
Expected: Success without errors.

**Step 5: Commit**
```
git add pyproject.toml
git commit -m "build(deps): add typer and redis for CLI commands"
```

---

## Task 1.2: Add REDIS_URL to Settings

**Files:**
- Modify: `crawl4r/core/config.py`
- Test: `tests/unit/test_config.py`

**Step 1: Write failing test**
Add a test in `tests/unit/test_config.py`:
```python
from crawl4r.core.config import Settings

def test_settings_default_redis_url() -> None:
    settings = Settings(watch_folder=".")
    assert settings.REDIS_URL == "redis://localhost:53379"
```

**Step 2: Run test to verify failure**
Run: `pytest tests/unit/test_config.py::test_settings_default_redis_url -v`
Expected: FAIL (attribute missing).

**Step 3: Implement minimal change**
Add field to `Settings` (use Google-style docstrings per Phase 1 tasks):
```python
REDIS_URL: str = Field(
    default="redis://localhost:53379",
    description="Redis connection URL for crawl queue coordination",
)
```

**Step 4: Run test to verify pass**
Run: `pytest tests/unit/test_config.py::test_settings_default_redis_url -v`
Expected: PASS.

**Step 5: Commit**
```
git add crawl4r/core/config.py tests/unit/test_config.py
git commit -m "feat(config): add REDIS_URL setting for queue coordination"
```

---

## Task 1.3: Create Service Data Models

**Files:**
- Create: `crawl4r/services/__init__.py`
- Create: `crawl4r/services/models.py`
- Create: `tests/unit/services/__init__.py`
- Create: `tests/unit/services/test_models.py`

**Step 1: Write failing tests**
Create `tests/unit/services/test_models.py`:
```python
from crawl4r.services.models import CrawlStatus, IngestResult, ScrapeResult

def test_crawl_status_enum_values() -> None:
    assert CrawlStatus.QUEUED.value == "QUEUED"
    assert CrawlStatus.RUNNING.value == "RUNNING"
    assert CrawlStatus.COMPLETED.value == "COMPLETED"
    assert CrawlStatus.FAILED.value == "FAILED"


def test_scrape_result_shape() -> None:
    result = ScrapeResult(
        url="https://example.com",
        success=True,
        markdown="# Title",
        status_code=200,
        error=None,
        metadata={"title": "Example"},
    )
    assert result.url == "https://example.com"
    assert result.markdown.startswith("#")
    assert result.success is True


def test_ingest_result_counts() -> None:
    result = IngestResult(
        crawl_id="crawl_test",
        success=False,
        urls_total=2,
        urls_failed=1,
        chunks_created=3,
        queued=False,
        error="1 failure",
    )
    assert result.urls_total == 2
    assert result.urls_failed == 1
    assert result.chunks_created == 3
    assert result.success is False
```

**Step 2: Run tests to verify failure**
Run: `pytest tests/unit/services/test_models.py -v`
Expected: FAIL (module missing).

**Step 3: Implement models and exports**
Create `crawl4r/services/models.py` with:
- `CrawlStatus` enum: QUEUED, RUNNING, COMPLETED, FAILED
- `ScrapeResult`, `MapResult`, `ExtractResult`, `ScreenshotResult`, `IngestResult`, `CrawlStatusInfo` dataclasses
- Google-style docstrings for each class
- Type hints for every field
- Standardized result fields across services:
  - `ScrapeResult`: `url`, `success`, `error`, `markdown`, `status_code`, `metadata`
  - `MapResult`: `url`, `success`, `error`, `links`, `internal_count`, `external_count`, `depth_reached`
  - `ExtractResult`: `url`, `success`, `error`, `data`
  - `ScreenshotResult`: `url`, `success`, `error`, `file_path`, `file_size`
  - `IngestResult`: `crawl_id`, `success`, `error`, `urls_total`, `urls_failed`, `chunks_created`, `queued`
  - `CrawlStatusInfo`: `crawl_id`, `status`, `error`, `started_at`, `finished_at`

Create `crawl4r/services/__init__.py` exporting all models:
```python
from crawl4r.services.models import (
    CrawlStatus,
    CrawlStatusInfo,
    ExtractResult,
    IngestResult,
    MapResult,
    ScrapeResult,
    ScreenshotResult,
)

__all__ = [
    "CrawlStatus",
    "CrawlStatusInfo",
    "ExtractResult",
    "IngestResult",
    "MapResult",
    "ScrapeResult",
    "ScreenshotResult",
]
```

**Step 4: Run tests to verify pass**
Run: `pytest tests/unit/services/test_models.py -v`
Expected: PASS.

**Step 5: Verify import**
Run: `python -c "from crawl4r.services.models import ScrapeResult, CrawlStatus, IngestResult; print('OK')"`
Expected: `OK`.

**Step 6: Commit**
```
git add crawl4r/services/__init__.py crawl4r/services/models.py tests/unit/services/__init__.py tests/unit/services/test_models.py
git commit -m "feat(services): add data models for service results"
```

---

## Task 1.4: Implement ScraperService

**Files:**
- Create: `crawl4r/services/scraper.py`
- Modify: `crawl4r/services/__init__.py`
- Create: `tests/unit/services/test_scraper.py`

**Step 1: Write failing tests**
Create `tests/unit/services/test_scraper.py`:
```python
import httpx
import pytest
import respx

from crawl4r.services.scraper import ScraperService


@respx.mock
@pytest.mark.asyncio
async def test_scrape_url_hits_md_endpoint() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(200, json={"markdown": "# Title", "status_code": 200})
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    result = await service.scrape_url("https://example.com")
    assert result.url == "https://example.com"
    assert result.markdown == "# Title"
    assert result.success is True


@respx.mock
@pytest.mark.asyncio
async def test_scrape_urls_batch_returns_results() -> None:
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/md?f=fit").mock(
        return_value=httpx.Response(200, json={"markdown": "# Ok", "status_code": 200})
    )
    service = ScraperService(endpoint_url="http://localhost:52004")
    results = await service.scrape_urls(["https://a.com", "https://b.com"], max_concurrent=2)
    assert len(results) == 2
```

**Step 2: Run tests to verify failure**
Run: `pytest tests/unit/services/test_scraper.py -v`
Expected: FAIL (service missing).

**Step 3: Implement ScraperService**
Create `crawl4r/services/scraper.py` with:
- `ScraperService` with a stored `_client: httpx.AsyncClient` for reuse and testability
- `scrape_url` uses `/md?f=fit` endpoint with JSON body `{ "url": url, "f": "fit" }`
- `scrape_urls` uses semaphore for concurrency
- CircuitBreaker integration (5 failures, 60s reset)
- Retry logic with backoff [1s, 2s, 4s]
- Google-style docstrings on public methods

**Step 4: Export service**
Update `crawl4r/services/__init__.py` to export `ScraperService`.

**Step 5: Run tests to verify pass**
Run: `pytest tests/unit/services/test_scraper.py -v`
Expected: PASS.

**Step 6: Verify import**
Run: `python -c "from crawl4r.services.scraper import ScraperService; print('OK')"`
Expected: `OK`.

**Step 7: Commit**
```
git add crawl4r/services/scraper.py crawl4r/services/__init__.py tests/unit/services/test_scraper.py
git commit -m "feat(services): implement ScraperService with circuit breaker"
```

---

## Task 1.5: Implement QueueManager

**Files:**
- Create: `crawl4r/services/queue.py`
- Modify: `crawl4r/services/__init__.py`
- Create: `tests/unit/services/test_queue.py`

**Step 1: Write failing tests**
Create `tests/unit/services/test_queue.py` using fakeredis/redis asyncio test utilities (if already in deps) or mock with `redis.asyncio` and `unittest.mock`:
```python
import asyncio
import pytest
from unittest.mock import AsyncMock

from crawl4r.services.models import CrawlStatus, CrawlStatusInfo
from crawl4r.services.queue import QueueManager


@pytest.mark.asyncio
async def test_enqueue_dequeue_round_trip() -> None:
    manager = QueueManager(redis_url="redis://localhost:53379")
    manager._client = AsyncMock()  # stub redis client
    manager._client.lpush = AsyncMock(return_value=1)
    manager._client.brpop = AsyncMock(return_value=(b"crawl_queue", b"crawl_id|https://example.com"))

    await manager.enqueue_crawl("crawl_id", ["https://example.com"])
    item = await manager.dequeue_crawl()
    assert item == ("crawl_id", ["https://example.com"])
```

**Step 2: Run tests to verify failure**
Run: `pytest tests/unit/services/test_queue.py -v`
Expected: FAIL (service missing).

**Step 3: Implement QueueManager**
Create `crawl4r/services/queue.py` with:
- Redis key constants (LOCK_KEY, QUEUE_KEY, STATUS_PREFIX, RECENT_LIST_KEY)
- Uses `redis.asyncio.Redis`
- `acquire_lock`, `release_lock`, `enqueue_crawl`, `dequeue_crawl`, `set_status`, `get_status`, `list_recent`, `get_active`, `close`
- TTLs: lock 1 hour, status 24 hours
- Google-style docstrings on public methods

**Step 4: Export QueueManager**
Update `crawl4r/services/__init__.py` to export `QueueManager`.

**Step 5: Run tests to verify pass**
Run: `pytest tests/unit/services/test_queue.py -v`
Expected: PASS.

**Step 6: Verify import**
Run: `python -c "from crawl4r.services.queue import QueueManager, LOCK_KEY; print('OK')"`
Expected: `OK`.

**Step 7: Commit**
```
git add crawl4r/services/queue.py crawl4r/services/__init__.py tests/unit/services/test_queue.py
git commit -m "feat(services): implement QueueManager for Redis coordination"
```

---

## Task 1.6: Quality Checkpoint (Services)

**Files:**
- None (verification only)

**Step 1: Run ruff**
Run: `ruff check crawl4r/services/`
Expected: PASS.

**Step 2: Run ty**
Run: `ty check crawl4r/services/`
Expected: PASS.

**Step 3: Fix issues if needed**
If fixes are required, repeat checks and commit:
```
git add crawl4r/services/ tests/unit/services/
git commit -m "chore(services): pass quality checkpoint"
```

---

## Task 1.7: Implement IngestionService (Basic)

**Files:**
- Create: `crawl4r/services/ingestion.py`
- Modify: `crawl4r/services/__init__.py`
- Create: `tests/unit/services/test_ingestion.py`

**Step 1: Write failing tests**
Create `tests/unit/services/test_ingestion.py` using mocks for TEI/Qdrant:
```python
import pytest
from unittest.mock import AsyncMock

from crawl4r.services.ingestion import generate_crawl_id, IngestionService


def test_generate_crawl_id_format() -> None:
    crawl_id = generate_crawl_id()
    assert crawl_id.startswith("crawl_")

@pytest.mark.asyncio
async def test_ingest_urls_returns_result() -> None:
    service = IngestionService(
        scraper=AsyncMock(),
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=AsyncMock(),
    )
    service.scraper.scrape_urls = AsyncMock(return_value=[])
    result = await service.ingest_urls(["https://example.com"])
    assert result.crawl_id.startswith("crawl_")
    assert result.success is True
```

**Step 2: Run tests to verify failure**
Run: `pytest tests/unit/services/test_ingestion.py -v`
Expected: FAIL (service missing).

**Step 3: Implement ingestion**
Create `crawl4r/services/ingestion.py` with:
- `generate_crawl_id()` using `crawl_<timestamp>_<random>`
- `IngestionService` constructor accepting scraper, tei client, vector store, queue manager (allow defaults when not provided)
- `ingest_urls`:
  - acquire lock or enqueue
  - call scraper
  - parse markdown with `MarkdownNodeParser`
  - embed nodes with `TEIClient`
  - dedupe via `vector_store.delete_by_url(url)`
  - upsert via `vector_store.upsert_vectors(...)`
  - update status in Redis
  - return `IngestResult` with `success`, `error`, and `queued` populated consistently
- Google-style docstrings on public methods

**Step 4: Export IngestionService**
Update `crawl4r/services/__init__.py` to export `IngestionService` and `generate_crawl_id`.

**Step 5: Run tests to verify pass**
Run: `pytest tests/unit/services/test_ingestion.py -v`
Expected: PASS.

**Step 6: Verify helper**
Run: `python -c "from crawl4r.services.ingestion import IngestionService, generate_crawl_id; print(generate_crawl_id())"`
Expected: prints `crawl_...`.

**Step 7: Commit**
```
git add crawl4r/services/ingestion.py crawl4r/services/__init__.py tests/unit/services/test_ingestion.py
git commit -m "feat(services): implement IngestionService for crawl pipeline"
```

---

## Task 1.8: Create Typer CLI Application

**Files:**
- Create: `crawl4r/cli/app.py`
- Modify: `crawl4r/cli/commands/__init__.py`
- Create: `crawl4r/cli/commands/scrape.py`
- Create: `crawl4r/cli/commands/crawl.py`
- Create: `crawl4r/cli/commands/status.py`
- Create: `tests/unit/cli/__init__.py`
- Create: `tests/unit/cli/test_app.py`

**Step 1: Write failing test**
Create `tests/unit/cli/test_app.py`:
```python
from crawl4r.cli.app import app

def test_app_name() -> None:
    assert app.info.name == "crawl4r"
```

**Step 2: Run test to verify failure**
Run: `pytest tests/unit/cli/test_app.py -v`
Expected: FAIL (module missing).

**Step 3: Implement Typer app**
Create `crawl4r/cli/app.py` with:
- `typer.Typer(no_args_is_help=True, name="crawl4r")`
- import command modules and register as subcommands

Create placeholder command files and exports in `crawl4r/cli/commands/__init__.py`.

**Step 4: Run test to verify pass**
Run: `pytest tests/unit/cli/test_app.py -v`
Expected: PASS.

**Step 5: Verify app**
Run: `python -c "from crawl4r.cli.app import app; print(app.info.name)"`
Expected: `crawl4r`.

**Step 6: Commit**
```
git add crawl4r/cli/app.py crawl4r/cli/commands/__init__.py crawl4r/cli/commands/scrape.py crawl4r/cli/commands/crawl.py crawl4r/cli/commands/status.py tests/unit/cli/__init__.py tests/unit/cli/test_app.py
git commit -m "feat(cli): create Typer application structure"
```

---

## Task 1.9: Implement Scrape Command

**Files:**
- Modify: `crawl4r/cli/commands/scrape.py`
- Modify: `crawl4r/cli/app.py`
- Create: `tests/unit/cli/test_scrape_command.py`

**Step 1: Write failing test**
Create `tests/unit/cli/test_scrape_command.py` with a minimal help invocation:
```python
from typer.testing import CliRunner

from crawl4r.cli.app import app

runner = CliRunner()

def test_scrape_help() -> None:
    result = runner.invoke(app, ["scrape", "--help"])
    assert result.exit_code == 0
    assert "scrape" in result.output
```

**Step 2: Run test to verify failure**
Run: `pytest tests/unit/cli/test_scrape_command.py -v`
Expected: FAIL (command missing).

**Step 3: Implement scrape command**
Implement in `crawl4r/cli/commands/scrape.py`:
- positional `urls: list[str]`
- `-f/--file` read URLs from file
- `-o/--output` path
- `-c/--concurrent`
- merge URLs, fail if empty
- use `ScraperService.scrape_urls`
- Rich progress bar for batch
- print markdown to stdout or file(s)
- summary of success/fail counts
- `raise typer.Exit(code=1)` if any failure

Register in `crawl4r/cli/app.py`.

**Step 4: Run test to verify pass**
Run: `pytest tests/unit/cli/test_scrape_command.py -v`
Expected: PASS.

**Step 5: Verify manual help**
Run: `python -m crawl4r.cli.app scrape --help`
Expected: help output.

**Step 6: Commit**
```
git add crawl4r/cli/commands/scrape.py crawl4r/cli/app.py tests/unit/cli/test_scrape_command.py
git commit -m "feat(cli): implement scrape command with batch support"
```

---

## Task 1.10: Implement Crawl Command

**Files:**
- Modify: `crawl4r/cli/commands/crawl.py`
- Modify: `crawl4r/cli/app.py`
- Create: `tests/unit/cli/test_crawl_command.py`

**Step 1: Write failing test**
Create `tests/unit/cli/test_crawl_command.py`:
```python
from typer.testing import CliRunner

from crawl4r.cli.app import app

runner = CliRunner()

def test_crawl_help() -> None:
    result = runner.invoke(app, ["crawl", "--help"])
    assert result.exit_code == 0
    assert "crawl" in result.output
```

**Step 2: Run test to verify failure**
Run: `pytest tests/unit/cli/test_crawl_command.py -v`
Expected: FAIL (command missing).

**Step 3: Implement crawl command**
Implement in `crawl4r/cli/commands/crawl.py`:
- positional `urls: list[str]`
- `-f/--file` for URL file
- `-d/--depth`
- use `IngestionService.ingest_urls`
- display crawl ID immediately
- show queue position if queued
- show progress for current URL
- print summary panel (urls processed, failed, chunks)
- exit code 1 on failures

Register in `crawl4r/cli/app.py`.

**Step 4: Run test to verify pass**
Run: `pytest tests/unit/cli/test_crawl_command.py -v`
Expected: PASS.

**Step 5: Verify manual help**
Run: `python -m crawl4r.cli.app crawl --help`
Expected: help output.

**Step 6: Commit**
```
git add crawl4r/cli/commands/crawl.py crawl4r/cli/app.py tests/unit/cli/test_crawl_command.py
git commit -m "feat(cli): implement crawl command with queue coordination"
```

---

## Task 1.11: Implement Status Command

**Files:**
- Modify: `crawl4r/cli/commands/status.py`
- Modify: `crawl4r/cli/app.py`
- Create: `tests/unit/cli/test_status_command.py`

**Step 1: Write failing test**
Create `tests/unit/cli/test_status_command.py`:
```python
from typer.testing import CliRunner

from crawl4r.cli.app import app

runner = CliRunner()

def test_status_help() -> None:
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0
    assert "status" in result.output
```

**Step 2: Run test to verify failure**
Run: `pytest tests/unit/cli/test_status_command.py -v`
Expected: FAIL (command missing).

**Step 3: Implement status command**
Implement in `crawl4r/cli/commands/status.py`:
- optional `crawl_id` argument
- `--list` and `--active`
- use `QueueManager.get_status`, `list_recent`, `get_active`
- color-code statuses (QUEUED yellow, RUNNING blue, COMPLETED green, FAILED red)

Register in `crawl4r/cli/app.py`.

**Step 4: Run test to verify pass**
Run: `pytest tests/unit/cli/test_status_command.py -v`
Expected: PASS.

**Step 5: Verify manual help**
Run: `python -m crawl4r.cli.app status --help`
Expected: help output.

**Step 6: Commit**
```
git add crawl4r/cli/commands/status.py crawl4r/cli/app.py tests/unit/cli/test_status_command.py
git commit -m "feat(cli): implement status command for crawl tracking"
```

---

## Task 1.12: Quality Checkpoint (CLI + Services)

**Files:**
- None (verification only)

**Step 1: Run ruff**
Run: `ruff check crawl4r/cli/ crawl4r/services/`
Expected: PASS.

**Step 2: Run ty**
Run: `ty check crawl4r/cli/ crawl4r/services/`
Expected: PASS.

**Step 3: Fix issues if needed**
If fixes are required, repeat checks and commit:
```
git add crawl4r/cli/ crawl4r/services/ tests/unit/cli/ tests/unit/services/
git commit -m "chore(cli): pass quality checkpoint"
```

---

## Task 1.13: POC End-to-End Checkpoint

**Files:**
- None (verification only)

**Step 1: Ask permission to manage services**
Confirm with user before running any Docker lifecycle commands.

**Step 2: Start services (if needed)**
Run: `docker compose up -d`

**Step 2: CLI scrape test**
Run: `python -m crawl4r.cli.app scrape https://example.com`
Expected: markdown output printed or saved.

**Step 3: CLI crawl test**
Run: `python -m crawl4r.cli.app crawl https://example.com`
Expected: crawl ID shown and summary printed.

**Step 4: CLI status test**
Run: `python -m crawl4r.cli.app status --list`
Expected: recent crawl shown in table.

**Step 5: Ensure sessions directory exists**
Run: `mkdir -p .docs/sessions`

**Step 6: Document issues**
Record any failures in `.docs/sessions/2026-01-19-web-crawl-cli-phase-1.md` with timestamped notes.

**Step 7: Commit (if new docs added)**
```
git add .docs/sessions/2026-01-19-web-crawl-cli-phase-1.md
git commit -m "feat(cli): complete POC for web crawling commands"
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-19-web-crawl-cli-phase-1.md`.

Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
