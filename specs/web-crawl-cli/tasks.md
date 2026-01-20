---
spec: web-crawl-cli
phase: tasks
created: 2026-01-18
---

# Tasks: Web Crawl CLI

## Overview

Implementation tasks for adding web crawling capabilities to the crawl4r CLI. Follows POC-first workflow with 5 phases: POC (core working), P0 Complete (watch + error handling), P1 Commands (map/extract/screenshot), Testing, and Quality Gates.

**Total Estimated Tasks:** 38 tasks
**Quality Checkpoints:** 8 intermediate + 3 final verification

---

## Phase 1: Make It Work (POC)

Focus: Get scrape, crawl, and status commands working end-to-end. Skip comprehensive tests, minimal error handling.

### Task 1.1: Add Dependencies to pyproject.toml

**Do:**
1. Add `typer[all]>=0.12.0` to dependencies (includes Rich for progress display)
2. Add `redis>=5.0.0` to dependencies (for queue coordination)
3. Add CLI entry point `crawl4r = "crawl4r.cli.app:app"` to `[project.scripts]`

**Files:**
- `pyproject.toml` - Modify - add dependencies and entry point

**Done when:**
- [ ] typer[all] and redis in dependencies list
- [ ] [project.scripts] section exists with crawl4r entry point
- [ ] `uv sync` succeeds without errors

**Verify:**
```bash
grep -E "typer|redis" pyproject.toml && grep -A1 "\[project.scripts\]" pyproject.toml
```

**Commit:** `build(deps): add typer and redis for CLI commands`

_Requirements: FR-012, FR-026_
_Design: Implementation Notes section_

---

### Task 1.2: Add REDIS_URL to Settings

**Do:**
1. Add `REDIS_URL: str = "redis://localhost:53379"` field to Settings class
2. Add Field description for documentation

**Files:**
- `crawl4r/core/config.py` - Modify - add REDIS_URL setting

**Done when:**
- [ ] REDIS_URL field exists with default value
- [ ] Field has description for documentation

**Verify:**
```bash
grep -A2 "REDIS_URL" crawl4r/core/config.py
```

**Commit:** `feat(config): add REDIS_URL setting for queue coordination`

_Requirements: FR-026_
_Design: Settings Addition section_

---

### Task 1.3: Create Service Data Models

**Do:**
1. Create `crawl4r/services/` directory structure
2. Create `crawl4r/services/__init__.py` with exports
3. Create `crawl4r/services/models.py` with data structures:
   - `CrawlStatus` enum (QUEUED, RUNNING, COMPLETED, FAILED)
   - `ScrapeResult` dataclass
   - `MapResult` dataclass
   - `ExtractResult` dataclass
   - `ScreenshotResult` dataclass
   - `IngestResult` dataclass
   - `CrawlStatusInfo` dataclass
4. Include Google-style docstrings for all classes

**Files:**
- `crawl4r/services/__init__.py` - Create - module exports
- `crawl4r/services/models.py` - Create - data structures

**Done when:**
- [ ] All 6 dataclasses and 1 enum defined
- [ ] Each has complete Google-style docstrings
- [ ] Type hints on all fields
- [ ] Imports work: `from crawl4r.services.models import ScrapeResult`

**Verify:**
```bash
python -c "from crawl4r.services.models import ScrapeResult, CrawlStatus, IngestResult; print('OK')"
```

**Commit:** `feat(services): add data models for service results`

_Requirements: FR-001, FR-008_
_Design: Data Structures section_

---

### Task 1.4: Implement ScraperService

**Do:**
1. Create `crawl4r/services/scraper.py` with ScraperService class
2. Implement `scrape_url(url: str) -> ScrapeResult` method using `/md?f=fit` endpoint
3. Implement `scrape_urls(urls: list[str], max_concurrent: int = 5) -> list[ScrapeResult]`
4. Integrate CircuitBreaker from existing codebase
5. Add retry logic with exponential backoff (1s, 2s, 4s)
6. Use httpx async client pattern from Crawl4AIReader

**Files:**
- `crawl4r/services/scraper.py` - Create - scraper service
- `crawl4r/services/__init__.py` - Modify - add ScraperService export

**Done when:**
- [ ] ScraperService class with scrape_url and scrape_urls methods
- [ ] Uses `/md?f=fit` endpoint (not `/crawl`)
- [ ] CircuitBreaker integration (5 failures, 60s timeout)
- [ ] Retry logic with backoff
- [ ] Google-style docstrings on all public methods

**Verify:**
```bash
python -c "from crawl4r.services.scraper import ScraperService; print('OK')"
```

**Commit:** `feat(services): implement ScraperService with circuit breaker`

_Requirements: FR-001, FR-002, FR-021, FR-022_
_Design: ScraperService section_

---

### Task 1.5: Implement QueueManager

**Do:**
1. Create `crawl4r/services/queue.py` with QueueManager class
2. Define Redis key constants (LOCK_KEY, QUEUE_KEY, STATUS_PREFIX, etc.)
3. Implement `acquire_lock(crawl_id: str, ttl: int = 3600) -> bool`
4. Implement `release_lock() -> None`
5. Implement `enqueue_crawl(crawl_id: str, urls: list[str]) -> int`
6. Implement `dequeue_crawl() -> tuple[str, list[str]] | None`
7. Implement `set_status(status: CrawlStatusInfo) -> None`
8. Implement `get_status(crawl_id: str) -> CrawlStatusInfo | None`
9. Implement `list_recent(limit: int = 10) -> list[CrawlStatusInfo]`
10. Implement `get_active() -> CrawlStatusInfo | None`
11. Use redis.asyncio for async operations

**Files:**
- `crawl4r/services/queue.py` - Create - queue manager
- `crawl4r/services/__init__.py` - Modify - add QueueManager export

**Done when:**
- [ ] All methods implemented per design
- [ ] Uses redis.asyncio for async operations
- [ ] Lock TTL of 1 hour, status TTL of 24 hours
- [ ] Close method for connection cleanup

**Verify:**
```bash
python -c "from crawl4r.services.queue import QueueManager, LOCK_KEY; print('OK')"
```

**Commit:** `feat(services): implement QueueManager for Redis coordination`

_Requirements: FR-009, AC-2.10 through AC-2.15_
_Design: QueueManager section_

---

### Task 1.6: [VERIFY] Quality checkpoint: ruff + ty

**Do:**
1. Run `ruff check crawl4r/services/`
2. Run `ty check crawl4r/services/`
3. Fix any lint or type errors

**Verify:**
```bash
ruff check crawl4r/services/ && ty check crawl4r/services/
```

**Done when:**
- [ ] ruff check passes with no errors
- [ ] ty check passes with no errors

**Commit:** `chore(services): pass quality checkpoint` (only if fixes needed)

---

### Task 1.7: Implement IngestionService (Basic)

**Do:**
1. Create `crawl4r/services/ingestion.py` with IngestionService class
2. Implement `generate_crawl_id() -> str` helper function
3. Implement `ingest_urls(urls: list[str], depth: int = 0, on_progress: callable | None = None) -> IngestResult`
4. Integrate with ScraperService for fetching
5. Integrate with existing TEIClient for embeddings
6. Integrate with existing VectorStoreManager for storage
7. Integrate with QueueManager for lock/status
8. Use MarkdownNodeParser for chunking
9. Call `delete_by_url()` before upserting (deduplication)

**Files:**
- `crawl4r/services/ingestion.py` - Create - ingestion service
- `crawl4r/services/__init__.py` - Modify - add IngestionService export

**Done when:**
- [ ] IngestionService class with ingest_urls method
- [ ] Generates crawl IDs in format `crawl_<timestamp>_<random>`
- [ ] Acquires lock or queues crawl
- [ ] Scrapes, chunks, embeds, and upserts
- [ ] Calls delete_by_url for deduplication
- [ ] Updates status in Redis

**Verify:**
```bash
python -c "from crawl4r.services.ingestion import IngestionService, generate_crawl_id; print(generate_crawl_id())"
```

**Commit:** `feat(services): implement IngestionService for crawl pipeline`

_Requirements: FR-008, FR-010, FR-011, AC-2.1 through AC-2.9_
_Design: IngestionService section_

---

### Task 1.8: Create Typer CLI Application

**Do:**
1. Create `crawl4r/cli/app.py` with Typer application
2. Set `no_args_is_help=True` for root app
3. Create `crawl4r/cli/commands/` directory if not exists
4. Create placeholder command modules (empty files for now)
5. Import and register command modules in app.py

**Files:**
- `crawl4r/cli/app.py` - Create - Typer entry point
- `crawl4r/cli/commands/__init__.py` - Modify - update exports
- `crawl4r/cli/commands/scrape.py` - Create - placeholder
- `crawl4r/cli/commands/crawl.py` - Create - placeholder
- `crawl4r/cli/commands/status.py` - Create - placeholder

**Done when:**
- [ ] Typer app created with name "crawl4r"
- [ ] Root app shows help when no args
- [ ] Command placeholders exist

**Verify:**
```bash
python -c "from crawl4r.cli.app import app; print(app.info.name)"
```

**Commit:** `feat(cli): create Typer application structure`

_Requirements: FR-012, FR-013_
_Design: CLI Application Structure section_

---

### Task 1.9: Implement Scrape Command

**Do:**
1. Implement `crawl4r/cli/commands/scrape.py` with scrape command
2. Accept positional URLs and `-f/--file` for URL file input
3. Accept `-o/--output` for file/directory output
4. Accept `-c/--concurrent` for concurrency limit (default 5)
5. Merge URL sources (positional + file)
6. Display Rich progress for batch operations
7. Print markdown to stdout or save to file(s)
8. Report summary for batch (success/fail counts)
9. Exit code 0 if all succeed, 1 if any fail
10. Register command in app.py

**Files:**
- `crawl4r/cli/commands/scrape.py` - Modify - implement command
- `crawl4r/cli/app.py` - Modify - register scrape command

**Done when:**
- [ ] `crawl4r scrape https://example.com` works
- [ ] `-f urls.txt` reads from file
- [ ] `-o output.md` saves to file
- [ ] Progress display during batch
- [ ] Summary shows success/fail counts

**Verify:**
```bash
python -m crawl4r.cli.app scrape --help
```

**Commit:** `feat(cli): implement scrape command with batch support`

_Requirements: US-1, AC-1.1 through AC-1.10, FR-014_
_Design: Scrape Command section_

---

### Task 1.10: Implement Crawl Command

**Do:**
1. Implement `crawl4r/cli/commands/crawl.py` with crawl command
2. Accept positional URLs and `-f/--file` for URL file input
3. Accept `-d/--depth` for link following depth (default 0)
4. Use IngestionService for crawl operation
5. Display crawl ID immediately
6. Show queue position if queued
7. Show progress during crawl (current URL, completed/total)
8. Display summary panel on completion
9. Exit code 0 if all succeed, 1 if any fail
10. Register command in app.py

**Files:**
- `crawl4r/cli/commands/crawl.py` - Modify - implement command
- `crawl4r/cli/app.py` - Modify - register crawl command

**Done when:**
- [ ] `crawl4r crawl https://example.com` works
- [ ] Shows crawl ID on start
- [ ] Shows queue position if lock held
- [ ] Progress shows current URL
- [ ] Summary shows URLs processed, failed, chunks

**Verify:**
```bash
python -m crawl4r.cli.app crawl --help
```

**Commit:** `feat(cli): implement crawl command with queue coordination`

_Requirements: US-2, AC-2.1 through AC-2.15, FR-015_
_Design: Crawl Command section_

---

### Task 1.11: Implement Status Command

**Do:**
1. Implement `crawl4r/cli/commands/status.py` with status command
2. Accept optional crawl_id positional argument
3. Accept `-l/--list` to list recent crawls
4. Accept `-a/--active` to show active crawl
5. Display detailed status for specific crawl
6. Display table for recent crawls list
7. Color-code status values (QUEUED=yellow, RUNNING=blue, COMPLETED=green, FAILED=red)
8. Register command in app.py

**Files:**
- `crawl4r/cli/commands/status.py` - Modify - implement command
- `crawl4r/cli/app.py` - Modify - register status command

**Done when:**
- [ ] `crawl4r status <id>` shows specific crawl
- [ ] `crawl4r status --list` shows table of recent
- [ ] `crawl4r status --active` shows running crawl
- [ ] Status values are color-coded

**Verify:**
```bash
python -m crawl4r.cli.app status --help
```

**Commit:** `feat(cli): implement status command for crawl tracking`

_Requirements: US-6, AC-6.1 through AC-6.8, FR-019_
_Design: Status Command section_

---

### Task 1.12: [VERIFY] Quality checkpoint: ruff + ty

**Do:**
1. Run `ruff check crawl4r/cli/` and `crawl4r/services/`
2. Run `ty check crawl4r/cli/` and `crawl4r/services/`
3. Fix any lint or type errors

**Verify:**
```bash
ruff check crawl4r/cli/ crawl4r/services/ && ty check crawl4r/cli/ crawl4r/services/
```

**Done when:**
- [ ] ruff check passes with no errors
- [ ] ty check passes with no errors

**Commit:** `chore(cli): pass quality checkpoint` (only if fixes needed)

---

### Task 1.13: POC Checkpoint - End-to-End Test

**Do:**
1. Ensure Docker services are running (Crawl4AI, Redis, Qdrant, TEI)
2. Test `crawl4r scrape https://example.com` returns markdown
3. Test `crawl4r crawl https://example.com` ingests to Qdrant
4. Test `crawl4r status --list` shows the crawl
5. Document any issues found

**Verify:**
```bash
# Start services if needed
docker compose up -d

# Test scrape
python -m crawl4r.cli.app scrape https://example.com

# Test crawl
python -m crawl4r.cli.app crawl https://example.com

# Test status
python -m crawl4r.cli.app status --list
```

**Done when:**
- [ ] Scrape returns markdown content
- [ ] Crawl completes and shows summary
- [ ] Status shows the completed crawl

**Commit:** `feat(cli): complete POC for web crawling commands`

_Requirements: All P0 acceptance criteria_

---

## Phase 2: Complete P0 Commands

Focus: Add watch command (refactor from main.py), complete error handling, and circuit breaker integration.

### Task 2.1: Implement Watch Command

**Do:**
1. Create `crawl4r/cli/commands/watch.py`
2. Refactor logic from `crawl4r/cli/main.py` into watch command
3. Accept `--folder` option to override WATCH_FOLDER from environment
4. Perform batch recovery on startup
5. Monitor for create, modify, delete events
6. Handle graceful shutdown on Ctrl+C
7. Show event count and queue depth in logs
8. Register command in app.py

**Files:**
- `crawl4r/cli/commands/watch.py` - Create - watch command
- `crawl4r/cli/app.py` - Modify - register watch command

**Done when:**
- [ ] `crawl4r watch` monitors default folder
- [ ] `--folder /path` overrides default
- [ ] Batch processes on startup
- [ ] Monitors file events
- [ ] Graceful Ctrl+C shutdown

**Verify:**
```bash
python -m crawl4r.cli.app watch --help
```

**Commit:** `feat(cli): implement watch command refactored from main.py`

_Requirements: US-7, AC-7.1 through AC-7.7, FR-020_
_Design: Watch Command section_

---

### Task 2.2: Add Signal Handler for Graceful Shutdown

**Do:**
1. Add signal handler in crawl command for SIGINT/SIGTERM
2. Release Redis lock on interrupt
3. Set crawl status to FAILED with "Interrupted by user" error
4. Ensure resources are cleaned up

**Files:**
- `crawl4r/cli/commands/crawl.py` - Modify - add signal handling

**Done when:**
- [ ] Ctrl+C during crawl releases lock
- [ ] Status set to FAILED on interrupt
- [ ] No orphaned locks after interrupt

**Verify:**
```bash
# Manual test: Start crawl, Ctrl+C, check status shows FAILED
```

**Commit:** `feat(cli): add graceful shutdown handling for crawl command`

_Requirements: FR-025, AC-2.14_
_Design: Edge Cases section_

---

### Task 2.3: Add Stale Lock Recovery

**Do:**
1. Enhance QueueManager.acquire_lock() to detect stale locks
2. Check if lock holder's status is FAILED
3. Force acquire lock if holder process is dead
4. Log recovery action for observability

**Files:**
- `crawl4r/services/queue.py` - Modify - add stale lock recovery

**Done when:**
- [ ] Stale locks from crashed processes are recovered
- [ ] Recovery is logged
- [ ] Normal lock contention still works

**Verify:**
```bash
# Manual test: Set a lock, set holder status to FAILED, try to acquire new lock
```

**Commit:** `feat(services): add stale lock recovery to QueueManager`

_Requirements: FR-025, AC-2.15_
_Design: Error Handling section_

---

### Task 2.4: Add Comprehensive Error Handling to Services

**Do:**
1. Add error handling for network errors in ScraperService
2. Add error handling for Redis connection errors in QueueManager
3. Add fallback to local-only mode when Redis unavailable
4. Add clear error messages for user-facing errors
5. Ensure batch operations continue on individual failures

**Files:**
- `crawl4r/services/scraper.py` - Modify - improve error handling
- `crawl4r/services/queue.py` - Modify - add Redis fallback
- `crawl4r/services/ingestion.py` - Modify - improve error handling

**Done when:**
- [ ] Network errors show clear messages
- [ ] Redis unavailable shows warning, continues without queue
- [ ] Batch operations don't abort on single failure

**Verify:**
```bash
# Stop Redis, run crawl, verify it continues without queue coordination
```

**Commit:** `feat(services): add comprehensive error handling and fallbacks`

_Requirements: FR-021, FR-022, FR-023, FR-024_
_Design: Error Handling section_

---

### Task 2.5: [VERIFY] Quality checkpoint: ruff + ty + tests

**Do:**
1. Run `ruff check .`
2. Run `ty check crawl4r/`
3. Run `pytest tests/unit/` (existing tests)
4. Fix any failures

**Verify:**
```bash
ruff check . && ty check crawl4r/ && pytest tests/unit/ -x
```

**Done when:**
- [ ] All lint checks pass
- [ ] All type checks pass
- [ ] Existing tests still pass

**Commit:** `chore: pass quality checkpoint` (only if fixes needed)

---

### Task 2.6: Add URL Validation

**Do:**
1. Add URL validation helper function
2. Validate URLs before processing in ScraperService
3. Validate URLs before processing in IngestionService
4. Return clear error for invalid URLs

**Files:**
- `crawl4r/services/scraper.py` - Modify - add URL validation
- `crawl4r/services/ingestion.py` - Modify - add URL validation

**Done when:**
- [ ] Invalid URLs rejected with clear error
- [ ] Valid http/https URLs accepted
- [ ] Validation reuses pattern from Crawl4AIReader

**Verify:**
```bash
python -c "from crawl4r.services.scraper import ScraperService; s = ScraperService(); import asyncio; print(asyncio.run(s.scrape_url('not-a-url')))"
```

**Commit:** `feat(services): add URL validation to scraper and ingestion`

_Requirements: FR-028_
_Design: Security Considerations section_

---

### Task 2.7: Validate Services on Startup

**Do:**
1. Add health check to ScraperService initialization
2. Add health check to IngestionService initialization
3. Check Crawl4AI /health endpoint
4. Fail fast with clear error if services unavailable

**Files:**
- `crawl4r/services/scraper.py` - Modify - add health check
- `crawl4r/services/ingestion.py` - Modify - add health check

**Done when:**
- [ ] ScraperService checks Crawl4AI health on init
- [ ] Clear error message if service unavailable
- [ ] Can be disabled for testing

**Verify:**
```bash
# Stop Crawl4AI, try to create ScraperService, verify error
```

**Commit:** `feat(services): validate service availability on startup`

_Requirements: FR-028_
_Design: Error Handling section_

---

### Task 2.8: P0 Complete Checkpoint

**Do:**
1. Verify all P0 commands work: scrape, crawl, status, watch
2. Verify queue coordination between terminals
3. Verify error handling and recovery
4. Document any remaining issues

**Verify:**
```bash
# Test all P0 commands
python -m crawl4r.cli.app scrape https://example.com
python -m crawl4r.cli.app crawl https://example.com
python -m crawl4r.cli.app status --active
python -m crawl4r.cli.app watch --help

# Test queue coordination (two terminals)
# Terminal 1: crawl4r crawl https://docs.python.org
# Terminal 2: crawl4r crawl https://docs.example.com (should queue)
```

**Done when:**
- [ ] All P0 commands functional
- [ ] Queue coordination working
- [ ] Error handling working

**Commit:** `feat(cli): complete P0 commands`

_Requirements: All P0 requirements_

---

## Phase 3: P1 Commands

Focus: Implement map, extract, and screenshot commands.

### Task 3.1: Implement MapperService

**Do:**
1. Create `crawl4r/services/mapper.py` with MapperService class
2. Implement `map_url(url: str, depth: int = 0, same_domain: bool = True) -> MapResult`
3. Use /crawl endpoint to get page links
4. Implement recursive depth crawling
5. Filter same-domain links by default
6. Deduplicate discovered URLs

**Files:**
- `crawl4r/services/mapper.py` - Create - mapper service
- `crawl4r/services/__init__.py` - Modify - add MapperService export

**Done when:**
- [ ] MapperService class with map_url method
- [ ] Extracts links from response
- [ ] Depth crawling works
- [ ] Same-domain filtering works

**Verify:**
```bash
python -c "from crawl4r.services.mapper import MapperService; print('OK')"
```

**Commit:** `feat(services): implement MapperService for URL discovery`

_Requirements: FR-003, FR-004_
_Design: MapperService section_

---

### Task 3.2: Implement Map Command

**Do:**
1. Create `crawl4r/cli/commands/map.py` with map command
2. Accept URL positional argument
3. Accept `--depth` for recursive discovery
4. Accept `--same-domain/--include-external` flags
5. Accept `-o` for output file
6. Display discovered URLs to stdout or file
7. Show count of unique URLs
8. Register command in app.py

**Files:**
- `crawl4r/cli/commands/map.py` - Create - map command
- `crawl4r/cli/app.py` - Modify - register map command

**Done when:**
- [ ] `crawl4r map https://example.com` lists links
- [ ] `--depth 2` follows links recursively
- [ ] `-o urls.txt` saves to file
- [ ] Shows unique URL count

**Verify:**
```bash
python -m crawl4r.cli.app map --help
```

**Commit:** `feat(cli): implement map command for URL discovery`

_Requirements: US-3, AC-3.1 through AC-3.8, FR-016_
_Design: Map Command section_

---

### Task 3.3: [VERIFY] Quality checkpoint

**Do:**
1. Run `ruff check crawl4r/services/mapper.py crawl4r/cli/commands/map.py`
2. Run `ty check crawl4r/services/mapper.py crawl4r/cli/commands/map.py`

**Verify:**
```bash
ruff check crawl4r/services/mapper.py crawl4r/cli/commands/map.py && ty check crawl4r/services/mapper.py crawl4r/cli/commands/map.py
```

**Done when:**
- [ ] All lint checks pass
- [ ] All type checks pass

**Commit:** `chore(services): pass quality checkpoint` (only if fixes needed)

---

### Task 3.4: Implement ExtractorService

**Do:**
1. Create `crawl4r/services/extractor.py` with ExtractorService class
2. Implement `extract_with_schema(url: str, schema: dict) -> ExtractResult`
3. Implement `extract_with_prompt(url: str, prompt: str) -> ExtractResult`
4. Use /llm/job endpoint for extraction
5. Add circuit breaker protection

**Files:**
- `crawl4r/services/extractor.py` - Create - extractor service
- `crawl4r/services/__init__.py` - Modify - add ExtractorService export

**Done when:**
- [ ] ExtractorService class with both methods
- [ ] Schema extraction works
- [ ] Prompt extraction works
- [ ] Circuit breaker integration

**Verify:**
```bash
python -c "from crawl4r.services.extractor import ExtractorService; print('OK')"
```

**Commit:** `feat(services): implement ExtractorService for structured extraction`

_Requirements: FR-005, FR-006_
_Design: ExtractorService section_

---

### Task 3.5: Implement Extract Command

**Do:**
1. Create `crawl4r/cli/commands/extract.py` with extract command
2. Accept URL positional argument
3. Accept `--schema` for JSON schema file (mutually exclusive with --prompt)
4. Accept `--prompt` for LLM extraction prompt
5. Accept `-o` for output file
6. Output valid JSON to stdout or file
7. Register command in app.py

**Files:**
- `crawl4r/cli/commands/extract.py` - Create - extract command
- `crawl4r/cli/app.py` - Modify - register extract command

**Done when:**
- [ ] `crawl4r extract <url> --schema schema.json` works
- [ ] `crawl4r extract <url> --prompt "extract..."` works
- [ ] Mutual exclusion enforced
- [ ] Output is valid JSON

**Verify:**
```bash
python -m crawl4r.cli.app extract --help
```

**Commit:** `feat(cli): implement extract command for structured data`

_Requirements: US-4, AC-4.1 through AC-4.7, FR-017_
_Design: Extract Command section_

---

### Task 3.6: Implement ScreenshotService

**Do:**
1. Create `crawl4r/services/screenshot.py` with ScreenshotService class
2. Implement `capture(url: str, output_path: Path, full_page: bool = False, wait: int = 0) -> ScreenshotResult`
3. Use /screenshot endpoint
4. Save base64 response to PNG file
5. Add circuit breaker protection

**Files:**
- `crawl4r/services/screenshot.py` - Create - screenshot service
- `crawl4r/services/__init__.py` - Modify - add ScreenshotService export

**Done when:**
- [ ] ScreenshotService class with capture method
- [ ] Saves PNG file from base64 response
- [ ] Full page option works
- [ ] Wait option works

**Verify:**
```bash
python -c "from crawl4r.services.screenshot import ScreenshotService; print('OK')"
```

**Commit:** `feat(services): implement ScreenshotService for page capture`

_Requirements: FR-007_
_Design: ScreenshotService section_

---

### Task 3.7: Implement Screenshot Command

**Do:**
1. Create `crawl4r/cli/commands/screenshot.py` with screenshot command
2. Accept URL positional argument
3. Accept `-o` for output file (default: {domain}.png)
4. Accept `--full-page` for full page capture
5. Accept `--wait N` for wait before capture
6. Report file path and size on success
7. Register command in app.py

**Files:**
- `crawl4r/cli/commands/screenshot.py` - Create - screenshot command
- `crawl4r/cli/app.py` - Modify - register screenshot command

**Done when:**
- [ ] `crawl4r screenshot https://example.com` saves PNG
- [ ] `-o page.png` saves to specified file
- [ ] `--full-page` captures entire page
- [ ] Shows file path and size

**Verify:**
```bash
python -m crawl4r.cli.app screenshot --help
```

**Commit:** `feat(cli): implement screenshot command for page capture`

_Requirements: US-5, AC-5.1 through AC-5.6, FR-018_
_Design: Screenshot Command section_

---

### Task 3.8: [VERIFY] Quality checkpoint: all services

**Do:**
1. Run `ruff check crawl4r/services/`
2. Run `ty check crawl4r/services/`
3. Verify all 5 services import correctly

**Verify:**
```bash
ruff check crawl4r/services/ && ty check crawl4r/services/ && python -c "from crawl4r.services import ScraperService, MapperService, ExtractorService, ScreenshotService, IngestionService, QueueManager; print('All services OK')"
```

**Done when:**
- [ ] All lint checks pass
- [ ] All type checks pass
- [ ] All services importable

**Commit:** `chore(services): pass quality checkpoint` (only if fixes needed)

---

### Task 3.9: P1 Complete Checkpoint

**Do:**
1. Verify all P1 commands work: map, extract, screenshot
2. Test with real URLs
3. Document any issues

**Verify:**
```bash
# Test all P1 commands
python -m crawl4r.cli.app map https://example.com
python -m crawl4r.cli.app extract https://example.com --prompt "extract main heading"
python -m crawl4r.cli.app screenshot https://example.com
```

**Done when:**
- [ ] Map command discovers URLs
- [ ] Extract command returns JSON
- [ ] Screenshot command saves PNG

**Commit:** `feat(cli): complete P1 commands`

_Requirements: All P1 requirements_

---

## Phase 4: Testing

Focus: Unit tests for services, integration tests with real services, CLI tests.

### Task 4.1: Unit Tests for ScraperService

**Do:**
1. Create `tests/unit/test_scraper_service.py`
2. Test scrape_url success case (mock httpx)
3. Test scrape_url failure cases (timeout, 4xx, 5xx)
4. Test circuit breaker opens after failures
5. Test retry logic
6. Test scrape_urls concurrency

**Files:**
- `tests/unit/test_scraper_service.py` - Create - unit tests

**Done when:**
- [ ] Tests for success path
- [ ] Tests for error paths
- [ ] Tests for circuit breaker
- [ ] Tests for retry logic
- [ ] 90%+ coverage for ScraperService

**Verify:**
```bash
pytest tests/unit/test_scraper_service.py -v --cov=crawl4r.services.scraper
```

**Commit:** `test(services): add unit tests for ScraperService`

_Requirements: NFR-017_
_Design: Test Strategy section_

---

### Task 4.2: Unit Tests for QueueManager

**Do:**
1. Create `tests/unit/test_queue_manager.py`
2. Test acquire_lock success and failure (mock redis)
3. Test release_lock
4. Test enqueue_crawl and dequeue_crawl
5. Test set_status and get_status
6. Test list_recent and get_active
7. Test stale lock recovery

**Files:**
- `tests/unit/test_queue_manager.py` - Create - unit tests

**Done when:**
- [ ] Tests for lock operations
- [ ] Tests for queue operations
- [ ] Tests for status operations
- [ ] Tests for stale lock recovery
- [ ] 85%+ coverage for QueueManager

**Verify:**
```bash
pytest tests/unit/test_queue_manager.py -v --cov=crawl4r.services.queue
```

**Commit:** `test(services): add unit tests for QueueManager`

_Requirements: NFR-017_
_Design: Test Strategy section_

---

### Task 4.3: Unit Tests for IngestionService

**Do:**
1. Create `tests/unit/test_ingestion_service.py`
2. Test ingest_urls with lock acquired (mock all deps)
3. Test ingest_urls queued when lock held
4. Test crawl ID generation format
5. Test progress callback invocation
6. Test deduplication call before upsert

**Files:**
- `tests/unit/test_ingestion_service.py` - Create - unit tests

**Done when:**
- [ ] Tests for ingestion flow
- [ ] Tests for queue behavior
- [ ] Tests for deduplication
- [ ] 90%+ coverage for IngestionService

**Verify:**
```bash
pytest tests/unit/test_ingestion_service.py -v --cov=crawl4r.services.ingestion
```

**Commit:** `test(services): add unit tests for IngestionService`

_Requirements: NFR-017_
_Design: Test Strategy section_

---

### Task 4.4: [VERIFY] Quality checkpoint: tests

**Do:**
1. Run all new unit tests
2. Verify coverage meets targets
3. Fix any failing tests

**Verify:**
```bash
pytest tests/unit/test_scraper_service.py tests/unit/test_queue_manager.py tests/unit/test_ingestion_service.py -v --cov=crawl4r.services --cov-report=term
```

**Done when:**
- [ ] All tests pass
- [ ] Coverage > 85% for services

**Commit:** `chore(tests): pass quality checkpoint` (only if fixes needed)

---

### Task 4.5: CLI Tests with CliRunner

**Do:**
1. Create `tests/unit/test_cli_commands.py`
2. Test scrape command argument parsing
3. Test crawl command argument parsing
4. Test status command argument parsing
5. Test file input (-f) handling
6. Test output (-o) handling
7. Mock services to avoid real network calls

**Files:**
- `tests/unit/test_cli_commands.py` - Create - CLI tests

**Done when:**
- [ ] Tests for argument parsing
- [ ] Tests for file/output handling
- [ ] Tests for error cases
- [ ] 80%+ coverage for CLI commands

**Verify:**
```bash
pytest tests/unit/test_cli_commands.py -v --cov=crawl4r.cli
```

**Commit:** `test(cli): add unit tests for CLI commands`

_Requirements: NFR-017_
_Design: Test Strategy section_

---

### Task 4.6: Integration Tests with Real Services

**Do:**
1. Create `tests/integration/test_cli_integration.py`
2. Mark tests with `@pytest.mark.integration`
3. Test scrape command with real Crawl4AI
4. Test crawl command with real pipeline
5. Test status command with real Redis
6. Skip if services unavailable

**Files:**
- `tests/integration/test_cli_integration.py` - Create - integration tests

**Done when:**
- [ ] Tests run against real services
- [ ] Tests skip gracefully if services down
- [ ] End-to-end flows validated

**Verify:**
```bash
pytest tests/integration/test_cli_integration.py -v -m integration
```

**Commit:** `test(cli): add integration tests for CLI commands`

_Requirements: NFR-017_
_Design: Test Strategy section_

---

### Task 4.7: Unit Tests for P1 Services

**Do:**
1. Create `tests/unit/test_mapper_service.py`
2. Create `tests/unit/test_extractor_service.py`
3. Create `tests/unit/test_screenshot_service.py`
4. Test each service's main methods with mocked dependencies

**Files:**
- `tests/unit/test_mapper_service.py` - Create - mapper tests
- `tests/unit/test_extractor_service.py` - Create - extractor tests
- `tests/unit/test_screenshot_service.py` - Create - screenshot tests

**Done when:**
- [ ] Tests for MapperService
- [ ] Tests for ExtractorService
- [ ] Tests for ScreenshotService
- [ ] 85%+ coverage for each

**Verify:**
```bash
pytest tests/unit/test_mapper_service.py tests/unit/test_extractor_service.py tests/unit/test_screenshot_service.py -v
```

**Commit:** `test(services): add unit tests for P1 services`

_Requirements: NFR-017_
_Design: Test Strategy section_

---

### Task 4.8: Testing Complete Checkpoint

**Do:**
1. Run all tests
2. Verify total coverage > 85%
3. Document any gaps

**Verify:**
```bash
pytest --cov=crawl4r --cov-report=term
```

**Done when:**
- [ ] All tests pass
- [ ] Coverage > 85% overall
- [ ] No critical gaps

**Commit:** `test: complete test coverage for web crawl CLI`

_Requirements: NFR-017_

---

## Phase 5: Quality Gates

### Task 5.1: V4 [VERIFY] Full local CI

**Do:**
1. Run complete local CI suite
2. Fix any issues found

**Verify:**
```bash
ruff check . && ty check crawl4r/ && pytest --cov=crawl4r --cov-report=term
```

**Done when:**
- [ ] ruff check passes
- [ ] ty check passes
- [ ] All tests pass
- [ ] Coverage > 85%

**Commit:** `chore: pass local CI` (if fixes needed)

---

### Task 5.2: Documentation Review

**Do:**
1. Verify all public APIs have Google-style docstrings
2. Verify all commands have help text
3. Update CLAUDE.md with new CLI commands
4. Add usage examples to docstrings

**Files:**
- `crawl4r/services/*.py` - Review docstrings
- `crawl4r/cli/commands/*.py` - Review docstrings
- `CLAUDE.md` - Update with CLI documentation

**Done when:**
- [ ] All public methods have docstrings
- [ ] All commands have --help text
- [ ] CLAUDE.md updated

**Verify:**
```bash
python -m crawl4r.cli.app --help
python -m crawl4r.cli.app scrape --help
python -m crawl4r.cli.app crawl --help
```

**Commit:** `docs: update documentation for web crawl CLI`

_Requirements: NFR-016_

---

### Task 5.3: V5 [VERIFY] CI pipeline passes

**Do:**
1. Push branch to origin
2. Create PR if not exists
3. Verify all CI checks pass

**Verify:**
```bash
git push -u origin $(git branch --show-current)
gh pr create --title "feat: add web crawl CLI commands" --body "Implements scrape, crawl, map, extract, screenshot, status, watch commands" || true
gh pr checks --watch
```

**Done when:**
- [ ] All CI checks green
- [ ] PR ready for review

**Commit:** None

---

### Task 5.4: V6 [VERIFY] AC checklist

**Do:**
1. Review requirements.md
2. Verify each acceptance criterion is met
3. Document verification for each

**Verify:**
```bash
# Manual verification against requirements.md
# Mark each AC as verified
```

**Done when:**
- [ ] All AC-1.x verified (scrape command)
- [ ] All AC-2.x verified (crawl command)
- [ ] All AC-3.x verified (map command)
- [ ] All AC-4.x verified (extract command)
- [ ] All AC-5.x verified (screenshot command)
- [ ] All AC-6.x verified (status command)
- [ ] All AC-7.x verified (watch command)

**Commit:** None

---

## Notes

### POC Shortcuts Taken
- Minimal error messages in Phase 1 (improved in Phase 2)
- No URL validation in Phase 1 (added in Phase 2)
- No service health checks in Phase 1 (added in Phase 2)
- Hardcoded retry delays (configurable later)

### Production TODOs
- Add configurable retry delays
- Add rate limiting beyond circuit breaker
- Add support for custom headers/cookies
- Add PDF/image extraction support
- Add sitemap.xml parsing

### Known Limitations
- Screenshot format is PNG only (Crawl4AI limitation)
- No JavaScript rendering configuration
- No robots.txt compliance enforcement
- Single-machine queue coordination only

### Dependencies
- Task 1.4 depends on 1.3 (models)
- Task 1.7 depends on 1.4, 1.5 (scraper, queue)
- Task 1.9-1.11 depend on 1.8 (CLI app)
- Phase 2 depends on Phase 1 completion
- Phase 4 depends on Phases 1-3 completion
