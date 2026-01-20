---
spec: web-crawl-cli
phase: requirements
created: 2026-01-18
---

# Requirements: Web Crawl CLI

## Overview

Add web crawling capabilities to the crawl4r CLI with commands for scraping, crawling with Qdrant ingestion, URL discovery, structured extraction, screenshots, and status tracking. The architecture separates service layer (reusable by CLI and future API) from CLI commands (thin Typer wrappers).

## Goal

Enable users to crawl web pages from the command line with full RAG pipeline integration, including markdown extraction, vector embedding, and Qdrant storage. Support both single-page operations and batch processing with cross-process queue coordination via Redis.

---

## User Stories

### US-1: Single Page Scraping

**As a** developer
**I want to** fetch a web page and get clean markdown
**So that** I can quickly inspect or save page content without ingesting to Qdrant

**Acceptance Criteria:**
- [ ] AC-1.1: `crawl4r scrape <url>` prints clean markdown to stdout
- [ ] AC-1.2: `crawl4r scrape <url> -o page.md` saves to specified file
- [ ] AC-1.3: `crawl4r scrape <url1> <url2>` processes multiple URLs
- [ ] AC-1.4: `crawl4r scrape -f urls.txt` reads URLs from file (one per line)
- [ ] AC-1.5: Both positional URLs and `-f` URLs are merged when both provided
- [ ] AC-1.6: Output directory (`-o ./output/`) creates files named by domain
- [ ] AC-1.7: Uses `/md?f=fit` endpoint for clean markdown (not raw HTML cruft)
- [ ] AC-1.8: Shows progress for batch operations (URL count, current URL)
- [ ] AC-1.9: Continues processing on individual URL failures (reports at end)
- [ ] AC-1.10: Exit code 0 if all succeed, 1 if any fail

---

### US-2: Crawl with Qdrant Ingestion

**As a** developer
**I want to** crawl URLs and ingest them directly to Qdrant
**So that** I can build a searchable knowledge base from web content

**Acceptance Criteria:**
- [ ] AC-2.1: `crawl4r crawl <url>` scrapes, chunks, embeds, and upserts to Qdrant
- [ ] AC-2.2: `crawl4r crawl <url1> <url2> -f urls.txt` merges all URL sources
- [ ] AC-2.3: Each crawl generates unique ID (`crawl_<timestamp>_<random>`)
- [ ] AC-2.4: Progress shows: URL being processed, chunks created, queue depth
- [ ] AC-2.5: `--depth N` follows same-domain links up to N levels deep
- [ ] AC-2.6: Deduplication removes existing URL data before re-ingesting
- [ ] AC-2.7: Uses configured Qdrant collection from environment
- [ ] AC-2.8: Summary shows: successful/failed URLs, total chunks ingested
- [ ] AC-2.9: Exit code 0 if all succeed, 1 if any fail

**Queue Coordination (Redis):**
- [ ] AC-2.10: Acquires lock before starting crawl (1-hour TTL)
- [ ] AC-2.11: If lock held, adds crawl to Redis queue and displays queue position
- [ ] AC-2.12: Updates status in Redis as crawl progresses
- [ ] AC-2.13: After completing, processes next queued crawl automatically
- [ ] AC-2.14: Releases lock on completion or error
- [ ] AC-2.15: Recovers from stale locks (detects dead processes)

---

### US-3: URL Discovery (Map)

**As a** developer
**I want to** discover all URLs on a page or site
**So that** I can plan crawl scope or generate URL lists for batch processing

**Acceptance Criteria:**
- [ ] AC-3.1: `crawl4r map <url>` lists all links on the page to stdout
- [ ] AC-3.2: Output includes: href, link text, base domain
- [ ] AC-3.3: `--depth N` recursively discovers URLs up to N levels
- [ ] AC-3.4: `-o urls.txt` saves discovered URLs to file
- [ ] AC-3.5: `--same-domain` filters to only same-domain links (default: true)
- [ ] AC-3.6: `--include-external` includes external domain links
- [ ] AC-3.7: Deduplicates discovered URLs
- [ ] AC-3.8: Shows count of unique URLs discovered

---

### US-4: Structured Data Extraction

**As a** developer
**I want to** extract structured data from web pages
**So that** I can get specific information in a machine-readable format

**Acceptance Criteria:**
- [ ] AC-4.1: `crawl4r extract <url> --schema schema.json` extracts per JSON schema
- [ ] AC-4.2: `crawl4r extract <url> --prompt "extract X"` extracts via LLM prompt
- [ ] AC-4.3: Either `--schema` or `--prompt` is required (mutually exclusive)
- [ ] AC-4.4: Output is valid JSON printed to stdout
- [ ] AC-4.5: `-o output.json` saves to file
- [ ] AC-4.6: Schema file must be valid JSON schema
- [ ] AC-4.7: Clear error message if extraction fails

---

### US-5: Screenshot Capture

**As a** developer
**I want to** capture screenshots of web pages
**So that** I can document page appearance or debug rendering issues

**Acceptance Criteria:**
- [ ] AC-5.1: `crawl4r screenshot <url>` saves as `{domain}.png` in current directory
- [ ] AC-5.2: `-o page.png` saves to specified filename
- [ ] AC-5.3: `--full-page` captures entire scrollable page
- [ ] AC-5.4: `--wait N` waits N seconds before capture (for JS rendering)
- [ ] AC-5.5: Output format is PNG only (Crawl4AI limitation)
- [ ] AC-5.6: Reports file path and size on success

---

### US-6: Crawl Status Tracking

**As a** developer
**I want to** check the status of crawl operations
**So that** I can monitor progress and troubleshoot failures

**Acceptance Criteria:**
- [ ] AC-6.1: `crawl4r status <crawl-id>` shows detailed status for specific crawl
- [ ] AC-6.2: Status display includes: status, progress, current URL, timestamps
- [ ] AC-6.3: `--list` shows recent crawls (last 10 by default)
- [ ] AC-6.4: `--active` shows currently running crawl (if any)
- [ ] AC-6.5: Status values: QUEUED, RUNNING, COMPLETED, FAILED
- [ ] AC-6.6: Shows queue position for QUEUED crawls
- [ ] AC-6.7: Shows results summary for COMPLETED crawls (success/fail counts)
- [ ] AC-6.8: Status data expires after 24 hours

---

### US-7: File Watch Mode

**As a** developer
**I want to** monitor a folder for file changes
**So that** I can automatically ingest local documents to Qdrant

**Acceptance Criteria:**
- [ ] AC-7.1: `crawl4r watch` monitors configured WATCH_FOLDER from environment
- [ ] AC-7.2: `--folder /path` overrides default watch folder
- [ ] AC-7.3: Processes new files on startup (batch recovery)
- [ ] AC-7.4: Monitors for create, modify, delete events
- [ ] AC-7.5: Graceful shutdown on Ctrl+C
- [ ] AC-7.6: Shows event count and queue depth in logs
- [ ] AC-7.7: Refactored from existing `crawl4r/cli/main.py` functionality

---

## Functional Requirements

### Service Layer

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-001 | ScraperService fetches clean markdown via `/md?f=fit` | P0 | Returns ScrapeResult with markdown, metadata, success flag |
| FR-002 | ScraperService supports batch operations with concurrency limit (5) | P0 | Processes N URLs concurrently without overwhelming service |
| FR-003 | MapperService discovers URLs via `/crawl` endpoint links | P0 | Returns MapResult with URL list, count, depth reached |
| FR-004 | MapperService supports recursive depth crawling | P1 | Follows links up to N levels, same-domain by default |
| FR-005 | ExtractorService extracts via `/llm/job` with JSON schema | P1 | Returns ExtractResult with structured data |
| FR-006 | ExtractorService extracts via `/llm/job` with prompt | P1 | Returns ExtractResult with LLM-generated data |
| FR-007 | ScreenshotService captures via `/screenshot` endpoint | P1 | Returns ScreenshotResult with base64 data, file path |
| FR-008 | IngestionService orchestrates scrape-chunk-embed-upsert pipeline | P0 | Returns IngestResult with chunk count, points upserted |
| FR-009 | IngestionService manages Redis queue for cross-process coordination | P0 | Supports lock, queue, status operations |
| FR-010 | IngestionService generates unique crawl IDs | P0 | Format: `crawl_<timestamp>_<random>` |
| FR-011 | IngestionService deduplicates by URL before ingesting | P0 | Calls delete_by_url before upserting new vectors |

### CLI Commands

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-012 | CLI entry point configured in pyproject.toml | P0 | `crawl4r = "crawl4r.cli.app:app"` enables CLI invocation |
| FR-013 | Typer app with subcommands for all operations | P0 | `crawl4r <command> --help` shows command-specific help |
| FR-014 | `scrape` command delegates to ScraperService | P0 | Thin wrapper with Rich progress display |
| FR-015 | `crawl` command delegates to IngestionService | P0 | Shows crawl ID, queue status, progress |
| FR-016 | `map` command delegates to MapperService | P1 | Lists URLs to stdout or file |
| FR-017 | `extract` command delegates to ExtractorService | P1 | Outputs JSON to stdout or file |
| FR-018 | `screenshot` command delegates to ScreenshotService | P1 | Saves PNG file, reports path |
| FR-019 | `status` command queries Redis for crawl status | P0 | Shows progress, queue position, results |
| FR-020 | `watch` command refactored from main.py | P0 | Maintains existing file monitoring functionality |

### Error Handling

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-021 | All services use CircuitBreaker for Crawl4AI calls | P0 | Opens after 5 failures, 60s timeout |
| FR-022 | Retry with exponential backoff for transient errors | P0 | Delays: 1s, 2s, 4s for normal operations |
| FR-023 | Batch operations continue on individual failures | P0 | Reports failures at end, does not abort |
| FR-024 | Clear error messages for user-facing errors | P0 | Includes URL, error type, suggestion |
| FR-025 | Lock recovery for stale Redis locks | P1 | Detects dead process, logs recovery |

### Configuration

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-026 | Services read endpoints from Settings (environment) | P0 | CRAWL4AI_ENDPOINT, TEI_ENDPOINT, QDRANT_URL, REDIS_URL |
| FR-027 | CLI options override environment defaults | P1 | `--endpoint`, `--collection` flags where appropriate |
| FR-028 | Validate service availability on startup | P0 | Check Crawl4AI health before batch operations |

---

## Non-Functional Requirements

### Performance

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-001 | Scrape command concurrency | Concurrent requests | 5 URLs simultaneously |
| NFR-002 | Crawl command throughput | Processing time | 1 URL at a time (sequential for Qdrant stability) |
| NFR-003 | Redis queue latency | Lock/queue operations | < 100ms per operation |
| NFR-004 | Status lookup time | Redis query | < 50ms for single status |
| NFR-005 | Batch scrape throughput | URLs per minute | > 30 URLs/min with concurrency |

### Reliability

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-006 | Circuit breaker threshold | Failures before open | 5 consecutive failures |
| NFR-007 | Circuit breaker recovery | Timeout before retry | 60 seconds |
| NFR-008 | Lock TTL | Maximum hold time | 1 hour (prevents stale locks) |
| NFR-009 | Status retention | Redis TTL | 24 hours |
| NFR-010 | Retry attempts | Max retries | 3 attempts with backoff |

### Observability

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-011 | Progress display | Update frequency | Per-URL for batch operations |
| NFR-012 | Structured logging | Log format | Consistent with existing crawl4r logging |
| NFR-013 | Exit codes | Semantic meaning | 0=success, 1=partial failure, 2=total failure |

### Maintainability

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-014 | Service layer separation | Coupling | Zero CLI imports in services |
| NFR-015 | Type hints | Coverage | 100% of public APIs |
| NFR-016 | Docstrings | Coverage | Google-style for all public functions |
| NFR-017 | Test coverage | Line coverage | > 85% for new code |

---

## Dependencies

### External Dependencies (New)

| Dependency | Version | Purpose |
|------------|---------|---------|
| `typer[all]` | >= 0.12.0 | CLI framework with Rich integration |
| `redis` | >= 5.0.0 | Queue coordination, status storage |

### Internal Dependencies (Existing)

| Component | Location | Reuse Purpose |
|-----------|----------|---------------|
| CircuitBreaker | `resilience/circuit_breaker.py` | Protect all HTTP calls |
| MetadataKeys | `core/metadata.py` | Consistent metadata field names |
| TEIClient | `storage/tei.py` | Embedding generation in IngestionService |
| VectorStoreManager | `storage/qdrant.py` | Vector storage in IngestionService |
| DocumentProcessor | `processing/processor.py` | Chunking and processing |
| Settings | `core/config.py` | Environment configuration |
| get_logger | `core/logger.py` | Structured logging |

### Service Dependencies (Runtime)

| Service | Endpoint | Required For |
|---------|----------|--------------|
| Crawl4AI | `http://localhost:52004` | scrape, crawl, map, extract, screenshot |
| TEI Embeddings | `http://100.74.16.82:52000` | crawl (embedding generation) |
| Qdrant | `http://localhost:52001` | crawl (vector storage), status (deduplication) |
| Redis | `redis://localhost:53379` | crawl queue, status tracking |

---

## Glossary

| Term | Definition |
|------|------------|
| **Crawl** | Full pipeline: fetch URL, chunk content, generate embeddings, store in Qdrant |
| **Scrape** | Fetch URL and return markdown content only (no storage) |
| **Map** | Discover all links on a page or site for planning crawl scope |
| **Extract** | Pull structured data from page using JSON schema or LLM prompt |
| **Crawl ID** | Unique identifier for a crawl operation (`crawl_<timestamp>_<random>`) |
| **Queue Position** | Position in Redis queue when a crawl is waiting for active crawl to complete |
| **Circuit Breaker** | Pattern that stops requests after repeated failures, preventing cascade |
| **Deduplication** | Removing existing vectors for a URL before re-ingesting |
| **Fit Filter** | Crawl4AI `/md?f=fit` parameter that extracts main content without nav/footer |

---

## Out of Scope

The following are explicitly NOT included in this specification:

1. **API endpoints** - Future work; services designed for reuse but API routes not implemented
2. **Authentication** - No auth for Crawl4AI or Redis (local development focus)
3. **Rate limiting** - Beyond circuit breaker; no token bucket or sliding window
4. **Distributed crawling** - Single-machine queue coordination only
5. **Scheduled crawls** - No cron or scheduled job support
6. **Crawl resumption** - Failed crawls must be restarted manually
7. **URL prioritization** - FIFO queue only, no priority ranking
8. **Robots.txt compliance** - Not enforced by CLI (user responsibility)
9. **Sitemap parsing** - Use `map --depth` instead of sitemap.xml
10. **PDF/image extraction** - Web pages only (markdown content)
11. **JavaScript rendering configuration** - Uses Crawl4AI defaults
12. **Custom headers/cookies** - Uses Crawl4AI defaults

---

## Success Criteria

This specification is complete when:

1. **All 7 commands functional**: scrape, crawl, map, extract, screenshot, status, watch
2. **Service layer separation verified**: Services have zero CLI imports
3. **Queue coordination working**: Multiple terminals can queue crawls via Redis
4. **Progress display implemented**: Rich-based progress for all batch operations
5. **Test coverage achieved**: > 85% for new service and CLI code
6. **Documentation complete**: All public APIs have Google-style docstrings
7. **Integration tests pass**: End-to-end tests with real services
8. **Ruff and ty clean**: No linting or type errors
