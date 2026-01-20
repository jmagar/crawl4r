# CLI Web Crawling Design

> Design document for adding web crawling commands to the crawl4r CLI.

## Overview

Add web crawling capabilities to the CLI with clear separation between service layer (reusable by API) and CLI commands (thin wrappers).

## Commands

| Command | Purpose | Default Behavior |
|---------|---------|------------------|
| `scrape` | Fetch single page | Print markdown to stdout |
| `crawl` | Fetch + ingest to Qdrant | Ingest to configured collection |
| `map` | URL discovery | List URLs to stdout |
| `extract` | Structured data extraction | Print JSON to stdout |
| `screenshot` | Capture page screenshot | Save as `{domain}.png` |
| `watch` | File monitoring (existing) | Monitor `WATCH_FOLDER` |
| `status` | Check crawl status | Show progress/results for crawl ID |

## Command Reference

```bash
# === SCRAPE (fetch only) ===
crawl4r scrape <url>                    # Print markdown to stdout
crawl4r scrape <url> -o page.md         # Save to file
crawl4r scrape <urls...> -o ./output/   # Batch to directory
crawl4r scrape -f urls.txt -o ./output/ # Batch from file

# === CRAWL (fetch + ingest) ===
crawl4r crawl <url>                     # Ingest to Qdrant
crawl4r crawl <urls...>                 # Batch ingest
crawl4r crawl -f urls.txt               # Batch from file
crawl4r crawl <url> --depth 2           # Deep crawl + ingest

# === MAP (URL discovery) ===
crawl4r map <url>                       # List URLs to stdout
crawl4r map <url> --depth 2             # Discover N levels deep
crawl4r map <url> -o urls.txt           # Save to file

# === EXTRACT (structured data) ===
crawl4r extract <url> --schema schema.json      # With JSON schema
crawl4r extract <url> --prompt "extract X"      # With prompt
crawl4r extract <url> -s schema.json -o out.json

# === SCREENSHOT ===
crawl4r screenshot <url>                # Save as {domain}.png
crawl4r screenshot <url> -o page.png    # Custom filename
crawl4r screenshot <url> --full-page    # Full page capture

# === WATCH (file monitoring) ===
crawl4r watch                           # Monitor WATCH_FOLDER
crawl4r watch --folder /path/to/docs    # Custom folder

# === STATUS (check crawl progress) ===
crawl4r status <crawl-id>               # Check crawl progress/results
crawl4r status --list                   # List recent crawls
crawl4r status --active                 # Show active crawl
```

## Architecture

### Directory Structure

```
crawl4r/
├── services/                    # Core business logic (used by CLI + API)
│   ├── __init__.py
│   ├── scraper.py              # Single page scraping
│   ├── mapper.py               # URL discovery/mapping
│   ├── extractor.py            # Structured data extraction
│   ├── screenshot.py           # Screenshot capture
│   └── ingestion.py            # Qdrant ingestion pipeline
│
├── cli/
│   ├── app.py                  # Typer app entry point
│   └── commands/
│       ├── scrape.py           # crawl4r scrape
│       ├── crawl.py            # crawl4r crawl
│       ├── map.py              # crawl4r map
│       ├── extract.py          # crawl4r extract
│       ├── screenshot.py       # crawl4r screenshot
│       └── watch.py            # crawl4r watch (refactored)
│
├── api/                        # FastAPI (future)
│   ├── app.py
│   └── routes/
│       ├── scrape.py           # Uses services/scraper.py
│       ├── crawl.py            # Uses services/ingestion.py
│       └── ...
```

### Service Layer

```python
# crawl4r/services/scraper.py
@dataclass
class ScrapeResult:
    url: str
    markdown: str
    html: str | None
    metadata: dict
    links: dict | None
    success: bool
    error: str | None

class ScraperService:
    """Single page scraping - used by CLI and API."""

    def __init__(self, endpoint_url: str = "http://localhost:52004"):
        self.endpoint_url = endpoint_url

    async def scrape(self, url: str, include_links: bool = False) -> ScrapeResult:
        """Fetch clean markdown via /md?f=fit"""
        ...

    async def scrape_batch(self, urls: list[str]) -> list[ScrapeResult]:
        """Batch scrape with concurrency control"""
        ...
```

```python
# crawl4r/services/mapper.py
@dataclass
class MapResult:
    seed_url: str
    urls: list[dict]  # [{href, text, title, base_domain}]
    total_found: int
    depth_reached: int
    # Note: scores (intrinsic_score, contextual_score) require LinkPreviewConfig

class MapperService:
    """URL discovery - used by CLI and API."""

    async def map(self, url: str, depth: int = 0) -> MapResult:
        """Discover URLs via /crawl endpoint links"""
        ...
```

```python
# crawl4r/services/extractor.py
@dataclass
class ExtractResult:
    url: str
    data: dict | list
    schema_used: dict | None
    prompt_used: str | None

class ExtractorService:
    """Structured extraction - used by CLI and API."""

    async def extract_with_schema(self, url: str, schema: dict) -> ExtractResult:
        """Extract via /llm/job with JSON schema"""
        ...

    async def extract_with_prompt(self, url: str, prompt: str) -> ExtractResult:
        """Extract via /llm/job with prompt only"""
        ...
```

```python
# crawl4r/services/screenshot.py
@dataclass
class ScreenshotResult:
    url: str
    path: Path
    base64_data: str  # Raw PNG base64 from API
    full_page: bool
    success: bool
    error: str | None = None
    # Note: Only PNG format supported by Crawl4AI

class ScreenshotService:
    """Screenshot capture - used by CLI and API."""

    async def capture(
        self,
        url: str,
        output: Path,
        full_page: bool = False,
        wait_for: float | None = None,
    ) -> ScreenshotResult:
        """Capture via /screenshot endpoint"""
        ...
```

```python
# crawl4r/services/ingestion.py
@dataclass
class IngestResult:
    url: str
    chunks: int
    points_upserted: int
    collection: str
    success: bool
    time_taken: float  # seconds
    error: str | None = None

@dataclass
class CrawlStatus:
    crawl_id: str
    status: str  # QUEUED, RUNNING, COMPLETED, FAILED
    urls: list[str]
    total_urls: int
    completed_urls: int
    current_url: str | None
    started_at: str | None  # ISO timestamp
    completed_at: str | None  # ISO timestamp
    results: list[IngestResult]
    queue_position: int | None  # If QUEUED

class IngestionService:
    """Qdrant ingestion with Redis queue coordination - used by CLI and API."""

    def __init__(
        self,
        scraper: ScraperService,
        tei_endpoint: str,
        qdrant_url: str,
        redis_url: str = "redis://localhost:53379",
    ):
        self.scraper = scraper
        self.redis = redis.from_url(redis_url)
        self.lock_key = "crawl4r:crawl:lock"
        self.queue_key = "crawl4r:crawl:queue"
        self.status_key_prefix = "crawl4r:status:"
        ...

    def generate_crawl_id(self) -> str:
        """Generate unique crawl ID (crawl_<timestamp>_<random>)"""
        ...

    async def acquire_lock(self, crawl_id: str, timeout: int = 3600) -> bool:
        """Acquire crawl lock with TTL (cross-process coordination)"""
        ...

    async def release_lock(self) -> None:
        """Release crawl lock"""
        ...

    async def add_to_queue(self, crawl_id: str, urls: list[str]) -> tuple[bool, int]:
        """Add crawl to Redis queue, return (success, queue_position)"""
        ...

    async def get_queue_size(self) -> int:
        """Get current queue size"""
        ...

    async def update_status(self, crawl_id: str, status: CrawlStatus) -> None:
        """Update crawl status in Redis (expires after 24 hours)"""
        ...

    async def get_status(self, crawl_id: str) -> CrawlStatus | None:
        """Get crawl status from Redis"""
        ...

    async def list_recent_crawls(self, limit: int = 10) -> list[CrawlStatus]:
        """List recent crawls (sorted by timestamp)"""
        ...

    async def get_active_crawl(self) -> CrawlStatus | None:
        """Get currently running crawl (if any)"""
        ...

    async def ingest(self, url: str) -> IngestResult:
        """Scrape → chunk → embed → upsert"""
        ...

    async def ingest_batch(
        self,
        urls: list[str],
        crawl_id: str,
    ) -> list[IngestResult]:
        """Sequential batch ingest with progress tracking (1 URL at a time)

        Updates status in Redis as it progresses.
        """
        ...

    async def process_queue(self) -> list[IngestResult]:
        """Process next crawl in Redis queue (called after batch completes)"""
        ...
```

### CLI Commands (Thin Wrappers)

```python
# crawl4r/cli/commands/scrape.py
import typer
from crawl4r.services.scraper import ScraperService

app = typer.Typer()

@app.command()
def scrape(
    urls: list[str] = typer.Argument(...),
    file: Path = typer.Option(None, "-f", "--file"),
    output: Path = typer.Option(None, "-o", "--output"),
):
    """Scrape URLs and output markdown."""
    service = ScraperService()
    # ... delegate to service
```

```python
# crawl4r/cli/commands/crawl.py
import typer
from rich.console import Console
from crawl4r.services.ingestion import IngestionService, CrawlStatus

app = typer.Typer()
console = Console()

@app.command()
def crawl(
    urls: list[str] = typer.Argument(...),
    file: Path = typer.Option(None, "-f", "--file"),
    depth: int = typer.Option(0, "--depth", help="Follow links N levels deep"),
):
    """Crawl URLs and ingest to Qdrant."""
    service = IngestionService(...)
    crawl_id = service.generate_crawl_id()

    # Merge URLs from positional args and file
    all_urls = urls + load_from_file(file) if file else urls

    # Try to acquire lock
    if not await service.acquire_lock(crawl_id):
        # Lock held by another process, add to queue
        success, position = await service.add_to_queue(crawl_id, all_urls)

        if success:
            console.print(f"✓ Successfully queued crawl", style="green")
            console.print(f"Crawl ID: {crawl_id}", style="cyan")
            console.print(f"Status: QUEUED (position {position} in queue)", style="yellow")
            console.print(f"\nCheck status: crawl4r status {crawl_id}", style="dim")

            # Update status in Redis
            await service.update_status(crawl_id, CrawlStatus(
                crawl_id=crawl_id,
                status="QUEUED",
                urls=all_urls,
                total_urls=len(all_urls),
                completed_urls=0,
                current_url=None,
                started_at=None,
                completed_at=None,
                results=[],
                queue_position=position,
            ))
        else:
            console.print("✗ Failed to queue crawl", style="red")
        return

    try:
        # Show crawl info
        console.print(f"Crawl ID: {crawl_id}", style="cyan")
        console.print(f"Status: RUNNING", style="green")

        # Process URLs
        results = await service.ingest_batch(all_urls, crawl_id)

        # Check queue and process if any
        queue_size = await service.get_queue_size()
        if queue_size > 0:
            console.print(f"\nQueue has {queue_size} crawls, processing next...")
            queue_results = await service.process_queue()
            results.extend(queue_results)
    finally:
        await service.release_lock()
```

```python
# crawl4r/cli/commands/status.py
import typer
from rich.console import Console
from rich.table import Table
from crawl4r.services.ingestion import IngestionService

app = typer.Typer()
console = Console()

@app.command()
def status(
    crawl_id: str = typer.Argument(None, help="Crawl ID to check"),
    list_crawls: bool = typer.Option(False, "--list", help="List recent crawls"),
    active: bool = typer.Option(False, "--active", help="Show active crawl"),
):
    """Check crawl status."""
    service = IngestionService(...)

    if list_crawls:
        # List recent crawls
        crawls = await service.list_recent_crawls(limit=10)
        table = Table(title="Recent Crawls")
        table.add_column("Crawl ID", style="cyan")
        table.add_column("Status")
        table.add_column("URLs")
        table.add_column("Progress")
        table.add_column("Started")

        for crawl in crawls:
            status_color = {
                "QUEUED": "yellow",
                "RUNNING": "green",
                "COMPLETED": "blue",
                "FAILED": "red"
            }.get(crawl.status, "white")

            table.add_row(
                crawl.crawl_id,
                f"[{status_color}]{crawl.status}[/{status_color}]",
                str(crawl.total_urls),
                f"{crawl.completed_urls}/{crawl.total_urls}",
                crawl.started_at or "Not started"
            )

        console.print(table)
        return

    if active:
        # Show active crawl
        active_crawl = await service.get_active_crawl()
        if not active_crawl:
            console.print("No active crawl", style="yellow")
            return
        crawl_id = active_crawl.crawl_id

    if not crawl_id:
        console.print("Error: Provide crawl ID or use --list/--active", style="red")
        raise typer.Exit(1)

    # Get status for specific crawl
    crawl_status = await service.get_status(crawl_id)
    if not crawl_status:
        console.print(f"Crawl {crawl_id} not found", style="red")
        raise typer.Exit(1)

    # Display status
    status_color = {
        "QUEUED": "yellow",
        "RUNNING": "green",
        "COMPLETED": "blue",
        "FAILED": "red"
    }.get(crawl_status.status, "white")

    console.print(f"\nCrawl ID: {crawl_status.crawl_id}", style="cyan")
    console.print(f"Status: [{status_color}]{crawl_status.status}[/{status_color}]")
    console.print(f"Progress: {crawl_status.completed_urls}/{crawl_status.total_urls} URLs")

    if crawl_status.queue_position:
        console.print(f"Queue Position: {crawl_status.queue_position}")

    if crawl_status.current_url:
        console.print(f"Current URL: {crawl_status.current_url}")

    if crawl_status.started_at:
        console.print(f"Started: {crawl_status.started_at}")

    if crawl_status.completed_at:
        console.print(f"Completed: {crawl_status.completed_at}")

    # Show results summary
    if crawl_status.results:
        successful = [r for r in crawl_status.results if r.success]
        failed = [r for r in crawl_status.results if not r.success]

        console.print(f"\nResults:")
        console.print(f"  ✓ Successful: {len(successful)}")
        console.print(f"  ✗ Failed: {len(failed)}")

        if successful:
            total_chunks = sum(r.chunks for r in successful)
            console.print(f"  Total chunks ingested: {total_chunks}")
```

## Crawl4AI API Endpoints Used

| Service | Endpoint | Purpose |
|---------|----------|---------|
| `ScraperService` | `POST /md` with `f=fit` | Clean markdown extraction |
| `MapperService` | `POST /crawl` | Get links for URL discovery |
| `ExtractorService` | `POST /llm/job` | Structured extraction with schema/prompt |
| `ScreenshotService` | `POST /screenshot` | Page screenshots |

### Why `/md?f=fit` for scraping

| Endpoint | Size | Quality |
|----------|------|---------|
| `/crawl` (raw_markdown) | ~89K | Nav/footer cruft |
| `/crawl` (fit_markdown) | 0 | Empty (not populated) |
| `/md?f=fit` | ~12K | Clean main content |

Always use `/md` with `f=fit` for clean markdown extraction.

## Design Decisions

1. **Service layer separation**: Business logic in `services/` for reuse by CLI and future API
2. **Typer for CLI**: Modern, type hints, auto-generated help
3. **Default to Qdrant ingest**: `crawl` command ingests by default (primary RAG use case)
4. **Always use fit filter**: Clean markdown without nav cruft
5. **Continue on error**: Batch operations report failures at end, don't fail fast
6. **Progress per URL**: Show status for each URL as it processes
7. **Deep crawling support**: `--depth` flag for following links (same-domain only)
8. **Both schema and prompt extraction**: Flexibility for structured data extraction

## Behavior Details

### URL Input Merging
When both positional URLs and `-f urls.txt` are provided, **merge both sources** and process all URLs:
```bash
crawl4r scrape https://a.com https://b.com -f urls.txt
# Processes: a.com, b.com, + all URLs from urls.txt
```

### Deep Crawling with `--depth`
- **Ingestion**: Ingest ALL discovered pages up to depth N (not just seed URL)
- **Domain restriction**: Only follow links within the seed URL's domain (prevents crawling entire web)
- **Example**: `crawl4r crawl https://docs.example.com --depth 2`
  - Crawls seed URL + all same-domain links up to 2 levels deep
  - Ingests all discovered pages to Qdrant

### Concurrency Control
- **`crawl` command**: Process 1 URL at a time (queue remaining if crawl running)
  - Prevents overwhelming Qdrant with concurrent ingestion
  - Shows queue status in progress display
  - **Queue mechanism**: Redis-based queue for cross-process coordination
    - Uses Redis list (`LPUSH`/`RPOP`) for persistent queue
    - Uses Redis key (`crawl4r:crawl:lock`) with TTL for active crawl detection
    - If crawl is in progress (lock exists), new URLs added to Redis queue
    - Progress shows: `"Crawling https://a.com... (2 queued)"`
    - **Cross-terminal support**: Multiple `crawl4r crawl` invocations coordinate via Redis
      - Terminal 1: `crawl4r crawl url1` (acquires lock, starts crawling)
      - Terminal 2: `crawl4r crawl url2` (detects lock, adds to queue, exits or watches)
    - Lock TTL: 1 hour (prevents stale locks if process crashes)
    - Queue key: `crawl4r:crawl:queue`
- **`scrape` command**: Process 5 URLs concurrently for batch operations
  - Faster batch scraping without overwhelming Crawl4AI service
  - Balanced throughput without rate limiting issues
  - No queue coordination needed (lightweight operation)

## Output Behavior

**Progress display:**
```
Crawling https://docs.example.com... (2 queued) ✓ 11 chunks
Crawling https://api.example.com... (1 queued) ✓ 8 chunks
Crawling https://bad-url.example... ✗ Connection timeout

Summary: 2/3 URLs ingested (19 chunks total)
Failed: https://bad-url.example
```

**Queueing behavior (cross-terminal):**
```bash
# Terminal 1: Start first crawl
crawl4r crawl https://docs.example.com --depth 2
# Output:
# Crawl ID: crawl_abc123def456
# Status: RUNNING
# Crawling https://docs.example.com...

# Terminal 2: While Terminal 1 is running, start second crawl
crawl4r crawl https://api.example.com
# Output:
# ✓ Successfully queued crawl
# Crawl ID: crawl_xyz789ghi012
# Status: QUEUED (position 1 in queue)
# Check status: crawl4r status crawl_xyz789ghi012

# Terminal 2: Check status later
crawl4r status crawl_xyz789ghi012
# Output:
# Crawl ID: crawl_xyz789ghi012
# Status: RUNNING
# Progress: 3/5 URLs completed
# Started: 2026-01-18 14:23:45
# Current: https://api.example.com/docs

# Terminal 1: After finishing docs.example.com, automatically processes queue
# Output: "Queue has 1 crawl, processing crawl_xyz789ghi012..."
# Output: "Crawling https://api.example.com..."
```

**Lock recovery:**
- Lock TTL set to 1 hour to prevent stale locks from crashed processes
- If lock exists but process is dead, next crawl detects and recovers:
  ```bash
  crawl4r crawl https://example.com
  # Output: "Stale lock detected (process 12345 not found), acquiring lock..."
  ```

**Exit codes:**
- `0`: All URLs processed successfully
- `1`: Some URLs failed (failures reported)

## Dependencies

- `typer` - CLI framework
- `rich` - Progress display (included with typer)
- `httpx` - Async HTTP client (existing)
- `redis` - Cross-process queue coordination (existing service)

## Reuse Existing Utilities

The following existing components should be reused (not duplicated):

| Utility | Location | Use In |
|---------|----------|--------|
| `CircuitBreaker` | `resilience/circuit_breaker.py` | All services (protect against cascading failures) |
| `MetadataKeys` | `core/metadata.py` | Consistent metadata field names |
| `TEIClient` | `storage/tei.py` | `IngestionService` for embeddings |
| `VectorStoreManager` | `storage/qdrant.py` | `IngestionService` for vector storage |
| Retry with backoff | Pattern in `tei.py` | All HTTP calls to Crawl4AI |
| Point ID generation | `qdrant.py._generate_point_id()` | `IngestionService` (use `SHA256(url:chunk_index)`) |

**Deduplication strategy**: Before ingesting a URL, call `VectorStoreManager.delete_by_url(source_url)` to remove old chunks (existing pattern).

## Entry Point

```toml
# pyproject.toml
[project.scripts]
crawl4r = "crawl4r.cli.app:app"
```

## Future API Integration

The API routes will be thin wrappers around the same services:

```python
# crawl4r/api/routes/scrape.py
from fastapi import APIRouter
from crawl4r.services.scraper import ScraperService

router = APIRouter()

@router.post("/scrape")
async def scrape(url: str):
    service = ScraperService()
    return await service.scrape(url)
```
