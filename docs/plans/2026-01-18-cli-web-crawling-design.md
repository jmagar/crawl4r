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

class IngestionService:
    """Qdrant ingestion - used by CLI and API."""

    def __init__(self, scraper: ScraperService, tei_endpoint: str, qdrant_url: str):
        self.scraper = scraper
        ...

    async def ingest(self, url: str) -> IngestResult:
        """Scrape → chunk → embed → upsert"""
        ...

    async def ingest_batch(self, urls: list[str]) -> list[IngestResult]:
        """Batch ingest with progress tracking"""
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
7. **Deep crawling support**: `--depth` flag for following links
8. **Both schema and prompt extraction**: Flexibility for structured data extraction

## Output Behavior

**Progress display:**
```
Crawling https://docs.example.com... ✓ 11 chunks
Crawling https://api.example.com... ✓ 8 chunks
Crawling https://bad-url.example... ✗ Connection timeout

Summary: 2/3 URLs ingested (19 chunks total)
Failed: https://bad-url.example
```

**Exit codes:**
- `0`: All URLs processed successfully
- `1`: Some URLs failed (failures reported)

## Dependencies

- `typer` - CLI framework
- `rich` - Progress display (included with typer)
- `httpx` - Async HTTP client (existing)

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
