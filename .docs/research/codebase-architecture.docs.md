# Crawl4r Codebase Architecture Research

## Overview

Crawl4r is a production-ready web crawling and RAG (Retrieval-Augmented Generation) ingestion pipeline built with Python 3.10+, featuring a comprehensive CLI interface, async-first architecture, and robust error handling. The codebase consists of ~9,800 lines of production code and ~25,700 lines of test code (786 tests, 87%+ coverage), demonstrating strong test-driven development practices.

**Key Characteristics:**
- Async-first design using asyncio throughout
- LlamaIndex integration for document processing and embeddings
- Circuit breaker pattern for fault tolerance
- Environment-aware configuration (Docker vs host)
- Modular architecture with clear separation of concerns

## Project Structure

```
crawl4r/
├── crawl4r/              # Main package (54 Python files, ~9,800 LOC)
│   ├── api/              # FastAPI REST API (skeleton only)
│   ├── cli/              # Typer CLI with 7 commands
│   ├── core/             # Infrastructure (config, logging, metadata)
│   ├── processing/       # Document processing pipeline
│   ├── readers/          # Data sources (crawl4ai, file_watcher)
│   ├── resilience/       # Fault tolerance (circuit breaker, retry)
│   ├── services/         # Business logic (ingestion, mapper, scraper)
│   └── storage/          # Backends (TEI, Qdrant, LlamaIndex wrappers)
├── tests/                # 86 test files, ~25,700 LOC, 786 tests
│   ├── unit/             # Fast, mocked tests (majority)
│   ├── integration/      # Real service tests (requires Docker)
│   └── performance/      # Load and memory tests
├── specs/                # Design specifications (web-crawl-cli, llamaindex-reader)
├── docs/                 # Completed implementation plans
└── examples/             # Usage examples
```

## Technology Stack (Actual Implementation)

### Core Dependencies
- **Framework**: Python 3.10+ with asyncio for all I/O operations
- **CLI**: Typer with Rich for interactive terminal output
- **HTTP Client**: httpx 0.28.1 for async requests
- **Validation**: Pydantic 2.0+ with pydantic-settings for configuration
- **LlamaIndex**: Core orchestration, node parsing, embeddings interface
- **Vector Store**: qdrant-client 1.16.0+ for async Qdrant operations
- **File Monitoring**: watchdog 6.0.0 for filesystem change detection
- **Queue**: redis 5.0+ for crawl coordination and status tracking

### Development Tools
- **Testing**: pytest with pytest-asyncio (786 tests, 87%+ coverage)
- **HTTP Mocking**: respx for testing async HTTP interactions
- **Linting**: Ruff with PEP 8 enforcement
- **Type Checking**: ty (fast type checker)
- **Coverage**: pytest-cov with .cache/.coverage tracking

### Infrastructure Services (Docker Compose)
- **crawl4r-embeddings**: TEI service (remote on steamy-wsl:52000, RTX 4070 12GB)
- **crawl4r-vectors**: Qdrant GPU-accelerated (localhost:52001)
- **crawl4r-db**: PostgreSQL 15 (localhost:53432, unused in current implementation)
- **crawl4r-cache**: Redis 7 (localhost:53379)
- **crawl4ai**: Crawl4AI service (localhost:52004)

## Architectural Patterns

### 1. Environment-Aware Service URLs
**Pattern**: Automatic detection of Docker vs host environment with appropriate URL selection
**Location**: `crawl4r/core/config.py` - `Settings.set_environment_aware_defaults()`
**Implementation**:
```python
# Detects /.dockerenv or /proc/1/cgroup for Docker
# Auto-configures:
# - Docker: redis://crawl4r-cache:6379
# - Host: redis://localhost:53379
```
**Rationale**: Enables same codebase to run in both CLI (host) and containerized (future API) contexts without configuration changes.

### 2. Circuit Breaker Pattern
**Pattern**: Prevent cascading failures when external services become unavailable
**Location**: `crawl4r/resilience/circuit_breaker.py`
**States**: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing recovery)
**Used In**:
- TEI embeddings client (5 failures → 60s timeout)
- Crawl4AIReader (5 failures → 60s timeout)
- MapperService (5 failures → 60s timeout)

**Implementation**:
```python
# Wraps async operations with fail-fast behavior
result = await circuit_breaker.call(lambda: risky_operation())
```

### 3. Retry Policy with Exponential Backoff
**Pattern**: Centralized retry logic for transient failures
**Location**: `crawl4r/resilience/retry.py`
**Configuration**: [1.0s, 2.0s, 4.0s] delays with jitter
**Handles**:
- Network errors (ConnectionError, TimeoutError)
- 5xx HTTP errors (server failures)
- Skips 4xx errors (client errors are permanent)

### 4. Metadata Key Constants
**Pattern**: Centralized metadata key definitions to prevent hardcoded strings
**Location**: `crawl4r/core/metadata.py` - `MetadataKeys` class
**Usage**:
```python
from crawl4r.core.metadata import MetadataKeys
doc.metadata[MetadataKeys.FILE_PATH]  # Not "file_path"
doc.metadata[MetadataKeys.CHUNK_INDEX]  # Not "chunk_index"
```
**Keys**:
- **LlamaIndex defaults**: FILE_PATH, FILE_NAME, FILE_TYPE, LAST_MODIFIED_DATE
- **Crawl4r chunking**: CHUNK_INDEX, SECTION_PATH, TOTAL_CHUNKS
- **Web crawl**: SOURCE_URL, SOURCE_TYPE, CRAWL_TIMESTAMP

### 5. LlamaIndex Integration
**Pattern**: Leverage LlamaIndex for document processing orchestration
**Components**:
- **SimpleDirectoryReader**: File loading with metadata extraction
- **MarkdownNodeParser**: Heading-aware chunking (512 tokens, 15% overlap)
- **BaseEmbedding wrapper**: TEIEmbedding wraps TEIClient for LlamaIndex compatibility
- **QdrantVectorStore**: Native LlamaIndex Qdrant integration
- **Instrumentation**: Custom events (DocumentProcessingStartEvent, EmbeddingBatchEvent)

**Pipeline Flow**:
```
SimpleDirectoryReader → MarkdownNodeParser → TEIEmbedding → QdrantVectorStore
                                              ↑
                                         TEIClient
                                    (Circuit Breaker)
```

### 6. Async-First Architecture
**Pattern**: All I/O operations use asyncio for non-blocking concurrency
**Conventions**:
- Service methods: `async def method_name()`
- HTTP clients: `httpx.AsyncClient` with persistent connections
- Redis: `redis.asyncio` client
- Concurrency control: `asyncio.Semaphore` for rate limiting

**Example Pattern**:
```python
async with httpx.AsyncClient() as client:
    tasks = [fetch_url(client, url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

### 7. Deterministic Point ID Generation
**Pattern**: SHA256 hashing for idempotent vector operations
**Location**: `crawl4r/storage/qdrant.py` - `generate_point_id()`
**Formula**: `sha256(f"{file_path_relative}_{chunk_index}")`
**Benefit**: Re-ingesting same file replaces existing vectors (no duplicates)

### 8. Pydantic Configuration Management
**Pattern**: Type-safe settings with validation and .env integration
**Location**: `crawl4r/core/config.py` - `Settings` class
**Features**:
- Field validators (chunk_overlap_percent: 0-50)
- Environment variable mapping (case-insensitive)
- Default values with validation
- .env file support via pydantic-settings

### 9. Redis Queue Coordination
**Pattern**: Distributed lock with stale lock recovery
**Location**: `crawl4r/services/queue.py` - `QueueManager`
**Implementation**:
- Lock TTL: 3600s (1 hour) with auto-expiration
- Atomic Lua script for stale lock recovery
- Status tracking (QUEUED → RUNNING → COMPLETED/FAILED)
- Recent crawl history (last 100)

**Lock Recovery Logic**:
```lua
-- Atomic check and recover if holder has terminal status
if current_holder == stale_crawl_id then
    DEL lock_key
    SET lock_key new_owner NX EX 3600
end
```

### 10. Modular Reader Architecture
**Pattern**: Pluggable data sources implementing LlamaIndex BasePydanticReader
**Readers**:
- **Crawl4AIReader**: Web crawling via Crawl4AI /md endpoint (f=fit for clean extraction)
- **FileWatcher**: Filesystem monitoring with 1-second debounce
- **Future**: RSS feeds, sitemap parsers, database connectors

**Common Interface**:
```python
async def aload_data(urls: list[str]) -> list[Document]:
    # Returns LlamaIndex Document objects
```

## CLI Commands (Implemented)

### Core Commands
1. **scrape**: Extract markdown from URLs (no ingestion)
2. **crawl**: Full ingestion pipeline with depth discovery
3. **map**: URL discovery with BFS traversal
4. **extract**: LLM-powered structured data extraction
5. **screenshot**: Full-page or viewport screenshots
6. **status**: Crawl job monitoring (Redis-backed)
7. **watch**: File monitoring with auto-ingestion

### Command Patterns

**URL Input Validation**:
```python
# All commands validate URLs before processing
from crawl4r.core.url_validation import validate_url
if not validate_url(url):
    typer.echo("Invalid URL")
    raise typer.Exit(code=1)
```

**Progress Reporting**:
```python
# Rich console for interactive feedback
from rich.console import Console
console = Console()
with console.status("Processing..."):
    result = await service.operation()
console.print(Panel(summary, title="Results"))
```

**Signal Handling** (crawl command):
```python
# Graceful shutdown on Ctrl+C
def _signal_handler():
    stop_event.set()
loop.add_signal_handler(signal.SIGINT, _signal_handler)
```

## Service Layer Architecture

### Service Dependencies

**IngestionService** (coordinator):
```
IngestionService
├── ScraperService (Crawl4AI HTTP wrapper)
├── TEIClient (embeddings with circuit breaker)
├── VectorStoreManager (Qdrant operations)
├── QueueManager (Redis coordination)
└── MarkdownNodeParser (LlamaIndex chunking)
```

**Service Initialization Pattern**:
```python
# Auto-init from Settings if not provided
def __init__(self, scraper=None, embeddings=None, ...):
    if scraper and embeddings and vector_store:
        self.scraper = scraper  # Injected (testing)
    else:
        settings = Settings(watch_folder=Path("."))
        self.scraper = ScraperService(settings.crawl4ai_base_url)
        # ... auto-init from config (production)
```

### Service Health Validation

**Pattern**: Optional service validation on startup
```python
# Ingestion service validates all dependencies
await ingestion_service.validate_services()
# Checks: Crawl4AI /health, TEI /health, Qdrant /readyz, Redis PING
```

## Storage Layer

### VectorStoreManager (Qdrant)
**File**: `crawl4r/storage/qdrant.py` (1,200+ LOC)
**Key Features**:
- Idempotent collection creation (VectorParams, Distance.COSINE)
- Payload indexing (FILE_PATH, SOURCE_URL, CHUNK_INDEX)
- Batch upsert with retry (BATCH_SIZE=100)
- Deletion by metadata filters (remove_file, remove_url)
- Deterministic point IDs (SHA256 hashing)

**Metadata Schema** (TypedDict):
```python
class VectorMetadata(TypedDict):
    file_path: str           # Required
    chunk_index: int         # Required
    chunk_text: str          # Required
    source_url: str          # Optional (web crawl)
    section_path: str        # Optional (heading hierarchy)
    tags: list[str]          # Optional (frontmatter)
```

### TEIClient (Embeddings)
**File**: `crawl4r/storage/tei.py` (500+ LOC)
**Key Features**:
- Persistent httpx.AsyncClient (connection pooling)
- Circuit breaker integration (fail-fast on service failures)
- Dimension validation (expected: 1024 for Qwen3-Embedding-0.6B)
- Batch size limits (max: 100 texts per request)
- Retry policy with exponential backoff

**Performance**:
- Local RTX 3050 8GB: 21 emb/s
- Remote RTX 4070 12GB: 59 emb/s (2.8x faster)
- Remote deployment: `steamy-wsl:/home/jmagar/compose/crawl4r/`

### TEIEmbedding (LlamaIndex Wrapper)
**File**: `crawl4r/storage/llama_embeddings.py`
**Purpose**: Adapts TEIClient to LlamaIndex BaseEmbedding interface
**Methods**:
- `_get_query_embedding(text: str) -> list[float]`
- `_get_text_embedding(text: str) -> list[float]`
- `_get_text_embeddings(texts: list[str]) -> list[list[float]]`

## Reader Implementations

### Crawl4AIReader
**File**: `crawl4r/readers/crawl4ai.py` (800+ LOC)
**Endpoint**: POST `/md` with `f=fit` filter (clean content, ~12K chars vs ~89K raw)
**Features**:
- Batch crawling with `asyncio.Semaphore` (max 5 concurrent)
- Circuit breaker protection (5 failures → 60s timeout)
- Deterministic Document IDs (SHA256 URL hashing)
- Metadata builder for Qdrant compatibility
- Order-preserving results (maintains URL order)
- VectorStoreManager integration for auto-deduplication

**Crawl4AI Endpoint Comparison**:
| Endpoint | Filter | Output | Use Case |
|----------|--------|--------|----------|
| `/md` | `f=fit` | 12K chars | **Recommended** - Clean main content |
| `/md` | `f=raw` | 89K chars | Full page with nav/footer |
| `/crawl` | N/A | Raw only | Legacy endpoint |

**Document Metadata**:
```python
{
    "source": "https://example.com",
    "source_url": "https://example.com",  # Required for Qdrant
    "source_type": "web_crawl",
    "title": "Page Title",
    "status_code": 200,
    "crawl_timestamp": "2026-01-20T10:30:00Z"
}
```

### FileWatcher
**File**: `crawl4r/readers/file_watcher.py` (600+ LOC)
**Patterns**: Watchdog observer with 1-second debounce
**Operations**:
- Batch processing on startup (existing files)
- Real-time monitoring for changes
- Deletion detection (removes stale vectors)
- Markdown-only filtering (*.md, *.markdown)

**Debounce Logic**:
```python
# Wait 1 second after last event before processing
while time.time() - last_event_time < debounce_seconds:
    await asyncio.sleep(0.1)
```

## Processing Pipeline

### Document Processor
**File**: `crawl4r/processing/processor.py` (400+ LOC)
**Pipeline Stages**:
1. Load documents (SimpleDirectoryReader or custom reader)
2. Parse into nodes (MarkdownNodeParser, 512 tokens, 15% overlap)
3. Generate embeddings (TEIClient batch, max 50 chunks)
4. Upsert to Qdrant (batch size 100)

**Batch Processing**:
```python
# Process 50 documents at a time in memory
DEFAULT_BATCH_CHUNK_SIZE = 50
MAX_EMBEDDING_BATCH_SIZE = 50  # TEI limit ~100
```

**Result Types**:
```python
@dataclass
class ProcessingResult:
    success: bool
    chunks_processed: int
    file_path: str
    document_ids: list[str]
    time_taken: float
    error: str | None

class BatchResult(list[ProcessingResult]):
    # Aggregate metrics
    total_documents: int
    successful: int
    failed: int
    total_chunks_processed: int
    total_time: float
    documents_per_second: float
    errors: list[tuple[str, str]]
```

## Resilience Patterns

### Circuit Breaker
**File**: `crawl4r/resilience/circuit_breaker.py` (200+ LOC)
**States**: CLOSED → OPEN (threshold failures) → HALF_OPEN (testing) → CLOSED/OPEN
**Configuration**:
- failure_threshold: 5 (consecutive failures)
- reset_timeout: 60.0 seconds
**Usage**: Wraps TEIClient, Crawl4AIReader, MapperService

### Retry Policy
**File**: `crawl4r/resilience/retry.py` (200+ LOC)
**Delays**: [1.0s, 2.0s, 4.0s] with jitter
**Retryable Errors**:
- httpx.NetworkError (connection failures)
- httpx.TimeoutException (request timeouts)
- httpx.HTTPStatusError (5xx server errors only)

### Failed Document Logging
**File**: `crawl4r/resilience/failed_docs.py`
**Format**: JSONL (JSON Lines)
**Fields**: file_path, error_message, timestamp
**Purpose**: Audit trail for debugging and manual recovery

## Configuration Management

### Settings Class
**File**: `crawl4r/core/config.py` (250 LOC)
**Environment Variables** (from .env):
```bash
# Required
WATCH_FOLDER=/path/to/docs

# Optional (auto-detected if not set)
TEI_ENDPOINT=http://100.74.16.82:52000  # Remote GPU
QDRANT_URL=http://localhost:52001
REDIS_URL=redis://localhost:53379
CRAWL4AI_BASE_URL=http://localhost:52004

# Chunking
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_PERCENT=15  # Validated: 0-50

# Performance
MAX_CONCURRENT_DOCS=10
BATCH_SIZE=50
```

### Docker vs Host Detection
**Method**: `is_running_in_docker()`
**Checks**:
1. `/.dockerenv` file exists
2. `/proc/1/cgroup` contains "docker"
3. `RUN_IN_DOCKER` env var set

## Testing Architecture

### Test Organization
```
tests/
├── unit/              # 70+ files, mocked dependencies
│   ├── cli/           # CLI command tests
│   ├── services/      # Service layer tests
│   └── test_*.py      # Component tests
├── integration/       # 10+ files, real services required
│   ├── test_e2e_*.py  # End-to-end pipelines
│   └── conftest.py    # Integration fixtures
├── performance/       # Load and memory tests
└── fixtures/          # Shared test data
    └── crawl4ai_responses.py  # Mock HTTP responses
```

### Testing Patterns

**HTTP Mocking (respx)**:
```python
@pytest.mark.asyncio
async def test_crawl4ai_reader(respx_mock):
    respx_mock.post("http://localhost:52004/md").mock(
        return_value=httpx.Response(200, json={
            "markdown": "# Test",
            "metadata": {"title": "Test Page"}
        })
    )
    reader = Crawl4AIReader()
    docs = await reader.aload_data(["https://example.com"])
```

**Integration Tests** (require Docker):
```python
@pytest.mark.integration
async def test_qdrant_integration():
    # Requires crawl4r-vectors running
    manager = VectorStoreManager(qdrant_url="http://localhost:52001")
    await manager.ensure_collection()
```

**Pytest Markers**:
- `@pytest.mark.integration`: Requires real services
- `@pytest.mark.asyncio`: Async test (auto-mode enabled)
- No marker: Unit test (fast, mocked)

### Coverage Configuration
**File**: `pyproject.toml`
```toml
[tool.coverage.run]
data_file = ".cache/.coverage"
omit = ["tests/*", ".cache/*"]

[tool.pytest.ini_options]
cache_dir = ".cache/.pytest_cache"
testpaths = ["tests"]
asyncio_mode = "auto"
```

## Common Patterns and Conventions

### Error Handling
**Pattern**: Fail fast with specific error messages
```python
# Validation in constructors
if not endpoint_url.startswith(("http://", "https://")):
    raise ValueError("Invalid endpoint URL")

# Operation errors with context
raise ValueError(f"Failed to fetch {url}: {status_code}")
```

### Logging
**Pattern**: Structured logging with module-level loggers
```python
logger = logging.getLogger(__name__)
logger.info("Processing %s documents", len(docs))
logger.warning("Service unavailable, retrying...")
logger.error("Failed to process %s: %s", file_path, error)
```

### Type Hints
**Pattern**: Complete type coverage with generic types
```python
from collections.abc import Callable, Awaitable
from typing import TypeVar

T = TypeVar("T")

async def operation(
    data: list[str],
    callback: Callable[[str], Awaitable[None]] | None = None
) -> list[T]:
    ...
```

### Context Managers
**Pattern**: Resource cleanup with async context managers
```python
class MapperService:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self._client.aclose()

# Usage
async with MapperService(endpoint_url) as mapper:
    result = await mapper.map_url(url)
```

## Gotchas and Edge Cases

### 1. Virtual Environment Requirement
**Issue**: Python commands fail if .venv not activated
**Location**: CLAUDE.md documentation
**Solution**: Always use `source .venv/bin/activate` or `.venv/bin/python`
**Reason**: No global `python` command on this system, dependencies in venv only

### 2. Remote TEI Service
**Issue**: TEI no longer runs locally (commented out in docker-compose.yaml)
**Location**: Lines 28-79 of docker-compose.yaml
**Current**: Remote GPU machine at 100.74.16.82:52000 (RTX 4070 12GB)
**Performance**: 2.8x faster (59 emb/s vs 21 emb/s on local RTX 3050 8GB)
**Deployment**: `steamy-wsl:/home/jmagar/compose/crawl4r/`

### 3. Docker Network Requirement
**Issue**: Services fail to start without external network
**Command**: `docker network create crawl4r`
**Reason**: All services connect to shared external network (not auto-created)

### 4. Metadata Key Legacy Support
**Issue**: Old code may use FILE_PATH_RELATIVE or FILE_PATH_ABSOLUTE
**Current**: FILE_PATH is primary (absolute path from SimpleDirectoryReader)
**Pattern**: Compute relative paths on demand when needed
```python
file_path = Path(doc.metadata[MetadataKeys.FILE_PATH])
relative_path = str(file_path.relative_to(watch_folder))
```

### 5. Crawl4AI Endpoint Choice
**Issue**: `/crawl` endpoint returns cruft, `/md?f=raw` returns full page with nav
**Best Practice**: Use `/md?f=fit` for clean main content (~12K vs ~89K chars)
**Location**: Crawl4AIReader default configuration

### 6. Redis Lock Expiration
**Issue**: Locks auto-expire after 3600 seconds (1 hour)
**Impact**: Long-running crawls (>1 hour) will lose lock mid-operation
**Mitigation**: LOCK_TTL_SECONDS constant, no auto-refresh implemented
**Location**: `crawl4r/services/queue.py`

### 7. PostgreSQL Unused
**Issue**: crawl4r-db service defined but not used in current implementation
**Status**: Placeholder for future features (user management, API keys)
**Location**: docker-compose.yaml line 102-118

### 8. Port Number Convention
**Rule**: ALL services use high ports (52000+, 53000+) to avoid conflicts
**Examples**:
- TEI: 52000 (not 80)
- Qdrant: 52001 (not 6333)
- Crawl4AI: 52004 (not 11235)
- Redis: 53379 (not 6379)
- Postgres: 53432 (not 5432)

### 9. LlamaIndex Instrumentation
**Pattern**: Custom events for observability (not yet used)
**Location**: `crawl4r/core/instrumentation.py`
**Events**: DocumentProcessingStartEvent, ChunkingEndEvent, EmbeddingBatchEvent
**Future**: OpenTelemetry integration planned (init_observability function exists)

### 10. Signal Handling (Windows Incompatibility)
**Issue**: `loop.add_signal_handler()` not supported on Windows
**Fallback**: Uses `signal.signal()` on Windows (limited functionality)
**Location**: `crawl4r/cli/commands/crawl.py` lines 92-99

## Development Workflow

### Running Tests
```bash
# Activate venv
source .venv/bin/activate

# All tests
pytest

# Unit tests only (fast)
pytest tests/unit/

# Integration tests (requires Docker services)
pytest tests/integration/ -m integration

# With coverage
pytest --cov=crawl4r --cov-report=term

# Specific test file
pytest tests/unit/test_crawl4ai_reader.py -v
```

### Quality Checks
```bash
# Linting
ruff check .

# Auto-fix issues
ruff check . --fix

# Type checking
ty check crawl4r/

# Format check
ruff format --check .
```

### Service Management
```bash
# Start all services
docker compose up -d

# Check service health
docker compose ps
docker compose logs -f crawl4ai

# Stop services
docker compose down

# Restart specific service
docker compose restart crawl4ai
```

### CLI Usage
```bash
# Activate venv first!
source .venv/bin/activate

# Scrape single URL
python -m crawl4r.cli.app scrape https://example.com

# Crawl with depth
python -m crawl4r.cli.app crawl https://docs.example.com --depth 2

# Map URLs
python -m crawl4r.cli.app map https://example.com --depth 1

# Check status
python -m crawl4r.cli.app status

# Watch folder
python -m crawl4r.cli.app watch --folder /path/to/docs
```

## Future Enhancements (Documented but Not Implemented)

### API Layer
**Status**: Skeleton only in `crawl4r/api/`
**Files**: app.py, routes/health.py, models/responses.py
**Planned**: FastAPI REST API for programmatic access

### OpenTelemetry Integration
**Status**: Code exists but not configured
**Location**: `crawl4r/core/instrumentation.py`
**Function**: `init_observability(enable_otel=True, otel_endpoint="...")`

### LLM Extraction Features
**Status**: Basic implementation in ExtractService
**Planned**: Advanced schema validation, multi-model support
**Dependencies**: Requires LLM provider API keys (OpenAI, Anthropic, etc.)

### Screenshot Service Advanced Features
**Status**: Basic implementation complete
**Planned**: Element selectors, scroll behavior, custom viewport sizes

## Critical Dependencies

### External Services (Must Be Running)
1. **Crawl4AI** (localhost:52004) - Web crawling and markdown extraction
2. **Qdrant** (localhost:52001) - Vector storage
3. **Redis** (localhost:53379) - Queue coordination
4. **TEI** (100.74.16.82:52000) - Embeddings (remote GPU machine)

### Python Package Highlights
- **llama-index-core**: 0.14.0+ (document processing, embeddings interface)
- **qdrant-client**: 1.16.0+ (async vector operations)
- **httpx**: 0.28.1 (async HTTP client)
- **typer**: 0.12.0+ (CLI framework)
- **pydantic**: 2.0+ (validation and settings)
- **watchdog**: 6.0.0 (file monitoring)
- **redis**: 5.0+ (async Redis client)

## File Count Summary
- **Production Code**: 54 Python files, ~9,800 lines
- **Test Code**: 86 Python files, ~25,700 lines
- **Test Count**: 786 tests
- **Coverage**: 87%+ (high confidence in codebase stability)

## Documentation Files
- **README.md**: User-facing documentation (CLI usage, installation)
- **CLAUDE.md**: AI assistant context (architecture, conventions, gotchas)
- **ENHANCEMENTS.md**: Future improvement ideas
- **specs/**: Design specifications for major features
- **docs/plans/complete/**: Completed implementation plans (historical)

## Key Takeaways for Development

1. **Always activate venv** before running Python commands
2. **Use environment-aware URLs** (Settings class handles Docker vs host)
3. **Prefer /md?f=fit endpoint** for clean Crawl4AI results
4. **Test coverage is high** (786 tests) - maintain this standard
5. **Circuit breakers protect** against service failures - don't bypass
6. **Redis coordination** enables distributed crawling (lock management)
7. **Metadata keys** use constants (MetadataKeys class) - no hardcoded strings
8. **Async patterns** are mandatory for I/O operations
9. **Type hints required** for all function signatures
10. **Remote TEI service** is intentional (better performance) - don't re-enable local
