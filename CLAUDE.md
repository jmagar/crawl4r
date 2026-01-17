# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

This repository contains the Crawl4r RAG ingestion pipeline implementation. The **Crawl4AIReader** component is complete and production-ready, providing LlamaIndex integration for web crawling.

Additional components are documented in `specs/rag-ingestion/` with complete requirements, design, and task breakdown for the Python-based RAG ingestion system.

## Infrastructure Commands

### Service Management

**Start all services:**
```bash
docker compose up -d
```

**Stop services:**
```bash
docker compose down
```

**View logs:**
```bash
docker compose logs -f [service-name]
```

**Required before starting:** Ensure Docker network exists:
```bash
docker network create crawl4r
```

### Required Environment Variables

The following must be set in `.env` before starting services:

- `TEI_HTTP_PORT` - TEI embeddings service port (default: 52000)
- `QDRANT_HTTP_PORT` - Qdrant HTTP port (default: 52001)
- `QDRANT_GRPC_PORT` - Qdrant gRPC port (default: 52002)
- `POSTGRES_PORT` - PostgreSQL port (default: 53432)
- `POSTGRES_PASSWORD` - PostgreSQL password (required, no default)
- `REDIS_PORT` - Redis port (default: 53379)
- `CRAWL4AI_PORT` - Crawl4AI service port (default: 52004)

**Optional TEI tuning variables:**
- `TEI_EMBEDDING_MODEL` (default: Qwen/Qwen3-Embedding-0.6B)
- `TEI_MAX_CONCURRENT_REQUESTS` (default: 128)
- `TEI_MAX_BATCH_TOKENS` (default: 131072)
- `TEI_MAX_BATCH_REQUESTS` (default: 32)
- `TEI_MAX_CLIENT_BATCH_SIZE` (default: 128)
- `TEI_POOLING` (default: last-token)
- `TEI_TOKENIZATION_WORKERS` (default: 8)

## Architecture Overview

### Services (Docker Compose)

The stack consists of 4 containerized services (TEI runs remotely on GPU machine), all connected via the external `crawl4r` network:

1. **crawl4r-embeddings** (TEI) - **REMOTE SERVICE ON GPU MACHINE**
   - **Location:** `steamy-wsl:52000` (RTX 4070 12GB)
   - **NOT running locally** - uses dedicated GPU machine for 2.8x better performance
   - Model: Qwen/Qwen3-Embedding-0.6B
   - Endpoint: `http://100.74.16.82:52000/embed`
   - See `steamy-wsl:/home/jmagar/compose/crawl4r/` for deployment files
   - **Performance:** 59 emb/s vs 21 emb/s on local 8GB GPU

2. **crawl4r-vectors** (Qdrant) - Vector database for storing embeddings (local)
   - GPU-accelerated (NVIDIA required)
   - HTTP: `${QDRANT_HTTP_PORT}` → 6333
   - gRPC: `${QDRANT_GRPC_PORT}` → 6334
   - Data: `/home/jmagar/appdata/crawl4r-vectors`

3. **crawl4r-db** (PostgreSQL 15) - Relational metadata storage (local)
   - Port: `${POSTGRES_PORT}` → 5432
   - Database: `crawl4r`
   - Data: `/home/jmagar/appdata/crawl4r-db`

4. **crawl4r-cache** (Redis 7) - Caching and message brokering (local)
   - Port: `${REDIS_PORT}` → 6379
   - Max memory: 2GB with LRU eviction
   - Data: `/home/jmagar/appdata/crawl4r-cache`

5. **crawl4ai** - Web crawling service (local)
   - Port: `${CRAWL4AI_PORT}` → 11235
   - Health endpoint: `http://localhost:${CRAWL4AI_PORT}/health`

All services have health checks configured and restart unless manually stopped.

## Crawl4AIReader - LlamaIndex Web Crawling

The `Crawl4AIReader` is a production-ready LlamaIndex reader for crawling web pages using the Crawl4AI service.

### Basic Usage

```python
from crawl4r.readers.crawl4ai import Crawl4AIReader

# Create reader with default configuration
reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

# Crawl single URL (async)
documents = await reader.aload_data(["https://example.com"])

# Crawl single URL (sync)
documents = reader.load_data(["https://example.com"])

# Batch crawl multiple URLs
urls = ["https://example.com", "https://example.org"]
documents = await reader.aload_data(urls)
```

### Configuration

```python
from crawl4r.readers.crawl4ai import Crawl4AIReader, Crawl4AIReaderConfig

# Custom configuration
config = Crawl4AIReaderConfig(
    endpoint_url="http://localhost:52004",
    max_concurrent_requests=10,
    max_retries=3,
    timeout=60.0,
    fail_on_error=False,  # Return None for failed URLs instead of raising
)

reader = Crawl4AIReader(**config.model_dump())
```

### Document Metadata

Each crawled document includes rich metadata:

```python
doc = documents[0]
print(doc.metadata)
# {
#     "source": "https://example.com",
#     "source_url": "https://example.com",  # Required for Qdrant indexing
#     "source_type": "web_crawl",
#     "title": "Example Domain",
#     "description": "Page description",
#     "status_code": 200,
#     "crawl_timestamp": "2024-01-15T10:30:00Z"
# }
```

### Deduplication with VectorStoreManager

Automatic deduplication removes existing URL data before re-crawling:

```python
from crawl4r.readers.crawl4ai import Crawl4AIReader
from crawl4r.storage.qdrant import VectorStoreManager

# Setup vector store for deduplication
vector_store = VectorStoreManager(
    collection_name="web_documents",
    qdrant_url="http://localhost:52001"
)

# Enable automatic deduplication (default: True)
reader = Crawl4AIReader(
    endpoint_url="http://localhost:52004",
    vector_store=vector_store,
    enable_deduplication=True  # Deletes existing data for URLs before crawling
)

# Re-crawl URLs - automatically removes old data first
documents = await reader.aload_data(["https://example.com"])
```

### Error Handling

The reader includes comprehensive error handling with retry logic:

- **Timeout errors**: Retries with exponential backoff (max 3 attempts)
- **Network errors**: Retries transient failures
- **5xx errors**: Retries server errors
- **4xx errors**: No retry (client errors are permanent)
- **Circuit breaker**: Opens after 5 consecutive failures, prevents cascade

```python
# Fail fast on errors
reader = Crawl4AIReader(
    endpoint_url="http://localhost:52004",
    fail_on_error=True  # Raises ValueError on any failure
)

# Graceful degradation
reader = Crawl4AIReader(
    endpoint_url="http://localhost:52004",
    fail_on_error=False  # Returns None for failed URLs
)

documents = await reader.aload_data(urls)
successful_docs = [doc for doc in documents if doc is not None]
```

### Integration with Pipeline

```python
from crawl4r.readers.crawl4ai import Crawl4AIReader
from crawl4r.processing.chunker import MarkdownChunker
from crawl4r.storage.tei import TEIEmbeddings
from crawl4r.storage.qdrant import VectorStoreManager

# Initialize components
reader = Crawl4AIReader(endpoint_url="http://localhost:52004")
chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
embeddings = TEIEmbeddings(endpoint_url="http://localhost:52000")
vector_store = VectorStoreManager(
    collection_name="web_documents",
    qdrant_url="http://localhost:52001"
)

# Crawl URLs
documents = await reader.aload_data(["https://docs.example.com"])

# Process each document
for doc in documents:
    if doc is None:
        continue

    # Chunk markdown content
    chunks = chunker.chunk(doc.text, filename=doc.metadata["source_url"])

    # Generate embeddings
    vectors = await embeddings.aembed_documents([chunk["chunk_text"] for chunk in chunks])

    # Store in Qdrant with web-specific metadata
    await vector_store.upsert_vectors(
        vectors=vectors,
        metadata=[
            {
                "source_url": doc.metadata["source_url"],
                "title": doc.metadata["title"],
                "chunk_index": chunk["chunk_index"],
                "section_path": chunk["section_path"]
            }
            for chunk in chunks
        ]
    )
```

### Health Check

The reader validates Crawl4AI service availability on initialization:

```python
try:
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")
    # Service is available
except ValueError as e:
    # Service unavailable: Crawl4AI service health check failed
    print(f"Service error: {e}")
```

### Testing

```bash
# Run unit tests (44 tests, mocked service)
pytest tests/unit/test_crawl4ai_reader.py -v

# Run integration tests (requires Crawl4AI service)
pytest tests/integration/test_crawl4ai_reader_integration.py -v -m integration

# Run E2E pipeline tests
pytest tests/integration/test_e2e_reader_pipeline.py -v -m integration

# Run with coverage
pytest tests/unit/test_crawl4ai_reader.py --cov=crawl4r.readers.crawl4ai --cov-report=term
```

### Planned Python Implementation

The RAG ingestion pipeline (not yet implemented) will be built with:

- **Framework:** Python 3.10+ with asyncio for non-blocking I/O
- **Document Processing:** LlamaIndex for orchestration, markdown-aware chunking
- **File Monitoring:** watchdog library with 1-second debounce
- **Embeddings:** Custom TEI client inheriting from LlamaIndex BaseEmbedding
- **Vector Store:** qdrant-client for direct Qdrant integration
- **Configuration:** Pydantic BaseSettings with .env support
- **Quality:** Ruff (linting), ty (type checking), pytest (testing)

**Key Design Principles:**
- Async-first architecture with asyncio
- Idempotent operations via deterministic point IDs (SHA256 of file_path + chunk_index)
- Circuit breaker pattern for TEI failures
- Batch processing on startup, real-time monitoring thereafter
- 512-token chunks with 15% overlap, heading-based splitting

## Specification Structure

All planning artifacts are in `specs/rag-ingestion/`:

- **requirements.md** - User stories with acceptance criteria
- **design.md** - Complete technical design (100KB+) with architecture diagrams, component specs, and edge case handling
- **tasks.md** - 47 tasks organized in 3 phases (POC → Production-Ready → Enterprise)
- **research.md** - Technical research and feasibility analysis
- **decisions.md** - Architectural decision records
- **technical-review.md** - Design review feedback

**When implementing, follow this order:**
1. Read `design.md` for architectural context
2. Reference `requirements.md` for specific acceptance criteria
3. Execute tasks from `tasks.md` in sequence (each includes verification steps)

## Development Workflow (When Implementation Starts)

### Project Structure

```
crawl4r/
├── crawl4r/                 # Main package
│   ├── core/                # Infrastructure (config, logger, quality)
│   ├── readers/             # Data sources (crawl4ai, file_watcher)
│   ├── processing/          # Document processing (chunker, processor)
│   ├── storage/             # Backends (embeddings, vector_store)
│   ├── resilience/          # Fault tolerance (circuit_breaker, recovery)
│   ├── cli/                 # Command-line interface (main.py)
│   └── api/                 # REST API (FastAPI skeleton)
├── tests/
│   ├── unit/                # Fast, isolated tests
│   └── integration/         # Tests with real services
├── examples/                # Usage examples
├── specs/                   # Design specifications
├── pyproject.toml           # Python dependencies (use uv)
└── .env                     # Environment config
```

### Testing Commands (Future)

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_embeddings.py

# Run with coverage
pytest --cov=crawl4r --cov-report=term

# Type checking
ty check crawl4r/

# Linting
ruff check .

# Auto-fix linting issues
ruff check . --fix
```

### Quality Gates

Every phase in `tasks.md` includes `[VERIFY]` checkpoints:
- Must pass `ruff check .` (no lint errors)
- Must pass `ty check crawl4r/` (no type errors)
- Integration tests must pass with real services running

## Critical Implementation Notes

### Port Management
All services use high ports (52000+) to avoid conflicts. Never use standard ports (80, 443, 3000, 5432, 6379).

### Embedding Dimensions
TEI endpoint returns 1024-dimensional vectors by default for Qwen3-Embedding-0.6B. Validation is **mandatory** before storing in Qdrant.

### Point ID Generation
Use deterministic SHA256 hashing: `hashlib.sha256(f"{file_path_relative}_{chunk_index}".encode()).hexdigest()` to enable idempotent re-ingestion.

### File Path Metadata
Store both:
- `file_path_relative` - Relative to watch folder (for deletion queries)
- `file_path_absolute` - Full path (for file access)

### Chunking Strategy
- **Primary split:** Markdown heading hierarchy (#, ##, ###)
- **Target size:** 512 tokens with 15% overlap (77 tokens)
- **Fallback:** Paragraph-level splitting for files without headings
- **Metadata:** Preserve section_path (e.g., "Guide > Installation > Requirements")

### Circuit Breaker Pattern
When TEI fails 3+ consecutive times:
1. Open circuit for 60 seconds
2. Queue documents instead of dropping
3. Log errors to `failed_documents.jsonl`
4. Resume when circuit closes

### Retry Strategy
- **Normal operations:** [1s, 2s, 4s] exponential backoff
- **Startup validation:** [5s, 10s, 20s] (services may be initializing)

## Current State vs. Planned State

**Current:** Infrastructure services defined, specifications complete, no Python code
**Next Step:** Task 1.1.1 - Create project structure, `pyproject.toml`, and configuration module

Before implementation, ensure:
1. Docker network `crawl4r` exists
2. `.env` file has all required variables
3. All services start successfully (`docker compose up -d`)
4. TEI embeddings endpoint responds: `curl http://localhost:52000/health`
5. Qdrant is ready: `curl http://localhost:52001/readyz`
