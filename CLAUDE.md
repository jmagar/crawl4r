# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Status

This repository contains infrastructure setup and specifications for the Crawl4r RAG ingestion pipeline. **No implementation code exists yet** - only Docker Compose configuration, specs, and planning artifacts.

The authoritative implementation plan is in `specs/rag-ingestion/` with complete requirements, design, and task breakdown for a Python-based RAG ingestion system.

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

The stack consists of 5 containerized services, all connected via the external `crawl4r` network:

1. **crawl4r-embeddings** (TEI) - HuggingFace Text Embeddings Inference for generating 1024-dim vectors
   - GPU-accelerated (NVIDIA required)
   - Model: Qwen/Qwen3-Embedding-0.6B
   - Endpoint: `http://crawl4r-embeddings:80/embed`
   - Host port: `${TEI_HTTP_PORT}` → container port 80

2. **crawl4r-vectors** (Qdrant) - Vector database for storing embeddings
   - GPU-accelerated (NVIDIA required)
   - HTTP: `${QDRANT_HTTP_PORT}` → 6333
   - gRPC: `${QDRANT_GRPC_PORT}` → 6334
   - Data: `/home/jmagar/appdata/crawl4r-vectors`

3. **crawl4r-db** (PostgreSQL 15) - Relational metadata storage
   - Port: `${POSTGRES_PORT}` → 5432
   - Database: `crawl4r`
   - Data: `/home/jmagar/appdata/crawl4r-db`

4. **crawl4r-cache** (Redis 7) - Caching and message brokering
   - Port: `${REDIS_PORT}` → 6379
   - Max memory: 2GB with LRU eviction
   - Data: `/home/jmagar/appdata/crawl4r-cache`

5. **crawl4ai** - Web crawling service
   - Port: `${CRAWL4AI_PORT}` → 11235
   - Health endpoint: `http://localhost:${CRAWL4AI_PORT}/health`

All services have health checks configured and restart unless manually stopped.

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

### Project Structure (Planned)

```
crawl4r/
├── rag_ingestion/          # Main package
│   ├── config.py           # Pydantic configuration
│   ├── logger.py           # Structured logging
│   ├── embeddings.py       # TEI client wrapper
│   ├── vector_store.py     # Qdrant manager
│   ├── processor.py        # Document chunking
│   ├── watcher.py          # File monitoring
│   ├── queue_manager.py    # Async queue + backpressure
│   └── quality.py          # Startup validation
├── tests/
│   ├── unit/               # Fast, isolated tests
│   └── integration/        # Tests with real services
├── data/
│   └── watched_folder/     # Default watch directory
├── pyproject.toml          # Python dependencies (use uv)
└── .env                    # Environment config
```

### Testing Commands (Future)

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_embeddings.py

# Run with coverage
pytest --cov=rag_ingestion --cov-report=term

# Type checking
ty check rag_ingestion/

# Linting
ruff check .

# Auto-fix linting issues
ruff check . --fix
```

### Quality Gates

Every phase in `tasks.md` includes `[VERIFY]` checkpoints:
- Must pass `ruff check .` (no lint errors)
- Must pass `ty check rag_ingestion/` (no type errors)
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
