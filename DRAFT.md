# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Crawl4r is a Python RAG ingestion pipeline for web crawling and document processing with vector storage. The system features async-first architecture, LlamaIndex integration, and comprehensive test coverage (786 tests, 87%+).

**Status**: Core functionality implemented with known bugs (see `.docs/research/cli-testing-report-2026-01-20.md`)
**Scale**: 9,800 lines of production code across 54 Python files
**Commands**: 7 CLI commands - **45% working**, 5 HIGH severity bugs
**Test Coverage**: 87% (excellent unit coverage, but E2E CLI testing needed)
**Infrastructure**: ✅ All services operational (docker-compose.yaml resolved)

## Virtual Environment - READ THIS FIRST

**CRITICAL: This project uses a virtual environment. ALWAYS activate it before running Python commands.**

### Activation Required

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Then you can use standard commands
python -m crawl4r.cli.app [command]
pytest
crawl4r [command]  # If installed via uv
```

**Without activating venv, you'll get:**
- Import errors (missing dependencies)
- Wrong Python version
- Command not found errors

## Quick Start

```bash
# 1. Start infrastructure services
docker compose up -d

# 2. Verify services are running
docker compose ps

# 3. Run CLI commands (example)
source .venv/bin/activate && python -m crawl4r.cli.app scrape https://example.com

# 4. Run tests
source .venv/bin/activate && pytest

# 5. Stop services
docker compose down
```

## Service Architecture

All services run on **high ports (52000+, 53000+)** to avoid conflicts. TEI embeddings run on a remote GPU machine for optimal performance.

| Service | Location | Port(s) | Purpose |
|---------|----------|---------|---------|
| **TEI Embeddings** | 100.74.16.82 (remote) | 52000 | Qwen3-Embedding-0.6B on RTX 4070 12GB |
| **Crawl4AI** | localhost | 52004 | Web crawling service |
| **Qdrant** | localhost | 52001 (HTTP), 52002 (gRPC) | Vector database |
| **Redis** | localhost | 53379 | Cache and message queue |
| **PostgreSQL** | localhost | 53432 | Metadata storage (reserved) |

**Network**: All services connect via external `crawl4r` Docker network (must be created before starting).

### Service Health Checks

```bash
# TEI embeddings (remote)
curl http://100.74.16.82:52000/health

# Crawl4AI
curl http://localhost:52004/health

# Qdrant
curl http://localhost:52001/readyz

# Redis
docker compose exec crawl4r-cache redis-cli ping
```

## Tech Stack

### Core Technologies
- **Python**: 3.11+ with asyncio (async-first throughout)
- **CLI Framework**: Typer with rich formatting
- **Document Processing**: LlamaIndex for orchestration
- **Embeddings**: TEI client (custom LlamaIndex integration)
- **Vector Store**: Qdrant with GPU acceleration
- **Web Crawling**: Crawl4AI service (use `/md?f=fit` endpoint)
- **HTTP Client**: httpx for async requests
- **Validation**: Pydantic BaseModel and BaseSettings

### Development Tools
- **Dependency Management**: uv (NOT pip, poetry, or pipenv)
- **Linting**: Ruff (replaces Black, isort, flake8)
- **Type Checking**: ty (extremely fast type checker)
- **Testing**: pytest with pytest-asyncio (auto-mode enabled)
- **Package Definition**: pyproject.toml (PEP 621 compliant)

## Project Structure

```
crawl4r/
├── crawl4r/                    # Main package (9,800 LOC)
│   ├── cli/                    # Typer CLI (7 commands)
│   ├── readers/                # Data sources (Crawl4AIReader, FileWatcher)
│   ├── services/               # Business logic (scraper, mapper)
│   ├── processing/             # Document processing and chunking
│   ├── storage/                # Backends (TEI client, Qdrant manager)
│   ├── resilience/             # Circuit breaker, retry, recovery
│   └── core/                   # Config, logging, metadata constants
├── tests/                      # 786 tests (87% coverage)
│   ├── unit/                   # Fast, mocked tests
│   └── integration/            # Requires Docker services
├── .docs/                      # Session logs and research
│   ├── sessions/               # Timestamped session logs
│   └── research/               # Architecture and analysis docs
├── specs/                      # Design specifications
├── pyproject.toml              # Dependencies and tool config
├── docker-compose.yaml         # Service orchestration
└── .env                        # Environment variables (gitignored)
```

**For detailed architecture**: See `.docs/research/codebase-architecture.docs.md` (58KB comprehensive guide)

## Critical Notes & Gotchas

### Port Management
- **ALWAYS use ports 53000+ for new services** (never 80, 443, 3000, 5432, 6379)
- Check availability before assigning: `ss -tuln | grep :PORT`
- Document all assignments in `.docs/services-ports.md`

### Embeddings
- TEI returns **1024-dimensional vectors** for Qwen3-Embedding-0.6B
- **ALWAYS validate dimensions before storing in Qdrant**
- Remote GPU machine provides 2.8x better performance (59 vs 21 emb/s)

### Crawl4AI Best Practices
- Use `/md` endpoint with `f=fit` parameter for clean content (~12K chars)
- Avoid `f=raw` or `/crawl` endpoint (returns 89K chars with navigation cruft)
- Default timeout: 60 seconds

### Metadata Management
- **Use constants from `crawl4r.core.metadata.MetadataKeys`** (never hardcode strings)
- Primary key: `MetadataKeys.FILE_PATH` (absolute path)
- Compute relative paths on demand when needed

### Point IDs for Qdrant
- Use deterministic SHA256 hashing: `hashlib.sha256(f"{file_path_relative}_{chunk_index}".encode()).hexdigest()`
- Enables idempotent re-ingestion without duplicates

### Circuit Breaker Pattern
- TEI, Qdrant, and Crawl4AI protected by circuit breakers
- Opens after 5 consecutive failures, prevents cascade
- **DON'T bypass circuit breaker logic**

### Retry Strategy
- Normal operations: [1s, 2s, 4s] exponential backoff
- Startup validation: [5s, 10s, 20s] (services may be initializing)

## Development Workflows

### Before Committing

```bash
# 1. Lint and auto-fix
ruff check . --fix

# 2. Type check
ty check crawl4r/

# 3. Run tests
pytest

# 4. Verify services are healthy
docker compose ps
```

All checks must pass before committing.

### Commit Message Format

```bash
git commit -m "type(scope): description

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only (fast)
pytest tests/unit/

# Integration tests (requires Docker services)
pytest tests/integration/ -m integration

# Specific test file
pytest tests/unit/test_crawl4ai_reader.py -v

# With coverage
pytest --cov=crawl4r --cov-report=term

# Watch mode (requires pytest-watch)
ptw tests/unit/
```

### Common CLI Commands

```bash
# Activate venv first
source .venv/bin/activate

# Scrape single URL
python -m crawl4r.cli.app scrape https://example.com

# Crawl with depth
python -m crawl4r.cli.app crawl https://example.com --max-depth 2

# Extract content with specific format
python -m crawl4r.cli.app extract https://example.com --format fit

# Watch directory for changes
python -m crawl4r.cli.app watch /path/to/docs

# Check service status
python -m crawl4r.cli.app status

# Take screenshot
python -m crawl4r.cli.app screenshot https://example.com
```

### Service Management

```bash
# Start all services
docker compose up -d

# Start specific service
docker compose up -d crawl4r-vectors

# View logs
docker compose logs -f [service-name]

# Restart service
docker compose restart [service-name]

# Stop all services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v
```

## Python Coding Standards

### Required Patterns
- **Async/Await**: Use for all I/O operations (HTTP, database, file)
- **Type Hints**: Required on all function signatures and class attributes
- **Docstrings**: Google-style for all public functions/classes (Args, Returns, Raises)
- **String Formatting**: f-strings only (not %, .format())
- **Imports**: Absolute imports from package root
- **Error Handling**: Fail fast with clear, specific error messages
- **Context Managers**: Use for resource management (files, connections, locks)

### Prohibited Patterns
- ❌ `any` type (use proper types or `unknown`)
- ❌ Bare `except:` clauses (catch specific exceptions)
- ❌ Mutable default arguments (use `None` and initialize in function)
- ❌ Global state (use dependency injection)
- ❌ Hardcoded metadata strings (use `MetadataKeys` constants)

### Configuration Management
- All config via Pydantic `BaseSettings` (reads from `.env`)
- No secrets in code, docs, logs, or commits
- Use `.env.example` as template (tracked in git)
- Validate on startup (fail fast if misconfigured)

## Environment Variables

**Required variables** (must be set in `.env`):

```bash
# Service Ports
TEI_HTTP_PORT=52000
QDRANT_HTTP_PORT=52001
QDRANT_GRPC_PORT=52002
POSTGRES_PORT=53432
POSTGRES_PASSWORD=<secure-password>
REDIS_PORT=53379
CRAWL4AI_PORT=52004

# TEI Configuration
TEI_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
TEI_MAX_CONCURRENT_REQUESTS=128
```

**Complete list**: See `.env.example` (50+ variables documented)

## Known Issues

**HIGH Severity Bugs** (5 total):
1. `status` command crashes on PARTIAL status (KeyError)
2. `scrape` flag parsing broken (--output, --concurrent treated as URLs)
3. `extract --prompt` fails with 422 error from Crawl4AI
4. `crawl --file` broken (requires positional URLs even with --file)
5. `watch` command has async/await bug in Qdrant validation

**MEDIUM Severity Bugs** (1 total):
6. `map` internal link discovery returns 0 URLs (broken filter logic)

**Complete Details**: See `.docs/research/cli-testing-report-2026-01-20.md`

## Repository Status & References

### Current State
- ✅ 786 tests passing with 87% unit test coverage
- ✅ Circuit breaker patterns implemented
- ✅ Async-first architecture throughout
- ✅ All infrastructure services operational (Crawl4AI, Qdrant, Redis, PostgreSQL)
- ⚠️ CLI commands 45% functional (real-world usage testing)
- ❌ 5 HIGH severity bugs remain

### Detailed Documentation
- **Architecture Guide**: `.docs/research/codebase-architecture.docs.md` (comprehensive 58KB guide)
- **Documentation Gap Analysis**: `.docs/research/documentation-implementation-gap-analysis.md`
- **Session Logs**: `.docs/sessions/` (timestamped development history)
- **Service Ports**: `.docs/services-ports.md` (port assignment registry)
- **README**: `README.md` (CLI usage and examples)

### Specifications (Architectural Reference)
- `specs/rag-ingestion/` - Complete design specs (treat as architectural guidance, not implementation status)
- `specs/web-crawl-cli/` - CLI implementation specs (Phases 1-4 complete)

## Quality Standards

### Test Coverage
- **Target**: 85%+ code coverage
- **Current**: 87%+ (exceeding target)
- **Strategy**: TDD with RED-GREEN-REFACTOR cycle
- **Isolation**: All tests independent, can run in any order

### Code Quality
- **Linting**: Zero ruff errors before commit
- **Type Safety**: Zero ty errors before commit
- **Complexity**: Max cyclomatic complexity 10
- **Function Size**: Max 50 lines per function

### Performance
- **Async Operations**: All I/O must be non-blocking
- **Batch Processing**: Process documents in configurable batches
- **Circuit Breakers**: Protect against service failures
- **Retry Logic**: Exponential backoff with jitter

## Notes for AI Assistants

### What to Do
- ✅ Read `.docs/research/codebase-architecture.docs.md` for comprehensive context
- ✅ Use `MetadataKeys` constants for all metadata access
- ✅ Follow async patterns consistently
- ✅ Activate venv before every Python command
- ✅ Run quality checks before suggesting commits
- ✅ Reference file:line numbers (e.g., `crawl4r/readers/crawl4ai.py:45`)

### What NOT to Do
- ❌ Assume specs in `specs/` reflect current implementation (most is already built)
- ❌ Bypass circuit breaker logic for "speed"
- ❌ Hardcode metadata keys (use constants)
- ❌ Use standard ports (80, 443, 3000, 5432, 6379)
- ❌ Skip venv activation
- ❌ Commit without running ruff, ty, and pytest

### Common Pitfalls
1. **Virtual Environment**: Forgetting to activate leads to import errors
2. **Metadata Keys**: Hardcoding strings instead of using `MetadataKeys` constants
3. **Port Conflicts**: Using standard ports instead of high ports (53000+)
4. **Crawl4AI Endpoint**: Using `/crawl` instead of `/md?f=fit` (returns 7x more data)
5. **Relative vs Absolute Paths**: `file_path` is absolute; compute relative on demand
6. **Embedding Dimensions**: Assuming 768 or 1536 instead of actual 1024

## Getting Help

### Documentation
- **Quick Reference**: This file (CLAUDE.md)
- **Deep Dive**: `.docs/research/codebase-architecture.docs.md`
- **CLI Help**: `python -m crawl4r.cli.app --help`
- **Test Examples**: Browse `tests/` directory for usage patterns

### Troubleshooting
1. Services not starting → Check Docker network exists: `docker network create crawl4r`
2. Import errors → Activate venv: `source .venv/bin/activate`
3. Port conflicts → Check availability: `ss -tuln | grep :PORT`
4. Test failures → Check services: `docker compose ps`
5. Type errors → Run ty: `ty check crawl4r/`

### Key Files to Reference
- `pyproject.toml` - Dependencies and tool configuration
- `docker-compose.yaml` - Service definitions
- `.env.example` - Environment variable template
- `README.md` - CLI usage and examples
