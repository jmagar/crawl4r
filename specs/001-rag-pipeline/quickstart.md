# Quickstart Guide: RAG Pipeline Local Development

**Feature**: 001-rag-pipeline
**Date**: 2025-01-11
**Target Audience**: Developers setting up local development environment

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Service Configuration](#3-service-configuration)
4. [Database Initialization](#4-database-initialization)
5. [Running the Application](#5-running-the-application)
6. [API Testing](#6-api-testing)
7. [Development Workflow](#7-development-workflow)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

### 1.1 Required Software

Install the following on your development machine:

| Software | Version | Purpose | Installation |
|----------|---------|---------|--------------|
| **Python** | 3.11+ | Application runtime | [python.org](https://python.org) |
| **uv** | latest | Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Docker** | 20.10+ | Container runtime | [docker.com](https://docker.com) |
| **Docker Compose** | 2.0+ | Multi-container orchestration | Included with Docker Desktop |
| **Git** | 2.30+ | Version control | [git-scm.com](https://git-scm.com) |
| **curl** | latest | API testing | Usually pre-installed (Linux/macOS) |

### 1.2 Hardware Requirements

- **CPU**: 4+ cores recommended (2 minimum)
- **RAM**: 8GB minimum, 16GB recommended (for TEI embedding service)
- **Disk**: 10GB free space (for Docker images + data)
- **GPU**: Optional (significantly speeds up embedding generation)

### 1.3 Verify Prerequisites

```bash
# Check versions
python --version    # Should show 3.11+
uv --version        # Should show latest
docker --version    # Should show 20.10+
docker compose version  # Should show 2.0+ (note: no hyphen)
git --version       # Should show 2.30+
```

---

## 2. Environment Setup

### 2.1 Clone Repository

```bash
git clone https://github.com/yourorg/crawl4r.git
cd crawl4r
git checkout 001-rag-pipeline
```

### 2.2 Create Environment File

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your preferred editor:

```bash
# === DATABASE ===
POSTGRES_DB=crawl4r
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here   # CHANGE THIS!
DATABASE_URL=postgresql+asyncpg://postgres:your_secure_password_here@localhost:53432/crawl4r

# === REDIS ===
REDIS_URL=redis://localhost:53379
REDIS_MAX_CONNECTIONS=50

# === QDRANT ===
QDRANT_URL=http://localhost:52002
QDRANT_COLLECTION=crawl4r
QDRANT_API_KEY=  # Optional, leave empty for local dev

# === TEI (Text Embeddings Inference) ===
TEI_URL=http://localhost:52010
TEI_MODEL=Qwen/Qwen3-0.6B-Embedding
TEI_MAX_BATCH_SIZE=128

# === CRAWL4AI ===
CRAWL4AI_URL=http://localhost:52001

# === API CONFIGURATION ===
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
SECRET_KEY=your_secret_key_here_generate_with_python3_-c_"import secrets; print(secrets.token_urlsafe(32))"
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# === LOGGING ===
LOG_LEVEL=INFO
LOG_FORMAT=json

# === WORKER CONFIGURATION ===
ARQ_WORKERS=2
ARQ_MAX_JOBS=10
```

**Generate Secret Key**:
```bash
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"
```

### 2.3 Install Python Dependencies

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
uv pip install -e ".[dev]"
```

**Expected** `pyproject.toml` dependencies (will be created during implementation):
```toml
[project]
name = "crawl4r"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "sqlalchemy>=2.0.0",
    "asyncpg>=0.29.0",
    "alembic>=1.13.0",
    "redis>=5.0.0",
    "qdrant-client>=1.9.0",
    "arq>=0.26.0",
    "httpx>=0.28.0",
    "uvicorn[standard]>=0.30.0",
    "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "mypy>=1.10.0",
    "ruff>=0.5.0",
]
```

---

## 3. Service Configuration

### 3.1 Docker Compose Services

The application uses 6 Docker services:

| Service | Port (Host→Container) | Purpose |
|---------|----------------------|---------|
| `postgres` | 53432→5432 | Document storage + FTS |
| `redis` | 53379→6379 | Cache + queues + rate limiting |
| `qdrant` | 52002→6333 | Vector storage |
| `tei` | 52010→80 | Embedding generation |
| `crawl4ai` | 52001→11235 | Web crawling |
| `api` | 52003→8000 | FastAPI application |
| `worker` | — | Background job processor |

### 3.2 Start External Services

**Option 1: Start all services** (recommended for first run):
```bash
docker compose up -d
```

**Option 2: Start only external services** (for Python development):
```bash
docker compose up -d postgres redis qdrant tei crawl4ai
```

### 3.3 Verify Services

Check all containers are running:
```bash
docker compose ps
```

Expected output:
```
NAME                COMMAND                  STATUS              PORTS
crawl4r-api         "uvicorn app.main:..."   Up (healthy)        0.0.0.0:52003->8000/tcp
crawl4r-crawl4ai    "/docker-entrypoint..."  Up (healthy)        0.0.0.0:52001->11235/tcp
crawl4r-postgres    "docker-entrypoint...."  Up (healthy)        0.0.0.0:53432->5432/tcp
crawl4r-qdrant      "/qdrant/qdrant"         Up (healthy)        0.0.0.0:52002->6333/tcp
crawl4r-redis       "docker-entrypoint...."  Up (healthy)        0.0.0.0:53379->6379/tcp
crawl4r-tei         "text-embeddings-..."    Up (healthy)        0.0.0.0:52010->80/tcp
crawl4r-worker      "arq app.workers.W..."   Up                  —
```

### 3.4 Check Service Health

```bash
# PostgreSQL
docker compose exec postgres pg_isready

# Redis
docker compose exec redis redis-cli ping

# Qdrant
curl http://localhost:52002/health

# TEI
curl http://localhost:52010/health

# Crawl4AI
curl http://localhost:52001/health
```

---

## 4. Database Initialization

### 4.1 Run Migrations

Apply database schema using Alembic:

```bash
# Create initial migration (first time only)
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head
```

### 4.2 Verify Schema

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U postgres -d crawl4r

# List tables
\dt

# Expected tables:
# - api_keys
# - collections
# - tags
# - documents
# - document_tags
# - chunks
# - crawl_jobs
# - deep_crawl_jobs
# - deep_crawl_frontier
# - discovery_jobs
# - webhooks
# - webhook_deliveries
# - crawl_configs
# - crawl_sources
# - domain_settings
# - proxy_configs
# - robots_txt_cache
# - canonical_urls

# Exit psql
\q
```

### 4.3 Initialize Qdrant Collection

Run the setup script (will be created during implementation):

```bash
python scripts/init_qdrant.py
```

Or manually via Python:
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, ScalarQuantizationConfig, ScalarType

client = QdrantClient(url="http://localhost:52002")

client.create_collection(
    collection_name="crawl4r",
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE,
        on_disk=True
    ),
    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    quantization_config=ScalarQuantizationConfig(
        type=ScalarType.INT8,
        quantile=0.99,
        always_ram=True
    )
)

print("✓ Qdrant collection 'crawl4r' created")
```

---

## 5. Running the Application

### 5.1 Development Mode (Hot Reload)

**Terminal 1: API Server**
```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Background Worker**
```bash
source .venv/bin/activate
arq app.workers.WorkerSettings
```

### 5.2 Production Mode (Docker)

```bash
# Build and start all services
docker compose up -d --build

# View logs
docker compose logs -f api worker
```

### 5.3 Verify Application

**Health Check**:
```bash
curl http://localhost:52003/health
# Expected: {"status":"healthy"}

curl http://localhost:52003/health/ready
# Expected: {"status":"ready","checks":{"postgres":true,"redis":true,...}}
```

**Interactive API Documentation**:
- **Swagger UI**: http://localhost:52003/docs
- **ReDoc**: http://localhost:52003/redoc

---

## 6. API Testing

### 6.1 Create API Key

```bash
# Generate API key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# Example output: dGhpc19pc19hX3NlY3VyZV9hcGlfa2V5X2V4YW1wbGU

# Insert into database (production: use API endpoint)
docker compose exec postgres psql -U postgres -d crawl4r -c "
INSERT INTO api_keys (key_hash, name, scopes, rate_limit_rpm)
VALUES (
    encode(digest('YOUR_API_KEY_HERE', 'sha256'), 'hex'),
    'Development Key',
    '{read,write,admin}',
    120
);"
```

### 6.2 Test Endpoints

**Upload Document**:
```bash
curl -X POST "http://localhost:52003/api/v1/documents" \
  -H "Authorization: Bearer YOUR_API_KEY_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/test-doc",
    "title": "Test Document",
    "content": "This is a test document for the RAG pipeline. It contains information about machine learning and neural networks.",
    "tags": ["test", "ml"]
  }'
```

**Search Documents**:
```bash
curl -X POST "http://localhost:52003/api/v1/search" \
  -H "Authorization: Bearer YOUR_API_KEY_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "limit": 10
  }'
```

**Submit Crawl Job**:
```bash
curl -X POST "http://localhost:52003/api/v1/crawl" \
  -H "Authorization: Bearer YOUR_API_KEY_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com"],
    "priority": "normal"
  }'
```

**Check Crawl Job Status**:
```bash
# Replace JOB_ID with actual job ID from previous response
curl "http://localhost:52003/api/v1/crawl/JOB_ID" \
  -H "Authorization: Bearer YOUR_API_KEY_HERE"
```

### 6.3 Test with Pytest

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/integration/test_search_flow.py -v

# Run specific test
pytest tests/unit/test_url_validator.py::test_ssrf_prevention -v
```

---

## 7. Development Workflow

### 7.1 TDD Cycle (MANDATORY)

**RED-GREEN-REFACTOR**:

1. **RED**: Write failing test
```python
# tests/unit/test_rrf_fusion.py
def test_rrf_fusion_combines_results():
    vector_results = [("doc1", 0.9), ("doc2", 0.8)]
    keyword_results = [("doc2", 0.95), ("doc3", 0.85)]

    fused = rrf_fusion(vector_results, keyword_results, k=60)

    assert fused[0][0] == "doc2"  # doc2 in both (highest score)
```

2. **GREEN**: Make test pass
```python
# app/services/search_service.py
def rrf_fusion(vector_results, keyword_results, k=60):
    scores = defaultdict(float)
    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] += 1 / (k + rank)
    for rank, (doc_id, _) in enumerate(keyword_results):
        scores[doc_id] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

3. **REFACTOR**: Improve while keeping tests green

### 7.2 Code Quality Checks

```bash
# Type checking
mypy app --strict

# Linting
ruff check app tests

# Formatting
ruff format app tests

# All checks together
pytest && mypy app --strict && ruff check app tests
```

### 7.3 Database Migrations

**Create Migration**:
```bash
# After modifying SQLAlchemy models
alembic revision --autogenerate -m "Add new field to documents"

# Review generated migration in alembic/versions/
# Edit if necessary

# Apply migration
alembic upgrade head
```

**Rollback Migration**:
```bash
# Rollback one migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade abc123
```

### 7.4 Git Workflow

```bash
# Create feature branch
git checkout -b feature/add-search-filters

# Make changes, run tests
pytest && mypy app --strict

# Commit (tests must pass)
git add .
git commit -m "Add domain filtering to search

- Added domain filter to SearchFilters model
- Updated Qdrant query to filter by domain
- Added tests for domain filtering
- All tests passing (85% coverage)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Push to remote
git push -u origin feature/add-search-filters

# Create pull request (via GitHub CLI or web)
gh pr create --title "Add domain filtering to search" --body "..."
```

---

## 8. Troubleshooting

### 8.1 Common Issues

#### **Problem**: Port already in use

```bash
# Check what's using the port
lsof -i :52003  # Replace with conflicting port
# or
ss -tuln | grep :52003

# Solution: Stop the conflicting process or change port in .env
```

#### **Problem**: Docker container unhealthy

```bash
# Check container logs
docker compose logs postgres
docker compose logs qdrant

# Restart specific service
docker compose restart postgres

# Rebuild if needed
docker compose up -d --build postgres
```

#### **Problem**: Database connection refused

```bash
# Verify PostgreSQL is running
docker compose ps postgres

# Check DATABASE_URL in .env matches docker-compose.yaml
# For Docker services: use service name (postgres:5432)
# For host machine: use localhost:53432

# Test connection
docker compose exec postgres pg_isready
```

#### **Problem**: Embedding generation slow

```bash
# Check TEI logs
docker compose logs tei

# If no GPU, expect 100-200ms latency per chunk
# With GPU (recommended): < 50ms per chunk

# Solution: Batch embeddings (32-128 chunks) to amortize overhead
```

#### **Problem**: Tests failing with "Event loop is closed"

```python
# Ensure pytest-asyncio is installed
# Add to pyproject.toml:
[tool.pytest.ini_options]
asyncio_mode = "auto"

# Or mark tests with:
@pytest.mark.asyncio
async def test_async_function():
    ...
```

### 8.2 Reset Everything

**Complete Reset** (deletes all data):
```bash
# Stop and remove all containers, volumes, networks
docker compose down -v

# Remove Python virtual environment
rm -rf .venv

# Remove database migrations
rm -rf alembic/versions/*.py

# Start fresh
docker compose up -d
source .venv/bin/activate
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

### 8.3 Debugging

**Enable Debug Logging**:
```bash
# Edit .env
LOG_LEVEL=DEBUG

# Restart services
docker compose restart api worker
```

**Interactive Debugging**:
```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Run with pytest
pytest tests/unit/test_search_service.py -s

# Or run in development mode
uvicorn app.main:app --reload
```

**Check Redis Cache**:
```bash
# Connect to Redis
docker compose exec redis redis-cli

# List all keys
KEYS crawl4r:*

# Check specific cache entry
GET crawl4r:cache:query:abc123...

# Clear all cache
FLUSHDB
```

---

## 9. Next Steps

### 9.1 Development Priorities

1. **Phase 1**: Implement core models (`app/core/models.py`)
2. **Phase 2**: Set up database layer (`app/storage/postgres.py`, `app/storage/qdrant.py`)
3. **Phase 3**: Implement services (`app/services/*.py`)
4. **Phase 4**: Build API endpoints (`app/api/v1/*.py`)
5. **Phase 5**: Add background workers (`app/workers/*.py`)
6. **Phase 6**: Integration tests (`tests/integration/*.py`)

### 9.2 Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Pydantic Docs**: https://docs.pydantic.dev/
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **ARQ Docs**: https://arq-docs.helpmanual.io/
- **Alembic Docs**: https://alembic.sqlalchemy.org/

### 9.3 Getting Help

- **GitHub Issues**: https://github.com/yourorg/crawl4r/issues
- **Discord**: https://discord.gg/crawl4r (if available)
- **Documentation**: `/docs/README.md`

---

**Status**: ✅ Quickstart guide complete
**Last Updated**: 2025-01-11
**Maintainer**: Development Team
