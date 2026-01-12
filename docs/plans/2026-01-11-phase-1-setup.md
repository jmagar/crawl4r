# Phase 1 Setup (Shared Infrastructure) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stand up the Phase 1 project skeleton, tooling config, and documentation scaffold for the crawl4r RAG pipeline.

**Architecture:** Initialize a backend-only Python 3.11+ monorepo structure (apps/packages not required yet), with core directories, uv-managed dependencies, lint/type/test tooling, env templates, and Docker Compose services. Keep root clean and align ports/services with spec.

**Tech Stack:** Python 3.11+, FastAPI 0.115+, Pydantic 2.x, uv, ruff, mypy (strict), pytest/pytest-asyncio, Docker Compose, PostgreSQL 15, Redis 7, Qdrant 1.9, TEI, Crawl4AI.

---

### Task 1: Create directory structure per plan.md (T001)

**Files:**
- Create: `app/.gitkeep`
- Create: `tests/.gitkeep`
- Create: `alembic/.gitkeep`
- Create: `scripts/.gitkeep`
- Create: `docs/.gitkeep`
- Create: `.docs/.gitkeep`

**Step 1: Verify directories do not yet exist**

Run: `ls app tests alembic scripts docs .docs`
Expected: FAIL with "No such file or directory" for missing directories.

**Step 2: Create directories and placeholder files**

Create directories and add `.gitkeep` files to ensure they are tracked:

```bash
mkdir -p app tests alembic scripts docs .docs
printf "" > app/.gitkeep
printf "" > tests/.gitkeep
printf "" > alembic/.gitkeep
printf "" > scripts/.gitkeep
printf "" > docs/.gitkeep
printf "" > .docs/.gitkeep
```

**Step 3: Verify directory structure**

Run: `ls app tests alembic scripts docs .docs`
Expected: Each directory listed without errors.

**Step 4: Commit**

```bash
git add app/.gitkeep tests/.gitkeep alembic/.gitkeep scripts/.gitkeep docs/.gitkeep .docs/.gitkeep
git commit -m "chore: add initial project directories"
```

---

### Task 2: Initialize Python 3.11+ project with pyproject.toml (T002)

**Files:**
- Create: `pyproject.toml`

**Step 1: Create minimal pyproject.toml**

Create: `pyproject.toml`

```toml
[project]
name = "crawl4r"
version = "0.1.0"
description = "Self-hosted RAG pipeline for intelligent web content retrieval"
requires-python = ">=3.11"
readme = "README.md"
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
  "pyyaml>=6.0.0",
]
```

**Step 2: Validate pyproject metadata**

Run: `python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb')); print(p['project']['name'])"`
Expected: `crawl4r`

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pyproject with core dependencies"
```

---

### Task 3: Configure linting, formatting, and mypy strict (T003)

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add ruff, mypy, pytest config**

Modify: `pyproject.toml`

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
extend-exclude = [".venv", ".cache", "alembic/versions"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "lf"

[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_configs = true
ignore_missing_imports = false
exclude = "(?x)(^\\.venv/|^\\.cache/|^alembic/versions/)"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 2: Validate tool sections exist**

Run: `python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb')); print('ruff' in p['tool'] and 'mypy' in p['tool'])"`
Expected: `True`

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: configure ruff mypy pytest"
```

---

### Task 4: Create .env.example template (T004)

**Files:**
- Create: `.env.example`

**Step 1: Create .env.example**

Create: `.env.example`

```bash
# === DATABASE ===
POSTGRES_DB=crawl4r
POSTGRES_USER=postgres
POSTGRES_PASSWORD=change_me
DATABASE_URL=postgresql+asyncpg://postgres:change_me@localhost:53432/crawl4r

# === REDIS ===
REDIS_URL=redis://localhost:53379
REDIS_MAX_CONNECTIONS=50

# === QDRANT ===
QDRANT_URL=http://localhost:52002
QDRANT_COLLECTION=crawl4r
QDRANT_API_KEY=

# === TEI ===
TEI_URL=http://localhost:52010
TEI_MODEL=Qwen/Qwen3-0.6B-Embedding
TEI_MAX_BATCH_SIZE=128

# === CRAWL4AI ===
CRAWL4AI_URL=http://localhost:52001

# === API ===
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
SECRET_KEY=change_me
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# === LOGGING ===
LOG_LEVEL=INFO
LOG_FORMAT=json

# === WORKERS ===
ARQ_WORKERS=2
ARQ_MAX_JOBS=10
```

**Step 2: Validate required variables exist**

Run: `rg "DATABASE_URL=|REDIS_URL=|QDRANT_URL=|TEI_URL=|CRAWL4AI_URL=|SECRET_KEY=" .env.example`
Expected: All entries present.

**Step 3: Commit**

```bash
git add .env.example
git commit -m "chore: add env template"
```

---

### Task 5: Create docker-compose.yaml with all services (T005)

**Files:**
- Create: `docker-compose.yaml`

**Step 1: Add docker-compose.yaml**

Create: `docker-compose.yaml`

```yaml
services:
  postgres:
    image: postgres:15-alpine
    container_name: crawl4r-postgres
    ports:
      - "53432:5432"
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - crawl4r_postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    container_name: crawl4r-redis
    ports:
      - "53379:6379"
    command: ["redis-server", "--maxmemory", "2gb", "--maxmemory-policy", "allkeys-lru"]
    volumes:
      - crawl4r_redis_data:/data

  # qdrant:
  #   image: qdrant/qdrant:v1.9
  #   container_name: crawl4r-qdrant
  #   ports:
  #     - "52002:6333"
  #   volumes:
  #     - crawl4r_qdrant_data:/qdrant/storage

  # tei:
  #   image: ghcr.io/huggingface/text-embeddings-inference:latest
  #   container_name: crawl4r-tei
  #   ports:
  #     - "52010:80"
  #   environment:
  #     MODEL_ID: ${TEI_MODEL}
  #   volumes:
  #     - crawl4r_tei_data:/data

  crawl4ai:
    image: unclecode/crawl4ai:latest
    container_name: crawl4r-crawl4ai
    ports:
      - "52001:11235"
    shm_size: "2gb"

volumes:
  crawl4r_postgres_data:
  crawl4r_redis_data:
  # crawl4r_qdrant_data:
  # crawl4r_tei_data:
```

**Step 2: Validate required services and ports**

Run: `rg "postgres:|redis:|qdrant:|tei:|crawl4ai:" docker-compose.yaml`
Expected: postgres, redis, crawl4ai listed; qdrant and tei commented out when external.

Run: `rg "53432:5432|53379:6379|52002:6333|52010:80|52001:11235" docker-compose.yaml`
Expected: All port mappings present; qdrant/tei mappings may be commented out.

**Step 3: Commit**

```bash
git add docker-compose.yaml
git commit -m "chore: add docker compose services"
```

---

### Task 6: Create README.md with overview and setup (T006)

**Files:**
- Create: `README.md`

**Step 1: Create README.md**

Create: `README.md`

```markdown
# crawl4r

## Overview

crawl4r is a self-hosted RAG pipeline that crawls web content, extracts clean text, stores documents in PostgreSQL with full-text search, and indexes embeddings in Qdrant for hybrid retrieval.

## Quick Start

1. Copy `.env.example` to `.env` and fill in values.
2. Start services: `docker compose up -d`
3. Create a virtual environment: `uv venv`
4. Install dependencies: `uv pip install -e ".[dev]"`
5. Run the API: `uvicorn app.main:app --reload`

## Development

- Lint: `ruff check app tests`
- Format: `ruff format app tests`
- Type check: `mypy app --strict`
- Test: `pytest`
```

**Step 2: Validate README sections**

Run: `rg "# crawl4r|## Overview|## Quick Start|## Development" README.md`
Expected: All required headings present.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add project readme"
```

---

### Task 7: Create .gitignore (T008)

**Files:**
- Create: `.gitignore`

**Step 1: Create .gitignore**

Create: `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.mypy_cache/
.cache/

# Env
.env

# Editors
.vscode/
.idea/

# OS
.DS_Store

# Node
node_modules/
```

**Step 2: Validate key ignore patterns**

Run: `rg "\.env|\.venv|__pycache__|\.cache/|node_modules/" .gitignore`
Expected: All patterns listed.

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add gitignore"
```

---

## Verification Checklist

- `ls app tests alembic scripts docs .docs`
- `python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb')); print(p['project']['name'])"`
- `python -c "import tomllib; p=tomllib.load(open('pyproject.toml','rb')); print('ruff' in p['tool'] and 'mypy' in p['tool'])"`
- `rg "DATABASE_URL=|REDIS_URL=|QDRANT_URL=|TEI_URL=|CRAWL4AI_URL=|SECRET_KEY=" .env.example`
- `rg "postgres:|redis:|qdrant:|tei:|crawl4ai:" docker-compose.yaml`
- `rg "# crawl4r|## Overview|## Quick Start|## Development" README.md`
- `rg "\.env|\.venv|__pycache__|\.cache/|node_modules/" .gitignore`

## Notes

- The plan avoids TDD for environment scaffolding; only verification checks are listed.
- Keep ports aligned to plan: Crawl4AI 52001, Qdrant 52002, TEI 52010, Redis 53379, Postgres 53432.
- Do not add `src/` directories; keep code directly under `app/` and tests under `tests/`.
