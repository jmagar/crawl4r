# Crawl4r Module Reorganization and Rename Plan

> **Organization Note:** When this plan is fully implemented and verified, move this file to `docs/plans/complete/` to keep the plans folder organized.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename `rag_ingestion/` to `crawl4r/` and reorganize into logical submodule structure for better maintainability and discoverability, with dedicated CLI and API components.

**Architecture:** Rename package from `rag_ingestion` to `crawl4r`. Move 14 Python files from flat root structure into 7 logical submodules (core, readers, processing, storage, resilience, cli, api). Move main.py to cli/ submodule. Create FastAPI skeleton for REST API. Use git mv to preserve history. All imports use new submodule paths - no backward compatibility layer.

**Tech Stack:** Python 3.13, pytest, git, FastAPI

---

## Current Structure (Flat)

```
rag_ingestion/              # TO BE RENAMED: crawl4r/
├── __init__.py
├── main.py
├── chunker.py
├── circuit_breaker.py
├── config.py
├── crawl4ai_reader.py
├── failed_docs.py
├── file_watcher.py
├── logger.py
├── processor.py
├── quality.py
├── recovery.py
├── tei_client.py
└── vector_store.py
```

## Target Structure (Organized)

```
crawl4r/                     # Renamed from rag_ingestion/
├── __init__.py              # Package marker
├── core/                    # Core infrastructure
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   └── quality.py
├── readers/                 # Input sources
│   ├── __init__.py
│   ├── crawl4ai.py         # Renamed from crawl4ai_reader.py
│   └── file_watcher.py
├── processing/              # Document processing
│   ├── __init__.py
│   ├── chunker.py
│   └── processor.py
├── storage/                 # Storage backends
│   ├── __init__.py
│   ├── embeddings.py       # Renamed from tei_client.py
│   └── vector_store.py
├── resilience/              # Error handling & recovery
│   ├── __init__.py
│   ├── circuit_breaker.py
│   ├── failed_docs.py
│   └── recovery.py
├── cli/                     # Command-line interface
│   ├── __init__.py
│   ├── main.py             # CLI entry point (moved from root)
│   └── commands/           # Individual CLI commands (future)
└── api/                     # REST API
    ├── __init__.py
    ├── app.py              # FastAPI application (future)
    ├── routes/             # API route handlers (future)
    └── models/             # Request/response models (future)
```

---

## Task 1: Create Test for New Structure

**Files:**
- Create: `tests/unit/test_module_structure.py`

**Step 1: Write test verifying new imports work**

```python
"""Test module reorganization - verify new import paths work."""
import pytest


def test_core_modules_importable():
    """Test core submodule imports."""
    from crawl4r.core.config import Settings
    from crawl4r.core.logger import get_logger
    from crawl4r.core.quality import QualityVerifier

    assert Settings is not None
    assert get_logger is not None
    assert QualityVerifier is not None


def test_readers_modules_importable():
    """Test readers submodule imports."""
    from crawl4r.readers.crawl4ai import Crawl4AIReader
    from crawl4r.readers.file_watcher import FileWatcher

    assert Crawl4AIReader is not None
    assert FileWatcher is not None


def test_processing_modules_importable():
    """Test processing submodule imports."""
    from crawl4r.processing.chunker import MarkdownChunker
    from crawl4r.processing.processor import DocumentProcessor

    assert MarkdownChunker is not None
    assert DocumentProcessor is not None


def test_storage_modules_importable():
    """Test storage submodule imports."""
    from crawl4r.storage.embeddings import TEIClient
    from crawl4r.storage.vector_store import VectorStoreManager

    assert TEIClient is not None
    assert VectorStoreManager is not None


def test_resilience_modules_importable():
    """Test resilience submodule imports."""
    from crawl4r.resilience.circuit_breaker import CircuitBreaker
    from crawl4r.resilience.failed_docs import FailedDocument
    from crawl4r.resilience.recovery import StateRecovery

    assert CircuitBreaker is not None
    assert FailedDocument is not None
    assert StateRecovery is not None


def test_cli_modules_importable():
    """Test CLI submodule imports."""
    from crawl4r.cli.main import main

    assert main is not None


def test_api_modules_exist():
    """Test API submodule structure exists."""
    import crawl4r.api

    assert crawl4r.api is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_module_structure.py -v`

Expected: FAIL - modules don't exist yet

**Step 3: Commit test**

```bash
git add tests/unit/test_module_structure.py
git commit -m "test: add module structure verification tests"
```

---

## Task 2: Rename Package from rag_ingestion to crawl4r

**Files:**
- Rename: `rag_ingestion/` → `crawl4r/`

**Step 1: Rename the directory**

```bash
git mv rag_ingestion crawl4r
```

**Step 2: Verify the rename**

Run: `ls -la crawl4r/`

Expected: Directory renamed, all files present

**Step 3: Update any absolute imports in __init__.py**

The root `__init__.py` likely has no imports yet, but verify:

Run: `grep "rag_ingestion" crawl4r/__init__.py`

Expected: No matches (or update if found)

**Step 4: Commit the rename**

```bash
git add -A
git commit -m "refactor: rename package from rag_ingestion to crawl4r"
```

---

## Task 3: Create Submodule Directories

**Files:**
- Create: `crawl4r/core/__init__.py`
- Create: `crawl4r/readers/__init__.py`
- Create: `crawl4r/processing/__init__.py`
- Create: `crawl4r/storage/__init__.py`
- Create: `crawl4r/resilience/__init__.py`
- Create: `crawl4r/cli/__init__.py`
- Create: `crawl4r/cli/commands/__init__.py`
- Create: `crawl4r/api/__init__.py`
- Create: `crawl4r/api/routes/__init__.py`
- Create: `crawl4r/api/models/__init__.py`

**Step 1: Create core submodule**

```bash
mkdir -p crawl4r/core
touch crawl4r/core/__init__.py
```

**Step 2: Create readers submodule**

```bash
mkdir -p crawl4r/readers
touch crawl4r/readers/__init__.py
```

**Step 3: Create processing submodule**

```bash
mkdir -p crawl4r/processing
touch crawl4r/processing/__init__.py
```

**Step 4: Create storage submodule**

```bash
mkdir -p crawl4r/storage
touch crawl4r/storage/__init__.py
```

**Step 5: Create resilience submodule**

```bash
mkdir -p crawl4r/resilience
touch crawl4r/resilience/__init__.py
```

**Step 6: Create cli submodule**

```bash
mkdir -p crawl4r/cli/commands
touch crawl4r/cli/__init__.py
touch crawl4r/cli/commands/__init__.py
```

**Step 7: Create api submodule**

```bash
mkdir -p crawl4r/api/routes crawl4r/api/models
touch crawl4r/api/__init__.py
touch crawl4r/api/routes/__init__.py
touch crawl4r/api/models/__init__.py
```

**Step 8: Verify directories created**

Run: `ls -la crawl4r/*/`

Expected: 7 subdirectories with empty __init__.py files

**Step 9: Commit structure**

```bash
git add crawl4r/core/ crawl4r/readers/ crawl4r/processing/ crawl4r/storage/ crawl4r/resilience/ crawl4r/cli/ crawl4r/api/
git commit -m "feat: create submodule directory structure with CLI and API"
```

---

## Task 4: Move Core Infrastructure Files

**Files:**
- Move: `crawl4r/config.py` → `crawl4r/core/config.py`
- Move: `crawl4r/logger.py` → `crawl4r/core/logger.py`
- Move: `crawl4r/quality.py` → `crawl4r/core/quality.py`

**Step 1: Move config.py**

```bash
git mv crawl4r/config.py crawl4r/core/config.py
```

**Step 2: Move logger.py**

```bash
git mv crawl4r/logger.py crawl4r/core/logger.py
```

**Step 3: Move quality.py**

```bash
git mv crawl4r/quality.py crawl4r/core/quality.py
```

**Step 4: Update imports in moved files**

Edit `crawl4r/core/config.py`:
```python
# No internal rag_ingestion imports - no changes needed
```

Edit `crawl4r/core/logger.py`:
```python
# No internal rag_ingestion imports - no changes needed
```

Edit `crawl4r/core/quality.py`:
```python
# Change imports
from crawl4r.core.logger import get_logger
from crawl4r.core.config import Settings
```

**Step 5: Commit moves**

```bash
git add -A
git commit -m "refactor: move core infrastructure to core/ submodule"
```

---

## Task 5: Move Reader Files

**Files:**
- Move: `crawl4r/crawl4ai_reader.py` → `crawl4r/readers/crawl4ai.py`
- Move: `crawl4r/file_watcher.py` → `crawl4r/readers/file_watcher.py`

**Step 1: Move and rename crawl4ai_reader.py**

```bash
git mv crawl4r/crawl4ai_reader.py crawl4r/readers/crawl4ai.py
```

**Step 2: Move file_watcher.py**

```bash
git mv crawl4r/file_watcher.py crawl4r/readers/file_watcher.py
```

**Step 3: Update imports in crawl4ai.py**

Edit `crawl4r/readers/crawl4ai.py`:
```python
# Change imports
from crawl4r.core.logger import get_logger
```

**Step 4: Update imports in file_watcher.py**

Edit `crawl4r/readers/file_watcher.py`:
```python
# Change imports
from crawl4r.core.config import Settings
from crawl4r.core.logger import get_logger
```

**Step 5: Commit moves**

```bash
git add -A
git commit -m "refactor: move readers to readers/ submodule"
```

---

## Task 6: Move Processing Files

**Files:**
- Move: `crawl4r/chunker.py` → `crawl4r/processing/chunker.py`
- Move: `crawl4r/processor.py` → `crawl4r/processing/processor.py`

**Step 1: Move chunker.py**

```bash
git mv crawl4r/chunker.py crawl4r/processing/chunker.py
```

**Step 2: Move processor.py**

```bash
git mv crawl4r/processor.py crawl4r/processing/processor.py
```

**Step 3: Update imports in chunker.py**

Edit `crawl4r/processing/chunker.py`:
```python
# Change imports
from crawl4r.core.logger import get_logger
```

**Step 4: Update imports in processor.py**

Edit `crawl4r/processing/processor.py`:
```python
# Change imports
from crawl4r.core.config import Settings
from crawl4r.core.logger import get_logger
from crawl4r.processing.chunker import MarkdownChunker
from crawl4r.storage.embeddings import TEIClient
from crawl4r.storage.vector_store import VectorStoreManager
from crawl4r.resilience.circuit_breaker import CircuitBreaker
from crawl4r.resilience.failed_docs import FailedDocumentTracker
```

**Step 5: Commit moves**

```bash
git add -A
git commit -m "refactor: move processing modules to processing/ submodule"
```

---

## Task 7: Move Storage Files

**Files:**
- Move: `crawl4r/tei_client.py` → `crawl4r/storage/embeddings.py`
- Move: `crawl4r/vector_store.py` → `crawl4r/storage/vector_store.py`

**Step 1: Move and rename tei_client.py**

```bash
git mv crawl4r/tei_client.py crawl4r/storage/embeddings.py
```

**Step 2: Move vector_store.py**

```bash
git mv crawl4r/vector_store.py crawl4r/storage/vector_store.py
```

**Step 3: Update imports in embeddings.py**

Edit `crawl4r/storage/embeddings.py`:
```python
# Change imports
from crawl4r.core.config import Settings
from crawl4r.core.logger import get_logger
from crawl4r.resilience.circuit_breaker import CircuitBreaker
```

**Step 4: Update imports in vector_store.py**

Edit `crawl4r/storage/vector_store.py`:
```python
# Change imports
from crawl4r.core.config import Settings
from crawl4r.core.logger import get_logger
```

**Step 5: Commit moves**

```bash
git add -A
git commit -m "refactor: move storage modules to storage/ submodule"
```

---

## Task 8: Move Resilience Files

**Files:**
- Move: `crawl4r/circuit_breaker.py` → `crawl4r/resilience/circuit_breaker.py`
- Move: `crawl4r/failed_docs.py` → `crawl4r/resilience/failed_docs.py`
- Move: `crawl4r/recovery.py` → `crawl4r/resilience/recovery.py`

**Step 1: Move circuit_breaker.py**

```bash
git mv crawl4r/circuit_breaker.py crawl4r/resilience/circuit_breaker.py
```

**Step 2: Move failed_docs.py**

```bash
git mv crawl4r/failed_docs.py crawl4r/resilience/failed_docs.py
```

**Step 3: Move recovery.py**

```bash
git mv crawl4r/recovery.py crawl4r/resilience/recovery.py
```

**Step 4: Update imports in circuit_breaker.py**

Edit `crawl4r/resilience/circuit_breaker.py`:
```python
# Change imports
from crawl4r.core.logger import get_logger
```

**Step 5: Update imports in failed_docs.py**

Edit `crawl4r/resilience/failed_docs.py`:
```python
# Change imports
from crawl4r.core.config import Settings
from crawl4r.core.logger import get_logger
```

**Step 6: Update imports in recovery.py**

Edit `crawl4r/resilience/recovery.py`:
```python
# Change imports
from crawl4r.core.config import Settings
from crawl4r.core.logger import get_logger
```

**Step 7: Commit moves**

```bash
git add -A
git commit -m "refactor: move resilience modules to resilience/ submodule"
```

---

## Task 9: Move main.py to CLI Submodule

**Files:**
- Move: `crawl4r/main.py` → `crawl4r/cli/main.py`

**Step 1: Move main.py to cli/**

```bash
git mv crawl4r/main.py crawl4r/cli/main.py
```

**Step 2: Update imports in cli/main.py**

Edit `crawl4r/cli/main.py` - replace all imports:

```python
"""RAG Ingestion Pipeline - CLI Entry Point."""
from crawl4r.core.config import Settings
from crawl4r.core.logger import get_logger
from crawl4r.core.quality import QualityVerifier
from crawl4r.readers.file_watcher import FileWatcher
from crawl4r.processing.processor import DocumentProcessor
from crawl4r.storage.embeddings import TEIClient
from crawl4r.storage.vector_store import VectorStoreManager
from crawl4r.resilience.recovery import StateRecovery
```

**Step 3: Verify cli/main.py has no syntax errors**

Run: `python -m py_compile crawl4r/cli/main.py`

Expected: No output (success)

**Step 4: Commit changes**

```bash
git add -A
git commit -m "refactor: move main.py to cli/ submodule"
```

---

## Task 10: Create API Skeleton Structure

**Files:**
- Create: `crawl4r/api/app.py`
- Create: `crawl4r/api/routes/health.py`
- Create: `crawl4r/api/models/responses.py`

**Step 1: Create FastAPI application skeleton**

Create `crawl4r/api/app.py`:

```python
"""RAG Ingestion Pipeline - REST API.

FastAPI application for RAG ingestion operations.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from crawl4r.core.config import Settings
from crawl4r.core.logger import get_logger

logger = get_logger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting RAG Ingestion API")
    yield
    # Shutdown
    logger.info("Shutting down RAG Ingestion API")


app = FastAPI(
    title="RAG Ingestion API",
    description="REST API for RAG document ingestion pipeline",
    version="0.1.0",
    lifespan=lifespan,
)
```

**Step 2: Create health check route**

Create `crawl4r/api/routes/health.py`:

```python
"""Health check endpoints."""
from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check():
    """Health check endpoint.

    Returns:
        dict: Service health status
    """
    return {"status": "healthy", "service": "rag-ingestion"}
```

**Step 3: Create response models**

Create `crawl4r/api/models/responses.py`:

```python
"""Pydantic response models for API."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    service: str
```

**Step 4: Update api/__init__.py to export app**

Edit `crawl4r/api/__init__.py`:

```python
"""RAG Ingestion API module."""
from crawl4r.api.app import app

__all__ = ["app"]
```

**Step 5: Verify API files have no syntax errors**

Run: `python -m py_compile crawl4r/api/app.py crawl4r/api/routes/health.py crawl4r/api/models/responses.py`

Expected: No output (success)

**Step 6: Commit API skeleton**

```bash
git add crawl4r/api/
git commit -m "feat: create API skeleton structure with FastAPI"
```

---

## Task 11: Update Examples Imports

**Files:**
- Modify: `examples/stress_test_pipeline.py`
- Modify: `examples/crawl4ai_reader_usage.py`

**Step 1: Update stress_test_pipeline.py imports**

Edit `examples/stress_test_pipeline.py` - replace imports at top:

```python
# Change from:
# from crawl4r.chunker import MarkdownChunker
# from crawl4r.crawl4ai_reader import Crawl4AIReader
# from crawl4r.logger import get_logger
# from crawl4r.tei_client import TEIClient
# from crawl4r.vector_store import VectorStoreManager

# To (using new structure):
from crawl4r.processing.chunker import MarkdownChunker
from crawl4r.readers.crawl4ai import Crawl4AIReader
from crawl4r.core.logger import get_logger
from crawl4r.storage.embeddings import TEIClient
from crawl4r.storage.vector_store import VectorStoreManager
```

**Step 2: Update crawl4ai_reader_usage.py imports**

Edit `examples/crawl4ai_reader_usage.py` - replace imports:

```python
# Change to new structure
from crawl4r.readers.crawl4ai import Crawl4AIReader
from crawl4r.core.logger import get_logger
from crawl4r.storage.vector_store import VectorStoreManager
from crawl4r.processing.chunker import MarkdownChunker
```

**Step 3: Verify examples run without import errors**

Run: `python -m py_compile examples/stress_test_pipeline.py examples/crawl4ai_reader_usage.py`

Expected: No output (success)

**Step 4: Commit changes**

```bash
git add examples/
git commit -m "refactor: update examples to use new module structure"
```

---

## Task 12: Update Test Imports

**Files:**
- Modify: All files in `tests/` that import from `rag_ingestion`

**Step 1: Find and list all test files with imports**

Run: `grep -r "from rag_ingestion" tests/ --include="*.py" -l`

Expected: List of test files to update

**Step 2: Update test imports systematically**

For each test file, replace old imports with new structure:

```python
# Old → New mappings:
# from crawl4r.config → from crawl4r.core.config
# from crawl4r.logger → from crawl4r.core.logger
# from crawl4r.quality → from crawl4r.core.quality
# from crawl4r.crawl4ai_reader → from crawl4r.readers.crawl4ai
# from crawl4r.file_watcher → from crawl4r.readers.file_watcher
# from crawl4r.chunker → from crawl4r.processing.chunker
# from crawl4r.processor → from crawl4r.processing.processor
# from crawl4r.tei_client → from crawl4r.storage.embeddings
# from crawl4r.vector_store → from crawl4r.storage.vector_store
# from crawl4r.circuit_breaker → from crawl4r.resilience.circuit_breaker
# from crawl4r.failed_docs → from crawl4r.resilience.failed_docs
# from crawl4r.recovery → from crawl4r.resilience.recovery
```

**Step 3: Run sed to batch-update imports**

```bash
# Update all test files
find tests/ -name "*.py" -type f -exec sed -i 's/from rag_ingestion\.config/from crawl4r.core.config/g' {} +
find tests/ -name "*.py" -type f -exec sed -i 's/from rag_ingestion\.logger/from crawl4r.core.logger/g' {} +
find tests/ -name "*.py" -type f -exec sed -i 's/from rag_ingestion\.quality/from crawl4r.core.quality/g' {} +
find tests/ -name "*.py" -type f -exec sed -i 's/from rag_ingestion\.crawl4ai_reader/from crawl4r.readers.crawl4ai/g' {} +
find tests/ -name "*.py" -type f -exec sed -i 's/from rag_ingestion\.file_watcher/from crawl4r.readers.file_watcher/g' {} +
find tests/ -name "*.py" -type f -exec sed -i 's/from rag_ingestion\.chunker/from crawl4r.processing.chunker/g' {} +
find tests/ -name "*.py" -type f -exec sed -i 's/from rag_ingestion\.processor/from crawl4r.processing.processor/g' {} +
find tests/ -name "*.py" -type f -exec sed -i 's/from rag_ingestion\.tei_client/from crawl4r.storage.embeddings/g' {} +
find tests/ -name "*.py" -type f -exec sed -i 's/from rag_ingestion\.vector_store/from crawl4r.storage.vector_store/g' {} +
find tests/ -name "*.py" -type f -exec sed -i 's/from rag_ingestion\.circuit_breaker/from crawl4r.resilience.circuit_breaker/g' {} +
find tests/ -name "*.py" -type f -exec sed -i 's/from rag_ingestion\.failed_docs/from crawl4r.resilience.failed_docs/g' {} +
find tests/ -name "*.py" -type f -exec sed -i 's/from rag_ingestion\.recovery/from crawl4r.resilience.recovery/g' {} +
```

**Step 4: Verify no syntax errors in tests**

Run: `python -m py_compile tests/**/*.py`

Expected: No errors

**Step 5: Commit changes**

```bash
git add tests/
git commit -m "refactor: update test imports for new module structure"
```

---

## Task 13: Run Full Test Suite

**Files:**
- Verify: All tests pass with new structure

**Step 1: Run module structure tests**

Run: `pytest tests/unit/test_module_structure.py -v`

Expected: ALL PASS - new imports work, backward compatibility works

**Step 2: Run all unit tests**

Run: `pytest tests/unit/ -v`

Expected: ALL PASS

**Step 3: Run integration tests**

Run: `pytest tests/integration/ -v -m integration`

Expected: ALL PASS

**Step 4: Run full test suite**

Run: `pytest tests/ -v`

Expected: ALL PASS

**Step 5: Document success**

If all tests pass, the refactoring is complete and verified.

---

## Task 14: Update Documentation

**Files:**
- Modify: `CLAUDE.md`
- Create: `crawl4r/README.md`

**Step 1: Document new structure in crawl4r/README.md**

Create `crawl4r/README.md`:

```markdown
# Crawl4r - RAG Ingestion Pipeline

Python package for ingesting documents into a RAG (Retrieval-Augmented Generation) vector database.

## Module Structure

```
crawl4r/
├── core/              # Core infrastructure
│   ├── config.py      # Configuration management (Settings)
│   ├── logger.py      # Structured logging (get_logger)
│   └── quality.py     # Startup validation (QualityVerifier)
├── readers/           # Input sources
│   ├── crawl4ai.py    # Web crawling (Crawl4AIReader)
│   └── file_watcher.py # File monitoring (FileWatcher)
├── processing/        # Document processing
│   ├── chunker.py     # Text chunking (MarkdownChunker)
│   └── processor.py   # Pipeline orchestration (DocumentProcessor)
├── storage/           # Storage backends
│   ├── embeddings.py  # TEI client (TEIClient)
│   └── vector_store.py # Qdrant manager (VectorStoreManager)
├── resilience/        # Error handling & recovery
│   ├── circuit_breaker.py # Circuit breaker pattern (CircuitBreaker)
│   ├── failed_docs.py     # Failed document tracking
│   └── recovery.py        # State recovery (StateRecovery)
├── cli/               # Command-line interface
│   ├── main.py        # CLI entry point
│   └── commands/      # CLI commands (future)
└── api/               # REST API
    ├── app.py         # FastAPI application
    ├── routes/        # API route handlers
    │   └── health.py  # Health check endpoints
    └── models/        # Request/response models
        └── responses.py # Pydantic response models
```

## Usage

### Usage

```python
# Import from specific submodules
from crawl4r.core.config import Settings
from crawl4r.readers.crawl4ai import Crawl4AIReader
from crawl4r.processing.chunker import MarkdownChunker
from crawl4r.storage.embeddings import TEIClient
from crawl4r.storage.vector_store import VectorStoreManager
```

## Migration Guide

Update all imports to use new submodule structure:

```python
# Old flat imports
from rag_ingestion.crawl4ai_reader import Crawl4AIReader
from rag_ingestion.config import Settings
from rag_ingestion.tei_client import TEIClient

# New submodule imports
from crawl4r.readers.crawl4ai import Crawl4AIReader
from crawl4r.core.config import Settings
from crawl4r.storage.embeddings import TEIClient
```

**No backward compatibility layer** - all code must use new import paths.

## Design Principles

- **Core**: Infrastructure shared across all modules
- **Readers**: Input sources (web, files, streams)
- **Processing**: Transform documents into chunks
- **Storage**: Persist embeddings and vectors
- **Resilience**: Handle failures gracefully
- **CLI**: Command-line interface for pipeline operations
- **API**: REST API for programmatic access
```

**Step 2: Update CLAUDE.md with new structure**

Edit section in `CLAUDE.md` about Python implementation:

```markdown
### Project Structure (Implemented)

```
crawl4r/
├── crawl4r/              # Main package
│   ├── core/                   # Core infrastructure
│   │   ├── config.py           # Pydantic configuration
│   │   ├── logger.py           # Structured logging
│   │   └── quality.py          # Startup validation
│   ├── readers/                # Input sources
│   │   ├── crawl4ai.py         # LlamaIndex web crawling
│   │   └── file_watcher.py     # File monitoring
│   ├── processing/             # Document processing
│   │   ├── chunker.py          # Markdown-aware chunking
│   │   └── processor.py        # Pipeline orchestration
│   ├── storage/                # Storage backends
│   │   ├── embeddings.py       # TEI client wrapper
│   │   └── vector_store.py     # Qdrant manager
│   ├── resilience/             # Error handling
│   │   ├── circuit_breaker.py  # Circuit breaker pattern
│   │   ├── failed_docs.py      # Failed document tracking
│   │   └── recovery.py         # State recovery
│   ├── cli/                    # Command-line interface
│   │   ├── main.py             # CLI entry point
│   │   └── commands/           # CLI commands
│   └── api/                    # REST API
│       ├── app.py              # FastAPI application
│       ├── routes/             # API route handlers
│       └── models/             # Request/response models
├── tests/
│   ├── unit/                   # Fast, isolated tests
│   └── integration/            # Tests with real services
├── examples/                   # Usage examples
└── docs/                       # Documentation
```
```

**Step 3: Commit documentation**

```bash
git add crawl4r/README.md CLAUDE.md
git commit -m "docs: document new module structure"
```

---

## Task 15: Final Verification and Cleanup

**Files:**
- Verify: No old files remain at root level

**Step 1: Check for leftover files at root**

Run: `ls crawl4r/*.py 2>/dev/null | grep -v "__init__.py" || echo "OK: Only __init__.py remains"`

Expected: "OK: Only __init__.py remains" (main.py moved to cli/)

**Step 2: Verify git history preserved**

Run: `git log --follow crawl4r/readers/crawl4ai.py | head -5`

Expected: Shows history from original crawl4ai_reader.py

**Step 3: Run final full test suite**

Run: `pytest tests/ -v --tb=short`

Expected: ALL PASS

**Step 4: Run type checking**

Run: `ty check crawl4r/`

Expected: No type errors

**Step 5: Run linting**

Run: `ruff check crawl4r/`

Expected: No linting errors

**Step 6: Create final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: final cleanup after module reorganization"
```

---

## Verification Checklist

- [ ] All 7 submodule directories created (core, readers, processing, storage, resilience, cli, api)
- [ ] All 14 files moved with git mv (history preserved)
- [ ] main.py moved to cli/ submodule
- [ ] API skeleton structure created with FastAPI
- [ ] All internal imports updated to use new submodule paths
- [ ] All example imports updated
- [ ] All test imports updated
- [ ] Module structure tests pass (including CLI and API)
- [ ] Full test suite passes
- [ ] No linting errors
- [ ] No type errors
- [ ] Documentation updated
- [ ] No orphaned files at root level (except __init__.py)

---

## Rollback Plan

If issues arise during implementation:

```bash
# Reset to before reorganization
git log --oneline | grep "module reorganization"
git reset --hard <commit-before-reorganization>

# Or revert specific commits
git revert <commit-hash>
```

---

## Post-Implementation

After successful reorganization:

1. Update CI/CD if it references specific file paths
2. Notify team about new import structure
3. Add deprecation warnings for old imports (future work)
4. Consider adding import linting rules to enforce new structure
5. Implement CLI commands in `cli/commands/` (future work)
6. Build out API routes in `api/routes/` (future work)
7. Add CLI entry point to `pyproject.toml` (e.g., `crawl4r = "crawl4r.cli.main:main"`)
