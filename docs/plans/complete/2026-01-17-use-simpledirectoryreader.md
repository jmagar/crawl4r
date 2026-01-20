# SimpleDirectoryReader Swap Implementation Plan

> **Organization Note:** When this plan is fully implemented and verified, move this file to `docs/plans/complete/` to keep the plans folder organized.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace manual file I/O in `DocumentProcessor` with LlamaIndex `SimpleDirectoryReader` and use its default metadata.

**Architecture:** Use `SimpleDirectoryReader(input_files=[...])` to load documents (and metadata) per file, then set deterministic IDs before passing documents into the ingestion pipeline.

**Tech Stack:** Python, LlamaIndex (`SimpleDirectoryReader`, `Document`), pytest.

---

## Metadata Key Mapping Strategy

SimpleDirectoryReader provides these default metadata keys:
- `file_path` - Absolute path to the file (replaces our `file_path_absolute`)
- `file_name` - Base filename with extension
- `file_type` - MIME type (e.g., `text/markdown`)
- `file_size` - Size in bytes
- `creation_date` - File creation timestamp
- `last_modified_date` - Last modification timestamp

**Key mapping:**
| Old Key | New Key | Notes |
|---------|---------|-------|
| `file_path_relative` | Computed from `file_path` | Derive via `Path(file_path).relative_to(watch_folder)` |
| `file_path_absolute` | `file_path` | Direct 1:1 replacement |

**Important:** The `file_path_relative` concept is still needed for Qdrant point ID generation (deterministic hashing) and user-facing metadata. Compute it from `file_path` at processing time.

---

### Task 0: Create MetadataKeys constants module

**Rationale:** Centralizing metadata key definitions prevents 200+ hardcoded string occurrences and makes future refactoring safer. This must be done BEFORE changing any existing code.

**Files:**
- Create: `crawl4r/core/metadata.py`
- Modify: `tests/unit/test_metadata.py` (new)

**Step 1: Write the failing test**

```python
# tests/unit/test_metadata.py
from crawl4r.core.metadata import MetadataKeys


def test_metadata_keys_has_file_path():
    """Verify MetadataKeys defines FILE_PATH constant."""
    assert hasattr(MetadataKeys, "FILE_PATH")
    assert MetadataKeys.FILE_PATH == "file_path"


def test_metadata_keys_has_file_name():
    """Verify MetadataKeys defines FILE_NAME constant."""
    assert hasattr(MetadataKeys, "FILE_NAME")
    assert MetadataKeys.FILE_NAME == "file_name"


def test_metadata_keys_has_chunk_index():
    """Verify MetadataKeys defines CHUNK_INDEX constant."""
    assert hasattr(MetadataKeys, "CHUNK_INDEX")
    assert MetadataKeys.CHUNK_INDEX == "chunk_index"


def test_metadata_keys_all_values_are_strings():
    """Verify all MetadataKeys values are strings."""
    for attr in dir(MetadataKeys):
        if not attr.startswith("_"):
            value = getattr(MetadataKeys, attr)
            assert isinstance(value, str), f"{attr} should be a string"
```

**Step 2: Run test to verify it fails (RED)**

Run: `pytest tests/unit/test_metadata.py -v`

Expected error:
```
ModuleNotFoundError: No module named 'crawl4r.core.metadata'
```

This fails because the metadata module doesn't exist yet.

**Step 3: Write minimal implementation**

Create `crawl4r/core/metadata.py`:

```python
"""Centralized metadata key definitions for the crawl4r pipeline.

This module provides a single source of truth for all metadata keys used
across document processing, storage, and retrieval. Using these constants
instead of hardcoded strings enables safe refactoring and IDE support.

Usage:
    from crawl4r.core.metadata import MetadataKeys

    doc.metadata[MetadataKeys.FILE_PATH]  # Instead of doc.metadata["file_path"]
"""


class MetadataKeys:
    """Constants for document metadata keys.

    These keys align with LlamaIndex SimpleDirectoryReader defaults where applicable.
    Custom keys (CHUNK_*) are crawl4r-specific additions.
    """

    # SimpleDirectoryReader defaults
    FILE_PATH = "file_path"  # Absolute path from SimpleDirectoryReader
    FILE_NAME = "file_name"  # Base filename with extension
    FILE_TYPE = "file_type"  # MIME type (e.g., "text/markdown")
    FILE_SIZE = "file_size"  # Size in bytes
    CREATION_DATE = "creation_date"  # File creation timestamp
    LAST_MODIFIED_DATE = "last_modified_date"  # Last modification timestamp

    # Crawl4r chunking metadata
    CHUNK_INDEX = "chunk_index"  # Position of chunk in document
    CHUNK_TEXT = "chunk_text"  # Raw text content of chunk
    SECTION_PATH = "section_path"  # Heading hierarchy (e.g., "Guide > Install")
    TOTAL_CHUNKS = "total_chunks"  # Total chunks in document

    # Web crawl metadata (from Crawl4AIReader)
    SOURCE_URL = "source_url"  # Original URL
    SOURCE_TYPE = "source_type"  # "web_crawl" or "local_file"
    TITLE = "title"  # Page/document title
    DESCRIPTION = "description"  # Page description
    STATUS_CODE = "status_code"  # HTTP status code
    CRAWL_TIMESTAMP = "crawl_timestamp"  # When crawled

    # Legacy keys (for migration compatibility - DEPRECATED)
    # These will be removed after Task 4 completes
    FILE_PATH_RELATIVE = "file_path_relative"  # DEPRECATED: Use FILE_PATH
    FILE_PATH_ABSOLUTE = "file_path_absolute"  # DEPRECATED: Use FILE_PATH
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_metadata.py -v`
Expected: PASS

**Step 5: Verify no regressions**

Run: `pytest tests/unit/ -v --tb=short`
Expected: All existing tests still pass (module is additive only)

**Step 6: Commit**

```bash
git add crawl4r/core/metadata.py tests/unit/test_metadata.py
git commit -m "feat: add MetadataKeys constants module for centralized metadata definitions"
```

---

### Task 0.5: Make point ID generation path-agnostic

**Rationale:** Point ID generation currently depends on `file_path_relative`. Before introducing SimpleDirectoryReader (which provides absolute paths), we must update `_generate_point_id()` to accept BOTH absolute and relative paths and always produce the same deterministic ID. This prevents breaking point IDs mid-migration.

**Files:**
- Modify: `crawl4r/storage/qdrant.py`
- Modify: `tests/unit/test_qdrant.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_qdrant.py
def test_generate_point_id_accepts_absolute_path(vector_store, tmp_path):
    """Verify _generate_point_id produces same ID from absolute or relative path."""
    watch_folder = tmp_path / "docs"
    watch_folder.mkdir()

    rel_path = "guide/install.md"
    abs_path = str(watch_folder / rel_path)
    chunk_index = 0

    # ID from relative path (current behavior)
    id_from_relative = vector_store._generate_point_id(rel_path, chunk_index)

    # ID from absolute path (new behavior) - should produce SAME ID
    id_from_absolute = vector_store._generate_point_id(abs_path, chunk_index, watch_folder=watch_folder)

    assert id_from_relative == id_from_absolute, "Point IDs must be stable regardless of path format"


def test_generate_point_id_fallback_when_not_under_watch_folder(vector_store, tmp_path):
    """Verify _generate_point_id handles paths outside watch_folder gracefully."""
    watch_folder = tmp_path / "docs"
    watch_folder.mkdir()

    # Path NOT under watch_folder
    external_path = "/some/other/location/file.md"
    chunk_index = 0

    # Should not raise, should use full path as fallback
    point_id = vector_store._generate_point_id(external_path, chunk_index, watch_folder=watch_folder)
    assert point_id is not None
```

**Step 2: Run test to verify it fails (RED)**

Run: `pytest tests/unit/test_qdrant.py::test_generate_point_id_accepts_absolute_path -v`

Expected error:
```
TypeError: _generate_point_id() got an unexpected keyword argument 'watch_folder'
```

This fails because the current signature is `_generate_point_id(self, file_path_relative: str, chunk_index: int)`.

**Step 3: Write minimal implementation**

Update `crawl4r/storage/qdrant.py`:

**First, add Path import at top of file (REQUIRED - qdrant.py does not currently import Path):**
```python
from pathlib import Path
```

**Then update the method:**
```python
def _generate_point_id(
    self,
    file_path: str,
    chunk_index: int,
    watch_folder: Path | None = None
) -> str:
    """Generate deterministic UUID from file path and chunk index.

    Accepts both absolute and relative paths. Always generates ID from
    relative path for consistency (if watch_folder provided).

    Args:
        file_path: Absolute or relative file path
        chunk_index: Position of chunk in document
        watch_folder: Base folder to compute relative path from (optional)

    Returns:
        Deterministic UUID string
    """
    path_obj = Path(file_path)

    # Compute relative path if absolute path provided
    if path_obj.is_absolute() and watch_folder is not None:
        try:
            rel_path = str(path_obj.relative_to(watch_folder))
        except ValueError:
            # Path not under watch_folder - use full path as fallback
            rel_path = file_path
    else:
        rel_path = file_path

    content = f"{rel_path}:{chunk_index}"
    hash_bytes = hashlib.sha256(content.encode()).digest()
    return str(uuid.UUID(bytes=hash_bytes[:16]))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_qdrant.py::test_generate_point_id_accepts_absolute_path tests/unit/test_qdrant.py::test_generate_point_id_fallback_when_not_under_watch_folder -v`
Expected: PASS

**Step 5: Verify no regressions**

Run: `pytest tests/unit/test_qdrant.py -v`
Expected: All existing tests still pass (backward compatible - watch_folder is optional)

**Step 5.5: Audit all `_generate_point_id` callsites (CRITICAL)**

Before committing, verify ALL existing callers are compatible with the new signature:

```bash
# Find all callers of _generate_point_id
rg "_generate_point_id" --type py -n
```

Expected callsites and compatibility status:
- `crawl4r/storage/qdrant.py` - Internal method, will be updated in Task 4.1
- `tests/unit/test_qdrant.py` - Tests, will be updated in Task 4.2

**Verification:** All callers must work WITHOUT `watch_folder` parameter (backward compatible).

If ANY caller would break, update the implementation in Step 3 to ensure backward compatibility.

**Step 6: Commit**

```bash
git add crawl4r/storage/qdrant.py tests/unit/test_qdrant.py
git commit -m "refactor: make point ID generation path-agnostic for SimpleDirectoryReader migration"
```

---

### Task 0.6: Verify point ID stability before proceeding (CRITICAL GATE)

**Rationale:** This is a CRITICAL verification gate. Before introducing SimpleDirectoryReader (which changes metadata keys), we MUST verify that point ID generation remains stable. If this task fails, DO NOT proceed to Task 1 - fix Task 0.5 first.

**Files:**
- Create: `tests/unit/test_point_id_stability_gate.py`

**Step 1: Write the stability verification test**

```python
# tests/unit/test_point_id_stability_gate.py
"""CRITICAL GATE: Verify point IDs remain stable after Task 0.5 changes.

This test MUST pass before proceeding to Task 1 (SimpleDirectoryReader integration).
If this test fails, point ID generation is broken and the migration will corrupt data.
"""
import pytest
from pathlib import Path


class TestPointIdStabilityGate:
    """Gate tests that MUST pass before Task 1."""

    def test_point_id_same_for_relative_and_absolute_paths(self, vector_store, tmp_path):
        """Point IDs must be identical whether computed from relative or absolute path."""
        watch_folder = tmp_path / "docs"
        watch_folder.mkdir()

        rel_path = "guide/install.md"
        abs_path = str(watch_folder / rel_path)
        chunk_index = 0

        # ID from relative path (OLD behavior - current production)
        id_from_relative = vector_store._generate_point_id(rel_path, chunk_index)

        # ID from absolute path (NEW behavior - with watch_folder)
        id_from_absolute = vector_store._generate_point_id(
            abs_path, chunk_index, watch_folder=watch_folder
        )

        assert id_from_relative == id_from_absolute, (
            f"CRITICAL GATE FAILURE: Point ID changed!\n"
            f"  Relative path ID: {id_from_relative}\n"
            f"  Absolute path ID: {id_from_absolute}\n"
            f"  DO NOT PROCEED TO TASK 1 - fix Task 0.5 first."
        )

    def test_backward_compatibility_without_watch_folder(self, vector_store):
        """Existing code without watch_folder must still work."""
        rel_path = "docs/readme.md"
        chunk_index = 0

        # This is how existing code calls the method (no watch_folder)
        point_id = vector_store._generate_point_id(rel_path, chunk_index)

        assert point_id is not None, "Must work without watch_folder parameter"
        assert len(point_id) == 36, "Must return valid UUID format"

    def test_point_id_deterministic(self, vector_store):
        """Same inputs must always produce same ID."""
        file_path = "docs/api.md"
        chunk_index = 3

        id1 = vector_store._generate_point_id(file_path, chunk_index)
        id2 = vector_store._generate_point_id(file_path, chunk_index)

        assert id1 == id2, "Point IDs must be deterministic"
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/unit/test_point_id_stability_gate.py -v`

Expected: ALL PASS - if any fail, DO NOT proceed to Task 1

**CRITICAL DECISION POINT:**
- ✅ If all tests pass → Proceed to Task 1
- 🔴 If any test fails → Go back to Task 0.5 and fix the implementation

**Step 3: Commit**

```bash
git add tests/unit/test_point_id_stability_gate.py
git commit -m "test: add critical point ID stability gate before SimpleDirectoryReader migration"
```

---

### Task 1: Add failing unit test for SimpleDirectoryReader usage

**Files:**
- Modify: `tests/unit/test_processor.py`

**Step 1: Write the failing test**

```python
from unittest.mock import patch
from crawl4r.processing.processor import DocumentProcessor


def test_processor_uses_simpledirectoryreader(config, vector_store, tei_client, tmp_path):
    file_path = tmp_path / "doc.md"
    file_path.write_text("# Title\n\nBody", encoding="utf-8")

    with patch("crawl4r.processing.processor.SimpleDirectoryReader") as reader_cls:
        processor = DocumentProcessor(
            config=config,
            vector_store=vector_store,
            tei_client=tei_client,
        )
        processor.process_document(file_path)
        reader_cls.assert_called_once()
```

**Step 2: Run test to verify it fails (RED)**

Run: `pytest tests/unit/test_processor.py::test_processor_uses_simpledirectoryreader -v`

Expected error:
```
AssertionError: Expected 'SimpleDirectoryReader' to be called once. Called 0 times.
```

This fails because `process_document()` currently uses `_load_markdown_file()` instead of `SimpleDirectoryReader`.

**Step 3: Write minimal implementation**

Update `crawl4r/processing/processor.py`:

**Location:** In `process_document()` method, replace lines 391-410 (the `_load_markdown_file` call and manual metadata extraction):

```python
# BEFORE (lines 391-410):
# content = await self._load_markdown_file(file_path)
# stat = file_path.stat()
# modification_date = datetime.fromtimestamp(stat.st_mtime).isoformat()
# filename = file_path.name
# ... (file_path_relative, file_path_absolute calculation)

# AFTER: Replace with SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader

# Load document via SimpleDirectoryReader (provides default metadata)
reader = SimpleDirectoryReader(input_files=[str(file_path)])
docs = reader.load_data()
if not docs:
    raise FileNotFoundError(f"SimpleDirectoryReader returned no documents for: {file_path}")
doc = docs[0]
content = doc.text

# SimpleDirectoryReader provides: file_path, file_name, file_type, file_size,
# creation_date, last_modified_date - all in doc.metadata

# Derive relative path for ID generation (preserves existing behavior)
abs_path = doc.metadata["file_path"]
try:
    rel_path = str(Path(abs_path).relative_to(self.config.watch_folder))
except ValueError:
    rel_path = abs_path  # Fallback if not under watch_folder

# Generate deterministic ID from relative path
doc.id_ = self._generate_document_id(rel_path)
```

**Also add import at top of file:**
```python
from llama_index.core import SimpleDirectoryReader
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_processor.py::test_processor_uses_simpledirectoryreader -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crawl4r/processing/processor.py tests/unit/test_processor.py
git commit -m "refactor: load documents via SimpleDirectoryReader"
```

---

### Task 2: Update metadata expectations to match SimpleDirectoryReader defaults

**Rationale:** After Task 1 introduces SimpleDirectoryReader, document metadata changes from custom keys (`file_path_relative`, `file_path_absolute`) to SimpleDirectoryReader defaults (`file_path`, `file_name`). Tests must be updated to expect the new keys.

**Files:**
- Modify: `tests/unit/test_processor.py`
- Modify: `tests/unit/test_processor_id_generation.py`

**Step 1: Write the failing test (NEW assertions only)**

Add NEW assertions that WILL fail with current implementation. Do NOT remove old assertions yet - that happens after GREEN.

```python
# tests/unit/test_processor.py
def test_process_document_uses_simpledirectoryreader_metadata(config, vector_store, tei_client, tmp_path):
    """Verify processed documents have SimpleDirectoryReader metadata keys."""
    file_path = tmp_path / "test.md"
    file_path.write_text("# Test\n\nBody content", encoding="utf-8")

    processor = DocumentProcessor(
        config=config,
        vector_store=vector_store,
        tei_client=tei_client,
    )

    # Process the document
    result = processor.process_document(file_path)

    # NEW assertions that WILL fail before implementation
    assert "file_path" in result.metadata, "Missing 'file_path' from SimpleDirectoryReader"
    assert "file_name" in result.metadata, "Missing 'file_name' from SimpleDirectoryReader"
    assert result.metadata["file_path"].endswith("test.md"), "file_path should be absolute path"
    assert result.metadata["file_name"] == "test.md", "file_name should be base filename"
```

**Step 2: Run test to verify it fails (RED)**

Run: `pytest tests/unit/test_processor.py::test_process_document_uses_simpledirectoryreader_metadata -v`

Expected error:
```
AssertionError: Missing 'file_path' from SimpleDirectoryReader
```

This fails because the current implementation uses `file_path_relative`/`file_path_absolute`, not SimpleDirectoryReader's `file_path`/`file_name` keys.

**Step 3: Write minimal implementation**

Update `crawl4r/processing/processor.py` to:
- Use metadata from SimpleDirectoryReader (already done in Task 1)
- Remove manual `file_path_relative` and `file_path_absolute` assignment
- Preserve deterministic ID assignment using relative path derived from `doc.metadata["file_path"]`:

```python
# In process_document method, after SimpleDirectoryReader loads the document:
abs_path = doc.metadata["file_path"]  # From SimpleDirectoryReader
rel_path = str(Path(abs_path).relative_to(self.config.watch_folder))

# Generate deterministic ID from relative path (preserves existing behavior)
doc.id_ = self._generate_document_id(rel_path)
```

**Step 4: Run test to verify it passes (GREEN)**

Run: `pytest tests/unit/test_processor.py::test_process_document_uses_simpledirectoryreader_metadata -v`
Expected: PASS

**Step 5: Add assertions verifying old keys are removed (REFACTOR)**

Now that implementation is GREEN, add assertions verifying old keys are gone:

```python
# Add to the same test function after the new key assertions:
    # OLD keys must NOT be present (verify migration complete)
    assert "file_path_relative" not in result.metadata, \
        "Old key 'file_path_relative' still present - migration incomplete"
    assert "file_path_absolute" not in result.metadata, \
        "Old key 'file_path_absolute' still present - migration incomplete"
```

Run: `pytest tests/unit/test_processor.py::test_process_document_uses_simpledirectoryreader_metadata -v`
Expected: PASS (old keys should already be removed by Step 3 implementation)

**Step 6: Update test_processor_id_generation.py**

Ensure ID generation tests use the new metadata format:

```python
# tests/unit/test_processor_id_generation.py
def test_document_id_stable_with_file_path_metadata(processor, tmp_path, watch_folder):
    """Verify document IDs are stable when using file_path metadata."""
    file_path = watch_folder / "docs/guide.md"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("# Guide", encoding="utf-8")

    # Process same file twice
    id1 = processor.process_document(file_path).id_
    id2 = processor.process_document(file_path).id_

    assert id1 == id2, "Document IDs must be deterministic"
```

**Step 7: Run all processor tests**

Run: `pytest tests/unit/test_processor.py tests/unit/test_processor_id_generation.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add tests/unit/test_processor.py tests/unit/test_processor_id_generation.py
git commit -m "test: align metadata expectations with SimpleDirectoryReader"
```

---

### Task 3: Update CLI and examples to avoid manual file loading

**Rationale:** The CLI must delegate file loading to DocumentProcessor (which now uses SimpleDirectoryReader). This ensures consistent metadata handling across the codebase.

**Files:**
- Modify: `crawl4r/cli/main.py`
- Create: `tests/unit/test_cli.py` (if not exists)
- Modify: `examples/*` (if exists)

**Step 1: Write the failing test (source code inspection for proper RED verification)**

```python
# tests/unit/test_cli.py
"""Tests verifying CLI delegates file loading to DocumentProcessor.

These tests use SOURCE CODE INSPECTION to ensure proper TDD RED verification.
Mock-based tests would pass prematurely before implementation changes.
"""
from pathlib import Path


def test_cli_does_not_manually_read_files():
    """Verify CLI source code doesn't manually read file contents.

    This test will FAIL if CLI contains:
    - file.read_text() or path.read_text()
    - open(file) patterns for reading content
    - Any direct file I/O that should be delegated to DocumentProcessor

    Expected RED state: If CLI currently reads files directly, this fails.
    Expected GREEN state: After delegating to DocumentProcessor, this passes.
    """
    cli_source = Path("crawl4r/cli/main.py").read_text()

    # Check for direct file reading patterns (not path construction)
    # These patterns indicate CLI is manually loading file content
    forbidden_patterns = [
        ".read_text()",      # Direct text reading
        ".read_bytes()",     # Direct binary reading
        "open(file",         # Manual file opening
        "with open(",        # Context manager file opening for reading
    ]

    for pattern in forbidden_patterns:
        assert pattern not in cli_source, (
            f"CLI should NOT use '{pattern}' - delegate file loading to DocumentProcessor.\n"
            f"RED state detected: CLI manually reads files instead of delegating."
        )


def test_cli_does_not_use_load_markdown_file():
    """Verify CLI source code doesn't call _load_markdown_file directly.

    Expected RED state: If CLI calls _load_markdown_file, this fails.
    Expected GREEN state: After removing _load_markdown_file calls, this passes.
    """
    cli_source = Path("crawl4r/cli/main.py").read_text()

    # CLI should NOT directly call the internal loader
    assert "_load_markdown_file" not in cli_source, (
        "CLI should delegate file loading to DocumentProcessor, not call _load_markdown_file.\n"
        "RED state detected: CLI uses internal loader instead of DocumentProcessor."
    )


def test_cli_uses_document_processor():
    """Verify CLI imports and uses DocumentProcessor for file processing.

    Expected RED state: If CLI doesn't import DocumentProcessor, this fails.
    Expected GREEN state: After adding DocumentProcessor usage, this passes.
    """
    cli_source = Path("crawl4r/cli/main.py").read_text()

    # CLI SHOULD use DocumentProcessor
    assert "DocumentProcessor" in cli_source, (
        "CLI should use DocumentProcessor for file processing.\n"
        "RED state detected: CLI doesn't reference DocumentProcessor."
    )
```

**Step 2: Run test to verify it fails (RED)**

Run: `pytest tests/unit/test_cli.py -v`

Expected error:
```
AssertionError: CLI should use DocumentProcessor for file processing.
RED state detected: CLI doesn't reference DocumentProcessor.
```

**Note:** This task is CONDITIONAL. If CLI already uses DocumentProcessor and doesn't have forbidden patterns, all 3 tests will pass. In that case:
- Mark this task as PASS (no changes needed)
- Skip to Step 5 for verification
- Document in commit: "verify: CLI already delegates to DocumentProcessor"

**Step 3: Write minimal implementation**

Update `crawl4r/cli/main.py` to ensure file loading is fully delegated:

- Remove any direct calls to `_load_markdown_file` or `file.read_text()`
- Pass file paths to `DocumentProcessor.process_document(path)`
- Let DocumentProcessor handle SimpleDirectoryReader internally

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_cli.py -v`
Expected: PASS

**Step 5: Verify examples (non-TDD - verification only)**

Run: `rg -n "_load_markdown_file" examples crawl4r/cli/main.py`
Expected: No remaining references to manual file loading.

Check if `examples/` directory exists:

```bash
ls -la examples/ 2>/dev/null || echo "No examples directory - skip this step"
```

If examples exist and manually load files, update them to use the public API.

**Step 6: Run full unit test suite**

Run: `pytest tests/unit/ -v --tb=short`
Expected: All tests pass

**Step 7: Commit**

```bash
git add crawl4r/cli/main.py tests/unit/test_cli.py
# Only add examples if directory exists and was modified:
# git add examples
git commit -m "refactor: CLI delegates file loading to DocumentProcessor"
```

---

### Task 4: Replace all file_path_relative references with MetadataKeys constants

**Rationale:** This is the largest refactoring task. After Task 0 establishes `MetadataKeys` constants, this task systematically replaces all hardcoded `"file_path_relative"` and `"file_path_absolute"` strings across the codebase.

**Scope:** 129 occurrences of `file_path_relative` and 12 occurrences of `file_path_absolute` across 12 Python files.

**Verified counts (as of plan creation):**

**Source code (4 files, 38 occurrences):**
- `crawl4r/storage/qdrant.py` (24 `file_path_relative`, 3 `file_path_absolute`)
- `crawl4r/processing/processor.py` (7 `file_path_relative`, 2 `file_path_absolute`)
- `crawl4r/resilience/recovery.py` (1 `file_path_relative`)
- `crawl4r/resilience/failed_docs.py` (6 `file_path_relative`)

**Unit tests (4 files, 72 occurrences):**
- `tests/unit/test_qdrant.py` (50 `file_path_relative`, 3 `file_path_absolute`)
- `tests/unit/test_processor.py` (1 `file_path_relative`, 1 `file_path_absolute`)
- `tests/unit/test_recovery.py` (16 `file_path_relative`)
- `tests/unit/test_failed_docs.py` (5 `file_path_relative`)

**Integration tests (4 files, 19 occurrences):**
- `tests/integration/test_e2e_pipeline.py` (7 `file_path_relative`, 1 `file_path_absolute`)
- `tests/integration/test_e2e_core.py` (8 `file_path_relative`, 1 `file_path_absolute`)
- `tests/integration/test_e2e_crawl_pipeline.py` (3 `file_path_relative`, 1 `file_path_absolute`)
- `tests/integration/test_e2e_reader_pipeline.py` (1 `file_path_relative`)

**Documentation (8+ files):**
- `CLAUDE.md` (~2 `file_path_relative`, ~1 `file_path_absolute`)
- `specs/rag-ingestion/design.md` (~19 `file_path_relative`, ~2 `file_path_absolute`)
- `specs/rag-ingestion/requirements.md` (~7 `file_path_relative`, ~2 `file_path_absolute`)
- `specs/rag-ingestion/tasks.md` (~15 `file_path_relative`, ~2 `file_path_absolute`)
- `specs/rag-ingestion/decisions.md` (~4 `file_path_relative`, ~2 `file_path_absolute`)
- `specs/rag-ingestion/technical-review.md` (~9 `file_path_relative`, ~2 `file_path_absolute`)
- `specs/llamaindex-crawl4ai-reader/*.md` (various files)

---

#### Task 4.1: Update unit tests to use MetadataKeys (TESTS FIRST - TDD)

**Rationale:** TDD requires updating test expectations BEFORE changing source code. Update tests to expect `MetadataKeys.FILE_PATH` instead of `"file_path_relative"`. Tests will initially PASS (since source still uses old keys), then we change source in Task 4.2 and verify tests still pass.

**Step 1: Update test files to import and use MetadataKeys**

For each test file, replace hardcoded strings with MetadataKeys constants:

```python
# tests/unit/test_qdrant.py
from crawl4r.core.metadata import MetadataKeys

# Before
metadata = {"file_path_relative": "docs/test.md"}

# After
metadata = {MetadataKeys.FILE_PATH: "/absolute/path/docs/test.md"}
```

**Files to update:**
- `tests/unit/test_qdrant.py` (53 replacements)
- `tests/unit/test_processor.py` (2 replacements)
- `tests/unit/test_recovery.py` (16 replacements)
- `tests/unit/test_failed_docs.py` (5 replacements)

**Step 2: Run unit tests to verify they FAIL (RED)**

Run: `pytest tests/unit/test_qdrant.py tests/unit/test_processor.py -v 2>&1 | head -50`

Expected error:
```
KeyError: 'file_path'
```

Tests now expect `MetadataKeys.FILE_PATH` but source code still uses `"file_path_relative"`.

**Step 3: Commit test changes (RED state committed)**

```bash
git add tests/unit/test_qdrant.py tests/unit/test_processor.py \
        tests/unit/test_recovery.py tests/unit/test_failed_docs.py
git commit -m "test: unit tests expect MetadataKeys constants (RED - source not updated yet)"
```

---

#### Task 4.2: Update source code to use MetadataKeys (GREEN)

**Step 1: Write failing tests for MetadataKeys imports**

Add tests verifying source files import `MetadataKeys`:

```python
# tests/unit/test_metadata_usage.py
"""Verify source code uses MetadataKeys constants instead of hardcoded strings."""
import subprocess
from pathlib import Path


def test_qdrant_imports_metadata_keys():
    """Verify qdrant.py imports MetadataKeys."""
    source = Path("crawl4r/storage/qdrant.py").read_text()
    assert "from crawl4r.core.metadata import MetadataKeys" in source, (
        "qdrant.py must import MetadataKeys for centralized metadata key definitions"
    )


def test_qdrant_no_hardcoded_file_path_relative():
    """Verify qdrant.py doesn't use hardcoded 'file_path_relative' strings."""
    result = subprocess.run(
        ["rg", '"file_path_relative"', "crawl4r/storage/qdrant.py", "--count"],
        capture_output=True,
        text=True,
    )
    match_count = int(result.stdout.strip()) if result.returncode == 0 else 0
    assert match_count == 0, (
        f"Found {match_count} hardcoded 'file_path_relative' strings in qdrant.py.\n"
        f"Replace with MetadataKeys.FILE_PATH or MetadataKeys.FILE_PATH_RELATIVE."
    )


def test_processor_imports_metadata_keys():
    """Verify processor.py imports MetadataKeys."""
    source = Path("crawl4r/processing/processor.py").read_text()
    assert "from crawl4r.core.metadata import MetadataKeys" in source, (
        "processor.py must import MetadataKeys for centralized metadata key definitions"
    )


def test_processor_no_hardcoded_file_path_relative():
    """Verify processor.py doesn't use hardcoded 'file_path_relative' strings."""
    result = subprocess.run(
        ["rg", '"file_path_relative"', "crawl4r/processing/processor.py", "--count"],
        capture_output=True,
        text=True,
    )
    match_count = int(result.stdout.strip()) if result.returncode == 0 else 0
    assert match_count == 0, (
        f"Found {match_count} hardcoded 'file_path_relative' strings in processor.py.\n"
        f"Replace with MetadataKeys.FILE_PATH."
    )
```

**Step 2: Run test to verify it fails (RED verification)**

Run: `pytest tests/unit/test_metadata_usage.py -v`

Expected error:
```
AssertionError: qdrant.py must import MetadataKeys for centralized metadata key definitions
```

**Step 3: Update crawl4r/storage/qdrant.py**

Replace all hardcoded metadata key strings:

```python
# Before
metadata = {"file_path_relative": str(rel_path), "file_path_absolute": str(abs_path)}

# After
from crawl4r.core.metadata import MetadataKeys

metadata = {
    MetadataKeys.FILE_PATH: str(abs_path),  # SimpleDirectoryReader format
    # Compute relative path for ID generation when needed:
    # rel_path = Path(abs_path).relative_to(watch_folder)
}
```

**Key changes in qdrant.py:**
- Import `MetadataKeys`
- Replace `"file_path_relative"` with `MetadataKeys.FILE_PATH` (using absolute path)
- Replace `"file_path_absolute"` with `MetadataKeys.FILE_PATH`
- Update filter queries to use `MetadataKeys.FILE_PATH`
- Add helper to compute relative paths when needed for ID generation

**Step 4: Update crawl4r/processing/processor.py**

```python
from crawl4r.core.metadata import MetadataKeys

# Replace hardcoded strings with constants
doc.metadata[MetadataKeys.FILE_PATH] = str(file_path)
doc.metadata[MetadataKeys.FILE_NAME] = file_path.name
doc.metadata[MetadataKeys.CHUNK_INDEX] = chunk_idx
```

**Step 5: Update crawl4r/resilience/recovery.py and failed_docs.py**

Apply same pattern - import MetadataKeys and replace strings.

**Step 6: Run ALL tests to verify pass (GREEN)**

Run: `pytest tests/unit/test_metadata_usage.py tests/unit/test_qdrant.py tests/unit/test_processor.py tests/unit/test_recovery.py tests/unit/test_failed_docs.py -v`
Expected: PASS (both usage tests AND unit tests now pass)

**Step 7: Commit**

```bash
git add crawl4r/storage/qdrant.py crawl4r/processing/processor.py \
        crawl4r/resilience/recovery.py crawl4r/resilience/failed_docs.py \
        tests/unit/test_metadata_usage.py
git commit -m "refactor: source code uses MetadataKeys constants"
```

---

#### Task 4.3: Update integration tests

**Step 0: Run tests to verify RED state (tests should FAIL)**

Before updating integration test files, verify the expected RED state:

Run: `pytest tests/integration/test_e2e_pipeline.py tests/integration/test_e2e_core.py -v -m integration 2>&1 | head -50`

Expected failure modes (at least one should occur):
- `KeyError: 'file_path_relative'` - Source code uses new keys but tests expect old keys
- `AssertionError` - Metadata assertions fail due to key mismatch

If tests PASS unexpectedly: Integration tests may already use MetadataKeys → verify and skip to Step 2.

**Step 1: Update integration test files**

Apply same MetadataKeys pattern to integration tests:

- `tests/integration/test_e2e_pipeline.py`
- `tests/integration/test_e2e_core.py`
- `tests/integration/test_e2e_crawl_pipeline.py`
- `tests/integration/test_e2e_reader_pipeline.py`

**Edge case test coverage to add:**

```python
def test_metadata_key_consistency():
    """Verify MetadataKeys.FILE_PATH matches SimpleDirectoryReader output."""
    from crawl4r.core.metadata import MetadataKeys
    from llama_index.core import SimpleDirectoryReader

    # Create temp file
    docs = SimpleDirectoryReader(input_files=[str(temp_file)]).load_data()
    assert MetadataKeys.FILE_PATH in docs[0].metadata


def test_relative_path_computation():
    """Verify relative paths can be derived from FILE_PATH for ID generation."""
    from pathlib import Path
    from crawl4r.core.metadata import MetadataKeys

    abs_path = "/home/user/docs/guide/install.md"
    watch_folder = "/home/user/docs"

    # Compute relative path from absolute
    rel_path = str(Path(abs_path).relative_to(watch_folder))
    assert rel_path == "guide/install.md"
```

**Step 2: Run integration tests**

Run: `pytest tests/integration/ -v -m integration`
Expected: PASS (or skip if services not running)

**Step 3: Commit**

```bash
git add tests/integration/
git commit -m "test: integration tests use MetadataKeys constants"
```

---

#### Task 4.4: Update documentation and specs

**Note:** This is a documentation-only task (non-TDD). No failing test required as documentation updates don't have executable tests.

**Step 1: Update CLAUDE.md**

Replace references to `file_path_relative` and `file_path_absolute` with `file_path`:

```markdown
### File Path Metadata

Store `file_path` (absolute path from SimpleDirectoryReader) in document metadata.
Compute relative paths on-demand when needed for:
- Deterministic point ID generation
- User-facing display
```

**Step 2: Update specs/rag-ingestion/*.md files**

For each spec file, update metadata key references to reflect new convention:

- `design.md` - Update metadata schema documentation
- `requirements.md` - Update acceptance criteria
- `tasks.md` - Update task descriptions
- `decisions.md` - Add ADR for metadata key change
- `technical-review.md` - Update review notes

**Step 3: Verify no hardcoded strings remain**

Run comprehensive grep to verify migration complete:

```bash
# Should return 0 matches in source and test files
rg -c "file_path_relative" crawl4r/ tests/ --type py
rg -c "file_path_absolute" crawl4r/ tests/ --type py

# Docs may still have references (acceptable for historical context)
rg -c "file_path_relative" specs/ docs/ CLAUDE.md
```

**Step 4: Commit**

```bash
git add CLAUDE.md specs/
git commit -m "docs: update specs to reflect MetadataKeys migration"
```

---

#### Task 4.6: Verify point ID stability after migration

**Rationale:** This is a CRITICAL verification task. Point IDs must remain identical before and after the metadata key migration to prevent duplicate chunks in Qdrant. If IDs change, deduplication breaks silently.

**Files:**
- Create: `tests/integration/test_point_id_stability.py`

**Step 1: Write the stability verification test**

```python
# tests/integration/test_point_id_stability.py
"""Critical: Verify point IDs remain stable after metadata key migration.

This test ensures that documents indexed BEFORE the metadata migration
produce the SAME point IDs as documents indexed AFTER the migration.
If this test fails, deduplication is broken and Qdrant will accumulate duplicates.
"""
import pytest
from pathlib import Path
from crawl4r.storage.qdrant import VectorStoreManager
from crawl4r.core.metadata import MetadataKeys


@pytest.fixture
def watch_folder(tmp_path):
    """Create a temporary watch folder."""
    folder = tmp_path / "docs"
    folder.mkdir()
    return folder


@pytest.mark.integration
class TestPointIdStability:
    """Verify point ID generation remains stable across metadata key changes."""

    def test_point_id_same_for_relative_and_absolute_paths(self, vector_store, watch_folder):
        """Point IDs must be identical whether computed from relative or absolute path."""
        rel_path = "guide/install.md"
        abs_path = str(watch_folder / rel_path)
        chunk_index = 0

        # Simulate OLD behavior (relative path directly)
        old_id = vector_store._generate_point_id(rel_path, chunk_index)

        # Simulate NEW behavior (absolute path + watch_folder)
        new_id = vector_store._generate_point_id(abs_path, chunk_index, watch_folder=watch_folder)

        assert old_id == new_id, (
            f"CRITICAL: Point ID changed after migration!\n"
            f"  Old ID (relative): {old_id}\n"
            f"  New ID (absolute): {new_id}\n"
            f"  This will cause duplicate chunks in Qdrant."
        )

    def test_point_id_deterministic_across_runs(self, vector_store, watch_folder):
        """Same file + chunk must always produce same ID."""
        file_path = "docs/api/reference.md"
        chunk_index = 5

        id1 = vector_store._generate_point_id(file_path, chunk_index)
        id2 = vector_store._generate_point_id(file_path, chunk_index)
        id3 = vector_store._generate_point_id(file_path, chunk_index)

        assert id1 == id2 == id3, "Point IDs must be deterministic"

    def test_point_id_differs_for_different_chunks(self, vector_store, watch_folder):
        """Different chunks of same file must have different IDs."""
        file_path = "docs/tutorial.md"

        id_chunk_0 = vector_store._generate_point_id(file_path, 0)
        id_chunk_1 = vector_store._generate_point_id(file_path, 1)
        id_chunk_2 = vector_store._generate_point_id(file_path, 2)

        assert len({id_chunk_0, id_chunk_1, id_chunk_2}) == 3, "Each chunk must have unique ID"

    def test_point_id_differs_for_different_files(self, vector_store, watch_folder):
        """Same chunk index in different files must have different IDs."""
        chunk_index = 0

        id_file_a = vector_store._generate_point_id("docs/a.md", chunk_index)
        id_file_b = vector_store._generate_point_id("docs/b.md", chunk_index)

        assert id_file_a != id_file_b, "Different files must have different IDs"

    def test_metadata_keys_file_path_produces_stable_id(self, vector_store, watch_folder):
        """Using MetadataKeys.FILE_PATH value must produce stable IDs."""
        # This simulates using doc.metadata[MetadataKeys.FILE_PATH]
        abs_path = str(watch_folder / "readme.md")
        chunk_index = 0

        # What the code will do after migration
        metadata = {MetadataKeys.FILE_PATH: abs_path}
        new_id = vector_store._generate_point_id(
            metadata[MetadataKeys.FILE_PATH],
            chunk_index,
            watch_folder=watch_folder
        )

        # What the code did before migration
        rel_path = "readme.md"
        old_id = vector_store._generate_point_id(rel_path, chunk_index)

        assert old_id == new_id, "MetadataKeys.FILE_PATH must produce stable IDs"
```

**Step 2: Run the stability tests**

Run: `pytest tests/integration/test_point_id_stability.py -v -m integration`
Expected: ALL PASS - if any fail, the migration has broken point ID generation

**Step 3: Verify with actual Qdrant (optional, requires services)**

If Qdrant is running, verify no duplicate points exist:

```bash
# Count unique point IDs in collection
curl -s http://localhost:52001/collections/crawl4r/points/count | jq '.result.count'

# After re-indexing same documents, count should be IDENTICAL
```

**Step 4: Commit**

```bash
git add tests/integration/test_point_id_stability.py
git commit -m "test: add critical point ID stability verification for metadata migration"
```

---

#### Task 4.5: Remove deprecated keys from MetadataKeys

**Step 1: After all migrations complete AND Task 4.6 passes, remove deprecated constants**

Update `crawl4r/core/metadata.py`:

```python
class MetadataKeys:
    # ... (keep all active keys)

    # REMOVED - Migration complete:
    # FILE_PATH_RELATIVE = "file_path_relative"
    # FILE_PATH_ABSOLUTE = "file_path_absolute"
```

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: PASS (no code should reference deprecated keys)

**Step 3: Final verification**

```bash
# Verify no Python code uses deprecated string literals
rg '"file_path_relative"' crawl4r/ tests/ --type py
rg '"file_path_absolute"' crawl4r/ tests/ --type py
# Expected: 0 matches
```

**Step 4: Commit**

```bash
git add crawl4r/core/metadata.py
git commit -m "refactor: remove deprecated FILE_PATH_RELATIVE and FILE_PATH_ABSOLUTE keys"
```

---

## Final Verification Checklist

After all tasks complete, verify:

- [ ] `pytest tests/unit/ -v` passes
- [ ] `pytest tests/integration/ -v -m integration` passes (with services)
- [ ] `pytest tests/integration/test_point_id_stability.py -v` passes (CRITICAL)
- [ ] `ruff check .` passes
- [ ] `ty check crawl4r/` passes
- [ ] No `file_path_relative` or `file_path_absolute` strings in Python code
- [ ] `MetadataKeys` is imported wherever metadata keys are used
- [ ] Documentation reflects new `file_path` convention
- [ ] Point IDs are stable (same file → same ID before and after migration)

Move this plan to `docs/plans/complete/` when finished.

---

## Task Execution Order Summary

Execute tasks in this exact order to prevent breaking changes:

```
Task 0: Create MetadataKeys constants module
    └── Additive only, no behavioral change

Task 0.5: Make point ID generation path-agnostic (CRITICAL)
    └── MUST complete before Task 0.6

Task 0.6: Verify point ID stability gate (CRITICAL GATE)
    └── MUST PASS before Task 1 - blocks migration if IDs unstable

Task 1: Add SimpleDirectoryReader integration
    └── Safe now that ID generation verified stable

Task 2: Update metadata expectations
    └── Tests expect file_path instead of file_path_relative

Task 3: Update CLI and examples
    └── Delegate file loading to DocumentProcessor

Task 4: Migrate to MetadataKeys constants (TDD order)
    ├── 4.1: Unit tests FIRST (tests expect new keys - RED)
    ├── 4.2: Source code (implement to make tests GREEN)
    ├── 4.3: Integration tests
    ├── 4.4: Documentation
    ├── 4.6: Verify point ID stability (CRITICAL - final verification)
    └── 4.5: Remove deprecated keys (LAST, after 4.6 passes)
```
