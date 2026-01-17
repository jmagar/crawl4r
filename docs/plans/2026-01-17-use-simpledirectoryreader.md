# SimpleDirectoryReader Swap Implementation Plan

> **Organization Note:** When this plan is fully implemented and verified, move this file to `docs/plans/complete/` to keep the plans folder organized.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace manual file I/O in `DocumentProcessor` with LlamaIndex `SimpleDirectoryReader` and use its default metadata.

**Architecture:** Use `SimpleDirectoryReader(input_files=[...])` to load documents (and metadata) per file, then set deterministic IDs before passing documents into the ingestion pipeline.

**Tech Stack:** Python, LlamaIndex (`SimpleDirectoryReader`, `Document`), pytest.

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

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_processor.py::test_processor_uses_simpledirectoryreader -v`
Expected: FAIL because `SimpleDirectoryReader` is not used yet.

**Step 3: Write minimal implementation**

- Update `crawl4r/processing/processor.py`:
  - Import `SimpleDirectoryReader`.
  - Add a helper `_load_document_via_reader(file_path: Path) -> Document`.
  - Replace `_load_markdown_file` usage with reader-based load.

```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_files=[str(file_path)])
docs = reader.load_data()
if not docs:
    raise FileNotFoundError(str(file_path))
doc = docs[0]
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

**Files:**
- Modify: `tests/unit/test_processor.py`
- Modify: `tests/unit/test_processor_id_generation.py`

**Step 1: Write the failing test**

Update existing metadata assertions to expect `file_path` and `file_name` keys.

```python
assert "file_path" in doc.metadata
assert "file_name" in doc.metadata
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_processor.py::test_process_document_success -v`
Expected: FAIL if tests still assert `file_path_relative` or `file_path_absolute`.

**Step 3: Write minimal implementation**

- Update tests to assert SimpleDirectoryReader metadata keys.
- Preserve deterministic ID assignment (uuid5) using the relative path derived from `doc.metadata["file_path"]`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_processor.py tests/unit/test_processor_id_generation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_processor.py tests/unit/test_processor_id_generation.py
git commit -m "test: align metadata expectations with SimpleDirectoryReader"
```

---

### Task 3: Update CLI and examples to avoid manual file loading

**Files:**
- Modify: `crawl4r/cli/main.py`
- Modify: `examples/*`

**Step 1: Update CLI to rely on DocumentProcessor load path**

Ensure CLI passes file paths only; `DocumentProcessor` handles `SimpleDirectoryReader`.

**Step 2: Verify examples**

Run: `rg -n "_load_markdown_file" examples crawl4r/cli/main.py`
Expected: No remaining references to manual file loading.

**Step 3: Commit**

```bash
git add crawl4r/cli/main.py examples
git commit -m "docs: update examples to SimpleDirectoryReader flow"
```
