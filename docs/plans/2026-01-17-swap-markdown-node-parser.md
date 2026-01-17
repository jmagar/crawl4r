# MarkdownNodeParser Swap Implementation Plan

> **Organization Note:** When this plan is fully implemented and verified, move this file to `docs/plans/complete/` to keep the plans folder organized.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace custom Markdown chunking + node parsing with LlamaIndex `MarkdownNodeParser` in the ingestion pipeline.

**Architecture:** Remove `MarkdownChunker` and `CustomMarkdownNodeParser` from the ingestion path and construct `MarkdownNodeParser` directly in `DocumentProcessor`. Update CLI, tests, and docs/examples to reference the LlamaIndex parser instead of the custom chunker.

**Tech Stack:** Python, LlamaIndex (`MarkdownNodeParser`, `IngestionPipeline`), pytest.

---

### Task 1: Add failing unit test for MarkdownNodeParser usage

**Files:**
- Modify: `tests/unit/test_llama_parser.py`
- Modify: `tests/unit/test_processor_pipeline.py`

**Step 1: Write the failing test**

```python
from llama_index.core.node_parser import MarkdownNodeParser
from crawl4r.processing.processor import DocumentProcessor


def test_processor_uses_markdown_node_parser(config, vector_store, tei_client):
    processor = DocumentProcessor(
        config=config,
        vector_store=vector_store,
        tei_client=tei_client,
    )
    assert isinstance(processor.node_parser, MarkdownNodeParser)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_processor_pipeline.py::test_processor_uses_markdown_node_parser -v`
Expected: FAIL with `TypeError` (unexpected ctor args) or assertion failure (custom parser used).

**Step 3: Write minimal implementation**

- Update `crawl4r/processing/processor.py`:
  - Remove `chunker` dependency from constructor and attributes.
  - Replace `CustomMarkdownNodeParser(...)` with `MarkdownNodeParser()`.
  - Keep `IngestionPipeline(transformations=[self.node_parser, self.embed_model])`.

```python
from llama_index.core.node_parser import MarkdownNodeParser

self.node_parser = MarkdownNodeParser()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_processor_pipeline.py::test_processor_uses_markdown_node_parser -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crawl4r/processing/processor.py tests/unit/test_processor_pipeline.py
git commit -m "refactor: switch processor to MarkdownNodeParser"
```

---

### Task 2: Remove custom Markdown chunker/parser from code paths

**Files:**
- Modify: `crawl4r/processing/processor.py`
- Modify: `crawl4r/cli/main.py`
- Modify: `crawl4r/processing/llama_parser.py`
- Modify: `crawl4r/processing/chunker.py`

**Step 1: Write the failing test**

Update CLI tests to expect `MarkdownNodeParser` instead of `MarkdownChunker`.

```python
# tests/unit/test_main.py
@patch("crawl4r.cli.main.MarkdownNodeParser")
def test_cli_uses_markdown_node_parser(mock_parser, ...):
    ...
    mock_parser.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_main.py::test_cli_uses_markdown_node_parser -v`
Expected: FAIL because CLI still constructs `MarkdownChunker`.

**Step 3: Write minimal implementation**

- Update `crawl4r/cli/main.py` to construct `MarkdownNodeParser()` instead of `MarkdownChunker` and remove unused imports.
- Deprecate/remove `CustomMarkdownNodeParser` in `crawl4r/processing/llama_parser.py`.
- Remove or archive `crawl4r/processing/chunker.py` if no longer referenced.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_main.py::test_cli_uses_markdown_node_parser -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crawl4r/cli/main.py crawl4r/processing/llama_parser.py crawl4r/processing/chunker.py tests/unit/test_main.py
git commit -m "refactor: replace custom chunker with MarkdownNodeParser"
```

---

### Task 3: Update test suite references to MarkdownChunker

**Files:**
- Modify: `tests/unit/test_chunker.py`
- Modify: `tests/unit/test_llama_parser.py`
- Modify: `tests/integration/*`
- Modify: `tests/performance/*`

**Step 1: Write the failing test**

Add a small integration test using `MarkdownNodeParser` directly (replacing `MarkdownChunker`).

```python
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document


def test_markdown_node_parser_basic():
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents([Document(text="# Title\n\nBody")])
    assert len(nodes) >= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_llama_parser.py::test_markdown_node_parser_basic -v`
Expected: FAIL until custom parser usage removed and imports updated.

**Step 3: Write minimal implementation**

- Replace `MarkdownChunker` tests with `MarkdownNodeParser` usage.
- Remove or rewrite `tests/unit/test_chunker.py` if chunker is deleted.
- Update integration/perf tests to construct `MarkdownNodeParser` instead of `MarkdownChunker`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_llama_parser.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_llama_parser.py tests/unit/test_chunker.py tests/integration tests/performance
git commit -m "test: migrate chunking tests to MarkdownNodeParser"
```

---

### Task 4: Update docs/examples to use MarkdownNodeParser

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `ENHANCEMENTS.md`
- Modify: `examples/*`

**Step 1: Update documentation snippets**

Replace `MarkdownChunker` usage examples with `MarkdownNodeParser`.

```python
from llama_index.core.node_parser import MarkdownNodeParser

node_parser = MarkdownNodeParser()
```

**Step 2: Verify docs consistency**

Run: `rg -n "MarkdownChunker" README.md CLAUDE.md ENHANCEMENTS.md examples`
Expected: No remaining references.

**Step 3: Commit**

```bash
git add README.md CLAUDE.md ENHANCEMENTS.md examples
git commit -m "docs: replace MarkdownChunker references with MarkdownNodeParser"
```
