# MarkdownNodeParser Swap Implementation Plan

> **📁 Organization Note:** When this plan is fully implemented and verified, move this file to `docs/plans/complete/` to keep the plans folder organized.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace custom Markdown chunking + node parsing with LlamaIndex `MarkdownNodeParser` in the ingestion pipeline.

**Architecture:** Remove `MarkdownChunker` and `CustomMarkdownNodeParser` from the ingestion path and construct `MarkdownNodeParser` directly in `DocumentProcessor`. Update CLI, tests, and docs/examples to reference the LlamaIndex parser instead of the custom chunker.

**Tech Stack:** Python, LlamaIndex (`MarkdownNodeParser`, `IngestionPipeline`), pytest.

---

### Task 0: Research LlamaIndex MarkdownNodeParser Feature Parity

**Files:**
- None (research only)

**Step 1: Verify MarkdownNodeParser exists and test basic functionality**

Run Python to verify the import and basic behavior:

```python
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document

parser = MarkdownNodeParser()
doc = Document(
    text="---\ntags: [python, test]\n---\n# Title\n\n## Section\n\nContent here.",
    metadata={"filename": "test.md"}
)
nodes = parser.get_nodes_from_documents([doc])

print(f"Nodes created: {len(nodes)}")
print(f"Node text sample: {nodes[0].text[:100] if nodes else 'No nodes'}")
print(f"Metadata keys: {list(nodes[0].metadata.keys()) if nodes else 'N/A'}")
```

**Step 2: Document feature comparison**

Create comparison matrix in this task's verification:

| Feature | MarkdownChunker | MarkdownNodeParser | Status |
|---------|----------------|-------------------|--------|
| Markdown splitting | ✅ | ✅ | Compatible |
| Heading hierarchy | ✅ section_path | Check output | Verify |
| Frontmatter parsing | ✅ YAML tags | Check output | Verify |
| Chunk size control | ✅ configurable | N/A (splits by headers) | Accept |
| Overlap percentage | ✅ 15% | N/A | Accept |

**Step 3: Accept trade-offs or abort**

**Objective Acceptance Criteria (all must pass to PROCEED):**

| Criteria | Requirement | Verified |
|----------|-------------|----------|
| Node creation | Parser produces ≥1 node from markdown with headings | [ ] |
| Heading hierarchy | Nodes preserve section structure (metadata or text) | [ ] |
| Frontmatter handling | YAML frontmatter doesn't corrupt node content | [ ] |
| Text preservation | Code blocks, lists, and formatting intact | [ ] |

**Decision Gate:**
- ✅ **PROCEED** if ALL criteria pass
- ❌ **ABORT** if ANY criteria fails - keep custom implementation

**[VERIFY]** Research complete. Document findings with filled criteria table before proceeding.

---

### Task 1: Add failing unit test for MarkdownNodeParser usage

**Files:**
- Modify: `tests/unit/test_processor_pipeline.py`

**Step 1: Write the failing test (RED)**

Add test to `tests/unit/test_processor_pipeline.py` using the existing `create_test_processor()` factory:

```python
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import Document


def test_processor_uses_markdown_node_parser() -> None:
    """DocumentProcessor should use LlamaIndex MarkdownNodeParser instead of custom parser."""
    processor = create_test_processor()

    # Verify type
    assert isinstance(processor.node_parser, MarkdownNodeParser), (
        f"Expected MarkdownNodeParser, got {type(processor.node_parser).__name__}"
    )


def test_markdown_node_parser_produces_nodes() -> None:
    """Verify MarkdownNodeParser produces nodes from markdown content."""
    processor = create_test_processor()

    test_doc = Document(
        text="# Title\n\nFirst paragraph.\n\n## Section\n\nSecond paragraph.",
        metadata={"filename": "test.md"}
    )

    nodes = processor.node_parser.get_nodes_from_documents([test_doc])

    # Verify nodes are produced
    assert len(nodes) > 0, "MarkdownNodeParser should produce at least one node"

    # Verify node has text content
    assert any(node.text.strip() for node in nodes), "Nodes should have text content"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_processor_pipeline.py::test_processor_uses_markdown_node_parser -v`
Expected: FAIL with assertion message containing:
```
Expected MarkdownNodeParser, got CustomMarkdownNodeParser
```

**Step 3: Commit failing test**

```bash
git add tests/unit/test_processor_pipeline.py
git commit -m "test: add failing test for MarkdownNodeParser usage (RED)"
```

---

### Task 2: Implement MarkdownNodeParser in DocumentProcessor (GREEN)

**Files:**
- Modify: `crawl4r/processing/processor.py`

**Step 1: Update processor to use MarkdownNodeParser**

- Update `crawl4r/processing/processor.py`:
  - Add import: `from llama_index.core.node_parser import MarkdownNodeParser`
  - Make `chunker` parameter optional with default `None` (deprecation path).
  - Replace `CustomMarkdownNodeParser(chunker=chunker)` with `MarkdownNodeParser()`.
  - Remove `MarkdownChunker` type hint from class attributes.
  - Keep `IngestionPipeline(transformations=[self.node_parser, self.embed_model])`.

```python
import warnings
from llama_index.core.node_parser import MarkdownNodeParser

# In __init__, make chunker optional with deprecation warning:
def __init__(
    self,
    config: Settings,
    vector_store: VectorStoreManager,
    chunker: MarkdownChunker | None = None,  # Deprecated, ignored
    tei_client: TEIClient | None = None,
    embed_model: BaseEmbedding | None = None,
    docstore: SimpleDocumentStore | None = None,
) -> None:
    # Emit deprecation warning if chunker is passed
    if chunker is not None:
        warnings.warn(
            "chunker parameter is deprecated and ignored. "
            "MarkdownNodeParser is now used automatically.",
            DeprecationWarning,
            stacklevel=2
        )
    self.node_parser = MarkdownNodeParser()
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/unit/test_processor_pipeline.py::test_processor_uses_markdown_node_parser -v`
Expected: PASS

**Step 3: Run full test suite to check for regressions**

Run: `pytest tests/unit/test_processor_pipeline.py -v`
Expected: All tests pass (chunker parameter is now optional).

> **⚠️ Note:** Some integration/performance tests may fail at this point due to using positional arguments in the old order `(config, tei_client, vector_store, chunker)`. This is expected - Task 3 migrations will fix these call sites. The unit tests should pass.

**Step 4: Commit implementation**

```bash
git add crawl4r/processing/processor.py
git commit -m "refactor: switch processor to MarkdownNodeParser (GREEN)"
```

---

### Task 3a: Remove MarkdownChunker from CLI (RED-GREEN)

**Files:**
- Modify: `crawl4r/cli/main.py`
- Modify: `tests/unit/test_main.py`

**Step 1: Write failing test (RED)**

Add test to `tests/unit/test_main.py`:

```python
def test_cli_does_not_use_markdown_chunker() -> None:
    """Verify CLI no longer imports or uses MarkdownChunker after migration."""
    import crawl4r.cli.main as main_module
    assert not hasattr(main_module, "MarkdownChunker"), (
        "CLI should not import MarkdownChunker after migration"
    )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_main.py::test_cli_does_not_use_markdown_chunker -v`
Expected: FAIL because `main.py` still imports `MarkdownChunker`.

**Step 3: Update CLI only (GREEN)**

Update `crawl4r/cli/main.py`:
- Remove `MarkdownChunker` import.
- Remove `chunker` variable and its construction.
- Update `DocumentProcessor` call to not pass `chunker`.
- Update `setup_components()` return type to remove `MarkdownChunker`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_main.py::test_cli_does_not_use_markdown_chunker -v`
Expected: PASS

**Step 5: Run CLI unit tests**

Run: `pytest tests/unit/test_main.py -v`
Expected: Some tests may fail due to chunker mocks - that's OK, we fix those next.

**Step 6: Commit CLI changes**

```bash
git add crawl4r/cli/main.py tests/unit/test_main.py
git commit -m "refactor: remove MarkdownChunker from CLI"
```

---

### Task 3b: Migrate CLI test mocks

**Files:**
- Modify: `tests/unit/test_main.py`

**Step 1: Update test_main.py mocks**

- Remove `"crawl4r.cli.main.MarkdownChunker"` from `COMMON_PATCHES`.
- Remove `mock_chunker` parameter from all test functions.
- Update test function signatures to remove chunker-related mocks.

**Step 2: Run tests to verify**

Run: `pytest tests/unit/test_main.py -v`
Expected: All tests pass.

**Step 3: Run full test suite regression check**

Run: `pytest tests/ -v --tb=short`
Expected: All previously passing tests still pass.

> **⚠️ Regression Gate:** After EACH Task 3 subtask, run the full test suite to catch regressions early. Do not proceed to the next subtask if tests fail unexpectedly.

**Step 4: Commit**

```bash
git add tests/unit/test_main.py
git commit -m "test: remove chunker mocks from CLI tests"
```

---

### Task 3c: Migrate processor pipeline test helpers

**Files:**
- Modify: `tests/unit/test_processor_pipeline.py`

**Step 1: Update create_test_processor() factory**

- Update `create_test_processor()` to not require/pass `chunker`.
- Remove `configure_chunker()` helper if no longer needed.

**Step 2: Run tests to verify**

Run: `pytest tests/unit/test_processor_pipeline.py -v`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/unit/test_processor_pipeline.py
git commit -m "test: update processor test factory to not use chunker"
```

---

### Task 3d: Migrate test_processor.py unit tests

**Files:**
- Modify: `tests/unit/test_processor.py`

**Step 1: Update test_processor.py**

This file has extensive chunker references (~50 occurrences). Update:
- Remove `configure_chunker()` helper function.
- Remove all `chunker = Mock()` and `configure_chunker(chunker)` lines.
- Update all `DocumentProcessor(config, tei_client, vector_store, chunker)` calls to not pass chunker.
- Update any assertions that reference `processor.chunker`.

**Step 2: Run tests to verify**

Run: `pytest tests/unit/test_processor.py -v`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/unit/test_processor.py
git commit -m "test: migrate test_processor.py from MarkdownChunker"
```

---

### Task 3e: Migrate module structure tests

**Files:**
- Modify: `tests/unit/test_module_structure.py`

**Step 1: Update module structure tests**

- Remove or update tests that verify `MarkdownChunker` import.
- Add test verifying `MarkdownNodeParser` is used instead.

**Step 2: Run tests to verify**

Run: `pytest tests/unit/test_module_structure.py -v`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/unit/test_module_structure.py
git commit -m "test: update module structure tests for MarkdownNodeParser"
```

---

### Task 3f: Migrate llama_parser tests

**Files:**
- Modify: `tests/unit/test_llama_parser.py`

**Step 1: Update llama_parser tests**

- Remove `MarkdownChunker` and `CustomMarkdownNodeParser` imports.
- Replace tests with `MarkdownNodeParser` equivalents or mark for deletion in Task 4.

**Step 2: Run tests to verify**

Run: `pytest tests/unit/test_llama_parser.py -v`
Expected: All tests pass (or file is empty/deleted).

**Step 3: Commit**

```bash
git add tests/unit/test_llama_parser.py
git commit -m "test: migrate llama_parser tests to MarkdownNodeParser"
```

---

### Task 3g: Migrate integration conftest and core tests

**Files:**
- Modify: `tests/integration/conftest.py`
- Modify: `tests/integration/test_e2e_core.py`

**Step 1: Update integration test conftest**

- Remove `MarkdownChunker` from conftest fixtures.
- Update `DocumentProcessor` fixture to not pass chunker.

**Step 2: Update test_e2e_core.py**

- Remove chunker references and update `DocumentProcessor` calls.

**Step 3: Run tests to verify**

Run: `pytest tests/integration/test_e2e_core.py -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add tests/integration/conftest.py tests/integration/test_e2e_core.py
git commit -m "test: migrate integration conftest and core tests from MarkdownChunker"
```

---

### Task 3h: Migrate integration pipeline and error tests

**Files:**
- Modify: `tests/integration/test_e2e_pipeline.py`
- Modify: `tests/integration/test_e2e_errors.py`

**Step 1: Update test files**

- Remove chunker references and update `DocumentProcessor` calls in both files.

**Step 2: Run tests to verify**

Run: `pytest tests/integration/test_e2e_pipeline.py tests/integration/test_e2e_errors.py -v`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/integration/test_e2e_pipeline.py tests/integration/test_e2e_errors.py
git commit -m "test: migrate integration pipeline and error tests from MarkdownChunker"
```

---

### Task 3i: Migrate integration crawl and reader tests

**Files:**
- Modify: `tests/integration/test_e2e_crawl_pipeline.py`
- Modify: `tests/integration/test_e2e_reader_pipeline.py`

**Step 1: Update test files**

- Remove chunker references and update `DocumentProcessor` calls in both files.

**Step 2: Run tests to verify**

Run: `pytest tests/integration/test_e2e_crawl_pipeline.py tests/integration/test_e2e_reader_pipeline.py -v`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/integration/test_e2e_crawl_pipeline.py tests/integration/test_e2e_reader_pipeline.py
git commit -m "test: migrate integration crawl and reader tests from MarkdownChunker"
```

---

### Task 3j: Migrate performance tests

**Files:**
- Modify: `tests/performance/test_e2e_performance.py`

**Step 1: Update performance tests**

- Remove `MarkdownChunker` imports and usage.
- Update `DocumentProcessor` instantiation.

**Step 2: Run performance tests to verify**

Run: `pytest tests/performance/ -v`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/performance/
git commit -m "test: migrate performance tests from MarkdownChunker"
```

---

### Task 3k: Migrate benchmark scripts

**Files:**
- Modify: `tests/run_benchmark.py`
- Modify: `tests/run_latency_benchmark.py`

**Step 1: Update benchmark scripts**

- Remove `MarkdownChunker` imports.
- Update to use `MarkdownNodeParser` or remove chunker usage.

**Step 2: Verify scripts run**

Run: `python tests/run_benchmark.py --help` (or similar dry-run)
Expected: No import errors.

**Step 3: Commit**

```bash
git add tests/run_benchmark.py tests/run_latency_benchmark.py
git commit -m "refactor: migrate benchmark scripts from MarkdownChunker"
```

---

### Task 3l: Migrate example scripts

**Files:**
- Modify: `examples/stress_test_pipeline.py`
- Modify: `examples/crawl4ai_reader_usage.py`

**Step 1: Update example scripts**

- Remove `MarkdownChunker` imports.
- Update to use `MarkdownNodeParser` in examples.

**Step 2: Verify scripts are syntactically correct**

Run: `python -m py_compile examples/stress_test_pipeline.py examples/crawl4ai_reader_usage.py`
Expected: No syntax errors.

**Step 3: Commit**

```bash
git add examples/
git commit -m "refactor: migrate example scripts from MarkdownChunker"
```

---

### Task 3m: Migrate integration watcher tests

**Files:**
- Modify: `tests/integration/test_e2e_watcher.py`

**Step 1: Update test_e2e_watcher.py**

- Remove `MarkdownChunker` imports and usage.
- Update `DocumentProcessor` instantiation to not pass chunker.

**Step 2: Run tests to verify**

Run: `pytest tests/integration/test_e2e_watcher.py -v`
Expected: All tests pass.

**Step 3: Run full test suite regression check**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add tests/integration/test_e2e_watcher.py
git commit -m "test: migrate integration watcher tests from MarkdownChunker"
```

---

### Task 4: Remove deprecated chunker files (RED-GREEN)

**Files:**
- Modify: `tests/unit/test_module_structure.py`
- Delete: `crawl4r/processing/chunker.py`
- Delete: `crawl4r/processing/llama_parser.py`
- Delete: `tests/unit/test_chunker.py`

**Step 1: Write failing test (RED)**

Add test to `tests/unit/test_module_structure.py`:

```python
import os


def test_deprecated_chunker_files_removed() -> None:
    """Verify deprecated chunker files no longer exist after migration."""
    deprecated_files = [
        "crawl4r/processing/chunker.py",
        "crawl4r/processing/llama_parser.py",
        "tests/unit/test_chunker.py",
    ]

    for filepath in deprecated_files:
        assert not os.path.exists(filepath), (
            f"Deprecated file should be removed: {filepath}"
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_module_structure.py::test_deprecated_chunker_files_removed -v`
Expected: FAIL with assertion message containing:
```
Deprecated file should be removed: crawl4r/processing/chunker.py
```

**Step 3: Verify no remaining imports (pre-deletion check)**

Run: `rg "from crawl4r.processing.chunker" --type py`
Run: `rg "from crawl4r.processing.llama_parser" --type py`
Run: `rg "import crawl4r.processing.chunker" --type py`
Run: `rg "MarkdownChunker" --type py`
Run: `rg "CustomMarkdownNodeParser" --type py`
Expected: No matches (all imports should be migrated in Task 3).

**Step 4: Delete deprecated files (GREEN)**

```bash
rm crawl4r/processing/chunker.py
rm crawl4r/processing/llama_parser.py
rm tests/unit/test_chunker.py
```

**Step 5: Verify files are deleted**

```bash
test ! -f crawl4r/processing/chunker.py && echo "chunker.py removed"
test ! -f crawl4r/processing/llama_parser.py && echo "llama_parser.py removed"
test ! -f tests/unit/test_chunker.py && echo "test_chunker.py removed"
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/unit/test_module_structure.py::test_deprecated_chunker_files_removed -v`
Expected: PASS

**Step 7: Run full test suite to verify no breakage**

Run: `pytest tests/ -v`
Expected: All tests pass.

**Step 8: Commit deletion**

```bash
git add -A
git commit -m "chore: remove deprecated MarkdownChunker and CustomMarkdownNodeParser"
```

---

### Task 5: Update docs/examples to use MarkdownNodeParser (RED-GREEN)

**Files:**
- Modify: `tests/unit/test_module_structure.py`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `ENHANCEMENTS.md`

**Step 1: Write failing test (RED)**

Add test to `tests/unit/test_module_structure.py`:

```python
import subprocess


def test_no_markdown_chunker_references_in_docs() -> None:
    """Verify no MarkdownChunker references remain in documentation."""
    result = subprocess.run(
        ["rg", "-l", "MarkdownChunker", "README.md", "CLAUDE.md", "ENHANCEMENTS.md"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, (
        f"MarkdownChunker still referenced in docs: {result.stdout.strip()}"
    )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_module_structure.py::test_no_markdown_chunker_references_in_docs -v`
Expected: FAIL with assertion message containing:
```
MarkdownChunker still referenced in docs: README.md
```
(or CLAUDE.md/ENHANCEMENTS.md depending on which is checked first)

**Step 3: Update documentation (GREEN)**

Update `README.md`:
- Replace `MarkdownChunker` usage examples with `MarkdownNodeParser`.
- Update import statements to use `from llama_index.core.node_parser import MarkdownNodeParser`.

Update `CLAUDE.md`:
- Replace chunker references in "Integration with Pipeline" section.
- Update architecture description.

Update `ENHANCEMENTS.md`:
- Mark MarkdownChunker as removed/replaced.
- Document migration to MarkdownNodeParser.

Example replacement:

```python
# OLD
from crawl4r.processing.chunker import MarkdownChunker
chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)

# NEW
from llama_index.core.node_parser import MarkdownNodeParser
node_parser = MarkdownNodeParser()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_module_structure.py::test_no_markdown_chunker_references_in_docs -v`
Expected: PASS

**Step 5: Verify docs consistency**

Run: `rg -n "MarkdownChunker" README.md CLAUDE.md ENHANCEMENTS.md`
Expected: No remaining references.

**Step 6: Commit**

```bash
git add README.md CLAUDE.md ENHANCEMENTS.md tests/unit/test_module_structure.py
git commit -m "docs: replace MarkdownChunker references with MarkdownNodeParser"
```

---

## Migration Checklist

Before marking complete, verify:

- [ ] Task 0: Feature parity researched and documented
- [ ] Task 1: Failing test written and committed
- [ ] Task 2: MarkdownNodeParser implemented in processor
- [ ] Task 3a-3m: All files migrated (CLI, tests, benchmarks, examples)
  - 3a: CLI main.py
  - 3b: CLI test mocks
  - 3c: Processor pipeline test helpers
  - 3d: test_processor.py unit tests
  - 3e: Module structure tests
  - 3f: LLama parser tests
  - 3g: Integration conftest + core tests
  - 3h: Integration pipeline + error tests
  - 3i: Integration crawl + reader tests
  - 3j: Performance tests
  - 3k: Benchmark scripts
  - 3l: Example scripts
  - 3m: Integration watcher tests
- [ ] Task 4: Deprecated files deleted after RED test
- [ ] Task 5: Documentation updated after RED test
- [ ] Full test suite passes: `pytest tests/ -v`
- [ ] No MarkdownChunker references remain: `rg "MarkdownChunker" --type py`
