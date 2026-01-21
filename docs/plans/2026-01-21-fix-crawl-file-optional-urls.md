# Fix Bug #4: `crawl --file` Optional URLs Implementation Plan

**Created:** 12:47:10 AM | 01/21/2026 (EST)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix `crawl --file urls.txt` command by making URLs positional argument optional when `--file` is provided.

**Architecture:** Modify the Typer argument signature to make `urls` optional (default `None`), then validate mutual exclusivity in the function body. This follows the same pattern as the `scrape` command which already works correctly.

**Tech Stack:** Python 3.11+, Typer, pytest, typer.testing.CliRunner

---

## Phase 1: Write Failing Tests (RED)

### Step 1: Write test for --file-only usage (should fail currently)

Create: `/home/jmagar/workspace/crawl4r/tests/unit/cli/test_crawl_file_option.py`

```python
"""Tests for crawl command --file option."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from typer.testing import CliRunner

from crawl4r.cli.app import app

runner = CliRunner()


def test_crawl_file_only_should_work(tmp_path: Path) -> None:
    """Test crawl --file urls.txt without positional URLs."""
    # Arrange
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://example.com\nhttps://example.org\n")

    mock_result = Mock()
    mock_result.crawl_id = "test-crawl-123"
    mock_result.urls_total = 2
    mock_result.urls_failed = 0
    mock_result.chunks_created = 10
    mock_result.success = True
    mock_result.queued = False

    # Act
    with patch("crawl4r.cli.commands.crawl.IngestionService") as mock_service:
        with patch("crawl4r.cli.commands.crawl.asyncio.run") as mock_run:
            mock_run.return_value = (mock_result, None)
            result = runner.invoke(app, ["crawl", "--file", str(urls_file)])

    # Assert
    assert result.exit_code == 0
    assert "Crawl ID: test-crawl-123" in result.output
    assert "URLs: 2" in result.output


def test_crawl_urls_only_should_work() -> None:
    """Test crawl URL1 URL2 without --file (backward compatibility)."""
    # Arrange
    mock_result = Mock()
    mock_result.crawl_id = "test-crawl-456"
    mock_result.urls_total = 2
    mock_result.urls_failed = 0
    mock_result.chunks_created = 10
    mock_result.success = True
    mock_result.queued = False

    # Act
    with patch("crawl4r.cli.commands.crawl.IngestionService") as mock_service:
        with patch("crawl4r.cli.commands.crawl.asyncio.run") as mock_run:
            mock_run.return_value = (mock_result, None)
            result = runner.invoke(
                app, ["crawl", "https://example.com", "https://example.org"]
            )

    # Assert
    assert result.exit_code == 0
    assert "Crawl ID: test-crawl-456" in result.output
    assert "URLs: 2" in result.output


def test_crawl_both_urls_and_file_should_merge(tmp_path: Path) -> None:
    """Test crawl URL --file urls.txt should merge both sources."""
    # Arrange
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://file.com\n")

    mock_result = Mock()
    mock_result.crawl_id = "test-crawl-789"
    mock_result.urls_total = 2  # 1 from CLI + 1 from file
    mock_result.urls_failed = 0
    mock_result.chunks_created = 10
    mock_result.success = True
    mock_result.queued = False

    # Act
    with patch("crawl4r.cli.commands.crawl.IngestionService") as mock_service:
        with patch("crawl4r.cli.commands.crawl.asyncio.run") as mock_run:
            mock_run.return_value = (mock_result, None)
            result = runner.invoke(
                app, ["crawl", "https://cli.com", "--file", str(urls_file)]
            )

    # Assert
    assert result.exit_code == 0
    assert "Crawl ID: test-crawl-789" in result.output


def test_crawl_no_urls_no_file_should_fail() -> None:
    """Test crawl with neither URLs nor --file should exit with error."""
    # Act
    result = runner.invoke(app, ["crawl"])

    # Assert
    assert result.exit_code == 1
    assert "No URLs provided" in result.output


def test_crawl_empty_file_should_fail(tmp_path: Path) -> None:
    """Test crawl --file empty.txt should exit with error."""
    # Arrange
    urls_file = tmp_path / "empty.txt"
    urls_file.write_text("")

    # Act
    result = runner.invoke(app, ["crawl", "--file", str(urls_file)])

    # Assert
    assert result.exit_code == 1
    assert "No URLs provided" in result.output


def test_crawl_nonexistent_file_should_fail() -> None:
    """Test crawl --file nonexistent.txt should exit with error."""
    # Act
    result = runner.invoke(app, ["crawl", "--file", "/does/not/exist.txt"])

    # Assert
    assert result.exit_code == 2  # Typer error code for bad parameter
    assert "URL file not found" in result.output
```

**Step 2: Run tests to verify they fail**

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_crawl_file_option.py::test_crawl_file_only_should_work -v`

Expected: FAIL with "Missing argument 'URLS...'" or similar

**Step 3: Run all new tests to see baseline failure**

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_crawl_file_option.py -v`

Expected: Multiple failures showing current behavior

---

## Phase 2: Implement Fix (GREEN)

### Step 4: Make URLs argument optional in crawl command

Modify: `/home/jmagar/workspace/crawl4r/crawl4r/cli/commands/crawl.py:20-24`

**OLD:**
```python
def crawl_command(
    urls: list[str] = typer.Argument(..., help="URLs to crawl"),
    file: Path | None = typer.Option(None, "-f", "--file", help="File containing URLs (one per line)"),
    depth: int = typer.Option(1, "-d", "--depth", help="Crawl depth (0=no discovery, 1+=recursive)"),
    external: bool = typer.Option(False, "--external", help="Include external links in discovery"),
) -> None:
```

**NEW:**
```python
def crawl_command(
    urls: list[str] | None = typer.Argument(None, help="URLs to crawl"),
    file: Path | None = typer.Option(None, "-f", "--file", help="File containing URLs (one per line)"),
    depth: int = typer.Option(1, "-d", "--depth", help="Crawl depth (0=no discovery, 1+=recursive)"),
    external: bool = typer.Option(False, "--external", help="Include external links in discovery"),
) -> None:
```

**Changes:**
1. Change `urls: list[str]` → `urls: list[str] | None`
2. Change `typer.Argument(...)` → `typer.Argument(None)`
3. Keep everything else the same

**Step 5: Run single test to verify basic fix works**

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_crawl_file_option.py::test_crawl_file_only_should_work -v`

Expected: PASS

**Step 6: Run all new tests to verify complete fix**

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_crawl_file_option.py -v`

Expected: All tests PASS

**Step 7: Run existing crawl command tests to ensure backward compatibility**

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_crawl_command.py -v`

Expected: All tests PASS

**Step 8: Verify the _merge_urls function handles None correctly**

The existing `_merge_urls` function at line 55-72 already handles this correctly:
- Line 58: `merged = [url.strip() for url in urls if url.strip()]` - works with empty list
- Line 26: `resolved_urls = _merge_urls(urls or [], file)` - converts None to empty list

No changes needed, but verify behavior is correct.

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_crawl_file_option.py::test_crawl_no_urls_no_file_should_fail -v`

Expected: PASS (validation already exists at line 27-29)

---

## Phase 3: Integration Testing

### Step 9: Write integration test with real file I/O

Add to: `/home/jmagar/workspace/crawl4r/tests/unit/cli/test_crawl_file_option.py`

```python
def test_crawl_file_integration_real_file(tmp_path: Path) -> None:
    """Integration test: crawl --file with real file system."""
    # Arrange
    urls_file = tmp_path / "integration_urls.txt"
    urls_file.write_text(
        "https://example.com\n"
        "https://example.org\n"
        "  https://example.net  \n"  # Test whitespace handling
        "\n"  # Test empty line
        "https://example.edu\n"
    )

    mock_result = Mock()
    mock_result.crawl_id = "integration-test"
    mock_result.urls_total = 4  # Should skip empty line
    mock_result.urls_failed = 0
    mock_result.chunks_created = 20
    mock_result.success = True
    mock_result.queued = False

    # Act
    with patch("crawl4r.cli.commands.crawl.IngestionService") as mock_service:
        with patch("crawl4r.cli.commands.crawl.asyncio.run") as mock_run:
            mock_run.return_value = (mock_result, None)
            result = runner.invoke(app, ["crawl", "--file", str(urls_file)])

    # Assert
    assert result.exit_code == 0
    assert "integration-test" in result.output
```

**Step 10: Run integration test**

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_crawl_file_option.py::test_crawl_file_integration_real_file -v`

Expected: PASS

---

## Phase 4: Edge Cases and Error Handling

### Step 11: Write test for whitespace-only URLs file

Add to: `/home/jmagar/workspace/crawl4r/tests/unit/cli/test_crawl_file_option.py`

```python
def test_crawl_file_whitespace_only_should_fail(tmp_path: Path) -> None:
    """Test crawl --file with only whitespace should fail."""
    # Arrange
    urls_file = tmp_path / "whitespace.txt"
    urls_file.write_text("   \n\n\t\n   ")

    # Act
    result = runner.invoke(app, ["crawl", "--file", str(urls_file)])

    # Assert
    assert result.exit_code == 1
    assert "No URLs provided" in result.output
```

**Step 12: Run edge case test**

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_crawl_file_option.py::test_crawl_file_whitespace_only_should_fail -v`

Expected: PASS (existing validation at line 27-29 should handle this)

**Step 13: Write test for file size limit**

Add to: `/home/jmagar/workspace/crawl4r/tests/unit/cli/test_crawl_file_option.py`

```python
def test_crawl_file_too_large_should_fail(tmp_path: Path) -> None:
    """Test crawl --file with >1MB file should fail."""
    # Arrange
    urls_file = tmp_path / "large.txt"
    # Create file slightly over 1MB
    large_content = "https://example.com\n" * 50000  # ~1.05MB
    urls_file.write_text(large_content)

    # Act
    result = runner.invoke(app, ["crawl", "--file", str(urls_file)])

    # Assert
    assert result.exit_code == 2  # Typer error code
    assert "URL file too large" in result.output
```

**Step 14: Run file size test**

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/test_crawl_file_option.py::test_crawl_file_too_large_should_fail -v`

Expected: PASS (existing validation at line 65-68 should handle this)

---

## Phase 5: Documentation and Verification

### Step 15: Update command help text to clarify optional usage

Modify: `/home/jmagar/workspace/crawl4r/crawl4r/cli/commands/crawl.py:20-24`

Update help text to document the optional nature:

```python
def crawl_command(
    urls: list[str] | None = typer.Argument(
        None, help="URLs to crawl (optional if --file is provided)"
    ),
    file: Path | None = typer.Option(
        None, "-f", "--file", help="File containing URLs (one per line)"
    ),
    depth: int = typer.Option(
        1, "-d", "--depth", help="Crawl depth (0=no discovery, 1+=recursive)"
    ),
    external: bool = typer.Option(
        False, "--external", help="Include external links in discovery"
    ),
) -> None:
    """Crawl URLs and ingest results into the vector store.

    Provide URLs either as arguments, via --file, or both:

      crawl https://example.com

      crawl --file urls.txt

      crawl https://example.com --file urls.txt
    """
```

**Step 16: Verify help text displays correctly**

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && python -m crawl4r.cli.app crawl --help`

Expected: Help text shows:
- "URLs to crawl (optional if --file is provided)"
- Usage examples in docstring

**Step 17: Run full test suite to ensure no regressions**

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/ -v`

Expected: All CLI tests PASS

**Step 18: Test manual invocations to verify real-world usage**

Create test file:
```bash
source /home/jmagar/workspace/crawl4r/.venv/bin/activate
echo "https://example.com" > /tmp/test_urls.txt
```

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && python -m crawl4r.cli.app crawl --file /tmp/test_urls.txt --depth 0 --help`

Expected: Help displays without errors

**Step 19: Verify type hints are correct**

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && .venv/bin/python -m ty check crawl4r/cli/commands/crawl.py`

Expected: No type errors

**Step 20: Run linter to ensure code quality**

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && .venv/bin/python -m ruff check crawl4r/cli/commands/crawl.py`

Expected: No lint errors

---

## Phase 6: Commit and Cleanup

### Step 21: Commit the implementation

Run:
```bash
cd /home/jmagar/workspace/crawl4r
source .venv/bin/activate
git add crawl4r/cli/commands/crawl.py tests/unit/cli/test_crawl_file_option.py
git commit -m "$(cat <<'EOF'
fix(cli): make URLs argument optional when --file is provided

Fixes Bug #4 where `crawl --file urls.txt` failed with "Missing argument 'URLS...'".

Changes:
- Made `urls` argument optional (default None) in crawl_command signature
- Updated help text to clarify optional usage with examples
- Existing validation already handles None case via `urls or []`

Backward compatibility:
- `crawl URL` still works (single URL)
- `crawl URL1 URL2` still works (multiple URLs)
- `crawl URL --file urls.txt` still works (merge both)

New functionality:
- `crawl --file urls.txt` now works (file-only mode)

Tests added:
- test_crawl_file_only_should_work
- test_crawl_urls_only_should_work
- test_crawl_both_urls_and_file_should_merge
- test_crawl_no_urls_no_file_should_fail
- test_crawl_empty_file_should_fail
- test_crawl_nonexistent_file_should_fail
- test_crawl_file_integration_real_file
- test_crawl_file_whitespace_only_should_fail
- test_crawl_file_too_large_should_fail

All tests pass. No regressions in existing functionality.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

**Step 22: Verify commit**

Run: `git log -1 --stat`

Expected: Shows commit with modified files and commit message

**Step 23: Final verification - run all tests**

Run: `source /home/jmagar/workspace/crawl4r/.venv/bin/activate && pytest tests/unit/cli/ -v --tb=short`

Expected: All tests PASS

---

## Verification Checklist

- [ ] Test `crawl --file urls.txt` works without positional URLs
- [ ] Test `crawl URL` still works (backward compatibility)
- [ ] Test `crawl URL1 URL2` still works (multiple URLs)
- [ ] Test `crawl URL --file urls.txt` merges both sources
- [ ] Test `crawl` with no arguments fails gracefully
- [ ] Test `crawl --file empty.txt` fails gracefully
- [ ] Test `crawl --file nonexistent.txt` fails with clear error
- [ ] Test file size limit (>1MB) still enforced
- [ ] Test whitespace handling in files
- [ ] Help text displays correctly with updated documentation
- [ ] No type errors (`ty check`)
- [ ] No lint errors (`ruff check`)
- [ ] All existing tests still pass
- [ ] Commit message follows conventional commits format

---

## Risk Assessment

**Low Risk Changes:**
- Single type annotation change (`list[str]` → `list[str] | None`)
- Single default value change (`...` → `None`)
- Help text updates (documentation only)

**No Breaking Changes:**
- Existing `urls or []` pattern handles None gracefully
- `_merge_urls` function already handles empty lists
- Validation logic already in place (line 27-29)

**Backward Compatibility:**
- All existing usage patterns continue to work
- Only adds new capability (file-only mode)

---

## Alternative Approaches Considered

**Option A: Make urls optional with default=None, validate in function** ✅ SELECTED
- Pros: Simple, minimal code change, follows scrape command pattern
- Cons: None significant

**Option B: Use Typer callback for custom validation**
- Pros: More explicit validation
- Cons: Over-engineered for this use case, more complex code

**Option C: Restructure to use mutually exclusive groups**
- Pros: Enforces mutual exclusivity at CLI level
- Cons: Breaking change to CLI interface, not supported well in Typer

---

## References

- Similar working implementation: `crawl4r/cli/commands/scrape.py:22-24`
- Existing validation: `crawl4r/cli/commands/crawl.py:27-29`
- URL merging logic: `crawl4r/cli/commands/crawl.py:55-72`
- Typer documentation: https://typer.tiangolo.com/tutorial/arguments/optional/
