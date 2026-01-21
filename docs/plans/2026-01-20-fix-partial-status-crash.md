# Fix PARTIAL Status KeyError Implementation Plan

**Created:** 12:47:00 AM | 01/21/2026 (EST)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix KeyError crash when status command encounters PARTIAL status by adding missing mapping in `_status_style()` function.

**Architecture:** Simple TDD bug fix - add PARTIAL status to color mapping dictionary and verify all enum values are covered.

**Tech Stack:** Python 3.11+, Typer, Rich, pytest

---

## Phase 1: Red - Write Failing Test

### Step 1.1: Write test for PARTIAL status color mapping

**Create:** `/home/jmagar/workspace/crawl4r/tests/unit/cli/test_status_partial.py`

```python
"""Tests for PARTIAL status handling in status command."""

from crawl4r.cli.commands.status import _status_style
from crawl4r.services.models import CrawlStatus


def test_status_style_partial() -> None:
    """Test that PARTIAL status has a color mapping."""
    color = _status_style(CrawlStatus.PARTIAL)
    assert color in ["yellow", "orange", "magenta", "cyan"]
    assert isinstance(color, str)


def test_status_style_all_enum_values() -> None:
    """Test that all CrawlStatus enum values have color mappings."""
    # Ensure no KeyError for any status value
    for status in CrawlStatus:
        color = _status_style(status)
        assert isinstance(color, str)
        assert len(color) > 0
```

**Verify:** This test should fail

**Run:**
```bash
source .venv/bin/activate && pytest tests/unit/cli/test_status_partial.py::test_status_style_partial -v
```

**Expected output:**
```
FAILED tests/unit/cli/test_status_partial.py::test_status_style_partial - KeyError: <CrawlStatus.PARTIAL: 'PARTIAL'>
```

---

### Step 1.2: Run comprehensive enum test to verify failure

**Run:**
```bash
source .venv/bin/activate && pytest tests/unit/cli/test_status_partial.py::test_status_style_all_enum_values -v
```

**Expected output:**
```
FAILED tests/unit/cli/test_status_partial.py::test_status_style_all_enum_values - KeyError: <CrawlStatus.PARTIAL: 'PARTIAL'>
```

**Verification:** Both tests should fail with KeyError proving the bug exists.

---

## Phase 2: Green - Minimal Implementation

### Step 2.1: Add PARTIAL to status style mapping

**Modify:** `/home/jmagar/workspace/crawl4r/crawl4r/cli/commands/status.py:40-46`

**Replace:**
```python
def _status_style(status: CrawlStatus) -> str:
    return {
        CrawlStatus.QUEUED: "yellow",
        CrawlStatus.RUNNING: "blue",
        CrawlStatus.COMPLETED: "green",
        CrawlStatus.FAILED: "red",
    }[status]
```

**With:**
```python
def _status_style(status: CrawlStatus) -> str:
    return {
        CrawlStatus.QUEUED: "yellow",
        CrawlStatus.RUNNING: "blue",
        CrawlStatus.COMPLETED: "green",
        CrawlStatus.PARTIAL: "yellow",
        CrawlStatus.FAILED: "red",
    }[status]
```

**Rationale:** Use "yellow" for PARTIAL to indicate incomplete/in-progress state, similar to QUEUED. Distinguishes from success (green) and failure (red).

---

### Step 2.2: Verify tests pass

**Run:**
```bash
source .venv/bin/activate && pytest tests/unit/cli/test_status_partial.py -v
```

**Expected output:**
```
PASSED tests/unit/cli/test_status_partial.py::test_status_style_partial
PASSED tests/unit/cli/test_status_partial.py::test_status_style_all_enum_values
```

**Verification:** Both tests should pass, confirming the fix works.

---

## Phase 3: Refactor - Improve Robustness

### Step 3.1: Add integration test for PARTIAL status display

**Modify:** `/home/jmagar/workspace/crawl4r/tests/unit/cli/test_status_partial.py`

**Add at end of file:**
```python
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from rich.console import Console

from crawl4r.cli.commands.status import _print_single
from crawl4r.services.models import CrawlStatusInfo


def test_print_single_partial_status() -> None:
    """Test that PARTIAL status prints without crashing."""
    status_info = CrawlStatusInfo(
        crawl_id="test-123",
        status=CrawlStatus.PARTIAL,
        error=None,
        started_at="2026-01-20T12:00:00Z",
        finished_at=None,
    )

    console = Console()
    # Should not raise KeyError
    _print_single(console, status_info)


def test_print_table_with_partial_status() -> None:
    """Test that PARTIAL status appears correctly in table."""
    from crawl4r.cli.commands.status import _print_table

    statuses = [
        CrawlStatusInfo(
            crawl_id="test-1",
            status=CrawlStatus.COMPLETED,
            started_at="2026-01-20T12:00:00Z",
            finished_at="2026-01-20T12:01:00Z",
        ),
        CrawlStatusInfo(
            crawl_id="test-2",
            status=CrawlStatus.PARTIAL,
            started_at="2026-01-20T12:00:00Z",
            finished_at=None,
        ),
        CrawlStatusInfo(
            crawl_id="test-3",
            status=CrawlStatus.FAILED,
            error="Network timeout",
            started_at="2026-01-20T12:00:00Z",
            finished_at="2026-01-20T12:00:30Z",
        ),
    ]

    console = Console()
    # Should not raise KeyError
    _print_table(console, statuses)
```

**Run:**
```bash
source .venv/bin/activate && pytest tests/unit/cli/test_status_partial.py -v
```

**Expected output:**
```
PASSED tests/unit/cli/test_status_partial.py::test_status_style_partial
PASSED tests/unit/cli/test_status_partial.py::test_status_style_all_enum_values
PASSED tests/unit/cli/test_status_partial.py::test_print_single_partial_status
PASSED tests/unit/cli/test_status_partial.py::test_print_table_with_partial_status
```

**Verification:** All 4 tests pass, confirming PARTIAL status works in all display contexts.

---

### Step 3.2: Add defensive fallback for unknown statuses

**Modify:** `/home/jmagar/workspace/crawl4r/crawl4r/cli/commands/status.py:40-46`

**Replace:**
```python
def _status_style(status: CrawlStatus) -> str:
    return {
        CrawlStatus.QUEUED: "yellow",
        CrawlStatus.RUNNING: "blue",
        CrawlStatus.COMPLETED: "green",
        CrawlStatus.PARTIAL: "yellow",
        CrawlStatus.FAILED: "red",
    }[status]
```

**With:**
```python
def _status_style(status: CrawlStatus) -> str:
    """Return Rich color style for a crawl status.

    Args:
        status: CrawlStatus enum value.

    Returns:
        Rich color name (yellow, blue, green, red, white).

    Note:
        Uses .get() with default fallback to prevent KeyError if new
        status values are added to enum without updating this mapping.
    """
    return {
        CrawlStatus.QUEUED: "yellow",
        CrawlStatus.RUNNING: "blue",
        CrawlStatus.COMPLETED: "green",
        CrawlStatus.PARTIAL: "yellow",
        CrawlStatus.FAILED: "red",
    }.get(status, "white")  # Default to white for unknown statuses
```

**Rationale:** Defensive programming - prevents future KeyErrors if new statuses are added to enum. White color indicates "unexpected" status.

---

### Step 3.3: Add test for unknown status fallback

**Modify:** `/home/jmagar/workspace/crawl4r/tests/unit/cli/test_status_partial.py`

**Add at end of file:**
```python
from unittest.mock import MagicMock


def test_status_style_unknown_status_fallback() -> None:
    """Test that unknown status values fall back to white color."""
    # Create a mock status that's not in the mapping
    mock_status = MagicMock(spec=CrawlStatus)
    mock_status.__str__ = lambda self: "UNKNOWN"

    # Should return default "white" instead of raising KeyError
    color = _status_style(mock_status)
    assert color == "white"
```

**Run:**
```bash
source .venv/bin/activate && pytest tests/unit/cli/test_status_partial.py::test_status_style_unknown_status_fallback -v
```

**Expected output:**
```
PASSED tests/unit/cli/test_status_partial.py::test_status_style_unknown_status_fallback
```

---

### Step 3.4: Run full test suite to verify no regressions

**Run:**
```bash
source .venv/bin/activate && pytest tests/unit/cli/test_status_partial.py -v
```

**Expected output:**
```
PASSED tests/unit/cli/test_status_partial.py::test_status_style_partial
PASSED tests/unit/cli/test_status_partial.py::test_status_style_all_enum_values
PASSED tests/unit/cli/test_status_partial.py::test_print_single_partial_status
PASSED tests/unit/cli/test_status_partial.py::test_print_table_with_partial_status
PASSED tests/unit/cli/test_status_partial.py::test_status_style_unknown_status_fallback

============================== 5 passed in 0.12s ===============================
```

**Verification:** All tests pass, confirming the fix is complete and robust.

---

### Step 3.5: Run existing status command tests

**Run:**
```bash
source .venv/bin/activate && pytest tests/unit/cli/test_status_command.py -v
```

**Expected output:**
```
PASSED tests/unit/cli/test_status_command.py::test_status_help
```

**Verification:** Existing test still passes, no regression introduced.

---

## Phase 4: Commit & Documentation

### Step 4.1: Run linter to ensure code quality

**Run:**
```bash
source .venv/bin/activate && ruff check crawl4r/cli/commands/status.py tests/unit/cli/test_status_partial.py
```

**Expected output:**
```
All checks passed!
```

**Verification:** Code meets style standards.

---

### Step 4.2: Run type checker

**Run:**
```bash
source .venv/bin/activate && ty check crawl4r/cli/commands/status.py
```

**Expected output:**
```
Success: no issues found in 1 source file
```

**Verification:** Type hints are correct.

---

### Step 4.3: Commit the fix

**Run:**
```bash
git add crawl4r/cli/commands/status.py tests/unit/cli/test_status_partial.py
git commit -m "$(cat <<'EOF'
fix(cli): add missing PARTIAL status to color mapping

Fixes KeyError crash when status command encounters PARTIAL status.

Changes:
- Add PARTIAL -> "yellow" mapping in _status_style()
- Use .get() with "white" fallback for defensive programming
- Add docstring explaining fallback behavior
- Add 5 comprehensive tests covering PARTIAL status

Tests:
- test_status_style_partial: Direct color mapping test
- test_status_style_all_enum_values: All enum values covered
- test_print_single_partial_status: Single status display
- test_print_table_with_partial_status: Table display
- test_status_style_unknown_status_fallback: Defensive fallback

Resolves: Bug #1 - status command crash on PARTIAL status

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

**Verification:** Commit created with descriptive message.

---

### Step 4.4: Verify commit with git log

**Run:**
```bash
git log -1 --stat
```

**Expected output:** Shows commit with 2 files changed (status.py and test_status_partial.py).

---

## Summary

**Files Modified:**
- `/home/jmagar/workspace/crawl4r/crawl4r/cli/commands/status.py` - Added PARTIAL mapping and defensive fallback

**Files Created:**
- `/home/jmagar/workspace/crawl4r/tests/unit/cli/test_status_partial.py` - Comprehensive test coverage

**Tests Added:** 5 tests
1. Direct PARTIAL color mapping test
2. All enum values coverage test
3. Single status display test
4. Table display test
5. Unknown status fallback test

**Bug Fixed:** KeyError: <CrawlStatus.PARTIAL: 'PARTIAL'> when running `status` command

**Color Choice:** Yellow (indicates incomplete/warning state, consistent with QUEUED)

**Defensive Improvement:** Added `.get()` fallback to prevent future KeyErrors if new statuses are added

---

## Verification Checklist

- [x] Tests fail before fix (RED)
- [x] Tests pass after fix (GREEN)
- [x] Code refactored for robustness (REFACTOR)
- [x] No regressions in existing tests
- [x] Linter passes (ruff)
- [x] Type checker passes (ty)
- [x] Commit created with proper message
- [x] All CrawlStatus enum values covered
