# Fix Watch Command Async/Await Bug Implementation Plan

**Created:** 12:47:00 AM | 01/21/2026 (EST)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix missing `await` in Qdrant validation causing `'coroutine' object has no attribute 'config'` error in watch command.

**Architecture:** The bug is in `crawl4r/core/quality.py` line 189 where `vector_store.client.get_collection()` is called without `await`. Since `client` is `AsyncQdrantClient`, the method returns a coroutine that must be awaited.

**Tech Stack:** Python 3.13, pytest, AsyncQdrantClient from qdrant-client

---

## Phase 1: Write Failing Test

### Step 1: Write test for async client.get_collection call

**Action:** Update test to verify `await` is used with AsyncQdrantClient

**File:** `tests/unit/test_quality.py:114-135`

**Change:**
```python
@pytest.mark.asyncio
async def test_validate_qdrant_connection(self) -> None:
    """Verify successful Qdrant connection validation passes."""
    from crawl4r.core.quality import QualityVerifier

    # Mock vector store with AsyncQdrantClient behavior
    vector_store = Mock()
    vector_store.collection_name = "test-collection"

    # Mock client.get_collection response (returns coroutine)
    mock_collection = Mock()
    mock_collection.config.params.vectors.size = 1024
    mock_collection.config.params.vectors.distance.name = "Cosine"

    # AsyncMock to simulate coroutine behavior
    vector_store.client = AsyncMock()
    vector_store.client.get_collection.return_value = mock_collection

    # Create verifier and validate connection
    verifier = QualityVerifier()
    result = await verifier.validate_qdrant_connection(vector_store)

    # Verify validation passed
    assert result is True
    vector_store.client.get_collection.assert_called_once_with(
        collection_name="test-collection"
    )
```

**Location:** Modify: `tests/unit/test_quality.py:114-135`

---

### Step 2: Update retry test to use AsyncMock

**Action:** Fix retry test to properly mock async client

**File:** `tests/unit/test_quality.py:138-169`

**Change:**
```python
@pytest.mark.asyncio
async def test_validate_qdrant_retries(self) -> None:
    """Verify Qdrant validation retries with exponential backoff."""
    from crawl4r.core.quality import QualityVerifier

    # Mock vector store: 2 failures, then success
    vector_store = Mock()
    vector_store.collection_name = "test-collection"

    # Mock successful collection info on 3rd try
    mock_collection = Mock()
    mock_collection.config.params.vectors.size = 1024
    mock_collection.config.params.vectors.distance.name = "Cosine"

    # Use AsyncMock for async client
    vector_store.client = AsyncMock()
    vector_store.client.get_collection.side_effect = [
        RuntimeError("Connection failed"),
        RuntimeError("Connection failed"),
        mock_collection,  # Success on 3rd
    ]

    # Mock asyncio.sleep to avoid actual delays
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        verifier = QualityVerifier()
        result = await verifier.validate_qdrant_connection(vector_store)

        # Verify succeeded after retries
        assert result is True
        assert vector_store.client.get_collection.call_count == 3

        # Verify retry delays: 5s, 10s (successful on 3rd, no 3rd delay)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(5)  # First retry delay
        mock_sleep.assert_any_call(10)  # Second retry delay
```

**Location:** Modify: `tests/unit/test_quality.py:138-169`

---

### Step 3: Update failure test to use AsyncMock

**Action:** Fix failure test to properly mock async client

**File:** `tests/unit/test_quality.py:172-192`

**Change:**
```python
@pytest.mark.asyncio
async def test_validate_qdrant_exits_on_failure(self) -> None:
    """Verify validation exits with code 1 after max retries."""
    from crawl4r.core.quality import QualityVerifier

    # Mock vector store: all attempts fail
    vector_store = Mock()
    vector_store.collection_name = "test-collection"

    # Use AsyncMock for async client
    vector_store.client = AsyncMock()
    vector_store.client.get_collection.side_effect = RuntimeError(
        "Connection failed"
    )

    # Mock sys.exit to capture exit call
    with (
        patch("crawl4r.core.quality.sys.exit") as mock_exit,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        verifier = QualityVerifier()
        await verifier.validate_qdrant_connection(vector_store)

        # Verify sys.exit(1) was called
        mock_exit.assert_called_once_with(1)
```

**Location:** Modify: `tests/unit/test_quality.py:172-192`

---

### Step 4: Update dimension check test to use AsyncMock

**Action:** Fix dimension check test to properly mock async client

**File:** `tests/unit/test_quality.py:195-213`

**Change:**
```python
@pytest.mark.asyncio
async def test_validate_qdrant_checks_dimensions(self) -> None:
    """Verify validation checks vector size matches expected."""
    from crawl4r.core.quality import QualityVerifier

    # Mock vector store with correct 1024-dimensional vectors
    vector_store = Mock()
    vector_store.collection_name = "test-collection"

    mock_collection = Mock()
    mock_collection.config.params.vectors.size = 1024
    mock_collection.config.params.vectors.distance.name = "Cosine"

    # Use AsyncMock for async client
    vector_store.client = AsyncMock()
    vector_store.client.get_collection.return_value = mock_collection

    verifier = QualityVerifier()
    result = await verifier.validate_qdrant_connection(vector_store)

    # Verify validation passed with correct dimensions
    assert result is True
```

**Location:** Modify: `tests/unit/test_quality.py:195-213`

---

### Step 5: Update wrong dimensions test to use AsyncMock

**Action:** Fix wrong dimensions test to properly mock async client

**File:** `tests/unit/test_quality.py:216-234`

**Change:**
```python
@pytest.mark.asyncio
async def test_validate_qdrant_rejects_wrong_dimensions(self) -> None:
    """Verify validation fails with wrong vector dimensions."""
    from crawl4r.core.quality import QualityVerifier

    # Mock vector store with wrong 768-dimensional vectors
    vector_store = Mock()
    vector_store.collection_name = "test-collection"

    mock_collection = Mock()
    mock_collection.config.params.vectors.size = 768
    mock_collection.config.params.vectors.distance.name = "Cosine"

    # Use AsyncMock for async client
    vector_store.client = AsyncMock()
    vector_store.client.get_collection.return_value = mock_collection

    verifier = QualityVerifier()

    # Verify validation raises error for wrong dimensions
    with pytest.raises(ValueError, match="dimension"):
        await verifier.validate_qdrant_connection(vector_store)
```

**Location:** Modify: `tests/unit/test_quality.py:216-234`

---

### Step 6: Run tests to verify they fail with the bug

**Action:** Run quality tests to confirm they expose the async/await bug

**Command:**
```bash
cd /home/jmagar/workspace/crawl4r && source .venv/bin/activate && pytest tests/unit/test_quality.py::TestQdrantConnectionValidation -v
```

**Expected Output:**
```
FAILED tests/unit/test_quality.py::TestQdrantConnectionValidation::test_validate_qdrant_connection
AttributeError: 'coroutine' object has no attribute 'config'
```

**Reason:** Tests now properly simulate AsyncQdrantClient behavior, exposing the missing `await`

---

### Step 7: Commit test changes (RED phase)

**Action:** Commit failing tests that expose the bug

**Command:**
```bash
cd /home/jmagar/workspace/crawl4r && git add tests/unit/test_quality.py && git commit -m "$(cat <<'EOF'
test(quality): update Qdrant tests to expose async/await bug

Update all Qdrant validation tests to use AsyncMock for client.get_collection,
properly simulating AsyncQdrantClient behavior. Tests now fail with
'coroutine' object has no attribute 'config' error, exposing the missing await
in quality.py line 189.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

**Expected:** Clean commit with failing tests

---

## Phase 2: Fix Implementation

### Step 8: Add await to client.get_collection call

**Action:** Fix the async/await bug in quality.py

**File:** `crawl4r/core/quality.py:189`

**Change:**
```python
                try:
                    collection_info = await vector_store.client.get_collection(  # type: ignore[attr-defined]
                        collection_name=vector_store.collection_name  # type: ignore[attr-defined]
                    )
                    info = {
                        "vector_size": collection_info.config.params.vectors.size,
                        "distance": collection_info.config.params.vectors.distance.name,
                    }
```

**Location:** Modify: `crawl4r/core/quality.py:189`

**Note:** Add `await` before `vector_store.client.get_collection()`

---

### Step 9: Run tests to verify fix (GREEN phase)

**Action:** Run quality tests to confirm they pass

**Command:**
```bash
cd /home/jmagar/workspace/crawl4r && source .venv/bin/activate && pytest tests/unit/test_quality.py::TestQdrantConnectionValidation -v
```

**Expected Output:**
```
PASSED tests/unit/test_quality.py::TestQdrantConnectionValidation::test_validate_qdrant_connection
PASSED tests/unit/test_quality.py::TestQdrantConnectionValidation::test_validate_qdrant_retries
PASSED tests/unit/test_quality.py::TestQdrantConnectionValidation::test_validate_qdrant_exits_on_failure
PASSED tests/unit/test_quality.py::TestQdrantConnectionValidation::test_validate_qdrant_checks_dimensions
PASSED tests/unit/test_quality.py::TestQdrantConnectionValidation::test_validate_qdrant_rejects_wrong_dimensions

5 passed
```

---

### Step 10: Run all quality tests to ensure no regression

**Action:** Run complete quality test suite

**Command:**
```bash
cd /home/jmagar/workspace/crawl4r && source .venv/bin/activate && pytest tests/unit/test_quality.py -v
```

**Expected Output:**
```
tests/unit/test_quality.py::TestTEIConnectionValidation::test_validate_tei_connection PASSED
tests/unit/test_quality.py::TestTEIConnectionValidation::test_validate_tei_retries PASSED
tests/unit/test_quality.py::TestTEIConnectionValidation::test_validate_tei_exits_on_failure PASSED
tests/unit/test_quality.py::TestTEIConnectionValidation::test_validate_tei_checks_dimensions PASSED
tests/unit/test_quality.py::TestTEIConnectionValidation::test_validate_tei_rejects_wrong_dimensions PASSED
tests/unit/test_quality.py::TestQdrantConnectionValidation::test_validate_qdrant_connection PASSED
tests/unit/test_quality.py::TestQdrantConnectionValidation::test_validate_qdrant_retries PASSED
tests/unit/test_quality.py::TestQdrantConnectionValidation::test_validate_qdrant_exits_on_failure PASSED
tests/unit/test_quality.py::TestQdrantConnectionValidation::test_validate_qdrant_checks_dimensions PASSED
tests/unit/test_quality.py::TestQdrantConnectionValidation::test_validate_qdrant_rejects_wrong_dimensions PASSED
tests/unit/test_quality.py::TestRuntimeQualityChecks::test_check_embedding_dimensions PASSED
tests/unit/test_quality.py::TestRuntimeQualityChecks::test_check_embedding_rejects_wrong_dims PASSED
tests/unit/test_quality.py::TestRuntimeQualityChecks::test_sample_embeddings_for_normalization PASSED
tests/unit/test_quality.py::TestRuntimeQualityChecks::test_check_normalization PASSED
tests/unit/test_quality.py::TestRuntimeQualityChecks::test_check_normalization_with_tolerance PASSED
tests/unit/test_quality.py::TestRuntimeQualityChecks::test_check_normalization_warns PASSED

16 passed
```

---

### Step 11: Commit the fix (GREEN phase)

**Action:** Commit implementation fix

**Command:**
```bash
cd /home/jmagar/workspace/crawl4r && git add crawl4r/core/quality.py && git commit -m "$(cat <<'EOF'
fix(quality): add missing await for AsyncQdrantClient.get_collection

Fix 'coroutine' object has no attribute 'config' error in watch command by
adding await to client.get_collection() call. The client attribute is
AsyncQdrantClient which returns coroutines for all methods.

Fixes: watch command startup validation
Related: tests/unit/test_quality.py

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

**Expected:** Clean commit with passing tests

---

## Phase 3: Integration Validation

### Step 12: Test watch command with Qdrant running

**Action:** Verify watch command starts without crashing

**Prerequisite:** Ensure Qdrant is running on localhost:52001

**Command:**
```bash
cd /home/jmagar/workspace/crawl4r && source .venv/bin/activate && timeout 10s python -m crawl4r.cli.app watch --folder /tmp/test-watch || true
```

**Expected Output:**
```
INFO | Starting RAG ingestion pipeline...
INFO | Watch folder: /tmp/test-watch
INFO | Collection: crawl4r
INFO | Validating service connections...
INFO | Validating TEI connection...
INFO | TEI validation passed
INFO | Validating Qdrant connection...
INFO | Qdrant validation passed
INFO | Ensuring collection exists...
INFO | Performing state recovery...
```

**Note:** No `'coroutine' object has no attribute 'config'` error

---

### Step 13: Verify Qdrant collection doesn't exist scenario

**Action:** Test validation when collection doesn't exist yet

**Command:**
```bash
cd /home/jmagar/workspace/crawl4r && source .venv/bin/activate && python -c "
import asyncio
from crawl4r.storage.qdrant import VectorStoreManager
from crawl4r.core.quality import QualityVerifier

async def test():
    vector_store = VectorStoreManager('http://localhost:52001', 'nonexistent-test-collection')
    verifier = QualityVerifier()
    result = await verifier.validate_qdrant_connection(vector_store)
    print(f'Validation result: {result}')

asyncio.run(test())
"
```

**Expected Output:**
```
INFO | Validating Qdrant connection...
INFO | Collection 'nonexistent-test-collection' doesn't exist yet, will be created on first use
Validation result: True
```

---

### Step 14: Run type checking

**Action:** Verify no type errors with the await fix

**Command:**
```bash
cd /home/jmagar/workspace/crawl4r && source .venv/bin/activate && ty check crawl4r/core/quality.py
```

**Expected Output:**
```
Success: no issues found in 1 source file
```

---

### Step 15: Run linting

**Action:** Verify code quality with ruff

**Command:**
```bash
cd /home/jmagar/workspace/crawl4r && source .venv/bin/activate && ruff check crawl4r/core/quality.py
```

**Expected Output:**
```
(no output = success)
```

---

### Step 16: Final commit - Update plan status

**Action:** Move plan to completed folder

**Command:**
```bash
cd /home/jmagar/workspace/crawl4r && git add docs/plans/2026-01-21-fix-watch-async-await-bug.md && git commit -m "$(cat <<'EOF'
docs: mark async/await bug fix plan as complete

Bug #5 fixed: Added missing await for AsyncQdrantClient.get_collection in
quality.py. Watch command now starts successfully with proper Qdrant validation.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
EOF
)"
```

**Expected:** Clean commit with documentation update

---

## Summary

**Total Steps:** 16 steps across 3 phases
**Estimated Time:** 20-25 minutes
**Risk Level:** Low (single-line fix with comprehensive tests)

**Key Changes:**
1. Updated 5 Qdrant validation tests to use AsyncMock
2. Added single `await` keyword in quality.py line 189
3. Verified fix with unit tests, integration test, and type checking

**Verification Criteria:**
- All unit tests pass (16 tests)
- Type checking passes (ty check)
- Linting passes (ruff check)
- Watch command starts without crashing
- No `'coroutine' object has no attribute 'config'` errors

**References:**
- Bug Report: Watch command async/await bug
- File: `crawl4r/core/quality.py:189`
- Tests: `tests/unit/test_quality.py:110-234`
