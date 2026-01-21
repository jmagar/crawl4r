# Medium Priority Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 4 medium-priority issues from comprehensive review: file_path consistency, async health checks, god object refactoring, and protocol consolidation.

**Architecture:** Fix data integrity bug (MEDIUM-000) first, then performance improvement (MEDIUM-001), then code quality refactoring (MEDIUM-002, MEDIUM-003).

**Tech Stack:** Python 3.10+, pytest, httpx, Pydantic, qdrant-client

---

## Task 1: MEDIUM-000 - Fix delete_by_file to Use Absolute Paths

**Files:**
- Modify: `crawl4r/readers/file_watcher.py:492`
- Modify: `crawl4r/readers/file_watcher.py:544`
- Modify: `tests/unit/test_qdrant.py:1541`
- Test: `tests/unit/test_file_watcher.py`

**Context:**
`VectorStoreManager.delete_by_file()` expects absolute paths (filters on `MetadataKeys.FILE_PATH` which is absolute). File watcher passes relative paths, causing silent deletion failures.

### Step 1: Write failing test for absolute path deletion

Add test to `tests/unit/test_file_watcher.py`:

```python
@pytest.mark.asyncio
async def test_handle_modify_passes_absolute_path_to_vector_store(
    tmp_path: Path, mock_processor: MagicMock, mock_vector_store: MagicMock
) -> None:
    """Verify _handle_modify passes absolute path to delete_by_file.

    Ensures:
    - delete_by_file receives absolute path, not relative
    - Matches metadata format from document ingestion
    """
    watch_folder = tmp_path / "docs"
    watch_folder.mkdir()
    test_file = watch_folder / "test.md"
    test_file.write_text("# Test")

    watcher = FileWatcher(
        watch_folder=watch_folder,
        processor=mock_processor,
        vector_store=mock_vector_store,
    )

    await watcher._handle_modify(test_file)

    # Should call delete_by_file with ABSOLUTE path
    mock_vector_store.delete_by_file.assert_called_once_with(str(test_file))
```

Add test for delete event:

```python
@pytest.mark.asyncio
async def test_handle_delete_passes_absolute_path_to_vector_store(
    tmp_path: Path, mock_processor: MagicMock, mock_vector_store: MagicMock
) -> None:
    """Verify _handle_delete passes absolute path to delete_by_file.

    Ensures:
    - delete_by_file receives absolute path, not relative
    - Matches metadata format from document ingestion
    """
    watch_folder = tmp_path / "docs"
    watch_folder.mkdir()
    test_file = watch_folder / "test.md"

    watcher = FileWatcher(
        watch_folder=watch_folder,
        processor=mock_processor,
        vector_store=mock_vector_store,
    )

    await watcher._handle_delete(test_file)

    # Should call delete_by_file with ABSOLUTE path
    mock_vector_store.delete_by_file.assert_called_once_with(str(test_file))
```

### Step 2: Run tests to verify they fail

Run:
```bash
pytest tests/unit/test_file_watcher.py::test_handle_modify_passes_absolute_path_to_vector_store -xvs
pytest tests/unit/test_file_watcher.py::test_handle_delete_passes_absolute_path_to_vector_store -xvs
```

Expected: FAIL - tests expect absolute path, but code passes relative path

### Step 3: Fix _handle_modify to pass absolute path

Modify `crawl4r/readers/file_watcher.py:492`:

```python
async def _handle_modify(self, file_path: Path) -> None:
    """Handle file modification event lifecycle.

    Deletes old vectors before re-processing to prevent stale data.
    Errors are logged but don't crash the watcher.
    """
    try:
        # Delete old vectors if vector store configured
        if self.vector_store is not None:
            deleted_count = await self.vector_store.delete_by_file(
                str(file_path)  # Pass absolute path, not relative
            )
            self.logger.info(
                f"Deleted {deleted_count} old vectors for {file_path}"
            )

        # Re-process document with updated content
        await self.processor.process_document(file_path)
    except FileNotFoundError:
        self.logger.warning(f"File not found during modification: {file_path}")
        raise
    except PermissionError:
        self.logger.error(f"Permission denied reading file: {file_path}")
        raise
    except Exception as e:
        self.logger.error(f"Failed to process modified file {file_path}: {e}")
        raise
```

### Step 4: Fix _handle_delete to pass absolute path

Modify `crawl4r/readers/file_watcher.py:544`:

```python
async def _handle_delete(self, file_path: Path) -> None:
    """Handle file deletion event lifecycle.

    Deletes vectors from store when file is deleted.
    Errors are logged for audit trail.
    """
    try:
        # Return early if no vector store
        if self.vector_store is None:
            return

        # Delete vectors using absolute path
        count = await self.vector_store.delete_by_file(str(file_path))

        # Log deletion count for audit trail
        self.logger.info(f"Deleted {count} vectors for {file_path}")
    except Exception as e:
        self.logger.error(f"Failed to delete vectors for {file_path}: {e}")
```

### Step 5: Update test assertion to use absolute path

Modify `tests/unit/test_qdrant.py:1541`:

```python
# Change from relative to absolute
count = await manager.delete_by_file("/home/user/docs/test.md")
```

### Step 6: Run all tests to verify fixes

Run:
```bash
pytest tests/unit/test_file_watcher.py -xvs
pytest tests/unit/test_qdrant.py::TestDeleteByFile -xvs
```

Expected: All tests PASS

### Step 7: Run integration test to verify end-to-end behavior

Run:
```bash
pytest tests/integration/test_qdrant_integration.py::test_qdrant_delete_by_file_path -xvs
```

Expected: PASS - deletion works with absolute paths

### Step 8: Commit changes

```bash
git add crawl4r/readers/file_watcher.py tests/unit/test_file_watcher.py tests/unit/test_qdrant.py
git commit -m "fix(file_watcher): use absolute paths for delete_by_file

- Pass absolute file_path to VectorStoreManager.delete_by_file()
- Fixes silent deletion failures due to path mismatch
- delete_by_file filters on MetadataKeys.FILE_PATH (absolute)
- Add tests verifying absolute path usage in modify/delete handlers

Resolves: MEDIUM-000"
```

---

## Task 2: MEDIUM-001 - Make Health Check Non-Blocking

**Files:**
- Modify: `crawl4r/readers/crawl4ai.py:253-270`
- Create: `tests/unit/test_crawl4ai_reader_async_init.py`

**Context:**
Synchronous health check in `__init__` blocks event loop for 10 seconds. Use async factory pattern for non-blocking initialization.

### Step 1: Write test for async factory pattern

Create `tests/unit/test_crawl4ai_reader_async_init.py`:

```python
"""Tests for Crawl4AIReader async initialization pattern."""

import pytest
from unittest.mock import AsyncMock, patch

from crawl4r.readers.crawl4ai import Crawl4AIReader


@pytest.mark.asyncio
async def test_create_async_validates_health_without_blocking() -> None:
    """Verify create() factory performs async health check.

    Ensures:
    - Health check uses AsyncClient (non-blocking)
    - Raises ValueError if service unreachable
    """
    with patch("crawl4r.readers.crawl4ai.httpx.AsyncClient") as mock_async:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = AsyncMock(
            return_value=AsyncMock(status_code=200)
        )
        mock_async.return_value = mock_client

        reader = await Crawl4AIReader.create(endpoint_url="http://localhost:52004")

        assert reader is not None
        assert reader.endpoint_url == "http://localhost:52004"


@pytest.mark.asyncio
async def test_create_async_raises_on_unhealthy_service() -> None:
    """Verify create() raises if service unreachable.

    Ensures:
    - Timeout errors raise ValueError
    - Error message includes endpoint URL
    """
    with patch("crawl4r.readers.crawl4ai.httpx.AsyncClient") as mock_async:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        mock_async.return_value = mock_client

        with pytest.raises(ValueError, match="Crawl4AI service unreachable"):
            await Crawl4AIReader.create(endpoint_url="http://localhost:52004")


def test_init_skips_health_check() -> None:
    """Verify __init__ no longer performs blocking health check.

    Ensures:
    - Direct instantiation works without service running
    - Health validation deferred to create() factory
    """
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")
    assert reader.endpoint_url == "http://localhost:52004"
```

### Step 2: Run tests to verify they fail

Run:
```bash
pytest tests/unit/test_crawl4ai_reader_async_init.py -xvs
```

Expected: FAIL - `create()` method doesn't exist yet

### Step 3: Add async factory method to Crawl4AIReader

Modify `crawl4r/readers/crawl4ai.py:253-270`:

```python
def __init__(
    self,
    endpoint_url: str = "http://localhost:52004",
    max_concurrent_requests: int = 10,
    max_retries: int = 3,
    timeout: float = 60.0,
    fail_on_error: bool = False,
    vector_store: Optional["VectorStoreManager"] = None,
    enable_deduplication: bool = True,
) -> None:
    """Initialize Crawl4AIReader (sync, no health check).

    For production use, prefer Crawl4AIReader.create() which validates
    service health asynchronously before returning the reader instance.

    Args:
        endpoint_url: Crawl4AI service URL
        max_concurrent_requests: Concurrent request limit
        max_retries: Retry attempts for failed requests
        timeout: Request timeout in seconds
        fail_on_error: Raise on any crawl error (vs. returning None)
        vector_store: Optional VectorStoreManager for deduplication
        enable_deduplication: Auto-delete existing URL data before crawl
    """
    self.endpoint_url = endpoint_url.rstrip("/")
    self.max_concurrent_requests = max_concurrent_requests
    self.max_retries = max_retries
    self.timeout = timeout
    self.fail_on_error = fail_on_error
    self.vector_store = vector_store
    self.enable_deduplication = enable_deduplication

    # Initialize circuit breaker
    self._circuit_breaker = CircuitBreaker(
        failure_threshold=5, timeout_seconds=60, logger=None
    )

    # Initialize structured logger
    self._logger = get_logger("crawl4r.readers.crawl4ai", log_level="INFO")

@classmethod
async def create(
    cls,
    endpoint_url: str = "http://localhost:52004",
    max_concurrent_requests: int = 10,
    max_retries: int = 3,
    timeout: float = 60.0,
    fail_on_error: bool = False,
    vector_store: Optional["VectorStoreManager"] = None,
    enable_deduplication: bool = True,
) -> "Crawl4AIReader":
    """Create Crawl4AIReader with async health validation.

    Validates service availability before returning reader instance.
    Recommended over direct __init__ for production use.

    Args:
        endpoint_url: Crawl4AI service URL
        max_concurrent_requests: Concurrent request limit
        max_retries: Retry attempts for failed requests
        timeout: Request timeout in seconds
        fail_on_error: Raise on any crawl error (vs. returning None)
        vector_store: Optional VectorStoreManager for deduplication
        enable_deduplication: Auto-delete existing URL data before crawl

    Returns:
        Initialized Crawl4AIReader instance

    Raises:
        ValueError: If service health check fails

    Example:
        >>> reader = await Crawl4AIReader.create(
        ...     endpoint_url="http://localhost:52004"
        ... )
        >>> docs = await reader.aload_data(["https://example.com"])
    """
    reader = cls(
        endpoint_url=endpoint_url,
        max_concurrent_requests=max_concurrent_requests,
        max_retries=max_retries,
        timeout=timeout,
        fail_on_error=fail_on_error,
        vector_store=vector_store,
        enable_deduplication=enable_deduplication,
    )

    # Async health check (non-blocking)
    if not await reader._validate_health():
        raise ValueError(
            f"Crawl4AI service unreachable at {reader.endpoint_url}/health"
        )

    return reader
```

### Step 4: Run tests to verify they pass

Run:
```bash
pytest tests/unit/test_crawl4ai_reader_async_init.py -xvs
```

Expected: All tests PASS

### Step 5: Update existing tests to use new pattern

Find tests that directly instantiate `Crawl4AIReader`:
```bash
grep -r "Crawl4AIReader(" tests/ --include="*.py"
```

Update tests to use factory or mock health check.

### Step 6: Run all Crawl4AIReader tests

Run:
```bash
pytest tests/unit/test_crawl4ai_reader.py -xvs
pytest tests/integration/test_crawl4ai_reader_integration.py -xvs
```

Expected: All tests PASS

### Step 7: Commit changes

```bash
git add crawl4r/readers/crawl4ai.py tests/unit/test_crawl4ai_reader_async_init.py
git commit -m "refactor(crawl4ai): add async factory for non-blocking init

- Add Crawl4AIReader.create() async factory method
- Remove blocking health check from __init__
- Prevents 10-second event loop blocks on initialization
- Existing code can still use __init__ for testing

Resolves: MEDIUM-001"
```

---

## Task 3: MEDIUM-003 - Consolidate VectorStoreProtocol

**Files:**
- Create: `crawl4r/core/interfaces.py`
- Modify: `crawl4r/core/__init__.py`
- Modify: `crawl4r/core/quality.py:34-39`
- Modify: `crawl4r/resilience/recovery.py:40-49`
- Test: `tests/unit/test_interfaces.py`

**Context:**
Two duplicate `VectorStoreProtocol` definitions exist. Consolidate into single canonical protocol in `crawl4r.core.interfaces`.

### Step 1: Write test for consolidated protocol

Create `tests/unit/test_interfaces.py`:

```python
"""Tests for crawl4r.core.interfaces module."""

from typing import Any
import pytest

from crawl4r.core.interfaces import VectorStoreProtocol


class MockVectorStore:
    """Mock implementation of VectorStoreProtocol for testing."""

    async def get_collection_info(self) -> dict[str, Any]:
        """Mock get_collection_info."""
        return {"vector_size": 1024, "distance": "Cosine"}

    async def scroll(self) -> list[dict[str, Any]]:
        """Mock scroll."""
        return [{"id": "1", "payload": {"file_path": "/test.md"}}]


def test_vector_store_protocol_has_required_methods() -> None:
    """Verify VectorStoreProtocol defines expected methods.

    Ensures:
    - Protocol has get_collection_info method
    - Protocol has scroll method
    """
    assert hasattr(VectorStoreProtocol, "get_collection_info")
    assert hasattr(VectorStoreProtocol, "scroll")


@pytest.mark.asyncio
async def test_mock_implements_vector_store_protocol() -> None:
    """Verify mock implementation satisfies protocol.

    Ensures:
    - Mock can be used as VectorStoreProtocol
    - Required methods are callable
    """
    store: VectorStoreProtocol = MockVectorStore()

    info = await store.get_collection_info()
    assert "vector_size" in info

    points = await store.scroll()
    assert isinstance(points, list)
```

### Step 2: Run test to verify it fails

Run:
```bash
pytest tests/unit/test_interfaces.py -xvs
```

Expected: FAIL - `crawl4r.core.interfaces` module doesn't exist

### Step 3: Create interfaces module with consolidated protocol

Create `crawl4r/core/interfaces.py`:

```python
"""Core protocol definitions for crawl4r components.

This module provides canonical protocol definitions used across the codebase
for type checking and dependency injection.
"""

from typing import Any, Protocol


class VectorStoreProtocol(Protocol):
    """Protocol defining expected interface for vector store operations.

    This protocol is implemented by VectorStoreManager and used by components
    that need vector store capabilities (QualityGate, StateRecovery).

    Methods required:
    - get_collection_info: Returns collection metadata (vector_size, distance)
    - scroll: Returns all points in collection for batch processing
    """

    async def get_collection_info(self) -> dict[str, Any]:
        """Get collection metadata including vector_size and distance.

        Returns:
            Dictionary with keys:
            - vector_size: Dimension of vectors (e.g., 1024)
            - distance: Distance metric (e.g., "Cosine")
        """
        ...

    async def scroll(self) -> list[dict[str, Any]]:
        """Scroll through all points in the collection.

        Returns:
            List of point dictionaries with payload data containing metadata
            like file_path, chunk_index, etc.
        """
        ...
```

### Step 4: Export protocol from core package

Modify `crawl4r/core/__init__.py`:

```python
"""Core infrastructure for crawl4r."""

from crawl4r.core.interfaces import VectorStoreProtocol

__all__ = [
    "VectorStoreProtocol",
]
```

### Step 5: Run test to verify it passes

Run:
```bash
pytest tests/unit/test_interfaces.py -xvs
```

Expected: All tests PASS

### Step 6: Update quality.py to use consolidated protocol

Modify `crawl4r/core/quality.py:34-39`:

```python
# Remove duplicate protocol definition
# Delete lines 34-39

# Add import at top of file
from crawl4r.core.interfaces import VectorStoreProtocol
```

### Step 7: Update recovery.py to use consolidated protocol

Modify `crawl4r/resilience/recovery.py:40-49`:

```python
# Remove duplicate protocol definition
# Delete lines 40-49

# Add import at top of file
from crawl4r.core.interfaces import VectorStoreProtocol
```

### Step 8: Run tests to verify no regressions

Run:
```bash
pytest tests/unit/test_quality.py -xvs
pytest tests/unit/test_recovery.py -xvs
```

Expected: All tests PASS

### Step 9: Verify no other protocol duplicates exist

Run:
```bash
grep -r "class VectorStoreProtocol" crawl4r/ --include="*.py"
```

Expected: Only `crawl4r/core/interfaces.py` should match

### Step 10: Commit changes

```bash
git add crawl4r/core/interfaces.py crawl4r/core/__init__.py crawl4r/core/quality.py crawl4r/resilience/recovery.py tests/unit/test_interfaces.py
git commit -m "refactor(core): consolidate VectorStoreProtocol into interfaces

- Create crawl4r.core.interfaces module for canonical protocols
- Move VectorStoreProtocol from quality.py and recovery.py
- Single source of truth for protocol definitions
- Export from crawl4r.core for easy imports

Resolves: MEDIUM-003"
```

---

## Task 4: MEDIUM-002 - Extract Components from Crawl4AIReader

**Files:**
- Create: `crawl4r/readers/crawl/url_validator.py`
- Create: `crawl4r/readers/crawl/http_client.py`
- Create: `crawl4r/readers/crawl/metadata_builder.py`
- Create: `crawl4r/readers/crawl/models.py`
- Create: `crawl4r/readers/crawl/__init__.py`
- Modify: `crawl4r/readers/crawl4ai.py`
- Test: `tests/unit/test_url_validator.py`
- Test: `tests/unit/test_http_client.py`
- Test: `tests/unit/test_metadata_builder.py`
- Test: `tests/unit/test_crawl_models.py`

**Context:**
Crawl4AIReader has 8+ responsibilities (god object). Extract into focused components: UrlValidator, HttpCrawlClient, MetadataBuilder, CrawlResult dataclass.

### Step 4.1: Extract CrawlResult dataclass

#### Step 1: Write test for CrawlResult model

Create `tests/unit/test_crawl_models.py`:

```python
"""Tests for crawl4r.readers.crawl.models module."""

import pytest
from datetime import datetime

from crawl4r.readers.crawl.models import CrawlResult


def test_crawl_result_success_creation() -> None:
    """Verify CrawlResult can represent successful crawl.

    Ensures:
    - Required fields: url, markdown, success
    - Optional fields: title, description, status_code
    - timestamp defaults to current time
    """
    result = CrawlResult(
        url="https://example.com",
        markdown="# Example\n\nContent here",
        title="Example Domain",
        description="Example description",
        status_code=200,
        success=True,
    )

    assert result.url == "https://example.com"
    assert result.markdown == "# Example\n\nContent here"
    assert result.title == "Example Domain"
    assert result.status_code == 200
    assert result.success is True
    assert isinstance(result.timestamp, datetime)


def test_crawl_result_failure_creation() -> None:
    """Verify CrawlResult can represent failed crawl.

    Ensures:
    - error field captures failure reason
    - success=False
    - markdown can be empty on failure
    """
    result = CrawlResult(
        url="https://example.com",
        markdown="",
        success=False,
        error="Connection timeout",
        status_code=0,
    )

    assert result.success is False
    assert result.error == "Connection timeout"
    assert result.markdown == ""
```

#### Step 2: Run test to verify it fails

Run:
```bash
pytest tests/unit/test_crawl_models.py -xvs
```

Expected: FAIL - module doesn't exist

#### Step 3: Create models module with CrawlResult

Create `crawl4r/readers/crawl/models.py`:

```python
"""Data models for web crawling operations."""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class CrawlResult:
    """Result of crawling a single URL.

    Attributes:
        url: Original URL that was crawled
        markdown: Extracted markdown content
        success: Whether crawl succeeded
        title: Page title (optional)
        description: Page description (optional)
        status_code: HTTP status code
        error: Error message if crawl failed
        timestamp: When crawl occurred (UTC)
    """

    url: str
    markdown: str
    success: bool
    title: str | None = None
    description: str | None = None
    status_code: int = 0
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

#### Step 4: Run test to verify it passes

Run:
```bash
pytest tests/unit/test_crawl_models.py -xvs
```

Expected: All tests PASS

#### Step 5: Commit CrawlResult model

```bash
git add crawl4r/readers/crawl/models.py tests/unit/test_crawl_models.py
git commit -m "feat(readers): add CrawlResult dataclass for crawl responses

- Create crawl4r.readers.crawl.models module
- Add CrawlResult with url, markdown, success, metadata fields
- Supports both successful and failed crawl representations

Part of MEDIUM-002 (god object refactoring)"
```

### Step 4.2: Extract UrlValidator

#### Step 1: Write tests for UrlValidator

Create `tests/unit/test_url_validator.py`:

```python
"""Tests for crawl4r.readers.crawl.url_validator module."""

import pytest

from crawl4r.readers.crawl.url_validator import UrlValidator, ValidationError


def test_validate_accepts_valid_https_url() -> None:
    """Verify validator accepts standard HTTPS URLs."""
    validator = UrlValidator()
    validator.validate("https://example.com")  # Should not raise


def test_validate_accepts_valid_http_url() -> None:
    """Verify validator accepts HTTP URLs."""
    validator = UrlValidator()
    validator.validate("http://example.com")  # Should not raise


def test_validate_rejects_non_http_scheme() -> None:
    """Verify validator rejects non-HTTP(S) schemes."""
    validator = UrlValidator()

    with pytest.raises(ValidationError, match="URL must use http or https"):
        validator.validate("ftp://example.com")


def test_validate_rejects_private_ip() -> None:
    """Verify validator rejects private IP addresses (SSRF protection)."""
    validator = UrlValidator(allow_private_ips=False)

    with pytest.raises(ValidationError, match="Private IP addresses not allowed"):
        validator.validate("http://192.168.1.1")


def test_validate_allows_private_ip_when_configured() -> None:
    """Verify validator allows private IPs when explicitly enabled."""
    validator = UrlValidator(allow_private_ips=True)
    validator.validate("http://192.168.1.1")  # Should not raise


def test_validate_rejects_localhost() -> None:
    """Verify validator rejects localhost (SSRF protection)."""
    validator = UrlValidator(allow_localhost=False)

    with pytest.raises(ValidationError, match="Localhost access not allowed"):
        validator.validate("http://localhost:8000")


def test_validate_allows_localhost_when_configured() -> None:
    """Verify validator allows localhost when explicitly enabled."""
    validator = UrlValidator(allow_localhost=True)
    validator.validate("http://localhost:8000")  # Should not raise
```

#### Step 2: Run tests to verify they fail

Run:
```bash
pytest tests/unit/test_url_validator.py -xvs
```

Expected: FAIL - module doesn't exist

#### Step 3: Create UrlValidator

Create `crawl4r/readers/crawl/url_validator.py`:

```python
"""URL validation with SSRF protection for web crawling."""

import ipaddress
from urllib.parse import urlparse


class ValidationError(ValueError):
    """Raised when URL validation fails."""

    pass


class UrlValidator:
    """Validates URLs and prevents SSRF attacks.

    Args:
        allow_private_ips: Allow private IP addresses (e.g., 192.168.x.x)
        allow_localhost: Allow localhost/127.0.0.1
    """

    def __init__(
        self,
        allow_private_ips: bool = False,
        allow_localhost: bool = False,
    ) -> None:
        self.allow_private_ips = allow_private_ips
        self.allow_localhost = allow_localhost

    def validate(self, url: str) -> None:
        """Validate URL and check for SSRF risks.

        Args:
            url: URL to validate

        Raises:
            ValidationError: If URL is invalid or poses SSRF risk
        """
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            raise ValidationError(f"URL must use http or https scheme: {url}")

        # Extract hostname
        hostname = parsed.hostname
        if not hostname:
            raise ValidationError(f"URL missing hostname: {url}")

        # Check for localhost
        if not self.allow_localhost and hostname in ("localhost", "127.0.0.1", "::1"):
            raise ValidationError(f"Localhost access not allowed: {url}")

        # Check for private IP addresses
        if not self.allow_private_ips:
            try:
                ip = ipaddress.ip_address(hostname)
                if ip.is_private:
                    raise ValidationError(f"Private IP addresses not allowed: {url}")
            except ValueError:
                # Not an IP address, hostname is fine
                pass
```

#### Step 4: Run tests to verify they pass

Run:
```bash
pytest tests/unit/test_url_validator.py -xvs
```

Expected: All tests PASS

#### Step 5: Commit UrlValidator

```bash
git add crawl4r/readers/crawl/url_validator.py tests/unit/test_url_validator.py
git commit -m "feat(readers): add UrlValidator with SSRF protection

- Create UrlValidator for URL validation
- SSRF protection: block private IPs and localhost by default
- Configurable allow_private_ips and allow_localhost flags
- Raises ValidationError on invalid or risky URLs

Part of MEDIUM-002 (god object refactoring)"
```

### Step 4.3: Extract MetadataBuilder

#### Step 1: Write tests for MetadataBuilder

Create `tests/unit/test_metadata_builder.py`:

```python
"""Tests for crawl4r.readers.crawl.metadata_builder module."""

import pytest
from datetime import datetime

from crawl4r.readers.crawl.metadata_builder import MetadataBuilder
from crawl4r.readers.crawl.models import CrawlResult
from crawl4r.core.metadata import MetadataKeys


def test_build_creates_complete_metadata() -> None:
    """Verify builder creates metadata with all expected fields."""
    builder = MetadataBuilder()

    result = CrawlResult(
        url="https://example.com/page",
        markdown="# Test Page",
        title="Test Page",
        description="A test page",
        status_code=200,
        success=True,
    )

    metadata = builder.build(result)

    assert metadata[MetadataKeys.SOURCE_URL] == "https://example.com/page"
    assert metadata[MetadataKeys.SOURCE_TYPE] == "web_crawl"
    assert metadata[MetadataKeys.TITLE] == "Test Page"
    assert metadata[MetadataKeys.DESCRIPTION] == "A test page"
    assert metadata[MetadataKeys.STATUS_CODE] == 200
    assert MetadataKeys.CRAWL_TIMESTAMP in metadata


def test_build_handles_missing_optional_fields() -> None:
    """Verify builder handles missing title/description gracefully."""
    builder = MetadataBuilder()

    result = CrawlResult(
        url="https://example.com",
        markdown="# Content",
        success=True,
        status_code=200,
    )

    metadata = builder.build(result)

    assert metadata[MetadataKeys.SOURCE_URL] == "https://example.com"
    assert metadata[MetadataKeys.TITLE] == ""
    assert metadata[MetadataKeys.DESCRIPTION] == ""
```

#### Step 2: Run tests to verify they fail

Run:
```bash
pytest tests/unit/test_metadata_builder.py -xvs
```

Expected: FAIL - module doesn't exist

#### Step 3: Create MetadataBuilder

Create `crawl4r/readers/crawl/metadata_builder.py`:

```python
"""Builds document metadata from crawl results."""

from crawl4r.core.metadata import MetadataKeys
from crawl4r.readers.crawl.models import CrawlResult


class MetadataBuilder:
    """Builds LlamaIndex document metadata from CrawlResult."""

    def build(self, result: CrawlResult) -> dict[str, str | int]:
        """Build metadata dictionary from crawl result.

        Args:
            result: CrawlResult from web crawl

        Returns:
            Metadata dictionary with source_url, title, description, etc.
        """
        return {
            MetadataKeys.SOURCE_URL: result.url,
            MetadataKeys.SOURCE_TYPE: "web_crawl",
            MetadataKeys.TITLE: result.title or "",
            MetadataKeys.DESCRIPTION: result.description or "",
            MetadataKeys.STATUS_CODE: result.status_code,
            MetadataKeys.CRAWL_TIMESTAMP: result.timestamp.isoformat(),
        }
```

#### Step 4: Run tests to verify they pass

Run:
```bash
pytest tests/unit/test_metadata_builder.py -xvs
```

Expected: All tests PASS

#### Step 5: Commit MetadataBuilder

```bash
git add crawl4r/readers/crawl/metadata_builder.py tests/unit/test_metadata_builder.py
git commit -m "feat(readers): add MetadataBuilder for crawl results

- Create MetadataBuilder to convert CrawlResult to metadata dict
- Uses MetadataKeys constants for consistency
- Handles optional fields (title, description)
- ISO format for crawl_timestamp

Part of MEDIUM-002 (god object refactoring)"
```

### Step 4.4: Extract HttpCrawlClient

#### Step 1: Write tests for HttpCrawlClient

Create `tests/unit/test_http_client.py`:

```python
"""Tests for crawl4r.readers.crawl.http_client module."""

import pytest
from unittest.mock import AsyncMock, patch

from crawl4r.readers.crawl.http_client import HttpCrawlClient
from crawl4r.readers.crawl.models import CrawlResult


@pytest.mark.asyncio
async def test_crawl_returns_success_result() -> None:
    """Verify client returns CrawlResult on successful crawl."""
    with patch("crawl4r.readers.crawl.http_client.httpx.AsyncClient") as mock_client:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "markdown": "# Test",
            "title": "Test Page",
            "description": "A test",
        }

        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_resp
        )

        client = HttpCrawlClient(endpoint_url="http://localhost:52004")
        result = await client.crawl("https://example.com")

        assert result.success is True
        assert result.url == "https://example.com"
        assert result.markdown == "# Test"
        assert result.title == "Test Page"


@pytest.mark.asyncio
async def test_crawl_returns_failure_on_timeout() -> None:
    """Verify client returns failure CrawlResult on timeout."""
    with patch("crawl4r.readers.crawl.http_client.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=Exception("Timeout")
        )

        client = HttpCrawlClient(endpoint_url="http://localhost:52004")
        result = await client.crawl("https://example.com")

        assert result.success is False
        assert "Timeout" in result.error
```

#### Step 2: Run tests to verify they fail

Run:
```bash
pytest tests/unit/test_http_client.py -xvs
```

Expected: FAIL - module doesn't exist

#### Step 3: Create HttpCrawlClient

Create `crawl4r/readers/crawl/http_client.py`:

```python
"""HTTP client for Crawl4AI service communication."""

import httpx
from typing import Optional

from crawl4r.readers.crawl.models import CrawlResult


class HttpCrawlClient:
    """HTTP client for Crawl4AI service.

    Args:
        endpoint_url: Crawl4AI service URL
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
    """

    def __init__(
        self,
        endpoint_url: str,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    async def crawl(self, url: str) -> CrawlResult:
        """Crawl URL using Crawl4AI service.

        Args:
            url: URL to crawl

        Returns:
            CrawlResult with markdown and metadata
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.endpoint_url}/md",
                    json={"url": url, "f": "fit"},
                )

                if response.status_code == 200:
                    data = response.json()
                    return CrawlResult(
                        url=url,
                        markdown=data.get("markdown", ""),
                        title=data.get("title"),
                        description=data.get("description"),
                        status_code=200,
                        success=True,
                    )
                else:
                    return CrawlResult(
                        url=url,
                        markdown="",
                        status_code=response.status_code,
                        success=False,
                        error=f"HTTP {response.status_code}",
                    )
        except Exception as e:
            return CrawlResult(
                url=url,
                markdown="",
                status_code=0,
                success=False,
                error=str(e),
            )
```

#### Step 4: Run tests to verify they pass

Run:
```bash
pytest tests/unit/test_http_client.py -xvs
```

Expected: All tests PASS

#### Step 5: Commit HttpCrawlClient

```bash
git add crawl4r/readers/crawl/http_client.py tests/unit/test_http_client.py
git commit -m "feat(readers): add HttpCrawlClient for service communication

- Create HttpCrawlClient for Crawl4AI HTTP requests
- Returns CrawlResult for both success and failure cases
- Handles timeouts and network errors gracefully
- Configurable timeout and retry settings

Part of MEDIUM-002 (god object refactoring)"
```

### Step 4.5: Create package init

#### Step 1: Create __init__.py for crawl package

Create `crawl4r/readers/crawl/__init__.py`:

```python
"""Web crawling components for Crawl4r."""

from crawl4r.readers.crawl.http_client import HttpCrawlClient
from crawl4r.readers.crawl.metadata_builder import MetadataBuilder
from crawl4r.readers.crawl.models import CrawlResult
from crawl4r.readers.crawl.url_validator import UrlValidator, ValidationError

__all__ = [
    "CrawlResult",
    "HttpCrawlClient",
    "MetadataBuilder",
    "UrlValidator",
    "ValidationError",
]
```

#### Step 2: Commit package init

```bash
git add crawl4r/readers/crawl/__init__.py
git commit -m "feat(readers): add crawl package exports

- Export CrawlResult, HttpCrawlClient, MetadataBuilder, UrlValidator
- Single import point for crawl components

Part of MEDIUM-002 (god object refactoring)"
```

### Step 4.6: Refactor Crawl4AIReader to use extracted components

#### Step 1: Update Crawl4AIReader to delegate to components

Modify `crawl4r/readers/crawl4ai.py` (abbreviated - show key changes):

```python
from crawl4r.readers.crawl import (
    CrawlResult,
    HttpCrawlClient,
    MetadataBuilder,
    UrlValidator,
    ValidationError,
)

class Crawl4AIReader(BaseReader):
    """Web crawler using Crawl4AI service (refactored)."""

    def __init__(self, ...) -> None:
        """Initialize reader with extracted components."""
        # ... existing init code ...

        # Delegate to extracted components
        self._url_validator = UrlValidator(
            allow_private_ips=False,
            allow_localhost=True,  # Allow for local dev
        )
        self._http_client = HttpCrawlClient(
            endpoint_url=endpoint_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._metadata_builder = MetadataBuilder()

    async def _crawl_single_url(self, url: str) -> Optional[Document]:
        """Crawl single URL (delegating to components)."""
        try:
            # Validate URL
            self._url_validator.validate(url)
        except ValidationError as e:
            self._logger.error(f"Invalid URL {url}: {e}")
            if self.fail_on_error:
                raise ValueError(f"Invalid URL: {e}") from e
            return None

        # Crawl using HTTP client
        result: CrawlResult = await self._http_client.crawl(url)

        if not result.success:
            self._logger.error(f"Crawl failed for {url}: {result.error}")
            if self.fail_on_error:
                raise ValueError(f"Crawl failed: {result.error}")
            return None

        # Build metadata
        metadata = self._metadata_builder.build(result)

        # Create document
        return Document(text=result.markdown, metadata=metadata)
```

#### Step 2: Run all Crawl4AIReader tests

Run:
```bash
pytest tests/unit/test_crawl4ai_reader.py -xvs
pytest tests/integration/test_crawl4ai_reader_integration.py -xvs
```

Expected: All tests PASS

#### Step 3: Verify line count reduction

Run:
```bash
wc -l crawl4r/readers/crawl4ai.py
```

Expected: Significantly fewer lines (500-600 vs. original 958)

#### Step 4: Commit refactored reader

```bash
git add crawl4r/readers/crawl4ai.py
git commit -m "refactor(readers): delegate Crawl4AIReader to extracted components

- Use UrlValidator for URL validation and SSRF protection
- Use HttpCrawlClient for service communication
- Use MetadataBuilder for metadata construction
- Reduce Crawl4AIReader from 958 to ~600 lines
- Single Responsibility Principle: reader orchestrates, components execute

Resolves: MEDIUM-002"
```

---

## Summary

This plan fixes 4 medium-priority issues:

1. **MEDIUM-000** (Data Integrity): Use absolute paths in `delete_by_file()` - prevents silent deletion failures
2. **MEDIUM-001** (Performance): Async health check - prevents event loop blocking
3. **MEDIUM-003** (Code Quality): Consolidate protocols - single source of truth
4. **MEDIUM-002** (Code Quality): Extract god object - improves testability and maintainability

**Total commits:** 11
**Total test files:** 6
**Estimated time:** 2-3 hours

All changes follow TDD (test-first), use precise file paths, and include exact commands with expected output.
