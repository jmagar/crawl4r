"""Integration test fixtures and configuration.

Provides pytest fixtures for integration tests that interact with real services
(TEI, Qdrant). Includes test configuration overrides, unique collection naming,
and automatic cleanup after tests complete.

Example:
    >>> @pytest.mark.integration
    >>> async def test_real_service(test_config, test_collection):
    ...     # test_config has test endpoints
    ...     # test_collection is unique per test
    ...     pass
"""

import asyncio
import tracemalloc
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest
import respx
from httpx import ConnectError, Response
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from watchdog.observers import Observer

from crawl4r.core.config import Settings
from crawl4r.processing.processor import DocumentProcessor
from crawl4r.readers.file_watcher import FileWatcher
from crawl4r.storage.qdrant import VectorStoreManager
from crawl4r.storage.tei import TEIClient


@pytest.fixture
def test_config(tmp_path: Path) -> Settings:
    """Test configuration with overridden endpoints.

    Creates a Settings instance with test-specific values:
    - Uses localhost endpoints for TEI and Qdrant
    - Points to a temporary watch folder
    - Uses environment variables or defaults for ports

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Settings instance configured for integration tests

    Example:
        >>> def test_example(test_config):
        ...     assert "localhost" in test_config.tei_endpoint
        ...     assert test_config.watch_folder.exists()
    """
    return Settings(
        watch_folder=tmp_path,
        tei_endpoint="http://localhost:52000",
        qdrant_url="http://localhost:52001",
        collection_name="test_collection",
        chunk_size_tokens=512,
        chunk_overlap_percent=15,
        max_concurrent_docs=5,
        queue_max_size=100,
        batch_size=10,
        log_level="DEBUG",
    )


@pytest.fixture
def test_collection() -> str:
    """Generate unique collection name per test.

    Creates a UUID-based collection name to ensure test isolation.
    Each test gets its own collection that won't conflict with others
    running in parallel.

    Returns:
        Unique collection name with 'test_' prefix

    Example:
        >>> def test_example(test_collection):
        ...     # test_collection = "test_12345678-1234-1234-1234-123456789abc"
        ...     assert test_collection.startswith("test_")
    """
    return f"test_{uuid.uuid4().hex[:16]}"


@pytest.fixture
async def cleanup_fixture(
    test_collection: str,
) -> AsyncGenerator[None, None]:
    """Delete test collection after test completes.

    Ensures Qdrant collections created during tests are cleaned up,
    even if the test fails. Connects to Qdrant using the test URL
    and deletes the collection in the teardown phase.

    Args:
        test_collection: Unique collection name from test_collection fixture

    Yields:
        None (setup phase has no value to yield)

    Example:
        >>> @pytest.mark.integration
        >>> async def test_example(test_collection, cleanup_fixture):
        ...     # Create collection and run test
        ...     pass
        ...     # Collection automatically deleted after test
    """
    # Setup phase - nothing to do
    yield

    # Teardown phase - delete test collection
    client = AsyncQdrantClient(url="http://localhost:52001")
    try:
        # Check if collection exists before trying to delete
        collections = await client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if test_collection in collection_names:
            await client.delete_collection(collection_name=test_collection)
    finally:
        await client.close()


@pytest.fixture
def sample_frontmatter_content() -> str:
    """Sample markdown with YAML frontmatter and tags.

    Returns:
        Markdown content with frontmatter metadata and tags

    Example:
        >>> def test_frontmatter(sample_frontmatter_content):
        ...     assert "---" in sample_frontmatter_content
        ...     assert "tags:" in sample_frontmatter_content
    """
    return """---
title: "Integration Testing Guide"
author: "Test Author"
date: "2025-01-15"
tags:
  - testing
  - integration
  - e2e
  - documentation
category: "Testing"
status: "published"
---

# Integration Testing Guide

## Overview

This document covers integration testing best practices for RAG ingestion pipelines.

## Key Concepts

### Test Isolation

Each test should run independently with its own test collection and temporary files.

### Service Dependencies

Integration tests require:
- TEI service for embeddings
- Qdrant for vector storage
- Watchdog for file monitoring

## Implementation

### Basic Test Structure

```python
@pytest.mark.integration
async def test_example(test_config, test_collection):
    # Arrange
    processor = DocumentProcessor(test_config)

    # Act
    await processor.process_document(file_path)

    # Assert
    results = await vector_store.search(query)
    assert len(results) > 0
```

### Error Handling

Always handle exceptions gracefully and clean up resources.

## Best Practices

1. Use unique collection names per test
2. Clean up test data after completion
3. Mock external services when appropriate
4. Test both success and failure paths
"""


@pytest.fixture
def sample_large_document() -> str:
    """Sample large document with 5000+ tokens across 15 sections.

    Returns:
        Large markdown document for chunking tests

    Example:
        >>> def test_large_doc(sample_large_document):
        ...     assert len(sample_large_document.split()) > 5000
    """
    sections = []
    for i in range(1, 16):
        sections.append(f"""
## Section {i}: Performance Testing at Scale

This section covers performance testing methodologies for RAG ingestion pipelines
at scale. When processing thousands of documents, several key factors influence
overall system performance and reliability.

### Throughput Considerations

The document processing throughput depends on several factors: embedding generation
speed, vector storage insertion rate, concurrent processing limits, and available
system resources. A well-optimized pipeline should achieve at least 0.5 documents
per second under normal load conditions.

### Memory Management

Memory consumption patterns vary based on document size and concurrent processing
limits. Large documents with many chunks can cause memory spikes during processing.
Implementing proper memory tracking with tracemalloc helps identify leaks early.
The system should maintain stable memory usage over extended processing runs.

### Concurrency Patterns

Async/await patterns enable efficient I/O-bound operations. The pipeline uses
asyncio for non-blocking operations during embedding generation and vector storage
insertion. Limiting concurrent document processing prevents resource exhaustion.
Typical configurations process 5-10 documents simultaneously.

### Error Recovery

Circuit breaker patterns protect against cascading failures when external services
become unavailable. The system implements exponential backoff with configurable
retry strategies. Failed documents are logged for later reprocessing without
blocking the entire pipeline.

### Quality Verification

Startup validation ensures all required services are healthy before processing
begins. Dimension validation prevents embedding mismatches that would cause
downstream errors. Idempotent operations allow safe reprocessing of documents
without duplicate vector storage.

### Monitoring and Observability

Structured logging provides visibility into pipeline operations and errors.
Key metrics include processing throughput, error rates, queue depths, and
memory usage patterns. Real-time monitoring helps identify performance
degradation before it impacts users.
""")

    return "\n".join(sections)


@pytest.fixture
def sample_unicode_content() -> str:
    """Sample markdown with special characters, emojis, and code blocks.

    Returns:
        Markdown with diverse character encoding scenarios

    Example:
        >>> def test_unicode(sample_unicode_content):
        ...     assert "🚀" in sample_unicode_content
        ...     assert "日本語" in sample_unicode_content
    """
    return """# Unicode Content Test 🚀

## International Characters

### Japanese
こんにちは世界！日本語のテキストです。

### Chinese
你好世界！这是中文文本。

### Arabic
مرحبا بالعام! هذا نص عربي.

### Russian
Привет мир! Это русский текст.

### Emoji Content 🎉

Testing various emoji: 🔥 💻 📊 🎯 ✅ ⚡ 🌟 💡 🚀 🎨

### Special Characters

Mathematical symbols: ∑ ∫ ∂ ∞ ≈ ≠ ≤ ≥ ∀ ∃

Currency symbols: $ € £ ¥ ₹ ₽

Arrows and symbols: → ← ↑ ↓ ⇒ ⇔ © ® ™

## Code Blocks

### Python Example

```python
def process_unicode(text: str) -> list[str]:
    \"\"\"Process text with unicode characters.\"\"\"
    return [char for char in text if ord(char) > 127]

# Test with emoji
result = process_unicode("Hello 🌍 World 🚀")
print(f"Found {len(result)} unicode characters")
```

### SQL Example

```sql
-- Query with unicode column names
SELECT
    用户名 AS username,
    電子郵件 AS email,
    註冊日期 AS registration_date
FROM users_表
WHERE 狀態 = 'active';
```

## Tables with Unicode

| Language | Greeting | Emoji |
|----------|----------|-------|
| English | Hello | 👋 |
| Spanish | Hola | 🇪🇸 |
| French | Bonjour | 🇫🇷 |
| German | Guten Tag | 🇩🇪 |
| Japanese | こんにちは | 🇯🇵 |

## Edge Cases

Combining characters: café, naïve, Zürich, São Paulo

Zero-width characters: ​‌‍ (invisible but present)

Right-to-left text: العربية עברית

Mixed direction: Hello مرحبا World
"""


@pytest.fixture
def generate_n_files() -> Callable[[Path, int, tuple[int, int]], list[Path]]:
    """Factory for generating N test markdown files with size distribution.

    Returns:
        Function that generates N files in tmp_path with content of varying sizes

    Example:
        >>> def test_bulk(tmp_path, generate_n_files):
        ...     files = generate_n_files(tmp_path, 10, (100, 500))
        ...     assert len(files) == 10
        ...     for file_path in files:
        ...         assert file_path.exists()
    """

    def _generate(
        tmp_path: Path, count: int, size_range: tuple[int, int] = (100, 1000)
    ) -> list[Path]:
        """Generate N markdown files with varying sizes.

        Args:
            tmp_path: Directory to create files in
            count: Number of files to generate
            size_range: (min_words, max_words) for content generation

        Returns:
            List of created file paths
        """
        import random

        files = []
        min_words, max_words = size_range

        for i in range(count):
            file_path = tmp_path / f"test_doc_{i:04d}.md"

            # Generate content with random size in range
            word_count = random.randint(min_words, max_words)
            words = [
                f"word{j % 1000}" for j in range(word_count)
            ]  # Cycle through 1000 unique words

            # Structure content with headings
            content = f"# Document {i}\n\n"
            content += f"## Section 1\n\n{' '.join(words[:word_count//2])}\n\n"
            content += f"## Section 2\n\n{' '.join(words[word_count//2:])}\n"

            file_path.write_text(content)
            files.append(file_path)

        return files

    return _generate


@pytest.fixture
def memory_tracker() -> Callable[[], AbstractContextManager[dict[str, int]]]:
    """Context manager for tracking memory usage with tracemalloc.

    Returns:
        Context manager that yields memory statistics

    Example:
        >>> def test_memory(memory_tracker):
        ...     with memory_tracker() as stats:
        ...         # Code under test
        ...         process_large_data()
        ...
        ...     peak_mb = stats['peak_mb']
        ...     assert peak_mb < 1000  # Less than 1GB
    """

    @contextmanager
    def _tracker():  # type: ignore[misc]
        """Track memory usage during context."""
        tracemalloc.start()
        stats: dict[str, int] = {}

        try:
            yield stats
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            stats["current_mb"] = current // 1024 // 1024
            stats["peak_mb"] = peak // 1024 // 1024

    return _tracker


@pytest.fixture
def performance_timer() -> Callable[[float], AbstractContextManager[dict[str, float]]]:
    """Context manager for timing assertions.

    Returns:
        Context manager that yields elapsed time and asserts max duration

    Example:
        >>> def test_performance(performance_timer):
        ...     with performance_timer(5.0) as stats:
        ...         # Code under test - must complete in <5 seconds
        ...         process_data()
        ...
        ...     elapsed = stats['elapsed']
        ...     assert elapsed < 5.0
    """

    @contextmanager
    def _timer(max_seconds: float):  # type: ignore[misc]
        """Time operation and assert max duration.

        Args:
            max_seconds: Maximum allowed duration in seconds
        """
        stats: dict[str, float] = {}
        start = perf_counter()

        try:
            yield stats
        finally:
            elapsed = perf_counter() - start
            stats["elapsed"] = elapsed

            if elapsed > max_seconds:
                pytest.fail(
                    f"Operation took {elapsed:.2f}s, exceeded limit of {max_seconds:.2f}s"
                )

    return _timer


@pytest.fixture
async def watcher_with_observer(
    tmp_path: Path, test_collection: str
) -> AsyncGenerator[tuple[Observer, asyncio.Queue[tuple[str, Path]], Path], None]:
    """Start watchdog Observer with FileWatcher and event queue.

    Initializes a complete file watching setup with:
    - Real watchdog Observer for file system events
    - FileWatcher handler with event queue for test verification
    - All pipeline components (TEI client, vector store, processor)
    - Temporary directory for isolated testing

    The event queue allows tests to verify that file events are detected
    and processed correctly without relying on fixed sleep delays.

    Returns:
        Tuple of (observer, event_queue, watch_path)

    Example:
        >>> async def test_watcher(watcher_with_observer):
        ...     observer, queue, watch_path = watcher_with_observer
        ...
        ...     # Create file - watcher will detect it
        ...     file = watch_path / "test.md"
        ...     file.write_text("# Test")
        ...
        ...     # Wait for event
        ...     event_type, event_path = await asyncio.wait_for(
        ...         queue.get(), timeout=3.0
        ...     )
        ...     assert event_type == "created"
        ...     assert event_path == file

    Notes:
        - Observer is started automatically and stopped in teardown
        - Event queue has maxsize=100 to prevent memory issues
        - Uses localhost endpoints for TEI and Qdrant
        - Creates temporary Qdrant collection that is cleaned up
    """
    import os

    # Get service endpoints
    tei_endpoint = os.getenv("TEI_ENDPOINT", "http://localhost:52000")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:52001")

    # Initialize configuration
    config = Settings(
        watch_folder=tmp_path,
        tei_endpoint=tei_endpoint,
        qdrant_url=qdrant_url,
        collection_name=test_collection,
        chunk_size_tokens=512,
        chunk_overlap_percent=15,
        max_concurrent_docs=5,
        queue_max_size=100,
    )

    # Initialize TEI client
    tei_client = TEIClient(
        endpoint_url=config.tei_endpoint,
        dimensions=1024,
        timeout=30.0,
        max_retries=3,
    )

    # Initialize vector store
    vector_store = VectorStoreManager(
        qdrant_url=config.qdrant_url,
        collection_name=config.collection_name,
        dimensions=1024,
    )

    # Create Qdrant collection for testing
    qdrant_client = AsyncQdrantClient(url=qdrant_url, timeout=30)
    try:
        await qdrant_client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    finally:
        await qdrant_client.close()

    # Initialize processor (uses internal MarkdownNodeParser by default)
    processor = DocumentProcessor(config, vector_store, tei_client=tei_client)

    # Create event queue for test verification
    event_queue: asyncio.Queue[tuple[str, Path]] = asyncio.Queue(maxsize=100)

    # Get the running event loop for bridging sync Observer with async processing
    loop = asyncio.get_running_loop()

    # Initialize file watcher with event queue and event loop
    watcher = FileWatcher(
        config=config,
        processor=processor,
        vector_store=vector_store,
        event_queue=event_queue,
        loop=loop,
    )

    # Create and start Observer
    observer = Observer()
    observer.schedule(watcher, str(tmp_path), recursive=True)
    observer.start()

    # Give observer time to initialize
    await asyncio.sleep(0.1)

    # Yield to test
    yield observer, event_queue, tmp_path

    # Teardown: stop observer
    observer.stop()
    observer.join(timeout=5.0)


async def wait_for_watcher_event(
    event_queue: asyncio.Queue[tuple[str, Path]],
    expected_type: str,
    expected_path: Path,
    timeout: float = 3.0,
) -> bool:
    """Poll event queue for matching event with deadline.

    Polls the event queue until a matching event is found or timeout is reached.
    Uses deadline-based polling to avoid flaky fixed sleep calls.

    This helper enables reliable watcher testing without race conditions. Instead
    of sleeping for a fixed duration and hoping the event arrived, we actively
    poll the queue with a deadline.

    Args:
        event_queue: asyncio.Queue containing (event_type, file_path) tuples
        expected_type: Expected event type ("created", "modified", "deleted")
        expected_path: Expected file path that triggered the event
        timeout: Maximum time to wait in seconds (default: 3.0)

    Returns:
        True if matching event found within timeout, False otherwise

    Example:
        >>> queue = asyncio.Queue()
        >>> path = Path("/tmp/test.md")
        >>>
        >>> # Simulate file creation
        >>> await queue.put(("created", path))
        >>>
        >>> # Wait for event
        >>> found = await wait_for_watcher_event(queue, "created", path, timeout=3.0)
        >>> assert found is True

    Notes:
        - Consumes events from queue while searching
        - Non-matching events are discarded (doesn't re-queue them)
        - Returns False on timeout rather than raising exception
        - Uses asyncio.wait_for for deadline enforcement
    """
    deadline = asyncio.get_event_loop().time() + timeout

    while asyncio.get_event_loop().time() < deadline:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break

        try:
            # Try to get event from queue with remaining timeout
            event_type, file_path = await asyncio.wait_for(
                event_queue.get(), timeout=min(remaining, 0.5)
            )

            # Check if this is the event we're looking for
            if event_type == expected_type and file_path == expected_path:
                return True

            # Not the event we want, continue polling

        except asyncio.TimeoutError:
            # No event available, continue polling
            continue

    # Timeout reached without finding matching event
    return False


@pytest.fixture
def mock_tei_unavailable() -> respx.MockRouter:
    """Mock TEI service to raise connection errors.

    Returns respx mock router that makes TEI endpoint raise ConnectError
    for all requests. This simulates the TEI service being down or unreachable.

    Returns:
        respx.MockRouter configured to raise connection errors

    Example:
        >>> @respx.mock
        >>> async def test_tei_failure(mock_tei_unavailable):
        ...     mock_tei_unavailable  # Activates the mock
        ...
        ...     # TEI requests will raise ConnectError
        ...     with pytest.raises(ConnectError):
        ...         await tei_client.embed(["test"])

    Notes:
        - Use with @respx.mock decorator on test function
        - Simulates complete service unavailability
        - Useful for testing circuit breaker activation
        - Tests retry logic and error handling
    """
    router = respx.mock
    router.post("http://localhost:52000/embed").mock(side_effect=ConnectError)
    router.get("http://localhost:52000/health").mock(side_effect=ConnectError)
    return router


@pytest.fixture
def mock_tei_invalid_dimensions() -> respx.MockRouter:
    """Mock TEI service returning invalid embedding dimensions.

    Returns respx mock router that makes TEI return 512-dimensional embeddings
    instead of expected 1024 dimensions. This tests dimension validation logic.

    Returns:
        respx.MockRouter configured to return wrong dimensions

    Example:
        >>> @respx.mock
        >>> async def test_dimension_mismatch(mock_tei_invalid_dimensions):
        ...     mock_tei_invalid_dimensions  # Activates the mock
        ...
        ...     # TEI returns 512 dims instead of 1024
        ...     result = await tei_client.embed(["test"])
        ...     # Should trigger validation error

    Notes:
        - Use with @respx.mock decorator on test function
        - Returns valid JSON with wrong embedding size
        - Tests dimension validation before storage
        - Prevents dimension mismatches in Qdrant
    """
    router = respx.mock

    # Mock health endpoint to succeed
    router.get("http://localhost:52000/health").mock(
        return_value=Response(200, json={"status": "ok"})
    )

    # Mock embed endpoint to return 512 dims instead of 1024
    def invalid_embeddings(*args: Any, **kwargs: Any) -> Response:
        # Return 512-dimensional embeddings (wrong size)
        embeddings = [[0.0] * 512]  # Should be 1024
        return Response(200, json={"embeddings": embeddings})

    router.post("http://localhost:52000/embed").mock(side_effect=invalid_embeddings)
    return router


@pytest.fixture
def mock_qdrant_unavailable() -> None:
    """Mock Qdrant service to be unavailable.

    Note: This fixture is a placeholder. Qdrant mocking is more complex
    because it uses its own client library (not httpx). Tests should use
    respx to mock the HTTP endpoints directly or skip Qdrant if unavailable.

    Returns:
        None

    Example:
        >>> @pytest.mark.skipif(not qdrant_available(), reason="Qdrant unavailable")
        >>> async def test_qdrant_failure(mock_qdrant_unavailable):
        ...     # Test behavior when Qdrant is down
        ...     pass

    Notes:
        - Qdrant client uses its own HTTP implementation
        - Use respx to mock Qdrant HTTP endpoints if needed
        - Or skip tests when Qdrant service is unavailable
        - Circuit breaker tests can use TEI unavailability instead
    """
    # Qdrant mocking is handled differently due to its custom client
    # Tests should use respx to mock specific Qdrant HTTP endpoints if needed
    pass


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers for integration tests.

    Adds custom markers to pytest's recognized markers for filtering:
    - integration: tests requiring real services (TEI, Qdrant)
    - slow: tests taking >30 seconds (skip by default)
    - watcher: file watcher integration tests
    - performance: load and performance tests

    Args:
        config: pytest configuration object

    Example:
        Run only integration tests:
        $ pytest -m integration

        Run integration tests excluding slow tests:
        $ pytest -m "integration and not slow"

        Run only watcher tests:
        $ pytest -m watcher
    """
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require real services)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (>30s runtime, skip by default)",
    )
    config.addinivalue_line(
        "markers",
        "watcher: marks tests as file watcher integration tests",
    )
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance/load tests",
    )
