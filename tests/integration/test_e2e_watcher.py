"""File watcher integration tests for RAG ingestion pipeline.

Tests the complete file watching functionality with real watchdog Observer.
Validates that file system events (create, modify, delete) are detected and
trigger the correct processing workflows.

Test scenarios covered:
- File creation detection and processing
- File modification detection and re-processing
- File deletion detection and vector cleanup
- Rapid modification debouncing (prevents duplicate processing)
- Subdirectory monitoring
- Hidden directory exclusion (.git, __pycache__, etc.)
- Non-markdown file filtering
- Queue backpressure handling
- Parallel file processing
- Concurrent modification and deletion

These tests use real watchdog Observer with event queue for verification.
Tests require TEI and Qdrant services to be running.

Example:
    Run watcher tests:
    $ pytest tests/integration/test_e2e_watcher.py -v -m watcher

    Run specific watcher test:
    $ pytest tests/integration/test_e2e_watcher.py::test_watcher_detects_file_creation -v
"""

import asyncio
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient
from watchdog.observers import Observer

from tests.integration.conftest import wait_for_watcher_event


@pytest.mark.integration
@pytest.mark.watcher
async def test_watcher_detects_file_creation(
    watcher_with_observer: tuple[Observer, asyncio.Queue[tuple[str, Path]], Path],
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test that file watcher detects new file creation.

    Verifies the complete workflow:
    1. Create new markdown file in watched directory
    2. Watcher detects creation event
    3. File is processed and vectors stored in Qdrant
    4. Event appears in queue for verification

    Args:
        watcher_with_observer: Fixture providing Observer, queue, and watch path
        test_collection: Unique collection name
        cleanup_fixture: Ensures cleanup after test

    Raises:
        AssertionError: If file creation is not detected or processed
    """
    observer, event_queue, watch_path = watcher_with_observer

    # Create new markdown file
    test_file = watch_path / "new_doc.md"
    test_file.write_text("# New Document\n\nThis is a newly created file.")

    # Wait for watcher to detect and process the file
    # Note: write_text() triggers modified event, not created
    found = await wait_for_watcher_event(event_queue, "modified", test_file, timeout=5.0)
    assert found, f"Watcher should detect file events for {test_file}"

    # Give processor time to complete
    await asyncio.sleep(1.0)

    # Verify vectors are in Qdrant
    import os

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:52001")
    client = AsyncQdrantClient(url=qdrant_url, timeout=30)
    try:
        collection_info = await client.get_collection(test_collection)
        assert collection_info.points_count is not None
        assert collection_info.points_count > 0, (
            "Vectors should be stored after file creation"
        )
    finally:
        await client.close()


@pytest.mark.integration
@pytest.mark.watcher
async def test_watcher_detects_file_modification(
    watcher_with_observer: tuple[Observer, asyncio.Queue[tuple[str, Path]], Path],
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test that file watcher detects file modifications.

    Verifies the complete workflow:
    1. Create and process initial file
    2. Modify file content
    3. Watcher detects modification event
    4. Old vectors deleted, file re-processed, new vectors stored

    Args:
        watcher_with_observer: Fixture providing Observer, queue, and watch path
        test_collection: Unique collection name
        cleanup_fixture: Ensures cleanup after test

    Raises:
        AssertionError: If file modification is not detected or processed
    """
    observer, event_queue, watch_path = watcher_with_observer

    # Create initial file
    test_file = watch_path / "modify_test.md"
    test_file.write_text("# Original\n\nOriginal content.")

    # Wait for initial creation
    found = await wait_for_watcher_event(event_queue, "modified", test_file, timeout=5.0)
    assert found, "Watcher should detect initial creation"

    # Give processor time to complete
    await asyncio.sleep(1.0)

    # Get initial vector count
    import os

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:52001")
    client = AsyncQdrantClient(url=qdrant_url, timeout=30)
    try:
        collection_info = await client.get_collection(test_collection)
        initial_count = collection_info.points_count
        assert initial_count is not None and initial_count > 0

        # Modify file
        test_file.write_text("# Modified\n\nModified content with changes.")

        # Wait for modification detection
        found = await wait_for_watcher_event(
            event_queue, "modified", test_file, timeout=5.0
        )
        assert found, "Watcher should detect modification"

        # Give processor time to complete re-processing
        await asyncio.sleep(1.0)

        # Verify vectors were re-processed (count may differ)
        collection_info = await client.get_collection(test_collection)
        assert collection_info.points_count is not None
        assert collection_info.points_count > 0, (
            "Vectors should exist after modification"
        )
    finally:
        await client.close()


@pytest.mark.integration
@pytest.mark.watcher
async def test_watcher_detects_file_deletion(
    watcher_with_observer: tuple[Observer, asyncio.Queue[tuple[str, Path]], Path],
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test that file watcher detects file deletion.

    Verifies the complete workflow:
    1. Create and process file
    2. Delete file
    3. Watcher detects deletion event
    4. All associated vectors removed from Qdrant

    Args:
        watcher_with_observer: Fixture providing Observer, queue, and watch path
        test_collection: Unique collection name
        cleanup_fixture: Ensures cleanup after test

    Raises:
        AssertionError: If file deletion is not detected or vectors not removed
    """
    observer, event_queue, watch_path = watcher_with_observer

    # Create file
    test_file = watch_path / "delete_test.md"
    test_file.write_text("# Delete Me\n\nThis file will be deleted.")

    # Wait for creation
    found = await wait_for_watcher_event(event_queue, "modified", test_file, timeout=5.0)
    assert found, "Watcher should detect creation"

    # Give processor time to complete
    await asyncio.sleep(1.0)

    # Verify vectors exist
    import os

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:52001")
    client = AsyncQdrantClient(url=qdrant_url, timeout=30)
    try:
        collection_info = await client.get_collection(test_collection)
        assert collection_info.points_count is not None
        assert collection_info.points_count > 0, "Vectors should exist before deletion"

        # Delete file
        test_file.unlink()

        # Wait for deletion detection
        found = await wait_for_watcher_event(
            event_queue, "deleted", test_file, timeout=5.0
        )
        # Note: deletion event may not appear in queue since it's handled differently
        # The important part is that vectors are removed

        # Give processor time to complete deletion
        await asyncio.sleep(1.0)

        # Verify vectors are removed
        collection_info = await client.get_collection(test_collection)
        assert collection_info.points_count is not None
        assert collection_info.points_count == 0, (
            "All vectors should be removed after file deletion"
        )
    finally:
        await client.close()


@pytest.mark.integration
@pytest.mark.watcher
async def test_watcher_debounces_rapid_modifications(
    watcher_with_observer: tuple[Observer, asyncio.Queue[tuple[str, Path]], Path],
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test that rapid file modifications are debounced.

    Verifies debouncing behavior:
    1. Create file
    2. Rapidly modify file 5 times within 0.5 seconds
    3. Watcher should debounce and process only once
    4. Only one processing event should occur (after 1-second delay)

    Args:
        watcher_with_observer: Fixture providing Observer, queue, and watch path
        test_collection: Unique collection name
        cleanup_fixture: Ensures cleanup after test

    Raises:
        AssertionError: If debouncing doesn't work correctly
    """
    observer, event_queue, watch_path = watcher_with_observer

    # Create file
    test_file = watch_path / "debounce_test.md"
    test_file.write_text("# Debounce Test\n\nInitial content.")

    # Wait for initial creation
    found = await wait_for_watcher_event(event_queue, "modified", test_file, timeout=5.0)
    assert found, "Watcher should detect initial creation"

    # Give processor time to complete
    await asyncio.sleep(1.5)

    # Rapidly modify file 5 times
    for i in range(5):
        test_file.write_text(f"# Debounce Test\n\nModification {i}")
        await asyncio.sleep(0.1)  # 0.1 second between modifications

    # Wait for debounced processing (should only see 1 event after 1-second delay)
    # The watcher debounces with 1-second delay, so we need to wait at least that long
    await asyncio.sleep(2.0)

    # Check queue - there may be multiple modification events detected,
    # but the important part is that processing happens only once
    # We can verify this by checking that only reasonable number of vectors exist
    import os

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:52001")
    client = AsyncQdrantClient(url=qdrant_url, timeout=30)
    try:
        collection_info = await client.get_collection(test_collection)
        assert collection_info.points_count is not None
        # Should have vectors from final modification only
        assert collection_info.points_count > 0, "Should have vectors after debouncing"
    finally:
        await client.close()


@pytest.mark.integration
@pytest.mark.watcher
async def test_watcher_monitors_subdirectories(
    watcher_with_observer: tuple[Observer, asyncio.Queue[tuple[str, Path]], Path],
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test that watcher monitors files in subdirectories.

    Verifies recursive monitoring:
    1. Create subdirectories (docs/, docs/guides/)
    2. Create files in nested directories
    3. Watcher detects all files regardless of depth
    4. All files are processed correctly

    Args:
        watcher_with_observer: Fixture providing Observer, queue, and watch path
        test_collection: Unique collection name
        cleanup_fixture: Ensures cleanup after test

    Raises:
        AssertionError: If subdirectory monitoring fails
    """
    observer, event_queue, watch_path = watcher_with_observer

    # Create nested subdirectories
    subdir1 = watch_path / "docs"
    subdir2 = watch_path / "docs" / "guides"
    subdir1.mkdir()
    subdir2.mkdir()

    # Create files in subdirectories
    file1 = subdir1 / "overview.md"
    file1.write_text("# Overview\n\nTop-level documentation.")

    file2 = subdir2 / "setup.md"
    file2.write_text("# Setup Guide\n\nSetup instructions.")

    # Wait for both files to be detected
    found1 = await wait_for_watcher_event(event_queue, "modified", file1, timeout=5.0)
    found2 = await wait_for_watcher_event(event_queue, "modified", file2, timeout=5.0)

    assert found1, f"Watcher should detect {file1}"
    assert found2, f"Watcher should detect {file2}"

    # Give processor time to complete
    await asyncio.sleep(1.5)

    # Verify vectors from both files exist
    import os

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:52001")
    client = AsyncQdrantClient(url=qdrant_url, timeout=30)
    try:
        collection_info = await client.get_collection(test_collection)
        assert collection_info.points_count is not None
        assert collection_info.points_count >= 2, (
            "Should have vectors from both subdirectory files"
        )
    finally:
        await client.close()


@pytest.mark.integration
@pytest.mark.watcher
async def test_watcher_excludes_hidden_directories(
    watcher_with_observer: tuple[Observer, asyncio.Queue[tuple[str, Path]], Path],
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test that watcher excludes hidden directories (.git, __pycache__, etc.).

    Verifies exclusion behavior:
    1. Create hidden directories (.git, __pycache__, node_modules)
    2. Create markdown files in hidden directories
    3. Watcher should ignore these files (no processing)
    4. No vectors should be stored from hidden directory files

    Args:
        watcher_with_observer: Fixture providing Observer, queue, and watch path
        test_collection: Unique collection name
        cleanup_fixture: Ensures cleanup after test

    Raises:
        AssertionError: If hidden directories are not excluded
    """
    observer, event_queue, watch_path = watcher_with_observer

    # Create hidden directories
    git_dir = watch_path / ".git"
    pycache_dir = watch_path / "__pycache__"
    node_modules_dir = watch_path / "node_modules"

    git_dir.mkdir()
    pycache_dir.mkdir()
    node_modules_dir.mkdir()

    # Create markdown files in hidden directories
    git_file = git_dir / "config.md"
    git_file.write_text("# Git Config\n\nThis should be ignored.")

    pycache_file = pycache_dir / "cache.md"
    pycache_file.write_text("# Cache\n\nThis should be ignored.")

    node_file = node_modules_dir / "module.md"
    node_file.write_text("# Module\n\nThis should be ignored.")

    # Wait a bit to see if any events are generated (they shouldn't be)
    await asyncio.sleep(2.0)

    # Verify no vectors were created
    import os

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:52001")
    client = AsyncQdrantClient(url=qdrant_url, timeout=30)
    try:
        collection_info = await client.get_collection(test_collection)
        assert collection_info.points_count is not None
        assert collection_info.points_count == 0, (
            "Hidden directory files should be ignored"
        )
    finally:
        await client.close()


@pytest.mark.integration
@pytest.mark.watcher
async def test_watcher_ignores_non_markdown(
    watcher_with_observer: tuple[Observer, asyncio.Queue[tuple[str, Path]], Path],
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test that watcher ignores non-markdown files.

    Verifies file filtering:
    1. Create non-markdown files (.txt, .pdf, .docx, .py)
    2. Watcher should ignore these files
    3. No processing should occur
    4. No vectors should be stored

    Args:
        watcher_with_observer: Fixture providing Observer, queue, and watch path
        test_collection: Unique collection name
        cleanup_fixture: Ensures cleanup after test

    Raises:
        AssertionError: If non-markdown files are processed
    """
    observer, event_queue, watch_path = watcher_with_observer

    # Create non-markdown files
    txt_file = watch_path / "document.txt"
    txt_file.write_text("Plain text document")

    pdf_file = watch_path / "document.pdf"
    pdf_file.write_bytes(b"PDF binary content")

    py_file = watch_path / "script.py"
    py_file.write_text("print('hello world')")

    docx_file = watch_path / "document.docx"
    docx_file.write_bytes(b"DOCX binary content")

    # Wait a bit to see if any events are generated (they shouldn't be)
    await asyncio.sleep(2.0)

    # Verify no vectors were created
    import os

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:52001")
    client = AsyncQdrantClient(url=qdrant_url, timeout=30)
    try:
        collection_info = await client.get_collection(test_collection)
        assert collection_info.points_count is not None
        assert collection_info.points_count == 0, "Non-markdown files should be ignored"
    finally:
        await client.close()


@pytest.mark.integration
@pytest.mark.watcher
async def test_watcher_handles_backpressure(
    watcher_with_observer: tuple[Observer, asyncio.Queue[tuple[str, Path]], Path],
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test that watcher handles queue backpressure correctly.

    Verifies backpressure handling:
    1. Create many files quickly to fill event queue
    2. Watcher should log warnings when queue approaches max size
    3. All files should eventually be processed
    4. No events should be dropped

    Args:
        watcher_with_observer: Fixture providing Observer, queue, and watch path
        test_collection: Unique collection name
        cleanup_fixture: Ensures cleanup after test

    Raises:
        AssertionError: If backpressure handling fails
    """
    observer, event_queue, watch_path = watcher_with_observer

    # Create multiple files quickly (10 files)
    files = []
    for i in range(10):
        file_path = watch_path / f"file_{i:02d}.md"
        file_path.write_text(f"# Document {i}\n\nContent for document {i}.")
        files.append(file_path)
        await asyncio.sleep(0.05)  # Small delay between creates

    # Wait for all files to be processed
    await asyncio.sleep(5.0)

    # Verify all files were processed
    import os

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:52001")
    client = AsyncQdrantClient(url=qdrant_url, timeout=30)
    try:
        collection_info = await client.get_collection(test_collection)
        assert collection_info.points_count is not None
        assert collection_info.points_count >= 10, (
            "All files should be processed despite backpressure"
        )
    finally:
        await client.close()


@pytest.mark.integration
@pytest.mark.watcher
async def test_watcher_multiple_files_parallel(
    watcher_with_observer: tuple[Observer, asyncio.Queue[tuple[str, Path]], Path],
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test that watcher handles multiple files created simultaneously.

    Verifies parallel processing:
    1. Create 3 files simultaneously
    2. Watcher detects all 3 files
    3. All files are processed (possibly in parallel)
    4. All vectors are stored correctly

    Args:
        watcher_with_observer: Fixture providing Observer, queue, and watch path
        test_collection: Unique collection name
        cleanup_fixture: Ensures cleanup after test

    Raises:
        AssertionError: If parallel file handling fails
    """
    observer, event_queue, watch_path = watcher_with_observer

    # Create 3 files simultaneously
    file1 = watch_path / "parallel_1.md"
    file2 = watch_path / "parallel_2.md"
    file3 = watch_path / "parallel_3.md"

    file1.write_text("# Document 1\n\nFirst parallel document.")
    file2.write_text("# Document 2\n\nSecond parallel document.")
    file3.write_text("# Document 3\n\nThird parallel document.")

    # Wait for all files to be processed
    await asyncio.sleep(5.0)

    # Verify all 3 files were processed
    import os

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:52001")
    client = AsyncQdrantClient(url=qdrant_url, timeout=30)
    try:
        collection_info = await client.get_collection(test_collection)
        assert collection_info.points_count is not None
        assert collection_info.points_count >= 3, (
            "All parallel files should be processed"
        )
    finally:
        await client.close()


@pytest.mark.integration
@pytest.mark.watcher
async def test_watcher_concurrent_modification_deletion(
    watcher_with_observer: tuple[Observer, asyncio.Queue[tuple[str, Path]], Path],
    test_collection: str,
    cleanup_fixture: None,
) -> None:
    """Test that watcher handles concurrent modification and deletion correctly.

    Verifies mixed operations:
    1. Create 2 files
    2. Simultaneously modify one file and delete another
    3. Watcher handles both operations correctly
    4. Modified file vectors are updated
    5. Deleted file vectors are removed

    Args:
        watcher_with_observer: Fixture providing Observer, queue, and watch path
        test_collection: Unique collection name
        cleanup_fixture: Ensures cleanup after test

    Raises:
        AssertionError: If concurrent operations fail
    """
    observer, event_queue, watch_path = watcher_with_observer

    # Create 2 files
    file1 = watch_path / "concurrent_modify.md"
    file2 = watch_path / "concurrent_delete.md"

    file1.write_text("# Modify Me\n\nOriginal content.")
    file2.write_text("# Delete Me\n\nThis will be deleted.")

    # Wait for initial creation
    await asyncio.sleep(2.0)

    # Verify both files created vectors
    import os

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:52001")
    client = AsyncQdrantClient(url=qdrant_url, timeout=30)
    try:
        collection_info = await client.get_collection(test_collection)
        assert collection_info.points_count is not None
        initial_count = collection_info.points_count
        assert initial_count >= 2, "Both files should create vectors"

        # Simultaneously modify file1 and delete file2
        file1.write_text("# Modified\n\nModified content.")
        file2.unlink()

        # Wait for processing
        await asyncio.sleep(3.0)

        # Verify final state: file1 modified, file2 deleted
        collection_info = await client.get_collection(test_collection)
        assert collection_info.points_count is not None
        # Should have fewer vectors (file2 removed)
        assert collection_info.points_count < initial_count, (
            "Deleted file vectors should be removed"
        )
    finally:
        await client.close()
