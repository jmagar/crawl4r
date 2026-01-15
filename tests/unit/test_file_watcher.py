"""Unit tests for file watcher with watchdog integration.

Tests for the file watcher that monitors a directory for markdown file changes,
triggers document processing on creation/modification, and handles deletion events.
Includes comprehensive debouncing, filtering, and error handling tests.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from rag_ingestion.file_watcher import FileWatcher


class TestFileWatcherInitialization:
    """Test FileWatcher initialization and setup."""

    def test_initializes_with_watch_folder(self) -> None:
        """Verify watcher initializes with watch folder from config."""
        config = Mock()
        config.watch_folder = Path("/data/docs")
        processor = Mock()

        watcher = FileWatcher(config=config, processor=processor)

        assert watcher.config is config
        assert watcher.processor is processor
        assert watcher.watch_folder == Path("/data/docs")

    def test_requires_processor_dependency(self) -> None:
        """Verify watcher requires processor for document processing."""
        config = Mock()
        config.watch_folder = Path("/data/docs")

        with pytest.raises(TypeError):
            FileWatcher(config=config)  # type: ignore[call-arg]

    def test_validates_watch_folder_exists(self) -> None:
        """Verify watcher validates watch folder exists at initialization."""
        config = Mock()
        config.watch_folder = Path("/nonexistent/folder")
        processor = Mock()

        with pytest.raises(ValueError, match="Watch folder does not exist"):
            FileWatcher(config=config, processor=processor)

    def test_validates_watch_folder_is_directory(self) -> None:
        """Verify watcher validates watch folder is a directory, not a file."""
        config = Mock()
        # Use a file path instead of directory
        config.watch_folder = Path(__file__)  # This test file
        processor = Mock()

        with pytest.raises(ValueError, match="Watch folder is not a directory"):
            FileWatcher(config=config, processor=processor)


class TestMarkdownFileDetection:
    """Test markdown file filtering and detection."""

    def test_detects_md_extension(self) -> None:
        """Verify watcher detects .md file extensions."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = Mock()

        watcher = FileWatcher(config=config, processor=processor)

        # Simulate file event with .md extension
        event = Mock()
        event.src_path = "/tmp/test.md"
        event.is_directory = False

        assert watcher._is_markdown_file(event) is True

    def test_detects_markdown_extension(self) -> None:
        """Verify watcher detects .markdown file extensions."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = Mock()

        watcher = FileWatcher(config=config, processor=processor)

        # Simulate file event with .markdown extension
        event = Mock()
        event.src_path = "/tmp/document.markdown"
        event.is_directory = False

        assert watcher._is_markdown_file(event) is True

    def test_ignores_non_markdown_extensions(self) -> None:
        """Verify watcher ignores non-markdown file extensions (.txt, .py, .json)."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = Mock()

        watcher = FileWatcher(config=config, processor=processor)

        # Test various non-markdown extensions
        for ext in [".txt", ".py", ".json", ".html", ".xml", ".pdf"]:
            event = Mock()
            event.src_path = f"/tmp/test{ext}"
            event.is_directory = False
            assert watcher._is_markdown_file(event) is False

    def test_ignores_directory_events(self) -> None:
        """Verify watcher ignores directory events (even if named .md)."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = Mock()

        watcher = FileWatcher(config=config, processor=processor)

        # Simulate directory event
        event = Mock()
        event.src_path = "/tmp/folder.md"  # Directory named with .md
        event.is_directory = True

        assert watcher._is_markdown_file(event) is False


class TestFileCreationEvents:
    """Test on_created event handler."""

    @pytest.mark.asyncio
    async def test_on_created_triggers_processing(self) -> None:
        """Verify on_created triggers document processing for markdown files."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        # Simulate file creation event
        event = Mock()
        event.src_path = "/tmp/new_doc.md"
        event.is_directory = False

        await watcher.on_created(event)

        # Wait for debounce delay
        import asyncio

        await asyncio.sleep(1.1)

        # Verify processor.process_document was called with correct path
        processor.process_document.assert_called_once()
        call_args = processor.process_document.call_args[0][0]
        assert call_args == Path("/tmp/new_doc.md")

    @pytest.mark.asyncio
    async def test_on_created_ignores_non_markdown(self) -> None:
        """Verify on_created ignores non-markdown file creation."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        # Simulate non-markdown file creation
        event = Mock()
        event.src_path = "/tmp/document.txt"
        event.is_directory = False

        await watcher.on_created(event)

        # Verify processor was NOT called
        processor.process_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_created_handles_processing_errors(self) -> None:
        """Verify on_created handles processing errors gracefully."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()
        processor.process_document.side_effect = RuntimeError("Processing failed")

        watcher = FileWatcher(config=config, processor=processor)

        # Simulate file creation event
        event = Mock()
        event.src_path = "/tmp/error_doc.md"
        event.is_directory = False

        # Should not raise exception
        await watcher.on_created(event)

        # Wait for debounce delay
        import asyncio

        await asyncio.sleep(1.1)

        # Verify error was logged (processor was called but failed)
        processor.process_document.assert_called_once()


class TestFileModificationEvents:
    """Test on_modified event handler."""

    @pytest.mark.asyncio
    async def test_on_modified_triggers_processing(self) -> None:
        """Verify on_modified triggers document processing for markdown files."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        # Simulate file modification event
        event = Mock()
        event.src_path = "/tmp/updated_doc.md"
        event.is_directory = False

        await watcher.on_modified(event)

        # Wait for debounce delay
        import asyncio

        await asyncio.sleep(1.1)

        # Verify processor.process_document was called
        processor.process_document.assert_called_once()
        call_args = processor.process_document.call_args[0][0]
        assert call_args == Path("/tmp/updated_doc.md")

    @pytest.mark.asyncio
    async def test_on_modified_ignores_non_markdown(self) -> None:
        """Verify on_modified ignores non-markdown file modifications."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        # Simulate non-markdown file modification
        event = Mock()
        event.src_path = "/tmp/config.json"
        event.is_directory = False

        await watcher.on_modified(event)

        # Verify processor was NOT called
        processor.process_document.assert_not_called()


class TestFileDeletionEvents:
    """Test on_deleted event handler."""

    @pytest.mark.asyncio
    async def test_on_deleted_removes_vectors(self) -> None:
        """Verify on_deleted removes vectors from Qdrant for markdown files."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = Mock()
        vector_store = AsyncMock()

        watcher = FileWatcher(
            config=config, processor=processor, vector_store=vector_store
        )

        # Simulate file deletion event
        event = Mock()
        event.src_path = "/tmp/deleted_doc.md"
        event.is_directory = False

        await watcher.on_deleted(event)

        # Verify vector_store.delete_by_file was called with relative path
        vector_store.delete_by_file.assert_called_once()
        # Extract file path from call
        call_args = vector_store.delete_by_file.call_args[0][0]
        assert "deleted_doc.md" in str(call_args)

    @pytest.mark.asyncio
    async def test_on_deleted_ignores_non_markdown(self) -> None:
        """Verify on_deleted ignores non-markdown file deletions."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = Mock()
        vector_store = AsyncMock()

        watcher = FileWatcher(
            config=config, processor=processor, vector_store=vector_store
        )

        # Simulate non-markdown file deletion
        event = Mock()
        event.src_path = "/tmp/data.csv"
        event.is_directory = False

        await watcher.on_deleted(event)

        # Verify vector_store was NOT called
        vector_store.delete_by_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_deleted_handles_errors(self) -> None:
        """Verify on_deleted handles deletion errors gracefully."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = Mock()
        vector_store = AsyncMock()
        vector_store.delete_by_file.side_effect = RuntimeError("Deletion failed")

        watcher = FileWatcher(
            config=config, processor=processor, vector_store=vector_store
        )

        # Simulate file deletion event
        event = Mock()
        event.src_path = "/tmp/problem_doc.md"
        event.is_directory = False

        # Should not raise exception
        await watcher.on_deleted(event)

        # Verify error was logged (vector_store was called but failed)
        vector_store.delete_by_file.assert_called_once()


class TestDebouncing:
    """Test debouncing logic to prevent rapid duplicate processing."""

    @pytest.mark.asyncio
    async def test_debounce_rapid_modify_events(self) -> None:
        """Verify rapid modify events are debounced (only process once)."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        # Simulate 5 rapid modification events within 1 second
        event = Mock()
        event.src_path = "/tmp/rapid_edit.md"
        event.is_directory = False

        for _ in range(5):
            await watcher.on_modified(event)

        # Wait for debounce timer to expire (1 second + buffer)
        import asyncio

        await asyncio.sleep(1.2)

        # Verify processor was called only ONCE (not 5 times)
        assert processor.process_document.call_count == 1

    @pytest.mark.asyncio
    async def test_debounce_uses_timer(self) -> None:
        """Verify debouncing uses 1-second delay."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        event = Mock()
        event.src_path = "/tmp/debounce_test.md"
        event.is_directory = False

        # Trigger event
        await watcher.on_modified(event)

        # Verify processor not called immediately
        processor.process_document.assert_not_called()

        # Wait for debounce delay
        import asyncio

        await asyncio.sleep(1.1)

        # Verify processor was called after delay
        processor.process_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_debounce_per_file(self) -> None:
        """Verify debouncing is per-file (different files debounced independently)."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        # Simulate rapid events for TWO different files
        event1 = Mock()
        event1.src_path = "/tmp/file1.md"
        event1.is_directory = False

        event2 = Mock()
        event2.src_path = "/tmp/file2.md"
        event2.is_directory = False

        # Trigger events for both files
        await watcher.on_modified(event1)
        await watcher.on_modified(event2)
        await watcher.on_modified(event1)  # file1 again
        await watcher.on_modified(event2)  # file2 again

        # Wait for debounce timers
        import asyncio

        await asyncio.sleep(1.2)

        # Verify processor called TWICE (once per file, not 4 times)
        assert processor.process_document.call_count == 2

    @pytest.mark.asyncio
    async def test_debounce_cancels_previous_timer(self) -> None:
        """Verify new event cancels previous task before starting new one."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        event = Mock()
        event.src_path = "/tmp/cancel_test.md"
        event.is_directory = False

        # First event
        await watcher.on_modified(event)

        # Second event before first completes (should cancel first)
        await watcher.on_modified(event)

        # Wait for debounce
        import asyncio

        await asyncio.sleep(1.2)

        # Verify processor was called only ONCE (second event cancelled first)
        assert processor.process_document.call_count == 1


class TestWatcherIntegration:
    """Test FileWatcher integration with DocumentProcessor."""

    @pytest.mark.asyncio
    async def test_uses_processor_for_document_processing(self) -> None:
        """Verify watcher uses processor.process_document for file processing."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        event = Mock()
        event.src_path = "/tmp/integration_test.md"
        event.is_directory = False

        await watcher.on_created(event)

        # Wait for debounce delay
        import asyncio

        await asyncio.sleep(1.1)

        # Verify correct method called
        processor.process_document.assert_called_once_with(
            Path("/tmp/integration_test.md")
        )

    @pytest.mark.asyncio
    async def test_handles_processor_exceptions(self) -> None:
        """Verify watcher handles processor exceptions without crashing."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()
        processor.process_document.side_effect = Exception("Unexpected error")

        watcher = FileWatcher(config=config, processor=processor)

        event = Mock()
        event.src_path = "/tmp/error_test.md"
        event.is_directory = False

        # Should NOT raise exception
        await watcher.on_created(event)

        # Wait for debounce delay
        import asyncio

        await asyncio.sleep(1.1)

        # Verify processor was called (but it failed)
        processor.process_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_correct_file_path(self) -> None:
        """Verify watcher passes correct Path object to processor."""
        config = Mock()
        config.watch_folder = Path("/data/documents")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        event = Mock()
        event.src_path = "/data/documents/subdir/test.md"
        event.is_directory = False

        await watcher.on_modified(event)

        # Wait for debounce delay
        import asyncio

        await asyncio.sleep(1.1)

        # Verify Path object passed correctly
        call_args = processor.process_document.call_args[0][0]
        assert isinstance(call_args, Path)
        assert call_args == Path("/data/documents/subdir/test.md")


class TestErrorHandling:
    """Test comprehensive error handling."""

    @pytest.mark.asyncio
    async def test_handles_file_not_found_during_processing(self) -> None:
        """Verify watcher handles FileNotFoundError gracefully."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()
        processor.process_document.side_effect = FileNotFoundError("File disappeared")

        watcher = FileWatcher(config=config, processor=processor)

        event = Mock()
        event.src_path = "/tmp/vanished.md"
        event.is_directory = False

        # Should not raise exception
        await watcher.on_created(event)

        # Wait for debounce delay
        import asyncio

        await asyncio.sleep(1.1)

        processor.process_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_permission_errors(self) -> None:
        """Verify watcher handles PermissionError gracefully."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()
        processor.process_document.side_effect = PermissionError("Access denied")

        watcher = FileWatcher(config=config, processor=processor)

        event = Mock()
        event.src_path = "/tmp/restricted.md"
        event.is_directory = False

        # Should not raise exception
        await watcher.on_modified(event)

        # Wait for debounce delay
        import asyncio

        await asyncio.sleep(1.1)

        processor.process_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_network_errors_during_deletion(self) -> None:
        """Verify watcher handles network errors during vector deletion."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = Mock()
        vector_store = AsyncMock()
        vector_store.delete_by_file.side_effect = RuntimeError("Network timeout")

        watcher = FileWatcher(
            config=config, processor=processor, vector_store=vector_store
        )

        event = Mock()
        event.src_path = "/tmp/network_error.md"
        event.is_directory = False

        # Should not raise exception
        await watcher.on_deleted(event)

        vector_store.delete_by_file.assert_called_once()


class TestDirectoryExclusions:
    """Test directory exclusion filtering."""

    @pytest.mark.asyncio
    async def test_ignore_git_directory(self) -> None:
        """Verify watcher ignores .git directory events."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        # Simulate event in .git directory
        event = Mock()
        event.src_path = "/tmp/.git/config.md"
        event.is_directory = False

        await watcher.on_created(event)

        # Wait for potential processing
        import asyncio

        await asyncio.sleep(1.1)

        # Verify processor was NOT called (.git should be excluded)
        processor.process_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignore_hidden_directories(self) -> None:
        """Verify watcher ignores hidden directories (.cache, .vscode, etc)."""
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        # Test various hidden directories
        hidden_paths = [
            "/tmp/.cache/test.md",
            "/tmp/.vscode/settings.md",
            "/tmp/.idea/config.md",
        ]

        for path in hidden_paths:
            event = Mock()
            event.src_path = path
            event.is_directory = False

            await watcher.on_modified(event)

        # Wait for potential processing
        import asyncio

        await asyncio.sleep(1.1)

        # Verify processor was NOT called for any hidden directory
        processor.process_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignore_build_directories(self) -> None:
        """Verify watcher ignores build/temp directories.

        Tests exclusion of __pycache__, node_modules, dist, build.
        """
        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        # Test various build/temp directories
        build_paths = [
            "/tmp/__pycache__/module.md",
            "/tmp/node_modules/package/README.md",
            "/tmp/dist/output.md",
            "/tmp/build/artifact.md",
        ]

        for path in build_paths:
            event = Mock()
            event.src_path = path
            event.is_directory = False

            await watcher.on_created(event)

        # Wait for potential processing
        import asyncio

        await asyncio.sleep(1.1)

        # Verify processor was NOT called for any build directory
        processor.process_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignore_symlinks(self) -> None:
        """Verify watcher ignores symlink files."""
        from unittest.mock import patch

        config = Mock()
        config.watch_folder = Path("/tmp")
        processor = AsyncMock()

        watcher = FileWatcher(config=config, processor=processor)

        # Simulate event for symlink file
        event = Mock()
        event.src_path = "/tmp/link.md"
        event.is_directory = False

        # Mock Path.is_symlink to return True for this path
        with patch("pathlib.Path.is_symlink", return_value=True):
            await watcher.on_modified(event)

        # Wait for potential processing
        import asyncio

        await asyncio.sleep(1.1)

        # Verify processor was NOT called (symlinks should be excluded)
        processor.process_document.assert_not_called()


class TestLifecycleHandlers:
    """Test suite for event lifecycle handler methods."""

    @pytest.mark.asyncio
    async def test_handle_create_event(self) -> None:
        """Verify _handle_create calls processor.process_document for new files."""
        config = Mock()
        config.watch_folder = Path("/data/docs")
        processor = AsyncMock()
        vector_store = Mock()

        watcher = FileWatcher(
            config=config, processor=processor, vector_store=vector_store
        )

        file_path = Path("/data/docs/new_file.md")

        # Call _handle_create
        await watcher._handle_create(file_path)

        # Verify processor.process_document was called
        processor.process_document.assert_called_once_with(file_path)

    @pytest.mark.asyncio
    async def test_handle_modify_event_deletes_old_vectors(self) -> None:
        """Verify _handle_modify deletes old vectors before re-ingestion."""
        config = Mock()
        config.watch_folder = Path("/data/docs")
        processor = AsyncMock()
        vector_store = Mock()
        vector_store.delete_by_file = Mock(return_value=5)

        watcher = FileWatcher(
            config=config, processor=processor, vector_store=vector_store
        )

        file_path = Path("/data/docs/modified_file.md")

        # Call _handle_modify
        await watcher._handle_modify(file_path)

        # Verify delete_by_file was called with relative path
        vector_store.delete_by_file.assert_called_once_with("modified_file.md")

    @pytest.mark.asyncio
    async def test_handle_modify_event_reprocesses(self) -> None:
        """Verify _handle_modify calls processor.process_document after deletion."""
        config = Mock()
        config.watch_folder = Path("/data/docs")
        processor = AsyncMock()
        vector_store = Mock()
        vector_store.delete_by_file = Mock(return_value=3)

        watcher = FileWatcher(
            config=config, processor=processor, vector_store=vector_store
        )

        file_path = Path("/data/docs/modified_file.md")

        # Call _handle_modify
        await watcher._handle_modify(file_path)

        # Verify processor.process_document was called after deletion
        processor.process_document.assert_called_once_with(file_path)
        # Verify deletion happened first (check call order)
        assert vector_store.delete_by_file.call_count == 1

    @pytest.mark.asyncio
    async def test_handle_delete_event_removes_vectors(self) -> None:
        """Verify _handle_delete calls vector_store.delete_by_file."""
        config = Mock()
        config.watch_folder = Path("/data/docs")
        processor = AsyncMock()
        vector_store = Mock()
        vector_store.delete_by_file = Mock(return_value=10)

        watcher = FileWatcher(
            config=config, processor=processor, vector_store=vector_store
        )

        file_path = Path("/data/docs/deleted_file.md")

        # Call _handle_delete
        await watcher._handle_delete(file_path)

        # Verify delete_by_file was called with relative path
        vector_store.delete_by_file.assert_called_once_with("deleted_file.md")

    @pytest.mark.asyncio
    async def test_handle_delete_event_logs_count(self) -> None:
        """Verify _handle_delete logs the count of deleted vectors."""
        from unittest.mock import patch

        config = Mock()
        config.watch_folder = Path("/data/docs")
        processor = AsyncMock()
        vector_store = Mock()
        vector_store.delete_by_file = Mock(return_value=15)

        watcher = FileWatcher(
            config=config, processor=processor, vector_store=vector_store
        )

        file_path = Path("/data/docs/deleted_file.md")

        # Mock logger
        with patch.object(watcher, "logger") as mock_logger:
            # Call _handle_delete
            await watcher._handle_delete(file_path)

            # Verify log message contains count
            # Should log something like "Deleted 15 vectors for deleted_file.md"
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "15" in call_args
            assert "deleted_file.md" in call_args
