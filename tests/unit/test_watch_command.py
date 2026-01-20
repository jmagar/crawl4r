"""Unit tests for watch command.

Tests the watch command functionality including:
- Help text display
- Folder option handling
- File system scanning
- Error handling
- Async command execution
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from crawl4r.cli.app import app
from crawl4r.cli.commands.watch import _watch_async, get_filesystem_files


def test_watch_help_shows_options() -> None:
    """Test that watch command help displays required options."""
    runner = CliRunner()
    result = runner.invoke(app, ["watch", "--help"])
    assert result.exit_code == 0
    assert "--folder" in result.output


class TestGetFilesystemFiles:
    """Tests for get_filesystem_files() function."""

    def test_finds_markdown_files(self, tmp_path: Path) -> None:
        """Test that markdown files are found in the watch folder."""
        # Create test markdown files
        (tmp_path / "test.md").write_text("# Test")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested/doc.md").write_text("# Doc")

        result = get_filesystem_files(tmp_path)

        assert len(result) == 2
        assert "test.md" in result
        assert "nested/doc.md" in result
        # Verify modification times are datetime objects
        assert all(isinstance(dt, datetime) for dt in result.values())

    def test_excludes_non_markdown_files(self, tmp_path: Path) -> None:
        """Test that non-markdown files are excluded."""
        (tmp_path / "test.md").write_text("# Test")
        (tmp_path / "other.txt").write_text("text")
        (tmp_path / "readme.rst").write_text("README")

        result = get_filesystem_files(tmp_path)

        assert len(result) == 1
        assert "test.md" in result
        assert "other.txt" not in result
        assert "readme.rst" not in result

    def test_handles_nested_directories(self, tmp_path: Path) -> None:
        """Test that nested directories are scanned recursively."""
        # Create deeply nested structure
        nested = tmp_path / "level1" / "level2" / "level3"
        nested.mkdir(parents=True)
        (nested / "deep.md").write_text("# Deep")

        result = get_filesystem_files(tmp_path)

        assert len(result) == 1
        assert "level1/level2/level3/deep.md" in result

    def test_empty_for_nonexistent_folder(self, tmp_path: Path) -> None:
        """Test that nonexistent folders return empty dict."""
        nonexistent = tmp_path / "missing"

        result = get_filesystem_files(nonexistent)

        assert len(result) == 0
        assert result == {}

    def test_empty_for_file_instead_of_folder(self, tmp_path: Path) -> None:
        """Test that passing a file instead of folder returns empty dict."""
        file_path = tmp_path / "file.md"
        file_path.write_text("# File")

        result = get_filesystem_files(file_path)

        assert len(result) == 0

    def test_empty_for_folder_without_markdown(self, tmp_path: Path) -> None:
        """Test that folders with no markdown files return empty dict."""
        (tmp_path / "test.txt").write_text("text")
        (tmp_path / "readme.rst").write_text("README")

        result = get_filesystem_files(tmp_path)

        assert len(result) == 0

    def test_modification_times_are_recent(self, tmp_path: Path) -> None:
        """Test that modification times are current."""
        (tmp_path / "test.md").write_text("# Test")

        result = get_filesystem_files(tmp_path)

        # Modification time should be within last minute
        mod_time = result["test.md"]
        time_diff = datetime.now() - mod_time
        assert time_diff.total_seconds() < 60


class TestWatchAsync:
    """Tests for _watch_async() function."""

    @pytest.mark.asyncio
    async def test_uses_folder_override(self) -> None:
        """Test that folder override is applied to config."""
        custom_folder = Path("/custom/path")

        # Mock all external dependencies
        with patch("crawl4r.cli.commands.watch.Settings") as mock_settings_cls, patch(
            "crawl4r.cli.commands.watch.configure_llama_settings"
        ), patch("crawl4r.cli.commands.watch.TEIClient"), patch(
            "crawl4r.cli.commands.watch.VectorStoreManager"
        ) as mock_vector_store, patch(
            "crawl4r.cli.commands.watch.QualityVerifier"
        ) as mock_quality, patch(
            "crawl4r.cli.commands.watch.StateRecovery"
        ) as mock_recovery, patch(
            "crawl4r.cli.commands.watch.DocumentProcessor"
        ), patch(
            "crawl4r.cli.commands.watch.FileWatcher"
        ), patch(
            "crawl4r.cli.commands.watch.Observer"
        ) as mock_observer_cls, patch(
            "crawl4r.cli.commands.watch.get_filesystem_files",
            return_value={},
        ):
            # Setup mock config instance
            mock_config = MagicMock()
            mock_config.watch_folder = Path("/default/path")
            mock_config.log_level = "INFO"
            mock_config.tei_endpoint = "http://localhost:52000"
            mock_config.qdrant_url = "http://localhost:52001"
            mock_config.collection_name = "test_collection"
            mock_settings_cls.return_value = mock_config

            # Setup mock async methods
            mock_quality_instance = mock_quality.return_value
            mock_quality_instance.validate_tei_connection = AsyncMock()
            mock_quality_instance.validate_qdrant_connection = AsyncMock()

            mock_vector_store_instance = mock_vector_store.return_value
            mock_vector_store_instance.ensure_collection = AsyncMock()

            mock_recovery_instance = mock_recovery.return_value
            mock_recovery_instance.get_files_to_process = AsyncMock(return_value=[])

            # Setup observer mock to raise KeyboardInterrupt on first join, return on second
            mock_observer = mock_observer_cls.return_value
            mock_observer.join.side_effect = [KeyboardInterrupt(), None]

            # Run the function
            await _watch_async(custom_folder)

            # Verify folder was set on config
            assert mock_config.watch_folder == custom_folder

    @pytest.mark.asyncio
    async def test_uses_default_folder_when_no_override(self) -> None:
        """Test that default folder from settings is used when no override."""
        default_folder = Path("/default/path")

        with patch("crawl4r.cli.commands.watch.Settings") as mock_settings_cls, patch(
            "crawl4r.cli.commands.watch.configure_llama_settings"
        ), patch("crawl4r.cli.commands.watch.TEIClient"), patch(
            "crawl4r.cli.commands.watch.VectorStoreManager"
        ) as mock_vector_store, patch(
            "crawl4r.cli.commands.watch.QualityVerifier"
        ) as mock_quality, patch(
            "crawl4r.cli.commands.watch.StateRecovery"
        ) as mock_recovery, patch(
            "crawl4r.cli.commands.watch.DocumentProcessor"
        ), patch(
            "crawl4r.cli.commands.watch.FileWatcher"
        ), patch(
            "crawl4r.cli.commands.watch.Observer"
        ) as mock_observer_cls, patch(
            "crawl4r.cli.commands.watch.get_filesystem_files",
            return_value={},
        ):
            # Setup mock config
            mock_config = MagicMock()
            mock_config.watch_folder = default_folder
            mock_config.log_level = "INFO"
            mock_config.tei_endpoint = "http://localhost:52000"
            mock_config.qdrant_url = "http://localhost:52001"
            mock_config.collection_name = "test_collection"
            mock_settings_cls.return_value = mock_config

            # Setup mock async methods
            mock_quality_instance = mock_quality.return_value
            mock_quality_instance.validate_tei_connection = AsyncMock()
            mock_quality_instance.validate_qdrant_connection = AsyncMock()

            mock_vector_store_instance = mock_vector_store.return_value
            mock_vector_store_instance.ensure_collection = AsyncMock()

            mock_recovery_instance = mock_recovery.return_value
            mock_recovery_instance.get_files_to_process = AsyncMock(return_value=[])

            # Setup observer to raise KeyboardInterrupt
            mock_observer = mock_observer_cls.return_value
            mock_observer.join.side_effect = [KeyboardInterrupt(), None]

            # Run with no override
            await _watch_async(None)

            # Verify default folder was not changed
            assert mock_config.watch_folder == default_folder

    @pytest.mark.asyncio
    async def test_handles_keyboard_interrupt_gracefully(self) -> None:
        """Test that KeyboardInterrupt triggers graceful shutdown."""
        with patch("crawl4r.cli.commands.watch.Settings") as mock_settings_cls, patch(
            "crawl4r.cli.commands.watch.configure_llama_settings"
        ), patch("crawl4r.cli.commands.watch.TEIClient"), patch(
            "crawl4r.cli.commands.watch.VectorStoreManager"
        ) as mock_vector_store, patch(
            "crawl4r.cli.commands.watch.QualityVerifier"
        ) as mock_quality, patch(
            "crawl4r.cli.commands.watch.StateRecovery"
        ) as mock_recovery, patch(
            "crawl4r.cli.commands.watch.DocumentProcessor"
        ), patch(
            "crawl4r.cli.commands.watch.FileWatcher"
        ), patch(
            "crawl4r.cli.commands.watch.Observer"
        ) as mock_observer_cls, patch(
            "crawl4r.cli.commands.watch.get_filesystem_files",
            return_value={},
        ):
            # Setup mock config
            mock_config = MagicMock()
            mock_config.watch_folder = Path("/test")
            mock_config.log_level = "INFO"
            mock_config.tei_endpoint = "http://localhost:52000"
            mock_config.qdrant_url = "http://localhost:52001"
            mock_config.collection_name = "test_collection"
            mock_settings_cls.return_value = mock_config

            # Setup mock async methods
            mock_quality_instance = mock_quality.return_value
            mock_quality_instance.validate_tei_connection = AsyncMock()
            mock_quality_instance.validate_qdrant_connection = AsyncMock()

            mock_vector_store_instance = mock_vector_store.return_value
            mock_vector_store_instance.ensure_collection = AsyncMock()

            mock_recovery_instance = mock_recovery.return_value
            mock_recovery_instance.get_files_to_process = AsyncMock(return_value=[])

            # Setup observer to raise KeyboardInterrupt
            mock_observer = mock_observer_cls.return_value
            mock_observer.join.side_effect = [KeyboardInterrupt(), None]

            # Should not raise exception
            await _watch_async(None)

            # Verify observer cleanup
            mock_observer.stop.assert_called_once()
            assert mock_observer.join.call_count == 2  # Once for wait, once for cleanup

    @pytest.mark.asyncio
    async def test_validates_services_on_startup(self) -> None:
        """Test that TEI and Qdrant connections are validated on startup."""
        with patch("crawl4r.cli.commands.watch.Settings") as mock_settings_cls, patch(
            "crawl4r.cli.commands.watch.configure_llama_settings"
        ), patch("crawl4r.cli.commands.watch.TEIClient") as mock_tei_cls, patch(
            "crawl4r.cli.commands.watch.VectorStoreManager"
        ) as mock_vector_store, patch(
            "crawl4r.cli.commands.watch.QualityVerifier"
        ) as mock_quality_cls, patch(
            "crawl4r.cli.commands.watch.StateRecovery"
        ) as mock_recovery, patch(
            "crawl4r.cli.commands.watch.DocumentProcessor"
        ), patch(
            "crawl4r.cli.commands.watch.FileWatcher"
        ), patch(
            "crawl4r.cli.commands.watch.Observer"
        ) as mock_observer_cls, patch(
            "crawl4r.cli.commands.watch.get_filesystem_files",
            return_value={},
        ):
            # Setup mock config
            mock_config = MagicMock()
            mock_config.watch_folder = Path("/test")
            mock_config.log_level = "INFO"
            mock_config.tei_endpoint = "http://localhost:52000"
            mock_config.qdrant_url = "http://localhost:52001"
            mock_config.collection_name = "test_collection"
            mock_settings_cls.return_value = mock_config

            # Setup mock clients
            mock_tei = mock_tei_cls.return_value
            mock_vector_store_instance = mock_vector_store.return_value

            # Setup mock quality verifier
            mock_quality = mock_quality_cls.return_value
            mock_quality.validate_tei_connection = AsyncMock()
            mock_quality.validate_qdrant_connection = AsyncMock()

            mock_vector_store_instance.ensure_collection = AsyncMock()

            mock_recovery_instance = mock_recovery.return_value
            mock_recovery_instance.get_files_to_process = AsyncMock(return_value=[])

            # Setup observer to raise KeyboardInterrupt
            mock_observer = mock_observer_cls.return_value
            mock_observer.join.side_effect = [KeyboardInterrupt(), None]

            # Run the function
            await _watch_async(None)

            # Verify validations were called
            mock_quality.validate_tei_connection.assert_called_once_with(mock_tei)
            mock_quality.validate_qdrant_connection.assert_called_once()

    @pytest.mark.asyncio
    async def test_processes_batch_on_startup(self) -> None:
        """Test that files are processed in batch on startup if recovery finds them."""
        with patch("crawl4r.cli.commands.watch.Settings") as mock_settings_cls, patch(
            "crawl4r.cli.commands.watch.configure_llama_settings"
        ), patch("crawl4r.cli.commands.watch.TEIClient"), patch(
            "crawl4r.cli.commands.watch.VectorStoreManager"
        ) as mock_vector_store, patch(
            "crawl4r.cli.commands.watch.QualityVerifier"
        ) as mock_quality, patch(
            "crawl4r.cli.commands.watch.StateRecovery"
        ) as mock_recovery, patch(
            "crawl4r.cli.commands.watch.DocumentProcessor"
        ) as mock_processor_cls, patch(
            "crawl4r.cli.commands.watch.FileWatcher"
        ), patch(
            "crawl4r.cli.commands.watch.Observer"
        ) as mock_observer_cls, patch(
            "crawl4r.cli.commands.watch.get_filesystem_files"
        ) as mock_get_files:
            # Setup mock config
            mock_config = MagicMock()
            mock_config.watch_folder = Path("/test")
            mock_config.log_level = "INFO"
            mock_config.tei_endpoint = "http://localhost:52000"
            mock_config.qdrant_url = "http://localhost:52001"
            mock_config.collection_name = "test_collection"
            mock_settings_cls.return_value = mock_config

            # Setup filesystem files
            mock_get_files.return_value = {"doc1.md": datetime.now()}

            # Setup mock async methods
            mock_quality_instance = mock_quality.return_value
            mock_quality_instance.validate_tei_connection = AsyncMock()
            mock_quality_instance.validate_qdrant_connection = AsyncMock()

            mock_vector_store_instance = mock_vector_store.return_value
            mock_vector_store_instance.ensure_collection = AsyncMock()

            # Recovery finds files to process
            mock_recovery_instance = mock_recovery.return_value
            mock_recovery_instance.get_files_to_process = AsyncMock(
                return_value=["doc1.md"]
            )

            # Setup processor to return success
            mock_processor = mock_processor_cls.return_value
            mock_result = MagicMock()
            mock_result.success = True
            mock_processor.process_batch = AsyncMock(return_value=[mock_result])

            # Setup observer to raise KeyboardInterrupt
            mock_observer = mock_observer_cls.return_value
            mock_observer.join.side_effect = [KeyboardInterrupt(), None]

            # Run the function
            await _watch_async(None)

            # Verify batch processing was called
            mock_processor.process_batch.assert_called_once()
            call_args = mock_processor.process_batch.call_args[0][0]
            assert len(call_args) == 1
            assert call_args[0] == Path("/test/doc1.md")

    @pytest.mark.asyncio
    async def test_starts_file_watcher_after_batch(self) -> None:
        """Test that file watcher is started after batch processing."""
        with patch("crawl4r.cli.commands.watch.Settings") as mock_settings_cls, patch(
            "crawl4r.cli.commands.watch.configure_llama_settings"
        ), patch("crawl4r.cli.commands.watch.TEIClient"), patch(
            "crawl4r.cli.commands.watch.VectorStoreManager"
        ) as mock_vector_store, patch(
            "crawl4r.cli.commands.watch.QualityVerifier"
        ) as mock_quality, patch(
            "crawl4r.cli.commands.watch.StateRecovery"
        ) as mock_recovery, patch(
            "crawl4r.cli.commands.watch.DocumentProcessor"
        ), patch(
            "crawl4r.cli.commands.watch.FileWatcher"
        ) as mock_watcher_cls, patch(
            "crawl4r.cli.commands.watch.Observer"
        ) as mock_observer_cls, patch(
            "crawl4r.cli.commands.watch.get_filesystem_files",
            return_value={},
        ):
            # Setup mock config
            mock_config = MagicMock()
            mock_config.watch_folder = Path("/test")
            mock_config.log_level = "INFO"
            mock_config.tei_endpoint = "http://localhost:52000"
            mock_config.qdrant_url = "http://localhost:52001"
            mock_config.collection_name = "test_collection"
            mock_settings_cls.return_value = mock_config

            # Setup mock async methods
            mock_quality_instance = mock_quality.return_value
            mock_quality_instance.validate_tei_connection = AsyncMock()
            mock_quality_instance.validate_qdrant_connection = AsyncMock()

            mock_vector_store_instance = mock_vector_store.return_value
            mock_vector_store_instance.ensure_collection = AsyncMock()

            mock_recovery_instance = mock_recovery.return_value
            mock_recovery_instance.get_files_to_process = AsyncMock(return_value=[])

            # Setup watcher and observer
            mock_watcher = mock_watcher_cls.return_value
            mock_observer = mock_observer_cls.return_value
            mock_observer.join.side_effect = [KeyboardInterrupt(), None]

            # Run the function
            await _watch_async(None)

            # Verify observer was started and scheduled
            mock_observer.schedule.assert_called_once_with(
                mock_watcher, "/test", recursive=True
            )
            mock_observer.start.assert_called_once()
            assert mock_observer.join.call_count == 2  # Once for wait, once for cleanup
