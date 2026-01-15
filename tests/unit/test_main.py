"""Unit tests for main orchestration module.

Tests for the main entry point that orchestrates the RAG ingestion pipeline:
loading config, validating services, recovering state, processing batches,
and starting the file watcher.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from rag_ingestion.main import main


class TestMainConfigLoading:
    """Test main() config loading and initialization."""

    @patch("rag_ingestion.main.Settings")
    def test_main_loads_config(self, mock_settings: Mock) -> None:
        """Verify Settings loaded from .env on startup.

        Requirements: FR-6 (Environment config)
        Design: Main Entry Point - Configuration Loading
        """
        mock_config = Mock()
        mock_settings.return_value = mock_config

        # This will fail until main.py exists
        with pytest.raises(ModuleNotFoundError):
            main()

        # When implemented, should call Settings()
        # mock_settings.assert_called_once()


class TestMainServiceValidation:
    """Test main() service validation on startup."""

    @patch("rag_ingestion.main.validate_tei_connection")
    @patch("rag_ingestion.main.validate_qdrant_connection")
    @patch("rag_ingestion.main.Settings")
    async def test_main_validates_services(
        self,
        mock_settings: Mock,
        mock_validate_qdrant: AsyncMock,
        mock_validate_tei: AsyncMock,
    ) -> None:
        """Verify TEI and Qdrant validation called on startup.

        Requirements: FR-7 (Service validation)
        Design: Main Entry Point - Startup Validation
        """
        mock_config = Mock()
        mock_settings.return_value = mock_config
        mock_validate_tei.return_value = None
        mock_validate_qdrant.return_value = None

        # This will fail until main.py exists
        with pytest.raises(ModuleNotFoundError):
            await main()

        # When implemented, should call both validators
        # mock_validate_tei.assert_called_once_with(mock_config)
        # mock_validate_qdrant.assert_called_once_with(mock_config)


class TestMainStateRecovery:
    """Test main() state recovery process."""

    @patch("rag_ingestion.main.StateRecovery")
    @patch("rag_ingestion.main.Settings")
    async def test_main_performs_state_recovery(
        self,
        mock_settings: Mock,
        mock_state_recovery_class: Mock,
    ) -> None:
        """Verify StateRecovery.get_files_to_process called on startup.

        Requirements: FR-8 (State recovery)
        Design: Main Entry Point - State Recovery
        """
        mock_config = Mock()
        mock_settings.return_value = mock_config

        mock_state_recovery = Mock()
        mock_state_recovery.get_files_to_process = AsyncMock(return_value=[])
        mock_state_recovery_class.return_value = mock_state_recovery

        # This will fail until main.py exists
        with pytest.raises(ModuleNotFoundError):
            await main()

        # When implemented, should call get_files_to_process
        # mock_state_recovery_class.assert_called_once()
        # mock_state_recovery.get_files_to_process.assert_called_once()


class TestMainBatchProcessing:
    """Test main() batch processing on startup."""

    @patch("rag_ingestion.main.DocumentProcessor")
    @patch("rag_ingestion.main.StateRecovery")
    @patch("rag_ingestion.main.Settings")
    async def test_main_processes_batch(
        self,
        mock_settings: Mock,
        mock_state_recovery_class: Mock,
        mock_processor_class: Mock,
    ) -> None:
        """Verify processor.process_batch called with recovered files.

        Requirements: FR-3 (Batch processing on startup)
        Design: Main Entry Point - Batch Processing
        """
        mock_config = Mock()
        mock_config.watch_folder = "/data/watched_folder"
        mock_settings.return_value = mock_config

        # Setup state recovery to return files
        mock_files = [
            Path("/data/watched_folder/doc1.md"),
            Path("/data/watched_folder/doc2.md"),
        ]
        mock_state_recovery = Mock()
        mock_state_recovery.get_files_to_process = AsyncMock(return_value=mock_files)
        mock_state_recovery_class.return_value = mock_state_recovery

        # Setup processor
        mock_processor = Mock()
        mock_processor.process_batch = AsyncMock()
        mock_processor_class.return_value = mock_processor

        # This will fail until main.py exists
        with pytest.raises(ModuleNotFoundError):
            await main()

        # When implemented, should call process_batch with recovered files
        # mock_processor.process_batch.assert_called_once_with(mock_files)


class TestMainWatcherStartup:
    """Test main() file watcher startup."""

    @patch("rag_ingestion.main.Observer")
    @patch("rag_ingestion.main.FileEventHandler")
    @patch("rag_ingestion.main.StateRecovery")
    @patch("rag_ingestion.main.Settings")
    async def test_main_starts_watcher(
        self,
        mock_settings: Mock,
        mock_state_recovery_class: Mock,
        mock_handler_class: Mock,
        mock_observer_class: Mock,
    ) -> None:
        """Verify watchdog Observer started after batch processing.

        Requirements: AC-1.5 (Watch mode after batch)
        Design: Main Entry Point - Watch Mode
        """
        mock_config = Mock()
        mock_config.watch_folder = "/data/watched_folder"
        mock_settings.return_value = mock_config

        # Setup state recovery to return no files
        mock_state_recovery = Mock()
        mock_state_recovery.get_files_to_process = AsyncMock(return_value=[])
        mock_state_recovery_class.return_value = mock_state_recovery

        # Setup observer
        mock_observer = Mock()
        mock_observer.start = Mock()
        mock_observer.join = Mock()
        mock_observer_class.return_value = mock_observer

        # This will fail until main.py exists
        with pytest.raises(ModuleNotFoundError):
            await main()

        # When implemented, should start observer
        # mock_observer_class.assert_called_once()
        # mock_observer.start.assert_called_once()


class TestMainShutdown:
    """Test main() shutdown handling."""

    @patch("rag_ingestion.main.Observer")
    @patch("rag_ingestion.main.Settings")
    async def test_main_handles_keyboard_interrupt(
        self,
        mock_settings: Mock,
        mock_observer_class: Mock,
    ) -> None:
        """Simulate Ctrl+C, verify clean shutdown.

        Requirements: FR-9 (Graceful shutdown)
        Design: Main Entry Point - Shutdown Handling
        """
        mock_config = Mock()
        mock_settings.return_value = mock_config

        # Setup observer to raise KeyboardInterrupt on join
        mock_observer = Mock()
        mock_observer.start = Mock()
        mock_observer.stop = Mock()
        mock_observer.join = Mock(side_effect=KeyboardInterrupt)
        mock_observer_class.return_value = mock_observer

        # This will fail until main.py exists
        with pytest.raises(ModuleNotFoundError):
            await main()

        # When implemented, should catch KeyboardInterrupt and stop observer
        # Should not re-raise the exception
        # mock_observer.stop.assert_called_once()
