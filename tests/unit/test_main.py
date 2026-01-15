"""Unit tests for main orchestration module.

Tests for the main entry point that orchestrates the RAG ingestion pipeline:
loading config, validating services, recovering state, processing batches,
and starting the file watcher.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from rag_ingestion.main import main


# Common patches for all tests
COMMON_PATCHES = [
    "rag_ingestion.main.Observer",
    "rag_ingestion.main.FileWatcher",
    "rag_ingestion.main.StateRecovery",
    "rag_ingestion.main.DocumentProcessor",
    "rag_ingestion.main.VectorStoreManager",
    "rag_ingestion.main.MarkdownChunker",
    "rag_ingestion.main.TEIClient",
    "rag_ingestion.main.QualityVerifier",
    "rag_ingestion.main.Settings",
]


def setup_common_mocks() -> tuple:
    """Setup common mock configuration for all tests.

    Returns:
        Tuple of (config, verifier, observer, state_recovery)
    """
    mock_config = Mock()
    mock_config.watch_folder = Path("/tmp/test")
    mock_config.log_level = "INFO"
    mock_config.tei_endpoint = "http://test:80"
    mock_config.qdrant_url = "http://test:6333"
    mock_config.collection_name = "test"
    mock_config.chunk_size_tokens = 512
    mock_config.chunk_overlap_percent = 15

    mock_verifier = Mock()
    mock_verifier.validate_tei_connection = AsyncMock()
    mock_verifier.validate_qdrant_connection = AsyncMock()

    mock_observer = Mock()
    mock_observer.schedule = Mock()
    mock_observer.start = Mock()
    mock_observer.stop = Mock()
    # join() raises KeyboardInterrupt to simulate Ctrl+C
    mock_observer.join = Mock(side_effect=[KeyboardInterrupt, None])

    mock_state_recovery = Mock()
    mock_state_recovery.get_files_to_process = AsyncMock(return_value=[])

    return mock_config, mock_verifier, mock_observer, mock_state_recovery


class TestMainConfigLoading:
    """Test main() config loading and initialization."""

    @patch("rag_ingestion.main.Observer")
    @patch("rag_ingestion.main.FileWatcher")
    @patch("rag_ingestion.main.StateRecovery")
    @patch("rag_ingestion.main.DocumentProcessor")
    @patch("rag_ingestion.main.VectorStoreManager")
    @patch("rag_ingestion.main.MarkdownChunker")
    @patch("rag_ingestion.main.TEIClient")
    @patch("rag_ingestion.main.QualityVerifier")
    @patch("rag_ingestion.main.Settings")
    async def test_main_loads_config(
        self,
        mock_settings: Mock,
        mock_verifier_class: Mock,
        mock_tei: Mock,
        mock_chunker: Mock,
        mock_vector_store: Mock,
        mock_processor: Mock,
        mock_state_recovery_class: Mock,
        mock_file_watcher: Mock,
        mock_observer_class: Mock,
    ) -> None:
        """Verify Settings loaded from .env on startup.

        Requirements: FR-6 (Environment config)
        Design: Main Entry Point - Configuration Loading
        """
        mock_config, mock_verifier, mock_observer, mock_state_recovery = (
            setup_common_mocks()
        )

        mock_settings.return_value = mock_config
        mock_verifier_class.return_value = mock_verifier
        mock_observer_class.return_value = mock_observer
        mock_state_recovery_class.return_value = mock_state_recovery

        # Call main
        await main()

        # Should call Settings()
        mock_settings.assert_called_once()


class TestMainServiceValidation:
    """Test main() service validation on startup."""

    @patch("rag_ingestion.main.Observer")
    @patch("rag_ingestion.main.FileWatcher")
    @patch("rag_ingestion.main.StateRecovery")
    @patch("rag_ingestion.main.DocumentProcessor")
    @patch("rag_ingestion.main.VectorStoreManager")
    @patch("rag_ingestion.main.MarkdownChunker")
    @patch("rag_ingestion.main.TEIClient")
    @patch("rag_ingestion.main.QualityVerifier")
    @patch("rag_ingestion.main.Settings")
    async def test_main_validates_services(
        self,
        mock_settings: Mock,
        mock_verifier_class: Mock,
        mock_tei: Mock,
        mock_chunker: Mock,
        mock_vector_store: Mock,
        mock_processor: Mock,
        mock_state_recovery_class: Mock,
        mock_file_watcher: Mock,
        mock_observer_class: Mock,
    ) -> None:
        """Verify TEI and Qdrant validation called on startup.

        Requirements: FR-7 (Service validation)
        Design: Main Entry Point - Startup Validation
        """
        mock_config, mock_verifier, mock_observer, mock_state_recovery = (
            setup_common_mocks()
        )

        mock_settings.return_value = mock_config
        mock_verifier_class.return_value = mock_verifier
        mock_observer_class.return_value = mock_observer
        mock_state_recovery_class.return_value = mock_state_recovery

        # Call main
        await main()

        # Should call both validators
        mock_verifier.validate_tei_connection.assert_called_once()
        mock_verifier.validate_qdrant_connection.assert_called_once()


class TestMainStateRecovery:
    """Test main() state recovery process."""

    @patch("rag_ingestion.main.Observer")
    @patch("rag_ingestion.main.FileWatcher")
    @patch("rag_ingestion.main.StateRecovery")
    @patch("rag_ingestion.main.DocumentProcessor")
    @patch("rag_ingestion.main.VectorStoreManager")
    @patch("rag_ingestion.main.MarkdownChunker")
    @patch("rag_ingestion.main.TEIClient")
    @patch("rag_ingestion.main.QualityVerifier")
    @patch("rag_ingestion.main.Settings")
    async def test_main_performs_state_recovery(
        self,
        mock_settings: Mock,
        mock_verifier_class: Mock,
        mock_tei: Mock,
        mock_chunker: Mock,
        mock_vector_store: Mock,
        mock_processor: Mock,
        mock_state_recovery_class: Mock,
        mock_file_watcher: Mock,
        mock_observer_class: Mock,
    ) -> None:
        """Verify StateRecovery.get_files_to_process called on startup.

        Requirements: FR-8 (State recovery)
        Design: Main Entry Point - State Recovery
        """
        mock_config, mock_verifier, mock_observer, mock_state_recovery = (
            setup_common_mocks()
        )

        mock_settings.return_value = mock_config
        mock_verifier_class.return_value = mock_verifier
        mock_observer_class.return_value = mock_observer
        mock_state_recovery_class.return_value = mock_state_recovery

        # Call main
        await main()

        # Should call get_files_to_process
        mock_state_recovery_class.assert_called_once()
        mock_state_recovery.get_files_to_process.assert_called_once()


class TestMainBatchProcessing:
    """Test main() batch processing on startup."""

    @patch("rag_ingestion.main.Observer")
    @patch("rag_ingestion.main.FileWatcher")
    @patch("rag_ingestion.main.StateRecovery")
    @patch("rag_ingestion.main.DocumentProcessor")
    @patch("rag_ingestion.main.VectorStoreManager")
    @patch("rag_ingestion.main.MarkdownChunker")
    @patch("rag_ingestion.main.TEIClient")
    @patch("rag_ingestion.main.QualityVerifier")
    @patch("rag_ingestion.main.Settings")
    async def test_main_processes_batch(
        self,
        mock_settings: Mock,
        mock_verifier_class: Mock,
        mock_tei: Mock,
        mock_chunker: Mock,
        mock_vector_store: Mock,
        mock_processor_class: Mock,
        mock_state_recovery_class: Mock,
        mock_file_watcher: Mock,
        mock_observer_class: Mock,
    ) -> None:
        """Verify processor.process_batch called with recovered files.

        Requirements: FR-3 (Batch processing on startup)
        Design: Main Entry Point - Batch Processing
        """
        mock_config, mock_verifier, mock_observer, _ = setup_common_mocks()

        # Override state recovery to return files
        mock_state_recovery = Mock()
        mock_files = ["doc1.md", "doc2.md"]
        mock_state_recovery.get_files_to_process = AsyncMock(return_value=mock_files)

        mock_processor = Mock()
        mock_processor.process_batch = AsyncMock()

        mock_settings.return_value = mock_config
        mock_verifier_class.return_value = mock_verifier
        mock_observer_class.return_value = mock_observer
        mock_state_recovery_class.return_value = mock_state_recovery
        mock_processor_class.return_value = mock_processor

        # Call main
        await main()

        # Should call process_batch with recovered files converted to absolute paths
        expected_files = [
            Path("/tmp/test/doc1.md"),
            Path("/tmp/test/doc2.md"),
        ]
        mock_processor.process_batch.assert_called_once_with(expected_files)


class TestMainWatcherStartup:
    """Test main() file watcher startup."""

    @patch("rag_ingestion.main.Observer")
    @patch("rag_ingestion.main.FileWatcher")
    @patch("rag_ingestion.main.StateRecovery")
    @patch("rag_ingestion.main.DocumentProcessor")
    @patch("rag_ingestion.main.VectorStoreManager")
    @patch("rag_ingestion.main.MarkdownChunker")
    @patch("rag_ingestion.main.TEIClient")
    @patch("rag_ingestion.main.QualityVerifier")
    @patch("rag_ingestion.main.Settings")
    async def test_main_starts_watcher(
        self,
        mock_settings: Mock,
        mock_verifier_class: Mock,
        mock_tei: Mock,
        mock_chunker: Mock,
        mock_vector_store: Mock,
        mock_processor: Mock,
        mock_state_recovery_class: Mock,
        mock_file_watcher: Mock,
        mock_observer_class: Mock,
    ) -> None:
        """Verify watchdog Observer started after batch processing.

        Requirements: AC-1.5 (Watch mode after batch)
        Design: Main Entry Point - Watch Mode
        """
        mock_config, mock_verifier, mock_observer, mock_state_recovery = (
            setup_common_mocks()
        )

        mock_settings.return_value = mock_config
        mock_verifier_class.return_value = mock_verifier
        mock_observer_class.return_value = mock_observer
        mock_state_recovery_class.return_value = mock_state_recovery

        # Call main
        await main()

        # Should start observer
        mock_observer_class.assert_called_once()
        mock_observer.start.assert_called_once()


class TestMainShutdown:
    """Test main() shutdown handling."""

    @patch("rag_ingestion.main.Observer")
    @patch("rag_ingestion.main.FileWatcher")
    @patch("rag_ingestion.main.StateRecovery")
    @patch("rag_ingestion.main.DocumentProcessor")
    @patch("rag_ingestion.main.VectorStoreManager")
    @patch("rag_ingestion.main.MarkdownChunker")
    @patch("rag_ingestion.main.TEIClient")
    @patch("rag_ingestion.main.QualityVerifier")
    @patch("rag_ingestion.main.Settings")
    async def test_main_handles_keyboard_interrupt(
        self,
        mock_settings: Mock,
        mock_verifier_class: Mock,
        mock_tei: Mock,
        mock_chunker: Mock,
        mock_vector_store: Mock,
        mock_processor: Mock,
        mock_state_recovery_class: Mock,
        mock_file_watcher: Mock,
        mock_observer_class: Mock,
    ) -> None:
        """Simulate Ctrl+C, verify clean shutdown.

        Requirements: FR-9 (Graceful shutdown)
        Design: Main Entry Point - Shutdown Handling
        """
        mock_config, mock_verifier, mock_observer, mock_state_recovery = (
            setup_common_mocks()
        )

        mock_settings.return_value = mock_config
        mock_verifier_class.return_value = mock_verifier
        mock_observer_class.return_value = mock_observer
        mock_state_recovery_class.return_value = mock_state_recovery

        # Call main - should not re-raise KeyboardInterrupt
        await main()

        # Should catch KeyboardInterrupt and stop observer
        mock_observer.stop.assert_called_once()
