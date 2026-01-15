"""Unit tests for logger module.

TDD RED Phase: All tests should FAIL initially (no implementation exists).
"""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

# This import will fail initially - that's expected in RED phase
from rag_ingestion.logger import get_logger


class TestLoggerCreation:
    """Test logger creation and handler setup."""

    def test_logger_creates_console_handler(self) -> None:
        """Test that logger creates a console handler with INFO level."""
        logger = get_logger("test_module", log_level="INFO")

        # Find console handler (StreamHandler)
        console_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(console_handlers) >= 1, "Expected at least one console handler"

        # Verify console handler level
        console_handler = console_handlers[0]
        assert console_handler.level == logging.INFO

    def test_logger_creates_rotating_file_handler(self) -> None:
        """Test that logger creates rotating file handler with correct settings."""
        from logging.handlers import RotatingFileHandler

        log_file = Path(".cache/test_logger.log")
        logger = get_logger("test_module", log_file=log_file)

        # Find rotating file handler
        file_handlers = [
            h for h in logger.handlers if isinstance(h, RotatingFileHandler)
        ]
        assert len(file_handlers) >= 1, "Expected at least one rotating file handler"

        # Verify rotating file handler settings
        file_handler = file_handlers[0]
        assert isinstance(file_handler, RotatingFileHandler)
        # 100MB = 104857600 bytes
        assert file_handler.maxBytes == 104857600, "Expected maxBytes to be 100MB"
        assert file_handler.backupCount == 5, "Expected 5 backup files"


class TestLoggerFormatting:
    """Test logger formatting configuration."""

    def test_logger_formats_human_readable(self) -> None:
        """Test that logger uses human-readable format with timestamp, level, module, message."""
        logger = get_logger("test_module")

        # Check that handlers have formatters
        for handler in logger.handlers:
            assert handler.formatter is not None, "Handler should have a formatter"

            # Get the format string
            format_str = handler.formatter._fmt

            # Verify format includes required components
            assert "%(asctime)s" in format_str, "Format should include timestamp"
            assert "%(levelname)s" in format_str, "Format should include log level"
            assert "%(name)s" in format_str, "Format should include module name"
            assert "%(message)s" in format_str, "Format should include message"


class TestLoggerLevels:
    """Test logger respects different log levels."""

    def test_logger_respects_log_level(self) -> None:
        """Test that logger respects DEBUG, INFO, WARNING, ERROR levels."""
        # Test DEBUG level
        logger_debug = get_logger("test_debug", log_level="DEBUG")
        assert logger_debug.level == logging.DEBUG

        # Test INFO level
        logger_info = get_logger("test_info", log_level="INFO")
        assert logger_info.level == logging.INFO

        # Test WARNING level
        logger_warning = get_logger("test_warning", log_level="WARNING")
        assert logger_warning.level == logging.WARNING

        # Test ERROR level
        logger_error = get_logger("test_error", log_level="ERROR")
        assert logger_error.level == logging.ERROR


class TestLoggerFileOutput:
    """Test logger file output configuration."""

    def test_logger_logs_to_correct_file(self) -> None:
        """Test that logger writes to the specified log file path."""
        from logging.handlers import RotatingFileHandler

        log_file = Path(".cache/test_custom.log")
        logger = get_logger("test_module", log_file=log_file)

        # Find rotating file handler
        file_handlers = [
            h for h in logger.handlers if isinstance(h, RotatingFileHandler)
        ]
        assert len(file_handlers) >= 1, "Expected at least one rotating file handler"

        # Verify file path
        file_handler = file_handlers[0]
        assert file_handler.baseFilename == str(
            log_file.resolve()
        ), f"Expected log file at {log_file}"


class TestLoggerWithConfig:
    """Test logger integration with Settings configuration."""

    def test_logger_uses_config_log_level(self) -> None:
        """Test that logger uses log level from config."""
        # Test with different log levels from config
        logger_info = get_logger("config_test", log_level="INFO")
        assert logger_info.level == logging.INFO

        logger_debug = get_logger("config_test", log_level="DEBUG")
        assert logger_debug.level == logging.DEBUG

    def test_logger_creates_log_directory(self) -> None:
        """Test that logger creates log file directory if it doesn't exist."""
        log_file = Path(".cache/nested/directory/test.log")

        # Ensure directory doesn't exist
        if log_file.parent.exists():
            import shutil
            shutil.rmtree(log_file.parent)

        # Create logger - should create directory
        logger = get_logger("test_module", log_file=log_file)

        # Verify directory was created
        assert log_file.parent.exists(), "Log directory should be created"
