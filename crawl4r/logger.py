"""Structured logging with rotation for RAG ingestion pipeline.

This module provides a configured logger with console and rotating file handlers.
Logs are human-readable with timestamp, level, module, and message.

Examples:
    >>> from rag_ingestion.logger import get_logger
    >>> logger = get_logger("rag_ingestion.processor")
    >>> logger.info("Processing document")
    2026-01-14 23:45:00,123 | INFO | rag_ingestion.processor | Processing document
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Human-readable log format with timestamp, level, module name, and message
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Default log file path (in .cache directory to keep project root clean)
DEFAULT_LOG_FILE = Path(".cache/rag_ingestion.log")

# Rotating file handler limits (100MB max file size, 5 backup files)
MAX_LOG_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
BACKUP_COUNT = 5


def get_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Path | None = None,
) -> logging.Logger:
    """Create and configure a logger with console and rotating file handlers.

    The logger includes:
    - Console handler: INFO level, writes to stderr
    - Rotating file handler: DEBUG level, 100MB max size, 5 backups
    - Human-readable format: timestamp | level | module | message

    Args:
        name: Logger name (typically module name like "rag_ingestion.processor")
        log_level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to INFO. Controls the logger's base level.
        log_file: Optional path to log file. If None, defaults to
            .cache/rag_ingestion.log. Parent directories are created automatically.

    Returns:
        Configured logging.Logger instance ready for use. The logger can be
        reconfigured by calling this function again with different parameters.

    Raises:
        ValueError: If log_level is not a valid logging level name.
        OSError: If log file directory cannot be created (rare, indicates
            filesystem permission issues).

    Examples:
        >>> logger = get_logger("rag_ingestion.processor")
        >>> logger.info("Processing document")
        2026-01-14 23:45:00,123 | INFO | rag_ingestion.processor | Processing document

        >>> logger = get_logger("rag_ingestion.embeddings", log_level="DEBUG")
        >>> logger.debug("Generating embeddings for chunk 0")
        2026-01-14 23:45:01,456 | DEBUG | rag_ingestion.embeddings | ...
    """
    # Create logger and set base level
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to allow reconfiguration
    # This is important for testing and when log_file changes between calls
    logger.handlers.clear()

    # Create formatter with human-readable format
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler (INFO level for user-facing output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler (DEBUG level for comprehensive file logging)
    if log_file is None:
        log_file = DEFAULT_LOG_FILE

    # Create log directory if it doesn't exist
    # parents=True creates all intermediate directories
    # exist_ok=True prevents errors if directory already exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_SIZE_BYTES,
        backupCount=BACKUP_COUNT,
    )
    file_handler.setLevel(logging.DEBUG)  # File captures all log levels
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
