"""Structured logging with rotation for RAG ingestion pipeline.

This module provides a configured logger with console and rotating file handlers.
Logs are human-readable with timestamp, level, module, and message.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Path | None = None,
) -> logging.Logger:
    """Create and configure a logger with console and file handlers.

    Args:
        name: Logger name (typically module name)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file (defaults to .cache/rag_ingestion.log)

    Returns:
        Configured logger instance

    Examples:
        >>> logger = get_logger("rag_ingestion.processor")
        >>> logger.info("Processing document")
        2026-01-14 23:45:00,123 | INFO | rag_ingestion.processor | Processing document
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to allow reconfiguration
    # This is important for testing and when log_file changes
    logger.handlers.clear()

    # Human-readable format with timestamp, level, module, message
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler (100MB max, 5 backups)
    if log_file is None:
        log_file = Path(".cache/rag_ingestion.log")

    # Create log directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)  # File logs everything
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
