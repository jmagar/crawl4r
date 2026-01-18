"""Failed document logging for RAG ingestion service.

Provides structured JSONL logging for documents that fail processing after retries.
Each failure is logged with complete context including file path, error details,
traceback, and retry count for debugging and monitoring purposes.

Example:
    >>> from pathlib import Path
    >>> from crawl4r.resilience.failed_docs import FailedDocLogger
    >>>
    >>> logger = FailedDocLogger(Path("failed_documents.jsonl"))
    >>> try:
    ...     process_document(Path("/data/doc.md"))
    ... except Exception as e:
    ...     logger.log_failure(
    ...         file_path=Path("/data/doc.md"),
    ...         event_type="modified",
    ...         error=e,
    ...         retry_count=3
    ...     )
"""

import json
import traceback as tb
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict


class FailedDocEntry(TypedDict):
    """Schema for JSONL log entries of failed document processing.

    Attributes:
        file_path: Absolute path to the failed document
        file_name: Name of the file (for readability)
        timestamp: ISO 8601 timestamp with timezone (UTC)
        event_type: File system event that triggered processing
            (created, modified, deleted)
        error_type: Exception class name (e.g., ValueError, OSError)
        error_message: Human-readable error message
        traceback: Full Python traceback string for debugging
        retry_count: Number of retry attempts before failure was logged
    """

    file_path: str
    file_name: str
    timestamp: str
    event_type: str
    error_type: str
    error_message: str
    traceback: str
    retry_count: int


class FailedDocLogger:
    """Logger for documents that fail processing after maximum retries.

    Writes structured JSONL entries to a log file with complete error context.
    Each entry includes file metadata, error details, traceback, and retry count.

    Attributes:
        log_path: Path to the JSONL log file

    Example:
        >>> logger = FailedDocLogger(Path("failed_documents.jsonl"))
        >>> error = ValueError("Invalid markdown format")
        >>> logger.log_failure(
        ...     file_path=Path("/data/watched_folder/docs/test.md"),
        ...     event_type="modified",
        ...     error=error,
        ...     retry_count=3
        ... )
    """

    def __init__(self, log_path: Path) -> None:
        """Initialize the failed document logger.

        Args:
            log_path: Path to the JSONL log file where failures will be recorded
        """
        self.log_path = log_path

    def log_failure(
        self,
        file_path: Path,
        event_type: str,
        error: Exception,
        retry_count: int,
    ) -> None:
        """Log a document processing failure with complete error context.

        Creates a JSONL entry with all required fields and appends it to the log file.
        The entry includes absolute file path, filename, timestamp, error details,
        full traceback, and retry count for comprehensive debugging.

        Args:
            file_path: Absolute path to the failed document
            event_type: Type of file event that triggered processing
                (created, modified, deleted)
            error: Exception that caused the failure
            retry_count: Number of retry attempts made before logging

        Example:
            >>> logger = FailedDocLogger(Path("failed_documents.jsonl"))
            >>> try:
            ...     process_markdown(Path("/data/watched_folder/doc.md"))
            ... except ValueError as e:
            ...     logger.log_failure(
            ...         file_path=Path("/data/watched_folder/doc.md"),
            ...         event_type="created",
            ...         error=e,
            ...         retry_count=2
            ...     )
        """
        # Build JSONL entry with all required fields
        entry: FailedDocEntry = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": tb.format_exc(),
            "retry_count": retry_count,
        }

        # Append to log file (create if doesn't exist)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
