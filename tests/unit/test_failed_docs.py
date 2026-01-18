"""Unit tests for failed document logging.

TDD RED Phase: All tests should FAIL initially (no implementation exists).
"""

import json
from datetime import datetime
from pathlib import Path

# This import will fail initially - that's expected in RED phase
from crawl4r.resilience.failed_docs import FailedDocLogger


class TestFailedDocLogger:
    """Test suite for failed document logging functionality."""

    def test_log_failed_document(self, tmp_path: Path) -> None:
        """Test logging a failed document with complete JSONL schema.

        Verifies that the logger writes a JSONL entry with all required fields:
        - file_path (absolute)
        - file_name
        - timestamp
        - event_type
        - error_type
        - error_message
        - traceback
        - retry_count
        """
        log_path = tmp_path / "failed_documents.jsonl"
        logger = FailedDocLogger(log_path)

        # Simulate a processing error
        test_file = Path("/data/watched_folder/docs/test.md")
        test_error = ValueError("Invalid markdown format")
        event_type = "modified"
        retry_count = 2

        # Log the failure
        logger.log_failure(
            file_path=test_file,
            event_type=event_type,
            error=test_error,
            retry_count=retry_count,
        )

        # Verify the JSONL entry was written
        assert log_path.exists()
        with open(log_path) as f:
            entry = json.loads(f.readline())

        # Verify all required fields
        assert entry["file_path"] == str(test_file)
        assert entry["file_name"] == "test.md"
        assert entry["event_type"] == event_type
        assert entry["error_type"] == "ValueError"
        assert entry["error_message"] == "Invalid markdown format"
        assert "traceback" in entry
        assert entry["retry_count"] == retry_count

        # Verify timestamp is ISO format
        timestamp = datetime.fromisoformat(entry["timestamp"])
        assert isinstance(timestamp, datetime)

    def test_failed_docs_log_path_from_config(self, tmp_path: Path) -> None:
        """Test that logger uses log path from configuration.

        Verifies that the FailedDocLogger accepts and uses a custom log path
        from the configuration settings.
        """
        custom_log_path = tmp_path / "custom_failed_docs.jsonl"
        logger = FailedDocLogger(custom_log_path)

        # Log a failure
        test_error = RuntimeError("Test error")
        logger.log_failure(
            file_path=Path("/data/test.md"),
            event_type="created",
            error=test_error,
            retry_count=1,
        )

        # Verify it wrote to the custom path
        assert custom_log_path.exists()
        assert not (tmp_path / "failed_documents.jsonl").exists()

    def test_failed_docs_append_mode(self, tmp_path: Path) -> None:
        """Test that multiple failures are appended to the log file.

        Verifies that subsequent failures are appended (not overwriting)
        and that the JSONL file contains all entries.
        """
        log_path = tmp_path / "failed_documents.jsonl"
        logger = FailedDocLogger(log_path)

        # Log first failure
        logger.log_failure(
            file_path=Path("/data/file1.md"),
            event_type="created",
            error=ValueError("Error 1"),
            retry_count=1,
        )

        # Log second failure
        logger.log_failure(
            file_path=Path("/data/file2.md"),
            event_type="modified",
            error=RuntimeError("Error 2"),
            retry_count=2,
        )

        # Verify both entries exist
        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])

        assert entry1["file_name"] == "file1.md"
        assert entry1["error_message"] == "Error 1"
        assert entry1["retry_count"] == 1

        assert entry2["file_name"] == "file2.md"
        assert entry2["error_message"] == "Error 2"
        assert entry2["retry_count"] == 2

    def test_failed_docs_skip_after_max_retries(self, tmp_path: Path) -> None:
        """Test logging document that failed after maximum retries.

        Verifies that when a document fails after exhausting all retry attempts,
        it's logged once with the final retry count (e.g., retry_count=3 for
        3 attempts).
        """
        log_path = tmp_path / "failed_documents.jsonl"
        logger = FailedDocLogger(log_path)

        test_file = Path("/data/problematic.md")
        test_error = OSError("Persistent file read error")
        max_retries = 3

        # Simulate final failure after max retries
        logger.log_failure(
            file_path=test_file,
            event_type="modified",
            error=test_error,
            retry_count=max_retries,
        )

        # Verify only one entry exists (not 3 separate ones)
        with open(log_path) as f:
            lines = f.readlines()

        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["file_name"] == "problematic.md"
        assert entry["error_type"] == "OSError"
        assert entry["error_message"] == "Persistent file read error"
        assert entry["retry_count"] == max_retries
