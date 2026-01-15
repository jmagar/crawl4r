"""Tests for state recovery module.

Tests state recovery including querying Qdrant for existing files and
comparing with filesystem to determine which files need processing.
"""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest


class TestQueryExistingFiles:
    """Test querying Qdrant for existing files."""

    @pytest.mark.asyncio
    async def test_query_existing_files_from_qdrant(self) -> None:
        """Verify querying Qdrant returns list of existing files."""
        # Import will fail since module doesn't exist yet
        from rag_ingestion.recovery import StateRecovery

        # Mock vector store with scroll API response containing 3 files
        vector_store = AsyncMock()
        vector_store.scroll.return_value = [
            {"payload": {"file_path_relative": "docs/file1.md"}},
            {"payload": {"file_path_relative": "docs/file2.md"}},
            {"payload": {"file_path_relative": "docs/file3.md"}},
        ]

        # Create recovery instance and query existing files
        recovery = StateRecovery()
        files = await recovery.query_existing_files(vector_store)

        # Verify extracted file paths
        assert len(files) == 3
        assert "docs/file1.md" in files
        assert "docs/file2.md" in files
        assert "docs/file3.md" in files
        vector_store.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_unique_file_paths(self) -> None:
        """Verify duplicate file paths from chunks are deduplicated."""
        from rag_ingestion.recovery import StateRecovery

        # Mock vector store with duplicate file_paths (multiple chunks per file)
        vector_store = AsyncMock()
        vector_store.scroll.return_value = [
            {"payload": {"file_path_relative": "docs/file1.md"}},
            {"payload": {"file_path_relative": "docs/file1.md"}},  # Chunk 2
            {"payload": {"file_path_relative": "docs/file2.md"}},
            {"payload": {"file_path_relative": "docs/file2.md"}},  # Chunk 2
            {"payload": {"file_path_relative": "docs/file2.md"}},  # Chunk 3
        ]

        recovery = StateRecovery()
        files = await recovery.query_existing_files(vector_store)

        # Verify only unique file paths returned
        assert len(files) == 2
        assert "docs/file1.md" in files
        assert "docs/file2.md" in files

    @pytest.mark.asyncio
    async def test_extract_modification_dates(self) -> None:
        """Verify extraction of modification dates for each file."""
        from rag_ingestion.recovery import StateRecovery

        # Mock vector store with modification_date payloads
        vector_store = AsyncMock()
        vector_store.scroll.return_value = [
            {
                "payload": {
                    "file_path_relative": "docs/file1.md",
                    "modification_date": "2026-01-01T10:00:00",
                }
            },
            {
                "payload": {
                    "file_path_relative": "docs/file1.md",
                    "modification_date": "2026-01-01T12:00:00",  # Latest
                }
            },
            {
                "payload": {
                    "file_path_relative": "docs/file2.md",
                    "modification_date": "2026-01-02T10:00:00",
                }
            },
        ]

        recovery = StateRecovery()
        file_dates = await recovery.get_file_modification_dates(vector_store)

        # Verify latest modification date per file
        assert len(file_dates) == 2
        assert file_dates["docs/file1.md"] == datetime.fromisoformat(
            "2026-01-01T12:00:00"
        )
        assert file_dates["docs/file2.md"] == datetime.fromisoformat(
            "2026-01-02T10:00:00"
        )


class TestFilesystemComparison:
    """Test comparing Qdrant state with filesystem."""

    @pytest.mark.asyncio
    async def test_compare_with_filesystem(self) -> None:
        """Verify comparison returns stale and new files to process."""
        from rag_ingestion.recovery import StateRecovery

        # Mock vector store with 3 files in Qdrant
        vector_store = AsyncMock()
        vector_store.scroll.return_value = [
            {
                "payload": {
                    "file_path_relative": "docs/file1.md",
                    "modification_date": "2026-01-01T10:00:00",  # Up-to-date
                }
            },
            {
                "payload": {
                    "file_path_relative": "docs/file2.md",
                    "modification_date": "2026-01-01T10:00:00",  # Stale
                }
            },
            {
                "payload": {
                    "file_path_relative": "docs/file3.md",
                    "modification_date": "2026-01-01T10:00:00",  # Up-to-date
                }
            },
        ]

        # Mock filesystem with 5 files (2 new, 1 stale, 2 up-to-date)
        filesystem_files = {
            "docs/file1.md": datetime(2026, 1, 1, 10, 0, 0),  # Up-to-date
            "docs/file2.md": datetime(2026, 1, 1, 12, 0, 0),  # Stale (newer)
            "docs/file3.md": datetime(2026, 1, 1, 10, 0, 0),  # Up-to-date
            "docs/file4.md": datetime(2026, 1, 1, 10, 0, 0),  # New
            "docs/file5.md": datetime(2026, 1, 1, 10, 0, 0),  # New
        }

        recovery = StateRecovery()
        files_to_process = await recovery.get_files_to_process(
            vector_store, filesystem_files
        )

        # Verify returns 1 stale + 2 new = 3 files
        assert len(files_to_process) == 3
        assert "docs/file2.md" in files_to_process  # Stale
        assert "docs/file4.md" in files_to_process  # New
        assert "docs/file5.md" in files_to_process  # New
        assert "docs/file1.md" not in files_to_process  # Up-to-date
        assert "docs/file3.md" not in files_to_process  # Up-to-date

    @pytest.mark.asyncio
    async def test_skip_up_to_date_files(self) -> None:
        """Verify files with Qdrant mod_date >= filesystem mod_date are skipped."""
        from rag_ingestion.recovery import StateRecovery

        # Mock vector store with files that are up-to-date
        vector_store = AsyncMock()
        vector_store.scroll.return_value = [
            {
                "payload": {
                    "file_path_relative": "docs/file1.md",
                    "modification_date": "2026-01-01T12:00:00",  # Same as filesystem
                }
            },
            {
                "payload": {
                    "file_path_relative": "docs/file2.md",
                    "modification_date": "2026-01-01T13:00:00",  # Newer than filesystem
                }
            },
        ]

        # Mock filesystem with same or older mod dates
        filesystem_files = {
            "docs/file1.md": datetime(2026, 1, 1, 12, 0, 0),  # Same
            "docs/file2.md": datetime(2026, 1, 1, 12, 0, 0),  # Older
        }

        recovery = StateRecovery()
        files_to_process = await recovery.get_files_to_process(
            vector_store, filesystem_files
        )

        # Verify no files to process (all up-to-date)
        assert len(files_to_process) == 0
