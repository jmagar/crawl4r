"""State recovery module for resuming ingestion after restart.

Queries Qdrant to determine which files have already been processed and compares
modification dates with filesystem to identify files needing reprocessing.

Features:
- Query existing files from Qdrant via scroll API
- Extract latest modification dates per file
- Compare Qdrant state with filesystem
- Skip up-to-date files to avoid redundant processing

Example:
    from crawl4r.resilience.recovery import StateRecovery
    from crawl4r.storage.qdrant import VectorStoreManager

    vector_store = VectorStoreManager("http://localhost:6333", "docs")
    recovery = StateRecovery()

    # Get filesystem files with modification dates
    filesystem_files = {
        "docs/file1.md": datetime(2026, 1, 1, 10, 0, 0),
        "docs/file2.md": datetime(2026, 1, 1, 12, 0, 0),
    }

    # Determine which files need processing
    files_to_process = await recovery.get_files_to_process(
        vector_store, filesystem_files
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Protocol

from crawl4r.core.metadata import MetadataKeys


class VectorStoreProtocol(Protocol):
    """Protocol defining expected interface for vector store scroll operations."""

    async def scroll(self) -> list[dict[str, Any]]:
        """Scroll through all points in the collection.

        Returns:
            List of point dictionaries with payload data
        """
        ...


class StateRecovery:
    """State recovery for resuming ingestion after restart.

    Queries Qdrant to determine existing files and compares modification dates
    with filesystem to identify files needing processing or reprocessing.

    Attributes:
        logger: Logger instance for recovery messages

    Example:
        recovery = StateRecovery()
        files_to_process = await recovery.get_files_to_process(
            vector_store, filesystem_files
        )
    """

    def __init__(self) -> None:
        """Initialize state recovery.

        Sets up logger for recovery operations.
        """
        self.logger = logging.getLogger(__name__)

    async def _query_qdrant_state(
        self, vector_store: VectorStoreProtocol
    ) -> list[dict[str, Any]]:
        """Query all points from Qdrant using scroll API.

        Args:
            vector_store: VectorStoreManager instance to query

        Returns:
            List of point dictionaries from Qdrant

        Raises:
            Exception: If Qdrant scroll API fails
        """
        return await vector_store.scroll()

    def _scan_filesystem(
        self, points: list[dict[str, Any]]
    ) -> tuple[set[str], dict[str, datetime]]:
        """Extract file paths and modification dates from Qdrant points.

        Args:
            points: List of point dictionaries from Qdrant

        Returns:
            Tuple of (unique file paths, file path to latest modification date mapping)
        """
        file_paths: set[str] = set()
        file_dates: dict[str, datetime] = {}

        for point in points:
            payload = point.get("payload", {})
            file_path = payload.get(MetadataKeys.FILE_PATH)
            mod_date_str = payload.get(MetadataKeys.LAST_MODIFIED_DATE)

            if file_path:
                file_paths.add(file_path)

            if file_path and mod_date_str:
                mod_date = datetime.fromisoformat(mod_date_str)

                # Keep the latest modification date for each file
                if file_path not in file_dates or mod_date > file_dates[file_path]:
                    file_dates[file_path] = mod_date

        return file_paths, file_dates

    def _compare_states(
        self, filesystem_files: dict[str, datetime], qdrant_dates: dict[str, datetime]
    ) -> tuple[list[str], int]:
        """Compare filesystem and Qdrant states to determine files needing processing.

        Args:
            filesystem_files: Dictionary mapping file paths to modification dates
            qdrant_dates: Dictionary mapping file paths to Qdrant modification dates

        Returns:
            Tuple of (list of files to process, count of skipped files)
        """
        files_to_process: list[str] = []
        skipped_count = 0

        for file_path, filesystem_date in filesystem_files.items():
            if file_path not in qdrant_dates:
                # New file - not in Qdrant
                files_to_process.append(file_path)
            elif filesystem_date > qdrant_dates[file_path]:
                # Stale file - filesystem is newer
                files_to_process.append(file_path)
            else:
                # Up-to-date file - skip
                skipped_count += 1

        return files_to_process, skipped_count

    async def query_existing_files(
        self, vector_store: VectorStoreProtocol
    ) -> list[str]:
        """Query Qdrant for all existing files.

        Uses scroll API to retrieve all file paths that have been ingested.
        Deduplicates file paths since each file may have multiple chunks.

        Args:
            vector_store: VectorStoreManager instance to query

        Returns:
            List of unique file paths that exist in Qdrant

        Raises:
            Exception: If Qdrant query fails

        Example:
            recovery = StateRecovery()
            existing_files = await recovery.query_existing_files(vector_store)
            # Returns: ["docs/file1.md", "docs/file2.md", "docs/file3.md"]
        """
        points = await self._query_qdrant_state(vector_store)
        file_paths, _ = self._scan_filesystem(points)
        return list(file_paths)

    async def get_file_modification_dates(
        self, vector_store: VectorStoreProtocol
    ) -> dict[str, datetime]:
        """Query Qdrant for modification dates of all files.

        Uses scroll API to retrieve modification dates for all ingested files.
        Returns the latest modification date for each file (in case of multiple chunks).

        Args:
            vector_store: VectorStoreManager instance to query

        Returns:
            Dictionary mapping file paths to latest modification dates

        Raises:
            Exception: If Qdrant query fails

        Example:
            recovery = StateRecovery()
            file_dates = await recovery.get_file_modification_dates(vector_store)
            # Returns: {"docs/file1.md": datetime(2026, 1, 1, 12, 0, 0), ...}
        """
        points = await self._query_qdrant_state(vector_store)
        _, file_dates = self._scan_filesystem(points)
        return file_dates

    async def get_files_to_process(
        self, vector_store: VectorStoreProtocol, filesystem_files: dict[str, datetime]
    ) -> list[str]:
        """Determine which files need processing based on modification dates.

        Compares Qdrant state with filesystem to identify:
        - New files: Files in filesystem but not in Qdrant
        - Stale files: Files in both, but filesystem mod_date > Qdrant mod_date
        - Up-to-date files: Skipped (Qdrant mod_date >= filesystem mod_date)

        Args:
            vector_store: VectorStoreManager instance to query
            filesystem_files: Dictionary mapping file paths to modification dates

        Returns:
            List of file paths that need processing

        Raises:
            Exception: If Qdrant query fails

        Example:
            filesystem_files = {
                "docs/file1.md": datetime(2026, 1, 1, 10, 0, 0),  # Up-to-date
                "docs/file2.md": datetime(2026, 1, 1, 12, 0, 0),  # Stale
                "docs/file3.md": datetime(2026, 1, 1, 10, 0, 0),  # New
            }
            recovery = StateRecovery()
            files = await recovery.get_files_to_process(vector_store, filesystem_files)
            # Returns: ["docs/file2.md", "docs/file3.md"]

        Notes:
            - Files with Qdrant mod_date >= filesystem mod_date are skipped
            - Logs count of skipped files
        """
        # Get modification dates from Qdrant
        qdrant_dates = await self.get_file_modification_dates(vector_store)

        # Compare and determine which files need processing
        files_to_process, skipped_count = self._compare_states(
            filesystem_files, qdrant_dates
        )

        # Log skipped files count
        if skipped_count > 0:
            self.logger.info(f"Skipped {skipped_count} up-to-date files")

        return files_to_process
