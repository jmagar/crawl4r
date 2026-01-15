"""State recovery module for resuming ingestion after restart.

Queries Qdrant to determine which files have already been processed and compares
modification dates with filesystem to identify files needing reprocessing.

Features:
- Query existing files from Qdrant via scroll API
- Extract latest modification dates per file
- Compare Qdrant state with filesystem
- Skip up-to-date files to avoid redundant processing

Example:
    from rag_ingestion.recovery import StateRecovery
    from rag_ingestion.vector_store import VectorStoreManager

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

import logging
from datetime import datetime


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
        """Initialize state recovery."""
        self.logger = logging.getLogger(__name__)

    async def query_existing_files(self, vector_store) -> list[str]:
        """Query Qdrant for all existing files.

        Uses scroll API to retrieve all file paths that have been ingested.
        Deduplicates file paths since each file may have multiple chunks.

        Args:
            vector_store: VectorStoreManager instance to query

        Returns:
            List of unique file paths that exist in Qdrant

        Example:
            recovery = StateRecovery()
            existing_files = await recovery.query_existing_files(vector_store)
            # Returns: ["docs/file1.md", "docs/file2.md", "docs/file3.md"]
        """
        # Query all points from Qdrant using scroll API
        points = await vector_store.scroll()

        # Extract unique file paths from payloads
        file_paths = set()
        for point in points:
            payload = point.get("payload", {})
            file_path = payload.get("file_path_relative")
            if file_path:
                file_paths.add(file_path)

        return list(file_paths)

    async def get_file_modification_dates(self, vector_store) -> dict[str, datetime]:
        """Query Qdrant for modification dates of all files.

        Uses scroll API to retrieve modification dates for all ingested files.
        Returns the latest modification date for each file (in case of multiple chunks).

        Args:
            vector_store: VectorStoreManager instance to query

        Returns:
            Dictionary mapping file paths to latest modification dates

        Example:
            recovery = StateRecovery()
            file_dates = await recovery.get_file_modification_dates(vector_store)
            # Returns: {"docs/file1.md": datetime(2026, 1, 1, 12, 0, 0), ...}
        """
        # Query all points from Qdrant using scroll API
        points = await vector_store.scroll()

        # Build dict of file_path -> latest modification_date
        file_dates: dict[str, datetime] = {}
        for point in points:
            payload = point.get("payload", {})
            file_path = payload.get("file_path_relative")
            mod_date_str = payload.get("modification_date")

            if file_path and mod_date_str:
                mod_date = datetime.fromisoformat(mod_date_str)

                # Keep the latest modification date for each file
                if file_path not in file_dates or mod_date > file_dates[file_path]:
                    file_dates[file_path] = mod_date

        return file_dates

    async def get_files_to_process(
        self, vector_store, filesystem_files: dict[str, datetime]
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
        files_to_process = []
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

        # Log skipped files count
        if skipped_count > 0:
            self.logger.info(f"Skipped {skipped_count} up-to-date files")

        return files_to_process
