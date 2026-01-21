"""Watch command for monitoring file system changes.

Monitors a directory for markdown file changes and automatically processes them
through the RAG ingestion pipeline. Performs startup batch recovery and runs
continuous file monitoring until interrupted.

Example:
    # Use default watch folder from settings
    crawl4r watch

    # Override watch folder
    crawl4r watch --folder /path/to/docs

Requirements:
    - FR-3: Batch processing on startup
    - FR-6: Environment configuration
    - FR-7: Service validation
    - FR-8: State recovery
    - FR-9: Graceful shutdown
    - AC-1.5: Watch mode after batch
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import typer
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from crawl4r.core.config import Settings
from crawl4r.core.quality import QualityVerifier
from crawl4r.core.quality import VectorStoreProtocol as QualityVectorStoreProtocol
from crawl4r.processing.processor import DocumentProcessor
from crawl4r.readers.file_watcher import FileWatcher
from crawl4r.resilience.recovery import StateRecovery
from crawl4r.resilience.recovery import (
    VectorStoreProtocol as RecoveryVectorStoreProtocol,
)
from crawl4r.storage.qdrant import VectorStoreManager
from crawl4r.storage.tei import TEIClient

app = typer.Typer(invoke_without_command=True)

# Module-level logger
logger = logging.getLogger(__name__)


def get_filesystem_files(watch_folder: Path) -> dict[str, datetime]:
    """Scan watch folder for markdown files with modification dates.

    Args:
        watch_folder: Directory to scan for markdown files

    Returns:
        Dict mapping relative file paths to modification datetimes

    Example:
        files = get_filesystem_files(Path("/data/docs"))
        # {"doc1.md": datetime(2026, 1, 15, 10, 0, 0), ...}
    """
    filesystem_files: dict[str, datetime] = {}

    if not watch_folder.exists() or not watch_folder.is_dir():
        return filesystem_files

    # Recursively find all markdown files
    for md_file in watch_folder.rglob("*.md"):
        if md_file.is_file():
            # Get relative path and modification time (UTC)
            relative_path = str(md_file.relative_to(watch_folder))
            mod_time = datetime.fromtimestamp(md_file.stat().st_mtime, tz=timezone.utc)
            filesystem_files[relative_path] = mod_time

    return filesystem_files


@app.callback()
def watch(
    folder: Path | None = typer.Option(
        None, "--folder", help="Override watch folder from settings"
    ),
) -> None:
    """Monitor directory for markdown changes and process automatically.

    Performs startup batch recovery to process any files that were added or modified
    while the watcher was not running, then starts continuous monitoring for real-time
    processing of file changes.

    Args:
        folder: Optional override for watch folder (defaults to settings)

    Example:
        # Use default watch folder
        crawl4r watch

        # Override watch folder
        crawl4r watch --folder /path/to/docs
    """
    asyncio.run(_watch_async(folder))


async def _watch_async(folder: Path | None) -> None:
    """Async implementation of watch command.

    Args:
        folder: Optional override for watch folder
    """
    # Load configuration
    # Pydantic BaseSettings loads watch_folder from environment via env_file.
    # Type checker expects call-arg but Pydantic handles this dynamically.
    config = Settings()  # type: ignore[call-arg]
    if folder is not None:
        config.watch_folder = folder

    # Setup logging (only if not already configured)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=config.log_level,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )

    # Configure LlamaIndex settings
    # Lazy import to avoid loading transformers for other CLI commands
    from crawl4r.core.llama_settings import configure_llama_settings

    configure_llama_settings(app_settings=config)

    logger.info("Starting RAG ingestion pipeline...")
    logger.info(f"Watch folder: {config.watch_folder}")
    logger.info(f"Collection: {config.collection_name}")

    # Initialize components with error handling for cleanup
    tei_client = None
    vector_store = None
    processor = None

    try:
        # Initialize services
        tei_client = TEIClient(config.tei_endpoint)
        vector_store = VectorStoreManager(
            config.qdrant_url,
            config.collection_name,
            dimensions=config.embedding_dimensions,
        )
        processor = DocumentProcessor(
            config=config,
            tei_client=tei_client,
            vector_store=vector_store,
        )
        quality_verifier = QualityVerifier(
            expected_dimensions=config.embedding_dimensions
        )

        # Run startup validations
        logger.info("Validating service connections...")
        await quality_verifier.validate_tei_connection(tei_client)
        await quality_verifier.validate_qdrant_connection(
            cast(QualityVectorStoreProtocol, vector_store)
        )
    except Exception:
        # Cleanup on validation failure
        if tei_client is not None and hasattr(tei_client, 'close'):
            await tei_client.close()
        if vector_store is not None and hasattr(vector_store, 'close'):
            await vector_store.close()
        raise

    # Ensure collection and indexes exist
    logger.info("Ensuring collection exists...")
    await vector_store.ensure_collection()

    # Perform state recovery
    logger.info("Performing state recovery...")
    state_recovery = StateRecovery()

    # Get filesystem files with modification dates
    filesystem_files = get_filesystem_files(config.watch_folder)
    logger.info(f"Found {len(filesystem_files)} files in watch folder")

    # Determine which files need processing
    files_to_process_relative = await state_recovery.get_files_to_process(
        cast(RecoveryVectorStoreProtocol, vector_store), filesystem_files
    )

    # Convert relative paths to absolute Path objects
    files_to_process = [
        config.watch_folder / relative_path
        for relative_path in files_to_process_relative
    ]

    logger.info(f"Files to process: {len(files_to_process)}")

    # Process batch if files to process
    if files_to_process:
        logger.info(f"Processing batch of {len(files_to_process)} files...")
        batch_results = await processor.process_batch(files_to_process)
        successful = sum(1 for r in batch_results if r.success)
        failed = sum(1 for r in batch_results if not r.success)
        logger.info(
            f"Batch processing complete: {successful} successful, {failed} failed"
        )
    else:
        logger.info("No files to process")

    # Start watchdog observer
    logger.info("Starting file watcher...")
    file_watcher = FileWatcher(
        config=config,
        processor=processor,
        vector_store=vector_store,
    )

    observer = Observer()
    observer.schedule(
        cast(FileSystemEventHandler, file_watcher),
        str(config.watch_folder),
        recursive=True,
    )
    observer.start()

    logger.info("File watcher started. Monitoring for changes...")
    logger.info("Press Ctrl+C to stop")

    # Run event processing loop
    try:
        observer.join()
    except KeyboardInterrupt:
        # Handle KeyboardInterrupt for clean shutdown
        logger.info("Received interrupt signal, shutting down...")
        observer.stop()
        observer.join(timeout=5.0)

        if observer.is_alive():
            logger.warning("Observer thread did not stop within 5s timeout")

        logger.info("Shutdown complete")
