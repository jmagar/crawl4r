"""Main entry point for RAG ingestion pipeline.

Orchestrates the complete document ingestion workflow including startup
validation, state recovery, batch processing, and real-time file monitoring.

Main orchestration flow:
1. Load configuration from environment
2. Initialize all components (embedder, vector store, chunker, processor)
3. Validate service connections (TEI, Qdrant)
4. Ensure collection and indexes exist
5. Perform state recovery to identify unprocessed files
6. Process batch of recovered files
7. Start watchdog observer for real-time monitoring
8. Run event processing loop
9. Handle graceful shutdown on KeyboardInterrupt

Example:
    python -m crawl4r.cli.main

Requirements:
    - FR-3: Batch processing on startup
    - FR-6: Environment configuration
    - FR-7: Service validation
    - FR-8: State recovery
    - FR-9: Graceful shutdown
    - AC-1.5: Watch mode after batch
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import cast

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from crawl4r.core.config import Settings
from crawl4r.core.llama_settings import configure_llama_settings
from crawl4r.core.quality import QualityVerifier
from crawl4r.core.quality import VectorStoreProtocol as QualityVectorStoreProtocol
from crawl4r.processing.chunker import MarkdownChunker
from crawl4r.processing.processor import DocumentProcessor
from crawl4r.readers.file_watcher import FileWatcher
from crawl4r.resilience.recovery import StateRecovery
from crawl4r.resilience.recovery import (
    VectorStoreProtocol as RecoveryVectorStoreProtocol,
)
from crawl4r.storage.tei import TEIClient
from crawl4r.storage.qdrant import VectorStoreManager

# Module-level logger for event processing loop
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
            # Get relative path and modification time
            relative_path = str(md_file.relative_to(watch_folder))
            mod_time = datetime.fromtimestamp(md_file.stat().st_mtime)
            filesystem_files[relative_path] = mod_time

    return filesystem_files


async def process_events_loop(
    event_queue: asyncio.Queue[dict[str, str]],
    processor: DocumentProcessor,
    vector_store: VectorStoreManager,
    watch_folder: Path | None = None,
) -> AsyncIterator[None]:
    """Process events from the queue in an infinite loop.

    Async generator that processes file system events from the queue,
    handling create, modify, and delete operations. Logs queue depth
    periodically and handles exceptions gracefully to ensure continuous
    operation.

    Args:
        event_queue: Asyncio queue containing events to process
        processor: DocumentProcessor instance for file processing
        vector_store: VectorStoreManager instance for vector operations
        watch_folder: Optional Path to watch folder for relative path calculations

    Yields:
        None after each event is processed (for test control)

    Example:
        queue = asyncio.Queue()
        await queue.put({"type": "created", "path": "/data/docs/new.md"})

        async for _ in process_events_loop(queue, processor, vector_store):
            # Event processed, can break for testing
            break

    Notes:
        - Runs indefinitely until cancelled
        - Logs queue depth to monitor backlog
        - Exceptions are logged and don't stop the loop
        - Calls event_queue.task_done() after each event
    """
    event_count = 0

    while True:
        try:
            # Get next event from queue
            event = await event_queue.get()

            # Log queue depth periodically
            event_count += 1
            logger.info("Queue depth: %d", event_queue.qsize())

            # Extract event details
            event_type = event.get("type", "")
            event_path_str = event.get("path", "")
            file_path = Path(event_path_str)

            # Process event based on type
            try:
                if event_type == "created":
                    await processor.process_document(file_path)
                elif event_type == "modified":
                    # Delete old vectors then reprocess
                    vector_store.delete_by_file(event_path_str)
                    await processor.process_document(file_path)
                elif event_type == "deleted":
                    # Delete vectors only
                    vector_store.delete_by_file(event_path_str)
                else:
                    logger.warning(f"Unknown event type: {event_type}")

            except Exception as e:
                # Log error but continue processing other events
                logger.error(
                    f"Error processing event {event_type} for {file_path}: {e}"
                )

            # Mark task as done
            event_queue.task_done()

            # Yield to allow tests to break after processing events
            yield

        except asyncio.CancelledError:
            # Task was cancelled, exit cleanly
            logger.info("Event processing loop cancelled")
            break
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unexpected error in event loop: {e}")
            # Continue processing


def setup_components(
    config: Settings,
) -> tuple[
    TEIClient,
    MarkdownChunker,
    VectorStoreManager,
    DocumentProcessor,
    QualityVerifier,
]:
    """Initialize all pipeline components with configuration.

    Creates instances of TEI client, chunker, vector store, processor,
    and quality verifier based on provided configuration settings.

    Args:
        config: Settings object with configuration from environment

    Returns:
        Tuple containing:
            - tei_client: TEI embedding client
            - chunker: Markdown document chunker
            - vector_store: Qdrant vector store manager
            - processor: Document processor orchestrator
            - quality_verifier: Service validation component

    Example:
        config = Settings()
        tei, chunker, store, proc, verifier = setup_components(config)
    """
    tei_client = TEIClient(config.tei_endpoint)
    chunker = MarkdownChunker(
        chunk_size_tokens=config.chunk_size_tokens,
        chunk_overlap_percent=config.chunk_overlap_percent,
    )
    vector_store = VectorStoreManager(
        config.qdrant_url,
        config.collection_name,
        dimensions=1024,  # Expected from Qwen3-Embedding-0.6B
    )
    processor = DocumentProcessor(
        config=config,
        tei_client=tei_client,
        vector_store=vector_store,
        chunker=chunker,
    )
    quality_verifier = QualityVerifier(expected_dimensions=1024)

    return tei_client, chunker, vector_store, processor, quality_verifier


async def run_startup_validations(
    quality_verifier: QualityVerifier,
    tei_client: TEIClient,
    vector_store: VectorStoreManager,
) -> None:
    """Execute all startup validation checks.

    Validates that TEI embedding service and Qdrant vector database
    are both accessible and responding correctly before processing begins.

    Args:
        quality_verifier: Validator for service health checks
        tei_client: TEI client to validate
        vector_store: Vector store to validate

    Raises:
        SystemExit: If any validation check fails

    Example:
        verifier = QualityVerifier(expected_dimensions=1024)
        tei = TEIClient("http://localhost:80")
        store = VectorStoreManager("http://localhost:6333", "docs", 1024)
        await run_startup_validations(verifier, tei, store)
    """
    module_logger = logging.getLogger(__name__)
    module_logger.info("Validating service connections...")
    await quality_verifier.validate_tei_connection(tei_client)
    await quality_verifier.validate_qdrant_connection(
        cast(QualityVectorStoreProtocol, vector_store)
    )


async def main() -> None:
    """Main entry point for RAG ingestion pipeline.

    Orchestrates the complete startup sequence, batch processing, and
    continuous file monitoring for the RAG ingestion system. Handles
    graceful shutdown on KeyboardInterrupt.

    Flow:
        1. Load configuration from environment
        2. Setup structured logging
        3. Initialize all components (TEI, Qdrant, processor, etc.)
        4. Run startup validations (service health checks)
        5. Ensure collection and indexes exist in Qdrant
        6. Perform state recovery (identify unprocessed files)
        7. Process batch of recovered files
        8. Start watchdog file observer
        9. Monitor for file changes until interrupted
        10. Gracefully shutdown on Ctrl+C

    Raises:
        SystemExit: If startup validation fails (TEI or Qdrant unavailable)

    Example:
        # Run from command line
        python -m crawl4r.cli.main

        # Or programmatically
        from crawl4r.cli.main import main
        asyncio.run(main())

    Requirements:
        - FR-3: Batch processing on startup
        - FR-6: Environment configuration
        - FR-7: Service validation
        - FR-8: State recovery
        - FR-9: Graceful shutdown
        - AC-1.5: Watch mode after batch processing
    """
    # 1. Load configuration from Settings()
    # Note: Pydantic BaseSettings loads watch_folder from environment
    config = Settings()  # type: ignore[call-arg]

    # 2. Setup logger
    module_logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # 3. Configure LlamaIndex settings (after logging is initialized)
    configure_llama_settings(app_settings=config)

    module_logger.info("Starting RAG ingestion pipeline...")
    module_logger.info(f"Watch folder: {config.watch_folder}")
    module_logger.info(f"Collection: {config.collection_name}")

    # 3. Initialize all components
    tei_client, chunker, vector_store, processor, quality_verifier = setup_components(
        config
    )

    # 4. Run startup validations
    await run_startup_validations(quality_verifier, tei_client, vector_store)

    # 5. Ensure collection and indexes exist
    module_logger.info("Ensuring collection exists...")
    vector_store.ensure_collection()

    # 6. Perform state recovery
    module_logger.info("Performing state recovery...")
    state_recovery = StateRecovery()

    # Get filesystem files with modification dates
    filesystem_files = get_filesystem_files(config.watch_folder)
    module_logger.info(f"Found {len(filesystem_files)} files in watch folder")

    # Determine which files need processing
    files_to_process_relative = await state_recovery.get_files_to_process(
        cast(RecoveryVectorStoreProtocol, vector_store), filesystem_files
    )

    # Convert relative paths to absolute Path objects
    files_to_process = [
        config.watch_folder / relative_path
        for relative_path in files_to_process_relative
    ]

    module_logger.info(f"Files to process: {len(files_to_process)}")

    # 7. Process batch if files to process
    if files_to_process:
        module_logger.info(f"Processing batch of {len(files_to_process)} files...")
        batch_results = await processor.process_batch(files_to_process)
        successful = sum(1 for r in batch_results if r.success)
        failed = sum(1 for r in batch_results if not r.success)
        module_logger.info(
            f"Batch processing complete: {successful} successful, {failed} failed"
        )
    else:
        module_logger.info("No files to process")

    # 8. Start watchdog observer
    module_logger.info("Starting file watcher...")
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

    module_logger.info("File watcher started. Monitoring for changes...")
    module_logger.info("Press Ctrl+C to stop")

    # 9. Run event processing loop
    try:
        observer.join()
    except KeyboardInterrupt:
        # 10. Handle KeyboardInterrupt for clean shutdown
        module_logger.info("Shutting down gracefully...")
        observer.stop()
        observer.join()
        module_logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
