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
    python -m rag_ingestion.main

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
from datetime import datetime
from pathlib import Path

from watchdog.observers import Observer

from rag_ingestion.chunker import MarkdownChunker
from rag_ingestion.config import Settings
from rag_ingestion.file_watcher import FileWatcher
from rag_ingestion.processor import DocumentProcessor
from rag_ingestion.quality import QualityVerifier
from rag_ingestion.recovery import StateRecovery
from rag_ingestion.tei_client import TEIClient
from rag_ingestion.vector_store import VectorStoreManager

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
    event_queue: asyncio.Queue,
    processor: DocumentProcessor,
    vector_store: VectorStoreManager,
    watch_folder: Path | None = None,
):
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
                    await vector_store.delete_by_file_path(event_path_str)
                    await processor.process_document(file_path)
                elif event_type == "deleted":
                    # Delete vectors only
                    await vector_store.delete_by_file_path(event_path_str)
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


async def main() -> None:
    """Main entry point for RAG ingestion pipeline.

    Orchestrates complete startup sequence, batch processing, and watch mode.
    Handles graceful shutdown on KeyboardInterrupt.

    Raises:
        SystemExit: If startup validation fails (TEI or Qdrant unavailable)

    Example:
        asyncio.run(main())
    """
    # 1. Load configuration from Settings()
    config = Settings()

    # 2. Setup logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info("Starting RAG ingestion pipeline...")
    logger.info(f"Watch folder: {config.watch_folder}")
    logger.info(f"Collection: {config.collection_name}")

    # 3. Initialize all components
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

    # 4. Run startup validations
    logger.info("Validating service connections...")
    await quality_verifier.validate_tei_connection(tei_client)
    await quality_verifier.validate_qdrant_connection(vector_store)

    # 5. Ensure collection and indexes exist
    logger.info("Ensuring collection exists...")
    vector_store.ensure_collection()

    # 6. Perform state recovery
    logger.info("Performing state recovery...")
    state_recovery = StateRecovery()

    # Get filesystem files with modification dates
    filesystem_files = get_filesystem_files(config.watch_folder)
    logger.info(f"Found {len(filesystem_files)} files in watch folder")

    # Determine which files need processing
    files_to_process_relative = await state_recovery.get_files_to_process(
        vector_store, filesystem_files
    )

    # Convert relative paths to absolute Path objects
    files_to_process = [
        config.watch_folder / relative_path
        for relative_path in files_to_process_relative
    ]

    logger.info(f"Files to process: {len(files_to_process)}")

    # 7. Process batch if files to process
    if files_to_process:
        logger.info(f"Processing batch of {len(files_to_process)} files...")
        batch_result = await processor.process_batch(files_to_process)
        logger.info(
            f"Batch processing complete: {batch_result.successful} successful, "
            f"{batch_result.failed} failed"
        )
    else:
        logger.info("No files to process")

    # 8. Start watchdog observer
    logger.info("Starting file watcher...")
    file_watcher = FileWatcher(
        config=config,
        processor=processor,
        vector_store=vector_store,
    )

    observer = Observer()
    observer.schedule(file_watcher, str(config.watch_folder), recursive=True)
    observer.start()

    logger.info("File watcher started. Monitoring for changes...")
    logger.info("Press Ctrl+C to stop")

    # 9. Run event processing loop
    try:
        observer.join()
    except KeyboardInterrupt:
        # 10. Handle KeyboardInterrupt for clean shutdown
        logger.info("Shutting down gracefully...")
        observer.stop()
        observer.join()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
