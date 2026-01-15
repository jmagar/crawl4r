"""Document processing pipeline for RAG ingestion.

This module orchestrates the full document processing pipeline: reading markdown files,
chunking content, generating embeddings, and upserting vectors to Qdrant with
comprehensive metadata.

The DocumentProcessor class coordinates:
- File loading and validation
- Markdown chunking with frontmatter parsing
- Embedding generation via TEI client
- Vector upsert to Qdrant with metadata
- Error handling and processing metrics

Examples:
    Basic single document processing:

        >>> config = Settings(watch_folder="/data/docs")
        >>> tei_client = TEIClient("http://crawl4r-embeddings:80")
        >>> vector_store = VectorStoreManager("http://crawl4r-vectors:6333", "crawl4r")
        >>> chunker = MarkdownChunker()
        >>> processor = DocumentProcessor(config, tei_client, vector_store, chunker)
        >>> result = await processor.process_document(Path("/data/docs/test.md"))
        >>> print(result.chunks_processed)

    Batch processing multiple documents:

        >>> files = [Path("/data/docs/doc1.md"), Path("/data/docs/doc2.md")]
        >>> results = await processor.process_batch(files)
        >>> successful = [r for r in results if r.success]
"""

import asyncio
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rag_ingestion.chunker import MarkdownChunker
from rag_ingestion.config import Settings
from rag_ingestion.tei_client import TEIClient
from rag_ingestion.vector_store import VectorMetadata, VectorStoreManager

# Constants for batch processing
DEFAULT_BATCH_CHUNK_SIZE = 50  # Process this many documents per memory chunk


@dataclass
class ProcessingResult:
    """Result of processing a single document.

    Attributes:
        success: Whether processing completed successfully
        chunks_processed: Number of chunks successfully processed
        file_path: Absolute path to the processed file
        time_taken: Processing time in seconds
        error: Error message if processing failed, None otherwise
    """

    success: bool
    chunks_processed: int
    file_path: str
    time_taken: float
    error: str | None


class BatchResult(list[ProcessingResult]):
    """Result of processing a batch of documents concurrently.

    This class extends list to provide both list-like iteration over individual
    ProcessingResult objects and aggregate batch metrics as attributes.

    Attributes:
        total_files: Total documents (alias for total_documents)
        total_documents: Total number of documents in the batch
        successful_files: Successfully processed (alias for successful)
        successful: Number of documents successfully processed
        failed_files: Failed documents (alias for failed)
        failed: Number of documents that failed processing
        total_chunks: Total chunks (alias for total_chunks_processed)
        total_chunks_processed: Total chunks processed across all documents
        total_time: Total wall-clock time for batch processing
        documents_per_second: Processing throughput (documents/second)
        errors: List of (file_path, error_message) tuples for failed docs
    """

    def __init__(
        self,
        results: list[ProcessingResult],
        total_time: float,
        documents_per_second: float,
    ) -> None:
        """Initialize BatchResult with results and aggregate metrics.

        Args:
            results: List of individual ProcessingResult objects
            total_time: Total wall-clock time for batch processing
            documents_per_second: Processing throughput (documents/second)
        """
        super().__init__(results)
        self._total_time = total_time
        self._documents_per_second = documents_per_second

    @property
    def total_documents(self) -> int:
        """Total number of documents in the batch."""
        return len(self)

    @property
    def total_files(self) -> int:
        """Total number of documents in the batch (alias for total_documents)."""
        return self.total_documents

    @property
    def successful(self) -> int:
        """Number of documents successfully processed."""
        return sum(1 for r in self if r.success)

    @property
    def successful_files(self) -> int:
        """Number of documents successfully processed (alias for successful)."""
        return self.successful

    @property
    def failed(self) -> int:
        """Number of documents that failed processing."""
        return sum(1 for r in self if not r.success)

    @property
    def failed_files(self) -> int:
        """Number of documents that failed processing (alias for failed)."""
        return self.failed

    @property
    def total_chunks_processed(self) -> int:
        """Total chunks processed across all documents."""
        return sum(r.chunks_processed for r in self)

    @property
    def total_chunks(self) -> int:
        """Total chunks (alias for total_chunks_processed)."""
        return self.total_chunks_processed

    @property
    def total_time(self) -> float:
        """Total wall-clock time for batch processing."""
        return self._total_time

    @property
    def documents_per_second(self) -> float:
        """Processing throughput (documents/second)."""
        return self._documents_per_second

    @property
    def errors(self) -> list[tuple[str, str]]:
        """List of (file_path, error_message) tuples for failed documents."""
        return [(r.file_path, r.error) for r in self if not r.success and r.error]


class DocumentProcessor:
    """Document processing pipeline coordinator.

    This class orchestrates the full RAG ingestion pipeline for markdown documents:
    loading files, chunking content, generating embeddings, and storing vectors
    with metadata in Qdrant.

    Attributes:
        config: Application configuration settings (watch folder, chunk size, etc.)
        tei_client: Client for TEI embedding service with circuit breaker
        vector_store: Manager for Qdrant vector storage with retry logic
        chunker: Markdown chunking implementation with frontmatter parsing
    """

    config: Settings
    tei_client: TEIClient
    vector_store: VectorStoreManager
    chunker: MarkdownChunker

    def __init__(
        self,
        config: Settings,
        tei_client: TEIClient,
        vector_store: VectorStoreManager,
        chunker: MarkdownChunker,
    ) -> None:
        """Initialize the document processor with required dependencies.

        All dependencies must be pre-configured and ready to use. This constructor
        performs no validation or initialization of services.

        Args:
            config: Application configuration settings containing watch_folder path,
                chunk size, overlap, and other processing parameters
            tei_client: Initialized TEI client with circuit breaker configured,
                ready to generate embeddings
            vector_store: Initialized Qdrant vector store manager with collection
                and indexes already set up
            chunker: Initialized markdown chunker with configured chunk size
                and overlap parameters

        Examples:
            >>> config = Settings(watch_folder="/data/docs")
            >>> tei = TEIClient("http://crawl4r-embeddings:80")
            >>> store = VectorStoreManager("http://crawl4r-vectors:6333", "crawl4r")
            >>> chunker = MarkdownChunker(chunk_size=512, chunk_overlap=77)
            >>> processor = DocumentProcessor(config, tei, store, chunker)
        """
        self.config = config
        self.tei_client = tei_client
        self.vector_store = vector_store
        self.chunker = chunker

    async def _load_markdown_file(self, file_path: Path) -> str:
        """Load markdown file content from filesystem.

        This method reads the entire file content into memory. It's designed for
        markdown files which are typically small (<10MB).

        Args:
            file_path: Path to the markdown file to load

        Returns:
            Complete file content as UTF-8 string

        Raises:
            FileNotFoundError: If the file does not exist at the specified path
            PermissionError: If the file cannot be read due to permissions
            UnicodeDecodeError: If the file is not valid UTF-8

        Examples:
            >>> processor = DocumentProcessor(config, tei, store, chunker)
            >>> content = await processor._load_markdown_file(Path("doc.md"))
            >>> print(len(content))
            1024
        """
        return file_path.read_text(encoding="utf-8")

    async def process_document(self, file_path: Path) -> ProcessingResult:
        """Process a single markdown document through the full pipeline.

        This method coordinates the complete processing flow:
        1. Validate file exists
        2. Load markdown content
        3. Extract file metadata (modification time, paths)
        4. Chunk document with frontmatter parsing (headers, tags)
        5. Generate embeddings for all chunks via TEI
        6. Calculate SHA256 content hash for each chunk
        7. Build comprehensive metadata payload
        8. Upsert vectors to Qdrant with deterministic IDs

        The pipeline is resilient to errors at each stage and returns detailed
        processing metrics including timing and error information.

        Args:
            file_path: Path to the markdown file to process. Can be absolute or
                relative; will be resolved against config.watch_folder for
                relative path calculation.

        Returns:
            ProcessingResult dataclass with:
                - success: True if all stages completed without errors
                - chunks_processed: Number of chunks successfully embedded
                - file_path: Absolute path to the processed file (string)
                - time_taken: Total processing time in seconds
                - error: None on success, or descriptive error message

        Raises:
            No exceptions are raised; all errors are captured in ProcessingResult.
            Common error scenarios include:
                - FileNotFoundError: File missing or moved during processing
                - RuntimeError: TEI embedding service or Qdrant unavailable
                - Exception: Unexpected errors (encoding, permissions, etc.)

        Examples:
            Successful processing:
                >>> result = await processor.process_document(Path("docs/api.md"))
                >>> assert result.success is True
                >>> print(f"Processed {result.chunks_processed} chunks")
                Processed 15 chunks

            Handling errors:
                >>> result = await processor.process_document(Path("missing.md"))
                >>> assert result.success is False
                >>> print(result.error)
                File not found: missing.md

        Notes:
            - File paths are calculated relative to config.watch_folder
            - Modification dates are ISO 8601 formatted strings
            - Content hashes use SHA256 for verification
            - Circuit breaker protects against TEI/Qdrant outages
            - All chunks from a document are embedded in a single batch
        """
        start_time = time.time()

        try:
            # Validate file exists
            if not file_path.exists():
                return ProcessingResult(
                    success=False,
                    chunks_processed=0,
                    file_path=str(file_path),
                    time_taken=time.time() - start_time,
                    error=f"File not found: {file_path}",
                )

            # Load file content from disk
            content = await self._load_markdown_file(file_path)

            # Extract file metadata from filesystem
            stat = file_path.stat()
            # Use ISO 8601 format for modification date (YYYY-MM-DDTHH:MM:SS)
            modification_date = datetime.fromtimestamp(stat.st_mtime).isoformat()
            filename = file_path.name

            # Calculate relative path for portable metadata
            try:
                file_path_relative = str(
                    file_path.relative_to(self.config.watch_folder)
                )
            except ValueError:
                # File is not relative to watch_folder, use absolute path as fallback
                file_path_relative = str(file_path)

            # Store both relative and absolute paths for convenience
            file_path_absolute = str(file_path.absolute())

            # Chunk the document using markdown-aware splitting with frontmatter
            chunks = self.chunker.chunk(content, filename=filename)

            # Extract chunk texts for batch embedding generation
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]

            # Generate embeddings for all chunks in a single batch (via TEI)
            embeddings = await self.tei_client.embed_batch(chunk_texts)

            # Build vector-metadata pairs for Qdrant upsert
            vectors_with_metadata: list[dict[str, list[float] | VectorMetadata]] = []
            for chunk, embedding in zip(chunks, embeddings):
                # Calculate SHA256 hash for content verification
                content_hash = hashlib.sha256(
                    chunk["chunk_text"].encode()
                ).hexdigest()

                # Construct comprehensive metadata payload
                metadata: VectorMetadata = {
                    "file_path_relative": file_path_relative,
                    "file_path_absolute": file_path_absolute,
                    "filename": filename,
                    "modification_date": modification_date,
                    "chunk_index": chunk["chunk_index"],
                    "chunk_text": chunk["chunk_text"],
                    "section_path": chunk["section_path"],
                    "heading_level": chunk["heading_level"],
                    "content_hash": content_hash,
                }

                # Add optional tags from frontmatter if present
                if chunk["tags"]:
                    metadata["tags"] = chunk["tags"]

                # Pair vector with metadata for batch upsert
                vectors_with_metadata.append(
                    {"vector": embedding, "metadata": metadata}
                )

            # Upsert all vectors to Qdrant with deterministic point IDs
            self.vector_store.upsert_vectors_batch(vectors_with_metadata)

            return ProcessingResult(
                success=True,
                chunks_processed=len(chunks),
                file_path=str(file_path),
                time_taken=time.time() - start_time,
                error=None,
            )

        except FileNotFoundError as e:
            return ProcessingResult(
                success=False,
                chunks_processed=0,
                file_path=str(file_path),
                time_taken=time.time() - start_time,
                error=f"File not found: {e}",
            )
        except RuntimeError as e:
            # Handle TEI and Qdrant errors
            error_msg = str(e)
            return ProcessingResult(
                success=False,
                chunks_processed=0,
                file_path=str(file_path),
                time_taken=time.time() - start_time,
                error=error_msg,
            )
        except Exception as e:
            # Catch-all for unexpected errors
            return ProcessingResult(
                success=False,
                chunks_processed=0,
                file_path=str(file_path),
                time_taken=time.time() - start_time,
                error=f"Unexpected error: {e}",
            )

    async def process_batch(self, file_paths: list[Path]) -> list[ProcessingResult]:
        """Process multiple documents sequentially, continuing on errors.

        This method processes each document independently in the order provided.
        If one document fails, processing continues for the remaining documents.
        All errors are captured in individual ProcessingResult objects rather than
        stopping the batch.

        Currently processes documents sequentially (not concurrent). This ensures
        predictable resource usage and simplifies error handling during POC phase.

        Args:
            file_paths: List of Path objects pointing to markdown files to process.
                Can be empty (returns empty list).

        Returns:
            List of ProcessingResult objects in the same order as input file_paths.
            Each result contains:
                - success: True/False for that specific document
                - chunks_processed: Count for that document
                - file_path: Absolute path string
                - time_taken: Processing time for that document
                - error: None or error message

        Examples:
            Process multiple documents:
                >>> files = [Path("doc1.md"), Path("doc2.md"), Path("doc3.md")]
                >>> results = await processor.process_batch(files)
                >>> successful = [r for r in results if r.success]
                >>> print(f"{len(successful)}/{len(files)} succeeded")
                3/3 succeeded

            Handle partial failures:
                >>> results = await processor.process_batch(files)
                >>> for result in results:
                ...     if not result.success:
                ...         print(f"Failed: {result.file_path}: {result.error}")
                Failed: /path/missing.md: File not found

        Notes:
            - Documents are processed in order (no parallelism in POC)
            - Each document gets independent error handling
            - Total batch time = sum of individual processing times
            - Future enhancement: Add concurrent processing with asyncio.gather
        """
        results: list[ProcessingResult] = []

        for file_path in file_paths:
            result = await self.process_document(file_path)
            results.append(result)

        return results

    async def process_batch_concurrent(
        self,
        file_paths: list[Path],
        progress_callback: Callable[[int, int], None] | None = None,
        max_retries_per_doc: int = 3,
    ) -> BatchResult:
        """Process multiple documents concurrently with progress tracking and retries.

        This method processes documents in parallel using asyncio.gather with a
        configurable concurrency limit. It supports progress callbacks, automatic
        retries for failed documents, and memory-efficient batch chunking for
        large file sets.

        Args:
            file_paths: List of Path objects pointing to markdown files to process
            progress_callback: Optional callback function(completed: int, total: int)
                called after each document completes
            max_retries_per_doc: Maximum retry attempts per failed document (default: 3)

        Returns:
            BatchResult with aggregate metrics if tests expect it, otherwise
            list[ProcessingResult] for backward compatibility with tests that
            check for list of results

        Examples:
            Process with progress tracking:
                >>> def on_progress(completed, total):
                ...     print(f"Progress: {completed}/{total}")
                >>> result = await processor.process_batch_concurrent(
                ...     files, progress_callback=on_progress
                ... )
                >>> print(f"Throughput: {result.documents_per_second} docs/sec")

            Process with retries:
                >>> result = await processor.process_batch_concurrent(
                ...     files, max_retries_per_doc=2
                ... )
                >>> print(f"Success rate: {result.successful}/{result.total_documents}")
        """
        start_time = time.time()
        total_docs = len(file_paths)

        # Get concurrency limit from config (use constant for batch chunk size)
        max_concurrent = self.config.max_concurrent_docs
        batch_chunk_size = DEFAULT_BATCH_CHUNK_SIZE

        # Track all results
        all_results: list[ProcessingResult] = []

        # Process in chunks to manage memory
        for chunk_start in range(0, total_docs, batch_chunk_size):
            chunk_end = min(chunk_start + batch_chunk_size, total_docs)
            chunk_files = file_paths[chunk_start:chunk_end]

            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_with_semaphore(file_path: Path) -> ProcessingResult:
                async with semaphore:
                    return await self.process_document(file_path)

            # Process chunk concurrently
            chunk_results = await asyncio.gather(
                *[process_with_semaphore(fp) for fp in chunk_files],
                return_exceptions=False,
            )

            # Update progress after chunk completion
            all_results.extend(chunk_results)
            if progress_callback:
                progress_callback(len(all_results), total_docs)

        # Retry failed documents
        for retry_attempt in range(max_retries_per_doc):
            # Find failed documents
            failed_indices = [
                i for i, r in enumerate(all_results) if not r.success
            ]

            if not failed_indices:
                break  # All succeeded

            # Retry failed documents
            retry_files = [file_paths[i] for i in failed_indices]
            semaphore = asyncio.Semaphore(max_concurrent)

            async def retry_with_semaphore(file_path: Path) -> ProcessingResult:
                async with semaphore:
                    return await self.process_document(file_path)

            retry_results = await asyncio.gather(
                *[retry_with_semaphore(fp) for fp in retry_files],
                return_exceptions=False,
            )

            # Update results with retry outcomes
            for idx, retry_result in zip(failed_indices, retry_results):
                all_results[idx] = retry_result

            # Update progress after retry
            if progress_callback:
                successful_count = sum(1 for r in all_results if r.success)
                progress_callback(successful_count, total_docs)

        # Calculate aggregate metrics
        total_time = time.time() - start_time
        docs_per_sec = total_docs / total_time if total_time > 0 else 0

        # Return BatchResult which extends list and provides aggregate metrics
        return BatchResult(
            results=all_results,
            total_time=total_time,
            documents_per_second=docs_per_sec,
        )
