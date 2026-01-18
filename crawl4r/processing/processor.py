import asyncio
import time
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from llama_index.core import Settings as LlamaSettings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

from crawl4r.core.config import Settings
from crawl4r.core.instrumentation import (
    DocumentProcessingEndEvent,
    DocumentProcessingStartEvent,
    dispatcher,
)
from crawl4r.core.metadata import MetadataKeys
from crawl4r.processing.chunker import MarkdownChunker
from crawl4r.storage.llama_embeddings import TEIEmbedding
from crawl4r.storage.qdrant import VectorStoreManager
from crawl4r.storage.tei import TEIClient

# Constants for batch processing
DEFAULT_BATCH_CHUNK_SIZE = 50  # Process this many documents per memory chunk
MAX_EMBEDDING_BATCH_SIZE = 50  # Max chunks per TEI embed_batch call (TEI limit ~100)


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
        tei_client: Client for TEI embedding service with circuit breaker (optional)
        vector_store: Manager for Qdrant vector storage with retry logic
        node_parser: LlamaIndex MarkdownNodeParser for chunking markdown content
        embed_model: LlamaIndex embedding model for generating embeddings
    """

    config: Settings
    tei_client: TEIClient | None
    vector_store: VectorStoreManager
    node_parser: MarkdownNodeParser
    embed_model: BaseEmbedding

    def __init__(
        self,
        config: Settings,
        vector_store: VectorStoreManager,
        chunker: MarkdownChunker | None = None,
        tei_client: TEIClient | None = None,
        embed_model: BaseEmbedding | None = None,
        docstore: SimpleDocumentStore | None = None,
    ) -> None:
        """Initialize the document processor with required dependencies.

        All dependencies must be pre-configured and ready to use. This constructor
        performs no validation or initialization of services.

        Args:
            config: Application configuration settings containing watch_folder path,
                chunk size, overlap, and other processing parameters
            vector_store: Initialized Qdrant vector store manager with collection
                and indexes already set up
            chunker: DEPRECATED. This parameter is ignored. MarkdownNodeParser
                is now used automatically for chunking.
            tei_client: Initialized TEI client with circuit breaker configured.
                Optional if embed_model is provided or Settings.embed_model is set.
            embed_model: LlamaIndex embedding model. If None, uses tei_client
                to create TEIEmbedding, or falls back to Settings.embed_model.
            docstore: Optional docstore for deduplication. If None, creates
                new SimpleDocumentStore for session-scoped deduplication.

        Raises:
            ValueError: If no embedding model is available (no embed_model,
                no tei_client, and Settings.embed_model is None).

        Examples:
            Using tei_client:
                >>> config = Settings(watch_folder="/data/docs")
                >>> tei = TEIClient("http://crawl4r-embeddings:80")
                >>> store = VectorStoreManager("http://qdrant:6333", "crawl4r")
                >>> processor = DocumentProcessor(
                ...     config, store, tei_client=tei
                ... )

            Using explicit embed_model:
                >>> from crawl4r.storage.llama_embeddings import TEIEmbedding
                >>> embed = TEIEmbedding(endpoint_url="http://tei:80")
                >>> processor = DocumentProcessor(
                ...     config, store, embed_model=embed
                ... )

            Using global Settings.embed_model:
                >>> from llama_index.core import Settings
                >>> Settings.embed_model = my_embed_model
                >>> processor = DocumentProcessor(config, store)
        """
        # Emit deprecation warning if chunker is passed
        if chunker is not None:
            warnings.warn(
                "chunker parameter is deprecated and ignored. "
                "MarkdownNodeParser is now used automatically.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.config = config
        self.tei_client = tei_client
        self.vector_store = vector_store
        self.docstore = docstore or SimpleDocumentStore()

        collection_name = str(config.collection_name)

        # Resolve embed model with fallback chain:
        # 1. Explicit embed_model parameter takes highest precedence
        # 2. Create TEIEmbedding from tei_client if provided
        # 3. Fall back to global Settings.embed_model (if explicitly set)
        # 4. Raise error if no embed model available
        #
        # Note: We check _embed_model (private attr) rather than embed_model (property)
        # because the property triggers default OpenAI embedding creation when accessed.
        if embed_model is not None:
            self.embed_model = embed_model
        elif tei_client is not None:
            self.embed_model = TEIEmbedding(client=tei_client)
        elif getattr(LlamaSettings, "_embed_model", None) is not None:
            # Safe to access public property now that we know _embed_model is set
            self.embed_model = LlamaSettings.embed_model
        else:
            raise ValueError(
                "Must provide embed_model, tei_client, or set Settings.embed_model"
            )

        self.node_parser = MarkdownNodeParser()

        # Initialize QdrantVectorStore using the existing client
        self.llama_vector_store = QdrantVectorStore(
            client=vector_store.client,
            collection_name=collection_name,
        )

        self.pipeline = IngestionPipeline(
            transformations=[self.node_parser, self.embed_model],
            vector_store=self.llama_vector_store,
            docstore=self.docstore,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )

    def _generate_document_id(self, file_path_relative: str) -> str:
        """Generate deterministic UUID from relative file path using SHA256.

        Creates a deterministic document ID by hashing the file path. This ensures
        that the same file path always results in the same document ID, enabling
        idempotent upserts in the vector store.

        The implementation uses SHA256 hash truncated to 128 bits (first 16 bytes)
        and converted to UUID format. This matches the pattern used in
        VectorStoreManager._generate_point_id() for consistency across the codebase.

        Args:
            file_path_relative: Relative path to the file from watch_folder

        Returns:
            UUID string derived from SHA256 hash of file path

        Notes:
            - Uses SHA256 for cryptographic-quality hash
            - Converts hash to UUID format for LlamaIndex compatibility
            - Same inputs always produce same UUID (deterministic)
            - Pattern matches qdrant.py::_generate_point_id() for consistency
        """
        import hashlib

        hash_bytes = hashlib.sha256(file_path_relative.encode()).digest()
        return str(uuid.UUID(bytes=hash_bytes[:16]))

    async def process_document(self, file_path: Path) -> ProcessingResult:
        """Process a single markdown document through the full pipeline.

        This method coordinates the complete processing flow:
        1. Validate file exists
        2. Load markdown content
        3. Extract file metadata (modification time, paths)
        4. Create LlamaIndex Document
        5. Run IngestionPipeline (chunk, embed, upsert)

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
            - IngestionPipeline handles chunking, embedding, and upserting
        """
        start_time = time.time()

        # Dispatch Start Event
        dispatcher.event(DocumentProcessingStartEvent(file_path=str(file_path)))

        try:
            # Validate file exists
            if not file_path.exists():
                error_msg = f"File not found: {file_path}"
                dispatcher.event(DocumentProcessingEndEvent(
                    file_path=str(file_path),
                    success=False,
                    error=error_msg
                ))
                return ProcessingResult(
                    success=False,
                    chunks_processed=0,
                    file_path=str(file_path),
                    time_taken=time.time() - start_time,
                    error=error_msg,
                )

            # Load document via SimpleDirectoryReader (provides default metadata)
            reader = SimpleDirectoryReader(input_files=[str(file_path)])
            docs = reader.load_data()
            if not docs:
                raise FileNotFoundError(
                    f"SimpleDirectoryReader returned no documents for: {file_path}"
                )
            doc = docs[0]

            # SimpleDirectoryReader provides: file_path, file_name, file_type,
            # file_size, creation_date, last_modified_date - all in doc.metadata

            # Derive relative path for ID generation (preserves existing behavior)
            abs_path = doc.metadata.get("file_path", str(file_path))
            try:
                file_path_relative = str(
                    Path(abs_path).relative_to(self.config.watch_folder)
                )
            except ValueError:
                # File is not relative to watch_folder, use absolute path as fallback
                file_path_relative = abs_path

            # TEMPORARY: Add legacy metadata keys for backward compatibility.
            # These will be removed in Task 4 when we migrate to MetadataKeys constants.
            # See: docs/plans/2026-01-17-use-simpledirectoryreader.md Task 4.5
            doc.metadata[MetadataKeys.FILE_PATH_RELATIVE] = file_path_relative
            doc.metadata[MetadataKeys.FILE_PATH_ABSOLUTE] = abs_path
            doc.metadata["filename"] = doc.metadata.get("file_name", file_path.name)
            # Use SimpleDirectoryReader's last_modified_date as modification_date
            doc.metadata["modification_date"] = doc.metadata.get(
                "last_modified_date", datetime.now().isoformat()
            )

            # Generate deterministic ID from relative path
            doc.id_ = self._generate_document_id(file_path_relative)

            # Run pipeline
            nodes = await self.pipeline.arun(documents=[doc])

            dispatcher.event(DocumentProcessingEndEvent(
                file_path=str(file_path),
                success=True
            ))

            return ProcessingResult(
                success=True,
                chunks_processed=len(nodes),
                file_path=str(file_path),
                time_taken=time.time() - start_time,
                error=None,
            )

        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            dispatcher.event(DocumentProcessingEndEvent(
                file_path=str(file_path),
                success=False,
                error=error_msg
            ))
            return ProcessingResult(
                success=False,
                chunks_processed=0,
                file_path=str(file_path),
                time_taken=time.time() - start_time,
                error=error_msg,
            )
        except RuntimeError as e:
            # Handle TEI and Qdrant errors
            error_msg = str(e)
            dispatcher.event(DocumentProcessingEndEvent(
                file_path=str(file_path),
                success=False,
                error=error_msg
            ))
            return ProcessingResult(
                success=False,
                chunks_processed=0,
                file_path=str(file_path),
                time_taken=time.time() - start_time,
                error=error_msg,
            )
        except Exception as e:
            # Catch-all for unexpected errors
            error_msg = f"Unexpected error: {e}"
            dispatcher.event(DocumentProcessingEndEvent(
                file_path=str(file_path),
                success=False,
                error=error_msg
            ))
            return ProcessingResult(
                success=False,
                chunks_processed=0,
                file_path=str(file_path),
                time_taken=time.time() - start_time,
                error=error_msg,
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
            failed_indices = [i for i, r in enumerate(all_results) if not r.success]

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
