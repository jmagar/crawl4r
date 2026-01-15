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

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rag_ingestion.chunker import MarkdownChunker
from rag_ingestion.config import Settings
from rag_ingestion.tei_client import TEIClient
from rag_ingestion.vector_store import VectorMetadata, VectorStoreManager


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


class DocumentProcessor:
    """Document processing pipeline coordinator.

    This class orchestrates the full RAG ingestion pipeline for markdown documents:
    loading files, chunking content, generating embeddings, and storing vectors
    with metadata in Qdrant.

    Attributes:
        config: Application configuration settings
        tei_client: Client for TEI embedding service
        vector_store: Manager for Qdrant vector storage
        chunker: Markdown chunking implementation
    """

    def __init__(
        self,
        config: Settings,
        tei_client: TEIClient,
        vector_store: VectorStoreManager,
        chunker: MarkdownChunker,
    ) -> None:
        """Initialize the document processor with required dependencies.

        Args:
            config: Application configuration settings
            tei_client: Client for TEI embedding service
            vector_store: Manager for Qdrant vector storage
            chunker: Markdown chunking implementation
        """
        self.config = config
        self.tei_client = tei_client
        self.vector_store = vector_store
        self.chunker = chunker

    async def _load_markdown_file(self, file_path: Path) -> str:
        """Load markdown file content from filesystem.

        Args:
            file_path: Path to the markdown file

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If file does not exist
        """
        return file_path.read_text(encoding="utf-8")

    async def process_document(self, file_path: Path) -> ProcessingResult:
        """Process a single markdown document through the full pipeline.

        This method coordinates the complete processing flow:
        1. Validate file exists
        2. Load markdown content
        3. Chunk document with frontmatter parsing
        4. Generate embeddings for all chunks
        5. Calculate content hash for each chunk
        6. Extract metadata (file paths, modification date, etc.)
        7. Upsert vectors to Qdrant with metadata

        Args:
            file_path: Path to the markdown file to process

        Returns:
            ProcessingResult with success status, metrics, and error info
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

            # Load file content
            content = await self._load_markdown_file(file_path)

            # Get file metadata
            stat = file_path.stat()
            modification_date = datetime.fromtimestamp(stat.st_mtime).isoformat()
            filename = file_path.name

            # Calculate relative path
            try:
                file_path_relative = str(
                    file_path.relative_to(self.config.watch_folder)
                )
            except ValueError:
                # File is not relative to watch_folder, use absolute path
                file_path_relative = str(file_path)

            file_path_absolute = str(file_path.absolute())

            # Chunk the document
            chunks = self.chunker.chunk(content, filename=filename)

            # Extract chunk texts for embedding
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]

            # Generate embeddings for all chunks
            embeddings = await self.tei_client.embed_batch(chunk_texts)

            # Prepare vectors with metadata for upsert
            vectors_with_metadata: list[dict[str, list[float] | VectorMetadata]] = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Calculate content hash for this chunk
                content_hash = hashlib.sha256(
                    chunk["chunk_text"].encode()
                ).hexdigest()

                # Build metadata
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

                # Add tags if present
                if chunk["tags"]:
                    metadata["tags"] = chunk["tags"]

                vectors_with_metadata.append(
                    {"vector": embedding, "metadata": metadata}
                )

            # Upsert vectors to Qdrant
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
        """Process multiple documents, continuing on errors.

        This method processes each document independently. If one document fails,
        processing continues for the remaining documents.

        Args:
            file_paths: List of file paths to process

        Returns:
            List of ProcessingResult objects, one per file
        """
        results: list[ProcessingResult] = []

        for file_path in file_paths:
            result = await self.process_document(file_path)
            results.append(result)

        return results
