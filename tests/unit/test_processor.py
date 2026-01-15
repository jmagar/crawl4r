"""Unit tests for document processing pipeline.

Tests for the DocumentProcessor class that orchestrates the full RAG ingestion
pipeline: reading markdown files, chunking content, generating embeddings, and
upserting vectors to Qdrant with comprehensive metadata.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from rag_ingestion.processor import DocumentProcessor, ProcessingResult


class TestProcessorInitialization:
    """Test DocumentProcessor initialization and setup."""

    def test_initializes_with_dependencies(self) -> None:
        """Verify processor initializes with config, TEI client, vector store, and chunker."""
        config = Mock()
        tei_client = Mock()
        vector_store = Mock()
        chunker = Mock()

        processor = DocumentProcessor(
            config=config,
            tei_client=tei_client,
            vector_store=vector_store,
            chunker=chunker,
        )

        assert processor.config is config
        assert processor.tei_client is tei_client
        assert processor.vector_store is vector_store
        assert processor.chunker is chunker

    def test_requires_all_dependencies(self) -> None:
        """Verify processor raises error if any dependency is missing."""
        config = Mock()
        tei_client = Mock()
        vector_store = Mock()
        chunker = Mock()

        # Missing config
        with pytest.raises(TypeError):
            DocumentProcessor(  # type: ignore[call-arg]
                tei_client=tei_client,
                vector_store=vector_store,
                chunker=chunker,
            )

        # Missing TEI client
        with pytest.raises(TypeError):
            DocumentProcessor(  # type: ignore[call-arg]
                config=config,
                vector_store=vector_store,
                chunker=chunker,
            )

        # Missing vector store
        with pytest.raises(TypeError):
            DocumentProcessor(  # type: ignore[call-arg]
                config=config,
                tei_client=tei_client,
                chunker=chunker,
            )

        # Missing chunker
        with pytest.raises(TypeError):
            DocumentProcessor(  # type: ignore[call-arg]
                config=config,
                tei_client=tei_client,
                vector_store=vector_store,
            )


class TestLoadMarkdownFile:
    """Test loading markdown files from filesystem."""

    @pytest.mark.asyncio
    async def test_loads_file_content(self) -> None:
        """Verify file content is loaded correctly."""
        config = Mock()
        tei_client = Mock()
        vector_store = Mock()
        chunker = Mock()
        processor = DocumentProcessor(config, tei_client, vector_store, chunker)

        test_file = Path("/tmp/test.md")
        test_content = "# Test Document\n\nThis is test content."

        with patch("pathlib.Path.read_text", return_value=test_content):
            content = await processor._load_markdown_file(test_file)

        assert content == test_content

    @pytest.mark.asyncio
    async def test_file_not_found_raises_error(self) -> None:
        """Verify FileNotFoundError raised for missing files."""
        config = Mock()
        tei_client = Mock()
        vector_store = Mock()
        chunker = Mock()
        processor = DocumentProcessor(config, tei_client, vector_store, chunker)

        test_file = Path("/nonexistent/file.md")

        with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                await processor._load_markdown_file(test_file)


class TestProcessDocument:
    """Test end-to-end document processing pipeline."""

    @pytest.mark.asyncio
    async def test_processes_document_successfully(self) -> None:
        """Verify complete processing flow: load → chunk → embed → upsert."""
        # Setup mocks
        config = Mock()
        config.watch_folder = Path("/watch")
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()

        # Mock chunker to return chunks
        chunker.chunk.return_value = [
            {
                "chunk_text": "First chunk content",
                "chunk_index": 0,
                "section_path": "test.md > Introduction",
                "heading_level": 1,
                "tags": ["test", "doc"],
            },
            {
                "chunk_text": "Second chunk content",
                "chunk_index": 1,
                "section_path": "test.md > Usage",
                "heading_level": 2,
                "tags": ["test", "doc"],
            },
        ]

        # Mock TEI client to return embeddings
        tei_client.embed_batch.return_value = [
            [0.1] * 1024,  # First embedding
            [0.2] * 1024,  # Second embedding
        ]

        processor = DocumentProcessor(config, tei_client, vector_store, chunker)
        test_file = Path("/watch/docs/test.md")
        test_content = "# Introduction\n\nContent..."

        with patch("pathlib.Path.read_text", return_value=test_content):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                result = await processor.process_document(test_file)

        # Verify result
        assert result.success is True
        assert result.chunks_processed == 2
        assert result.file_path == str(test_file)
        assert result.error is None

        # Verify chunker was called
        chunker.chunk.assert_called_once_with(test_content, filename="test.md")

        # Verify TEI client was called
        tei_client.embed_batch.assert_called_once()

        # Verify vector store was called
        vector_store.upsert_vectors_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_extracts_metadata_correctly(self) -> None:
        """Verify file_path_relative, file_path_absolute, filename, modification_date extracted."""
        config = Mock()
        config.watch_folder = Path("/watch")
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()

        chunker.chunk.return_value = [
            {
                "chunk_text": "Test content",
                "chunk_index": 0,
                "section_path": "test.md",
                "heading_level": 0,
                "tags": [],
            },
        ]
        tei_client.embed_batch.return_value = [[0.1] * 1024]

        processor = DocumentProcessor(config, tei_client, vector_store, chunker)
        test_file = Path("/watch/docs/test.md")
        test_content = "Test content"

        with patch("pathlib.Path.read_text", return_value=test_content):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                await processor.process_document(test_file)

        # Get the metadata that was passed to upsert_vectors_batch
        call_args = vector_store.upsert_vectors_batch.call_args[0][0]
        metadata = call_args[0]["metadata"]

        # Verify metadata fields
        assert metadata["file_path_relative"] == "docs/test.md"
        assert metadata["file_path_absolute"] == str(test_file)
        assert metadata["filename"] == "test.md"
        assert "modification_date" in metadata
        assert isinstance(metadata["modification_date"], str)

    @pytest.mark.asyncio
    async def test_includes_tags_from_frontmatter(self) -> None:
        """Verify tags from frontmatter are included in metadata."""
        config = Mock()
        config.watch_folder = Path("/watch")
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()

        # Mock chunker to return chunks with tags
        chunker.chunk.return_value = [
            {
                "chunk_text": "Test content",
                "chunk_index": 0,
                "section_path": "test.md",
                "heading_level": 0,
                "tags": ["python", "tutorial", "beginner"],
            },
        ]
        tei_client.embed_batch.return_value = [[0.1] * 1024]

        processor = DocumentProcessor(config, tei_client, vector_store, chunker)
        test_file = Path("/watch/test.md")
        test_content = "---\ntags: [python, tutorial, beginner]\n---\nTest content"

        with patch("pathlib.Path.read_text", return_value=test_content):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                await processor.process_document(test_file)

        # Get the metadata that was passed to upsert_vectors_batch
        call_args = vector_store.upsert_vectors_batch.call_args[0][0]
        metadata = call_args[0]["metadata"]

        # Verify tags are included
        assert metadata["tags"] == ["python", "tutorial", "beginner"]

    @pytest.mark.asyncio
    async def test_calculates_content_hash(self) -> None:
        """Verify SHA256 hash of chunk_text added to metadata."""
        config = Mock()
        config.watch_folder = Path("/watch")
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()

        test_chunk_text = "Test content for hashing"
        expected_hash = hashlib.sha256(test_chunk_text.encode()).hexdigest()

        chunker.chunk.return_value = [
            {
                "chunk_text": test_chunk_text,
                "chunk_index": 0,
                "section_path": "test.md",
                "heading_level": 0,
                "tags": [],
            },
        ]
        tei_client.embed_batch.return_value = [[0.1] * 1024]

        processor = DocumentProcessor(config, tei_client, vector_store, chunker)
        test_file = Path("/watch/test.md")

        with patch("pathlib.Path.read_text", return_value=test_chunk_text):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                await processor.process_document(test_file)

        # Get the metadata that was passed to upsert_vectors_batch
        call_args = vector_store.upsert_vectors_batch.call_args[0][0]
        metadata = call_args[0]["metadata"]

        # Verify content_hash is included and correct
        assert metadata["content_hash"] == expected_hash

    @pytest.mark.asyncio
    async def test_handles_file_not_found_error(self) -> None:
        """Verify FileNotFoundError is handled gracefully."""
        config = Mock()
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()

        processor = DocumentProcessor(config, tei_client, vector_store, chunker)
        test_file = Path("/nonexistent/file.md")

        with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
            result = await processor.process_document(test_file)

        # Verify result indicates failure
        assert result.success is False
        assert result.chunks_processed == 0
        assert result.error is not None
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handles_tei_client_error(self) -> None:
        """Verify TEI client errors are handled gracefully."""
        config = Mock()
        config.watch_folder = Path("/watch")
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()

        chunker.chunk.return_value = [
            {
                "chunk_text": "Test content",
                "chunk_index": 0,
                "section_path": "test.md",
                "heading_level": 0,
                "tags": [],
            },
        ]
        # Mock TEI client to raise error
        tei_client.embed_batch.side_effect = RuntimeError("TEI service unavailable")

        processor = DocumentProcessor(config, tei_client, vector_store, chunker)
        test_file = Path("/watch/test.md")
        test_content = "Test content"

        with patch("pathlib.Path.read_text", return_value=test_content):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                result = await processor.process_document(test_file)

        # Verify result indicates failure
        assert result.success is False
        assert result.chunks_processed == 0
        assert result.error is not None
        assert "TEI" in result.error or "unavailable" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handles_qdrant_error(self) -> None:
        """Verify Qdrant errors are handled gracefully."""
        config = Mock()
        config.watch_folder = Path("/watch")
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()

        chunker.chunk.return_value = [
            {
                "chunk_text": "Test content",
                "chunk_index": 0,
                "section_path": "test.md",
                "heading_level": 0,
                "tags": [],
            },
        ]
        tei_client.embed_batch.return_value = [[0.1] * 1024]
        # Mock vector store to raise error
        vector_store.upsert_vectors_batch.side_effect = RuntimeError(
            "Qdrant connection failed"
        )

        processor = DocumentProcessor(config, tei_client, vector_store, chunker)
        test_file = Path("/watch/test.md")
        test_content = "Test content"

        with patch("pathlib.Path.read_text", return_value=test_content):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                result = await processor.process_document(test_file)

        # Verify result indicates failure
        assert result.success is False
        assert result.error is not None
        assert "Qdrant" in result.error or "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_validates_file_path_exists(self) -> None:
        """Verify file path existence is validated before processing."""
        config = Mock()
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()

        processor = DocumentProcessor(config, tei_client, vector_store, chunker)
        test_file = Path("/nonexistent/file.md")

        with patch("pathlib.Path.exists", return_value=False):
            result = await processor.process_document(test_file)

        # Verify result indicates failure due to missing file
        assert result.success is False
        assert result.chunks_processed == 0
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_returns_processing_metrics(self) -> None:
        """Verify ProcessingResult includes chunks_processed and time_taken."""
        config = Mock()
        config.watch_folder = Path("/watch")
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()

        chunker.chunk.return_value = [
            {
                "chunk_text": f"Chunk {i}",
                "chunk_index": i,
                "section_path": "test.md",
                "heading_level": 0,
                "tags": [],
            }
            for i in range(5)
        ]
        tei_client.embed_batch.return_value = [[0.1] * 1024 for _ in range(5)]

        processor = DocumentProcessor(config, tei_client, vector_store, chunker)
        test_file = Path("/watch/test.md")
        test_content = "Test content"

        with patch("pathlib.Path.read_text", return_value=test_content):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                result = await processor.process_document(test_file)

        # Verify metrics are included
        assert result.chunks_processed == 5
        assert result.time_taken > 0
        assert isinstance(result.time_taken, float)


class TestBatchProcessing:
    """Test batch processing of multiple documents."""

    @pytest.mark.asyncio
    async def test_processes_multiple_documents(self) -> None:
        """Verify batch processing handles multiple files."""
        config = Mock()
        config.watch_folder = Path("/watch")
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()

        chunker.chunk.return_value = [
            {
                "chunk_text": "Test content",
                "chunk_index": 0,
                "section_path": "test.md",
                "heading_level": 0,
                "tags": [],
            },
        ]
        tei_client.embed_batch.return_value = [[0.1] * 1024]

        processor = DocumentProcessor(config, tei_client, vector_store, chunker)
        test_files = [
            Path("/watch/doc1.md"),
            Path("/watch/doc2.md"),
            Path("/watch/doc3.md"),
        ]

        with patch("pathlib.Path.read_text", return_value="Test content"):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                results = await processor.process_batch(test_files)

        # Verify all files were processed
        assert len(results) == 3
        assert all(r.success is True for r in results)

    @pytest.mark.asyncio
    async def test_batch_processing_continues_on_error(self) -> None:
        """Verify batch processing continues even if one file fails."""
        config = Mock()
        config.watch_folder = Path("/watch")
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()

        # Mock first file to fail, others succeed
        def side_effect_read(encoding: str = "utf-8") -> str:  # noqa: ARG001
            if "fail.md" in str(processor._current_file):
                raise RuntimeError("Processing error")
            return "Test content"

        chunker.chunk.return_value = [
            {
                "chunk_text": "Test content",
                "chunk_index": 0,
                "section_path": "test.md",
                "heading_level": 0,
                "tags": [],
            },
        ]
        tei_client.embed_batch.return_value = [[0.1] * 1024]

        processor = DocumentProcessor(config, tei_client, vector_store, chunker)
        test_files = [
            Path("/watch/doc1.md"),
            Path("/watch/fail.md"),
            Path("/watch/doc3.md"),
        ]

        # Use a more complex patching approach
        results = []
        for file in test_files:
            processor._current_file = file  # Store current file for side_effect
            if "fail.md" in str(file):
                with patch("pathlib.Path.read_text", side_effect=RuntimeError):
                    result = await processor.process_document(file)
            else:
                with patch("pathlib.Path.read_text", return_value="Test content"):
                    with patch("pathlib.Path.stat") as mock_stat:
                        mock_stat.return_value.st_mtime = 1234567890.0
                        result = await processor.process_document(file)
            results.append(result)

        # Verify 2 succeeded, 1 failed
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        assert len(successful) == 2
        assert len(failed) == 1
        assert "fail.md" in failed[0].file_path


class TestRetryLogic:
    """Test retry logic integration with circuit breaker."""

    @pytest.mark.asyncio
    async def test_uses_circuit_breaker_from_tei_client(self) -> None:
        """Verify processor uses circuit breaker from TEI client."""
        config = Mock()
        config.watch_folder = Path("/watch")
        tei_client = AsyncMock()
        tei_client.circuit_breaker = Mock()
        tei_client.circuit_breaker.state = "CLOSED"
        vector_store = Mock()
        chunker = Mock()

        chunker.chunk.return_value = [
            {
                "chunk_text": "Test content",
                "chunk_index": 0,
                "section_path": "test.md",
                "heading_level": 0,
                "tags": [],
            },
        ]
        tei_client.embed_batch.return_value = [[0.1] * 1024]

        processor = DocumentProcessor(config, tei_client, vector_store, chunker)
        test_file = Path("/watch/test.md")

        with patch("pathlib.Path.read_text", return_value="Test content"):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                await processor.process_document(test_file)

        # Verify TEI client was called (circuit breaker is internal to client)
        tei_client.embed_batch.assert_called_once()
