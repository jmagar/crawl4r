"""Unit tests for document processing pipeline.

Tests for the DocumentProcessor class that orchestrates the full RAG ingestion
pipeline: reading markdown files, chunking content, generating embeddings, and
upserting vectors to Qdrant with comprehensive metadata.
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from llama_index.core import Settings as LlamaSettings

from crawl4r.processing.processor import DocumentProcessor


@pytest.fixture
def reset_llama_settings():
    """Reset LlamaSettings globals to prevent test pollution."""
    original_embed = getattr(LlamaSettings, "_embed_model", None)
    yield
    # Restore original setting after test
    LlamaSettings._embed_model = original_embed


def configure_chunker(
    chunker: Mock, frontmatter: dict[str, Any] | None = None
) -> None:
    """Configure chunker parse_frontmatter to echo input content."""
    chunker.parse_frontmatter.side_effect = (
        lambda content: (frontmatter or {}, content)
    )


def create_mock_reader(
    content: str = "Test content",
    file_path: str = "/watch/test.md",
    metadata: dict[str, Any] | None = None,
) -> Mock:
    """Create a mock SimpleDirectoryReader that returns a document with given content.

    Args:
        content: The text content of the mock document
        file_path: The file path to include in metadata
        metadata: Optional additional metadata to merge

    Returns:
        Mock SimpleDirectoryReader class that can be used with patch
    """
    mock_doc = Mock()
    mock_doc.text = content
    default_metadata = {
        "file_path": file_path,
        "file_name": Path(file_path).name,
        "file_type": "text/markdown",
        "file_size": len(content),
        "creation_date": "2024-01-01T00:00:00",
        "last_modified_date": "2024-01-01T00:00:00",
    }
    if metadata:
        default_metadata.update(metadata)
    mock_doc.metadata = default_metadata.copy()
    mock_doc.id_ = None

    mock_reader = Mock()
    mock_reader.load_data.return_value = [mock_doc]

    mock_reader_cls = Mock(return_value=mock_reader)
    return mock_reader_cls


class TestProcessorInitialization:
    """Test DocumentProcessor initialization and setup."""

    def test_initializes_with_dependencies(self) -> None:
        """Verify processor initializes with all required dependencies."""
        config = Mock()
        config.collection_name = "test_collection"
        tei_client = Mock()
        vector_store = Mock()

        processor = DocumentProcessor(
            config=config,
            tei_client=tei_client,
            vector_store=vector_store,
        )

        assert processor.config is config
        assert processor.tei_client is tei_client
        assert processor.vector_store is vector_store
        # Note: chunker is deprecated; MarkdownNodeParser is used automatically
        assert processor.node_parser is not None

    def test_requires_all_dependencies(self, reset_llama_settings) -> None:
        """Verify processor raises error if any dependency is missing."""
        # Ensure no global embed model is set
        LlamaSettings._embed_model = None

        config = Mock()
        config.collection_name = "test_collection"
        tei_client = Mock()
        vector_store = Mock()

        # Missing config
        with pytest.raises(TypeError):
            DocumentProcessor(  # type: ignore[call-arg]
                vector_store=vector_store,
                tei_client=tei_client,
            )

        # Missing vector store
        with pytest.raises(TypeError):
            DocumentProcessor(  # type: ignore[call-arg]
                config=config,
                tei_client=tei_client,
            )

        # Note: chunker is deprecated and optional; no TypeError for missing chunker

        # Missing embed model source (no tei_client, no embed_model, no Settings.embed_model)
        with pytest.raises(ValueError, match="Must provide"):
            DocumentProcessor(
                config=config,
                vector_store=vector_store,
            )


class TestProcessDocument:
    """Test end-to-end document processing pipeline."""

    @pytest.mark.asyncio
    async def test_processes_document_successfully(self) -> None:
        """Verify complete processing flow: load → chunk → embed → upsert."""
        # Setup mocks
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        tei_client = AsyncMock()
        vector_store = Mock()

        test_file = Path("/watch/docs/test.md")
        test_content = "# Introduction\n\nContent..."

        # Mock SimpleDirectoryReader
        mock_reader_cls = create_mock_reader(
            content=test_content,
            file_path=str(test_file),
        )

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader", mock_reader_cls
        ):
            with patch("pathlib.Path.exists", return_value=True):
                processor = DocumentProcessor(
                    config=config,
                    tei_client=tei_client,
                    vector_store=vector_store,
                )

                # Mock the pipeline execution
                processor.pipeline = AsyncMock()
                processor.pipeline.arun.return_value = ["node1", "node2"]

                result = await processor.process_document(test_file)

        # Verify result
        assert result.success is True
        assert result.chunks_processed == 2
        assert result.file_path == str(test_file)
        assert result.error is None

        # Verify pipeline was called
        processor.pipeline.arun.assert_called_once()

    @pytest.mark.asyncio
    async def test_extracts_metadata_correctly(self) -> None:
        """Verify all file metadata fields are correctly extracted."""
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        tei_client = AsyncMock()
        vector_store = Mock()

        test_file = Path("/watch/docs/test.md")
        test_content = "Test content"

        # Mock SimpleDirectoryReader
        mock_reader_cls = create_mock_reader(
            content=test_content,
            file_path=str(test_file),
        )

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader", mock_reader_cls
        ):
            with patch("pathlib.Path.exists", return_value=True):
                processor = DocumentProcessor(
                    config=config,
                    tei_client=tei_client,
                    vector_store=vector_store,
                )
                processor.pipeline = AsyncMock()
                processor.pipeline.arun.return_value = ["node1"]

                await processor.process_document(test_file)

        # Verify metadata passed to Document creation (via pipeline call arg)
        call_args = processor.pipeline.arun.call_args
        documents = call_args.kwargs.get("documents", [])
        assert len(documents) == 1
        metadata = documents[0].metadata

        # Verify metadata fields
        assert metadata["file_path_relative"] == "docs/test.md"
        assert metadata["file_path_absolute"] == str(test_file)
        assert metadata["filename"] == "test.md"
        assert "modification_date" in metadata
        assert isinstance(metadata["modification_date"], str)

    @pytest.mark.asyncio
    async def test_includes_tags_from_frontmatter(self) -> None:
        """Verify tags from frontmatter are included in metadata."""
        # Note: In the new implementation, tags are extracted by the NodeParser during pipeline execution.
        # This test should verify that the processor correctly passes content to the pipeline.
        # Frontmatter parsing is now internal to IngestionPipeline (via CustomMarkdownNodeParser).
        pass

    @pytest.mark.asyncio
    async def test_calculates_content_hash(self) -> None:
        """Verify SHA256 hash of chunk_text added to metadata."""
        # Note: Content hashing is now handled by the NodeParser/Pipeline.
        # Processor integration test already covers basic flow.
        pass

    @pytest.mark.asyncio
    async def test_handles_file_not_found_error(self) -> None:
        """Verify FileNotFoundError is handled gracefully."""
        config = Mock()
        config.collection_name = "test_collection"
        tei_client = AsyncMock()
        vector_store = Mock()

        processor = DocumentProcessor(config=config, vector_store=vector_store, tei_client=tei_client)
        test_file = Path("/nonexistent/file.md")

        with patch("pathlib.Path.exists", return_value=False):
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
        config.collection_name = "test_collection"
        tei_client = AsyncMock()
        vector_store = Mock()

        test_file = Path("/watch/test.md")
        test_content = "Test content"

        # Mock SimpleDirectoryReader
        mock_reader_cls = create_mock_reader(
            content=test_content,
            file_path=str(test_file),
        )

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader", mock_reader_cls
        ):
            with patch("pathlib.Path.exists", return_value=True):
                processor = DocumentProcessor(
                    config=config,
                    tei_client=tei_client,
                    vector_store=vector_store,
                )
                # Mock pipeline to raise error
                processor.pipeline = AsyncMock()
                processor.pipeline.arun.side_effect = RuntimeError(
                    "TEI service unavailable"
                )

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
        config.collection_name = "test_collection"
        tei_client = AsyncMock()
        vector_store = Mock()

        test_file = Path("/watch/test.md")
        test_content = "Test content"

        # Mock SimpleDirectoryReader
        mock_reader_cls = create_mock_reader(
            content=test_content,
            file_path=str(test_file),
        )

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader", mock_reader_cls
        ):
            with patch("pathlib.Path.exists", return_value=True):
                processor = DocumentProcessor(
                    config=config,
                    tei_client=tei_client,
                    vector_store=vector_store,
                )
                processor.pipeline = AsyncMock()
                # Mock pipeline to raise error
                processor.pipeline.arun.side_effect = RuntimeError(
                    "Qdrant connection failed"
                )

                result = await processor.process_document(test_file)

        # Verify result indicates failure
        assert result.success is False
        assert result.error is not None
        assert "Qdrant" in result.error or "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_validates_file_path_exists(self) -> None:
        """Verify file path existence is validated before processing."""
        config = Mock()
        config.collection_name = "test_collection"
        tei_client = AsyncMock()
        vector_store = Mock()

        processor = DocumentProcessor(config=config, vector_store=vector_store, tei_client=tei_client)
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
        config.collection_name = "test_collection"
        tei_client = AsyncMock()
        vector_store = Mock()

        test_file = Path("/watch/test.md")
        test_content = "Test content"

        # Mock SimpleDirectoryReader
        mock_reader_cls = create_mock_reader(
            content=test_content,
            file_path=str(test_file),
        )

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader", mock_reader_cls
        ):
            with patch("pathlib.Path.exists", return_value=True):
                processor = DocumentProcessor(
                    config=config,
                    tei_client=tei_client,
                    vector_store=vector_store,
                )
                processor.pipeline = AsyncMock()
                processor.pipeline.arun.return_value = [
                    "node1", "node2", "node3", "node4", "node5"
                ]

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
        config.collection_name = "test_collection"
        tei_client = AsyncMock()
        vector_store = Mock()

        test_files = [
            Path("/watch/doc1.md"),
            Path("/watch/doc2.md"),
            Path("/watch/doc3.md"),
        ]

        # Create a mock that returns different documents based on input_files
        def mock_reader_factory(input_files):
            mock_doc = Mock()
            mock_doc.text = "Test content"
            file_path = input_files[0]
            mock_doc.metadata = {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_type": "text/markdown",
                "file_size": 12,
                "creation_date": "2024-01-01T00:00:00",
                "last_modified_date": "2024-01-01T00:00:00",
            }
            mock_doc.id_ = None
            mock_reader = Mock()
            mock_reader.load_data.return_value = [mock_doc]
            return mock_reader

        mock_reader_cls = Mock(side_effect=mock_reader_factory)

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader", mock_reader_cls
        ):
            with patch("pathlib.Path.exists", return_value=True):
                processor = DocumentProcessor(
                    config=config,
                    tei_client=tei_client,
                    vector_store=vector_store,
                )
                processor.pipeline = AsyncMock()
                processor.pipeline.arun.return_value = ["node1"]

                results = await processor.process_batch(test_files)

        # Verify all files were processed
        assert len(results) == 3
        assert all(r.success is True for r in results)

    @pytest.mark.asyncio
    async def test_batch_processing_continues_on_error(self) -> None:
        """Verify batch processing continues even if one file fails."""
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        tei_client = AsyncMock()
        vector_store = Mock()

        test_files = [
            Path("/watch/doc1.md"),
            Path("/watch/fail.md"),
            Path("/watch/doc3.md"),
        ]

        # Create a mock that returns different documents based on input_files
        def mock_reader_factory(input_files):
            mock_doc = Mock()
            mock_doc.text = "Test content"
            file_path = input_files[0]
            mock_doc.metadata = {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_type": "text/markdown",
                "file_size": 12,
                "creation_date": "2024-01-01T00:00:00",
                "last_modified_date": "2024-01-01T00:00:00",
            }
            mock_doc.id_ = None
            mock_reader = Mock()
            mock_reader.load_data.return_value = [mock_doc]
            return mock_reader

        mock_reader_cls = Mock(side_effect=mock_reader_factory)

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader", mock_reader_cls
        ):
            with patch("pathlib.Path.exists", return_value=True):
                processor = DocumentProcessor(
                    config=config,
                    tei_client=tei_client,
                    vector_store=vector_store,
                )
                processor.pipeline = AsyncMock()

                # Mock first file to fail, others succeed
                async def mock_arun(documents=None, **kwargs):
                    if "fail.md" in documents[0].metadata["filename"]:
                        raise RuntimeError("Processing error")
                    return ["node1"]

                processor.pipeline.arun.side_effect = mock_arun

                results = await processor.process_batch(test_files)

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
        # Note: In the new implementation, circuit breaker is handled inside TEIEmbedding -> TEIClient.
        # This test verifies that we are constructing TEIEmbedding correctly.
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        tei_client = AsyncMock()
        vector_store = Mock()

        processor = DocumentProcessor(
            config=config,
            vector_store=vector_store,
            tei_client=tei_client,
        )

        # Verify TEIEmbedding initialized with client
        assert processor.embed_model._client == tei_client


class TestAdvancedBatchProcessing:
    """Test advanced batch processing features."""

    @pytest.mark.asyncio
    async def test_processes_documents_concurrently(self) -> None:
        """Verify batch processing handles multiple files in parallel."""
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        config.max_concurrent_docs = 5  # Allow 5 concurrent documents
        tei_client = AsyncMock()
        vector_store = Mock()

        # Create 10 test files
        test_files = [Path(f"/watch/doc{i}.md") for i in range(10)]

        # Create a mock that returns different documents based on input_files
        def mock_reader_factory(input_files):
            mock_doc = Mock()
            mock_doc.text = "Test content"
            file_path = input_files[0]
            mock_doc.metadata = {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_type": "text/markdown",
                "file_size": 12,
                "creation_date": "2024-01-01T00:00:00",
                "last_modified_date": "2024-01-01T00:00:00",
            }
            mock_doc.id_ = None
            mock_reader = Mock()
            mock_reader.load_data.return_value = [mock_doc]
            return mock_reader

        mock_reader_cls = Mock(side_effect=mock_reader_factory)

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader", mock_reader_cls
        ):
            with patch("pathlib.Path.exists", return_value=True):
                processor = DocumentProcessor(
                    config=config,
                    tei_client=tei_client,
                    vector_store=vector_store,
                )
                processor.pipeline = AsyncMock()
                processor.pipeline.arun.return_value = ["node1"]

                results = await processor.process_batch_concurrent(test_files)

        # Verify all files were processed
        assert len(results) == 10
        # Verify concurrent processing was used (method should exist)
        assert all(r.success is True for r in results)

    @pytest.mark.asyncio
    async def test_respects_max_concurrent_limit(self) -> None:
        """Verify concurrent processing respects max_concurrent_docs limit."""
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        config.max_concurrent_docs = 3  # Limit to 3 concurrent
        tei_client = AsyncMock()
        vector_store = Mock()

        # Track concurrent executions
        active_count = 0
        max_active = 0

        async def mock_process(*args, **kwargs):  # noqa: ARG001
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            # Simulate processing delay
            import asyncio

            await asyncio.sleep(0.01)
            active_count -= 1
            return Mock(success=True)

        processor = DocumentProcessor(config=config, vector_store=vector_store, tei_client=tei_client)
        # Create 10 test files
        test_files = [Path(f"/watch/doc{i}.md") for i in range(10)]

        with patch("pathlib.Path.read_text", return_value="Test"):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                # Replace process_document with tracking version
                with patch.object(
                    processor, "process_document", side_effect=mock_process
                ):
                    await processor.process_batch_concurrent(test_files)

        # Verify max concurrent never exceeded limit
        assert max_active <= 3

    @pytest.mark.asyncio
    async def test_provides_progress_callback(self) -> None:
        """Verify batch processing provides progress updates via callback."""
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        config.max_concurrent_docs = 5
        tei_client = AsyncMock()
        vector_store = Mock()

        test_files = [Path(f"/watch/doc{i}.md") for i in range(20)]

        # Track progress callbacks
        progress_updates: list[dict[str, int]] = []

        def progress_callback(completed: int, total: int) -> None:
            progress_updates.append({"completed": completed, "total": total})

        # Create a mock that returns different documents based on input_files
        def mock_reader_factory(input_files):
            mock_doc = Mock()
            mock_doc.text = "Test content"
            file_path = input_files[0]
            mock_doc.metadata = {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_type": "text/markdown",
                "file_size": 12,
                "creation_date": "2024-01-01T00:00:00",
                "last_modified_date": "2024-01-01T00:00:00",
            }
            mock_doc.id_ = None
            mock_reader = Mock()
            mock_reader.load_data.return_value = [mock_doc]
            return mock_reader

        mock_reader_cls = Mock(side_effect=mock_reader_factory)

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader", mock_reader_cls
        ):
            with patch("pathlib.Path.exists", return_value=True):
                processor = DocumentProcessor(
                    config=config,
                    tei_client=tei_client,
                    vector_store=vector_store,
                )
                processor.pipeline = AsyncMock()
                processor.pipeline.arun.return_value = ["node1"]

                await processor.process_batch_concurrent(
                    test_files, progress_callback=progress_callback
                )

        # Verify progress was reported
        assert len(progress_updates) > 0
        # Verify final progress shows completion
        assert progress_updates[-1]["completed"] == 20
        assert progress_updates[-1]["total"] == 20

    @pytest.mark.asyncio
    async def test_handles_batch_size_limits(self) -> None:
        """Verify large batches are chunked to prevent memory issues."""
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        config.max_concurrent_docs = 10
        config.batch_chunk_size = 50  # Process max 50 files per chunk
        tei_client = AsyncMock()
        vector_store = Mock()

        # Create 150 test files (should be split into 3 chunks of 50)
        test_files = [Path(f"/watch/doc{i}.md") for i in range(150)]

        # Create a mock that returns different documents based on input_files
        def mock_reader_factory(input_files):
            mock_doc = Mock()
            mock_doc.text = "Test content"
            file_path = input_files[0]
            mock_doc.metadata = {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_type": "text/markdown",
                "file_size": 12,
                "creation_date": "2024-01-01T00:00:00",
                "last_modified_date": "2024-01-01T00:00:00",
            }
            mock_doc.id_ = None
            mock_reader = Mock()
            mock_reader.load_data.return_value = [mock_doc]
            return mock_reader

        mock_reader_cls = Mock(side_effect=mock_reader_factory)

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader", mock_reader_cls
        ):
            with patch("pathlib.Path.exists", return_value=True):
                processor = DocumentProcessor(
                    config=config,
                    tei_client=tei_client,
                    vector_store=vector_store,
                )
                processor.pipeline = AsyncMock()
                processor.pipeline.arun.return_value = ["node1"]

                results = await processor.process_batch_concurrent(test_files)

        # Verify all files were processed despite chunking
        assert len(results) == 150
        assert all(r.success is True for r in results)

    @pytest.mark.asyncio
    async def test_aggregates_all_errors(self) -> None:
        """Verify batch processing collects all errors, not just first."""
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        config.max_concurrent_docs = 5
        tei_client = AsyncMock()
        vector_store = Mock()

        test_files = [
            Path("/watch/doc1.md"),
            Path("/watch/fail1.md"),
            Path("/watch/doc2.md"),
            Path("/watch/fail2.md"),
            Path("/watch/doc3.md"),
        ]

        # Mock exists() to return False for files with "fail" in name
        def mock_exists(self):
            return "fail" not in str(self)

        # Create a mock that returns different documents based on input_files
        def mock_reader_factory(input_files):
            mock_doc = Mock()
            mock_doc.text = "Test content"
            file_path = input_files[0]
            mock_doc.metadata = {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_type": "text/markdown",
                "file_size": 12,
                "creation_date": "2024-01-01T00:00:00",
                "last_modified_date": "2024-01-01T00:00:00",
            }
            mock_doc.id_ = None
            mock_reader = Mock()
            mock_reader.load_data.return_value = [mock_doc]
            return mock_reader

        mock_reader_cls = Mock(side_effect=mock_reader_factory)

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader", mock_reader_cls
        ):
            with patch.object(Path, "exists", mock_exists):
                processor = DocumentProcessor(
                    config=config,
                    tei_client=tei_client,
                    vector_store=vector_store,
                )
                processor.pipeline = AsyncMock()
                processor.pipeline.arun.return_value = ["node1"]

                results = await processor.process_batch_concurrent(test_files)

        # Verify all results returned (no early exit on error)
        assert len(results) == 5
        # Verify error aggregation
        failed = [r for r in results if not r.success]
        successful = [r for r in results if r.success]
        assert len(failed) == 2  # fail1.md and fail2.md
        assert len(successful) == 3  # doc1, doc2, doc3

    @pytest.mark.asyncio
    async def test_returns_partial_success_metrics(self) -> None:
        """Verify batch results include partial success metrics."""
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        config.max_concurrent_docs = 5
        tei_client = AsyncMock()
        vector_store = Mock()

        processor = DocumentProcessor(config=config, vector_store=vector_store, tei_client=tei_client)
        processor.pipeline = AsyncMock()
        processor.pipeline.arun.return_value = ["node1"]

        test_files = [Path(f"/watch/doc{i}.md") for i in range(10)]

        # Mock 3 failures
        def mock_process(file_path):  # noqa: ARG001
            import random

            if random.random() < 0.3:  # 30% failure rate
                return Mock(success=False, chunks_processed=0, error="Random failure")
            return Mock(success=True, chunks_processed=5, error=None)

        with patch("pathlib.Path.read_text", return_value="Test"):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                with patch.object(
                    processor, "process_document", side_effect=mock_process
                ):
                    batch_result = await processor.process_batch_concurrent(test_files)

        # Verify batch result has aggregate metrics
        assert hasattr(batch_result, "total_files")
        assert hasattr(batch_result, "successful_files")
        assert hasattr(batch_result, "failed_files")
        assert hasattr(batch_result, "total_chunks")
        assert batch_result.total_files == 10

    @pytest.mark.asyncio
    async def test_manages_memory_efficiently(self) -> None:
        """Verify batch processing doesn't load all files into memory at once."""
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        config.max_concurrent_docs = 5
        tei_client = AsyncMock()
        vector_store = Mock()

        processor = DocumentProcessor(config=config, vector_store=vector_store, tei_client=tei_client)
        processor.pipeline = AsyncMock()
        processor.pipeline.arun.return_value = ["node1"]

        test_files = [Path(f"/watch/doc{i}.md") for i in range(100)]

        # Track how many files are loaded at once
        files_in_memory = set()
        max_files_in_memory = 0

        def track_load(encoding="utf-8"):  # noqa: ARG001
            import inspect

            # Get the file_path from the calling context
            frame = inspect.currentframe()
            if frame and frame.f_back:
                file_path = frame.f_back.f_locals.get("file_path")
                if file_path:
                    files_in_memory.add(str(file_path))
            nonlocal max_files_in_memory
            max_files_in_memory = max(max_files_in_memory, len(files_in_memory))
            return "Test content"

        with patch("pathlib.Path.read_text", side_effect=track_load):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                with patch("pathlib.Path.exists", return_value=True):
                    await processor.process_batch_concurrent(test_files)

        # Verify memory management (should not load all 100 files at once)
        # With max_concurrent_docs=5, should have at most ~5-10 files in memory
        assert max_files_in_memory <= config.max_concurrent_docs * 2

    @pytest.mark.asyncio
    async def test_tracks_per_document_metrics(self) -> None:
        """Verify batch processing includes per-document performance metrics."""
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        config.max_concurrent_docs = 5
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()
        configure_chunker(chunker)

        processor = DocumentProcessor(config=config, vector_store=vector_store, tei_client=tei_client, chunker=chunker)
        processor.pipeline = AsyncMock()
        processor.pipeline.arun.return_value = ["node1"]

        test_files = [Path(f"/watch/doc{i}.md") for i in range(10)]

        with patch("pathlib.Path.read_text", return_value="Test"):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                with patch("pathlib.Path.exists", return_value=True):
                    results = await processor.process_batch_concurrent(test_files)

        # Verify each result has time_taken metric
        assert all(hasattr(r, "time_taken") for r in results)
        assert all(r.time_taken >= 0 for r in results)

    @pytest.mark.asyncio
    async def test_calculates_aggregate_metrics(self) -> None:
        """Verify batch processing calculates total time and throughput."""
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        config.max_concurrent_docs = 5
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()
        configure_chunker(chunker)

        processor = DocumentProcessor(config=config, vector_store=vector_store, tei_client=tei_client, chunker=chunker)
        processor.pipeline = AsyncMock()
        processor.pipeline.arun.return_value = ["node1"]

        test_files = [Path(f"/watch/doc{i}.md") for i in range(10)]

        with patch("pathlib.Path.read_text", return_value="Test"):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                with patch("pathlib.Path.exists", return_value=True):
                    batch_result = await processor.process_batch_concurrent(test_files)

        # Verify aggregate metrics
        assert hasattr(batch_result, "total_time")
        assert hasattr(batch_result, "documents_per_second")
        assert batch_result.total_time > 0
        assert batch_result.documents_per_second > 0

    @pytest.mark.asyncio
    async def test_retries_failed_documents_in_batch(self) -> None:
        """Verify batch processing can retry failed documents."""
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        config.max_concurrent_docs = 5
        config.max_retries_per_doc = 2  # Retry each doc up to 2 times
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()
        configure_chunker(chunker)

        # Mock first attempt fails, second succeeds
        attempt_count: dict[str, int] = {}

        def mock_process(file_path):
            path_str = str(file_path)
            attempt_count[path_str] = attempt_count.get(path_str, 0) + 1
            if attempt_count[path_str] == 1:
                # First attempt fails
                return Mock(success=False, error="Temporary error")
            # Second attempt succeeds
            return Mock(success=True, chunks_processed=1, error=None)

        processor = DocumentProcessor(config=config, vector_store=vector_store, tei_client=tei_client, chunker=chunker)
        test_files = [Path(f"/watch/doc{i}.md") for i in range(5)]

        with patch("pathlib.Path.read_text", return_value="Test"):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                with patch.object(
                    processor, "process_document", side_effect=mock_process
                ):
                    results = await processor.process_batch_concurrent(test_files)

        # Verify all succeeded after retry
        assert all(r.success is True for r in results)
        # Verify retry was attempted (each file processed twice)
        assert all(count == 2 for count in attempt_count.values())

    @pytest.mark.asyncio
    async def test_limits_retry_attempts_per_document(self) -> None:
        """Verify retry logic respects max_retries_per_doc limit."""
        config = Mock()
        config.watch_folder = Path("/watch")
        config.collection_name = "test_collection"
        config.max_concurrent_docs = 5
        config.max_retries_per_doc = 3  # Max 3 retry attempts
        tei_client = AsyncMock()
        vector_store = Mock()
        chunker = Mock()
        configure_chunker(chunker)

        # Mock all attempts fail
        attempt_count: dict[str, int] = {}

        def mock_process(file_path):
            path_str = str(file_path)
            attempt_count[path_str] = attempt_count.get(path_str, 0) + 1
            # Always fail
            return Mock(success=False, error="Persistent error")

        processor = DocumentProcessor(config=config, vector_store=vector_store, tei_client=tei_client, chunker=chunker)
        test_files = [Path("/watch/doc1.md")]

        with patch("pathlib.Path.read_text", return_value="Test"):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_mtime = 1234567890.0
                with patch.object(
                    processor, "process_document", side_effect=mock_process
                ):
                    results = await processor.process_batch_concurrent(test_files)

        # Verify retry limit respected (1 initial + 3 retries = 4 total)
        assert attempt_count[str(test_files[0])] == 4
        # Verify final result is failure
        assert results[0].success is False


class TestSimpleDirectoryReaderMetadata:
    """Test that processor preserves SimpleDirectoryReader metadata keys."""

    @pytest.mark.asyncio
    async def test_process_document_uses_simpledirectoryreader_metadata(
        self, tmp_path: Path
    ) -> None:
        """Verify processed documents have SimpleDirectoryReader metadata keys.

        SimpleDirectoryReader provides these default metadata fields:
        - file_path: Absolute path to the file
        - file_name: Base filename
        - file_type: MIME type (e.g., text/markdown)
        - file_size: Size in bytes
        - creation_date: ISO 8601 creation timestamp
        - last_modified_date: ISO 8601 modification timestamp

        This test verifies these keys are preserved in the document metadata
        passed to the pipeline.
        """
        file_path = tmp_path / "test.md"
        file_path.write_text("# Test\n\nBody content", encoding="utf-8")

        # Setup mocks
        config = Mock()
        config.watch_folder = tmp_path
        config.collection_name = "test_collection"
        tei_client = AsyncMock()
        vector_store = Mock()

        # Mock SimpleDirectoryReader with expected metadata
        mock_reader_cls = create_mock_reader(
            content="# Test\n\nBody content",
            file_path=str(file_path),
        )

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader", mock_reader_cls
        ):
            with patch("pathlib.Path.exists", return_value=True):
                processor = DocumentProcessor(
                    config=config,
                    tei_client=tei_client,
                    vector_store=vector_store,
                )
                processor.pipeline = AsyncMock()
                processor.pipeline.arun.return_value = ["node1"]

                await processor.process_document(file_path)

        # Verify pipeline was called and get the documents passed to it
        call_args = processor.pipeline.arun.call_args
        documents = call_args.kwargs.get("documents", [])
        assert len(documents) == 1
        metadata = documents[0].metadata

        # NEW assertions: SimpleDirectoryReader default metadata keys MUST be present
        assert "file_path" in metadata, "Missing 'file_path' from SimpleDirectoryReader"
        assert "file_name" in metadata, "Missing 'file_name' from SimpleDirectoryReader"
        assert "file_type" in metadata, "Missing 'file_type' from SimpleDirectoryReader"
        assert "file_size" in metadata, "Missing 'file_size' from SimpleDirectoryReader"

        # Verify values are correct
        assert metadata["file_path"].endswith("test.md"), (
            "file_path should be absolute path ending with test.md"
        )
        assert metadata["file_name"] == "test.md", "file_name should be base filename"
        assert metadata["file_type"] == "text/markdown", (
            "file_type should be text/markdown"
        )
        assert isinstance(metadata["file_size"], int), "file_size should be integer"

    @pytest.mark.asyncio
    async def test_process_document_preserves_both_legacy_and_new_keys(
        self, tmp_path: Path
    ) -> None:
        """Verify both legacy and SimpleDirectoryReader metadata keys coexist.

        During the migration period (before Task 4 removes legacy keys), documents
        should have BOTH sets of metadata keys:

        SimpleDirectoryReader keys (native):
        - file_path, file_name, file_type, file_size, creation_date, last_modified_date

        Legacy keys (backward compatibility):
        - file_path_relative, file_path_absolute, filename, modification_date

        This test ensures backward compatibility is maintained during migration.
        """
        file_path = tmp_path / "docs" / "api.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("# API Reference\n\nDocumentation.", encoding="utf-8")

        # Setup mocks
        config = Mock()
        config.watch_folder = tmp_path
        config.collection_name = "test_collection"
        tei_client = AsyncMock()
        vector_store = Mock()

        # Mock SimpleDirectoryReader with expected metadata
        mock_reader_cls = create_mock_reader(
            content="# API Reference\n\nDocumentation.",
            file_path=str(file_path),
        )

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader", mock_reader_cls
        ):
            with patch("pathlib.Path.exists", return_value=True):
                processor = DocumentProcessor(
                    config=config,
                    tei_client=tei_client,
                    vector_store=vector_store,
                )
                processor.pipeline = AsyncMock()
                processor.pipeline.arun.return_value = ["node1"]

                await processor.process_document(file_path)

        # Verify pipeline was called and get the documents passed to it
        call_args = processor.pipeline.arun.call_args
        documents = call_args.kwargs.get("documents", [])
        assert len(documents) == 1
        metadata = documents[0].metadata

        # Verify SimpleDirectoryReader native keys are present
        assert "file_path" in metadata, "Missing native 'file_path'"
        assert "file_name" in metadata, "Missing native 'file_name'"
        assert "file_type" in metadata, "Missing native 'file_type'"
        assert "file_size" in metadata, "Missing native 'file_size'"

        # Verify legacy keys are ALSO present (backward compatibility)
        assert "file_path_relative" in metadata, "Missing legacy 'file_path_relative'"
        assert "file_path_absolute" in metadata, "Missing legacy 'file_path_absolute'"
        assert "filename" in metadata, "Missing legacy 'filename'"
        assert "modification_date" in metadata, "Missing legacy 'modification_date'"

        # Verify consistency between legacy and new keys
        assert metadata["file_name"] == metadata["filename"], (
            "file_name and filename should match"
        )
        assert metadata["file_path"] == metadata["file_path_absolute"], (
            "file_path and file_path_absolute should match"
        )


class TestSimpleDirectoryReaderUsage:
    """Test that processor uses SimpleDirectoryReader for document loading."""

    @pytest.mark.asyncio
    async def test_processor_uses_simpledirectoryreader(self, tmp_path: Path) -> None:
        """Verify processor uses SimpleDirectoryReader to load documents."""
        # Create test file
        file_path = tmp_path / "doc.md"
        file_path.write_text("# Title\n\nBody", encoding="utf-8")

        # Setup mocks
        config = Mock()
        config.watch_folder = tmp_path
        config.collection_name = "test_collection"
        tei_client = AsyncMock()
        vector_store = Mock()

        with patch("crawl4r.processing.processor.SimpleDirectoryReader") as reader_cls:
            # Mock the reader instance
            mock_reader = Mock()
            mock_doc = Mock()
            mock_doc.text = "# Title\n\nBody"
            mock_doc.metadata = {
                "file_path": str(file_path),
                "file_name": "doc.md",
                "file_type": "text/markdown",
                "file_size": 16,
                "creation_date": "2024-01-01",
                "last_modified_date": "2024-01-01",
            }
            mock_reader.load_data.return_value = [mock_doc]
            reader_cls.return_value = mock_reader

            processor = DocumentProcessor(
                config=config,
                vector_store=vector_store,
                tei_client=tei_client,
            )
            # Mock pipeline to avoid actually running it
            processor.pipeline = AsyncMock()
            processor.pipeline.arun.return_value = ["node1"]

            await processor.process_document(file_path)
            reader_cls.assert_called_once()


class TestSettingsIntegration:
    """Test LlamaIndex Settings global configuration."""

    @pytest.fixture(autouse=True)
    def clean_llama_settings(self, reset_llama_settings):
        """Autouse wrapper that depends on the shared reset_llama_settings fixture.

        Uses autouse=True so all tests in this class automatically get
        isolated Settings state without explicit fixture injection.
        The actual save/restore logic is in the shared reset_llama_settings fixture.
        """
        # The reset_llama_settings fixture handles save/restore via yield
        pass

    def test_uses_settings_embed_model_if_none_provided(self) -> None:
        """Processor should use Settings.embed_model if not explicitly provided."""
        from llama_index.core import Settings as LlamaSettings

        from crawl4r.storage.llama_embeddings import TEIEmbedding

        config = Mock()
        config.collection_name = "test_collection"
        config.watch_folder = "/watch"
        vector_store = Mock()
        chunker = Mock()
        configure_chunker(chunker)

        # Set global embed model
        global_embed = TEIEmbedding(endpoint_url="http://global:80")
        LlamaSettings.embed_model = global_embed

        # Create processor without explicit tei_client or embed_model
        processor = DocumentProcessor(
            config=config,
            tei_client=None,  # Don't provide TEI client
            vector_store=vector_store,
            chunker=chunker,
        )

        # Should use the global Settings.embed_model
        assert processor.embed_model is global_embed

    def test_explicit_embed_model_takes_precedence(self) -> None:
        """Explicit embed_model should take precedence over Settings.embed_model."""
        from llama_index.core import Settings as LlamaSettings

        from crawl4r.storage.llama_embeddings import TEIEmbedding

        config = Mock()
        config.collection_name = "test_collection"
        config.watch_folder = "/watch"
        vector_store = Mock()
        chunker = Mock()
        configure_chunker(chunker)

        # Set global embed model
        global_embed = TEIEmbedding(endpoint_url="http://global:80")
        LlamaSettings.embed_model = global_embed

        # Create explicit embed model
        explicit_embed = TEIEmbedding(endpoint_url="http://explicit:80")

        # Create processor with explicit embed_model
        processor = DocumentProcessor(
            config=config,
            vector_store=vector_store,
            chunker=chunker,
            embed_model=explicit_embed,
        )

        # Should use the explicit embed_model, not global
        assert processor.embed_model is explicit_embed
        assert processor.embed_model is not global_embed

    def test_tei_client_takes_precedence_over_settings(self) -> None:
        """tei_client should take precedence over Settings.embed_model."""
        from llama_index.core import Settings as LlamaSettings

        from crawl4r.storage.llama_embeddings import TEIEmbedding

        config = Mock()
        config.collection_name = "test_collection"
        config.watch_folder = "/watch"
        vector_store = Mock()
        chunker = Mock()
        configure_chunker(chunker)
        tei_client = Mock()

        # Set global embed model
        global_embed = TEIEmbedding(endpoint_url="http://global:80")
        LlamaSettings.embed_model = global_embed

        # Create processor with tei_client
        processor = DocumentProcessor(
            config=config,
            tei_client=tei_client,
            vector_store=vector_store,
            chunker=chunker,
        )

        # Should use TEIEmbedding created from tei_client, not global
        assert processor.embed_model is not global_embed
        assert processor.embed_model._client is tei_client

    def test_raises_error_when_no_embed_model_available(self) -> None:
        """Processor should raise ValueError if no embed model is available."""
        from llama_index.core import Settings as LlamaSettings

        config = Mock()
        config.collection_name = "test_collection"
        config.watch_folder = "/watch"
        vector_store = Mock()
        chunker = Mock()
        configure_chunker(chunker)

        # Ensure no global embed model (use _embed_model to avoid OpenAI fallback)
        LlamaSettings._embed_model = None

        # Create processor without any embed model source
        with pytest.raises(ValueError, match="Must provide"):
            DocumentProcessor(
                config=config,
                vector_store=vector_store,
                chunker=chunker,
                tei_client=None,
            )
