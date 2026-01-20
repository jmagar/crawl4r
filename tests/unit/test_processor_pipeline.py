# tests/unit/test_processor_pipeline.py
"""Unit tests for DocumentProcessor pipeline configuration with docstore.

Tests for the IngestionPipeline integration and docstore that enables
LlamaIndex's native deduplication via doc_id hashing using SimpleDocumentStore.
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from llama_index.core import Document
from llama_index.core.ingestion import DocstoreStrategy
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore

from crawl4r.processing.processor import DocumentProcessor


def create_test_processor(
    docstore: SimpleDocumentStore | None = None,
) -> DocumentProcessor:
    """Create a DocumentProcessor with mocked dependencies for testing.

    Args:
        docstore: Optional docstore to inject for deduplication testing.
            If None, the processor will create its own SimpleDocumentStore.

    Returns:
        DocumentProcessor with mocked config, tei_client, and vector_store.
    """
    config = Mock()
    config.collection_name = "test_collection"
    config.watch_folder = Path("/watch")
    config.max_concurrent_docs = 5

    tei_client = Mock()
    vector_store = Mock()
    vector_store.client = Mock()  # Required for QdrantVectorStore initialization

    kwargs: dict[str, Any] = {
        "config": config,
        "tei_client": tei_client,
        "vector_store": vector_store,
    }
    if docstore is not None:
        kwargs["docstore"] = docstore

    return DocumentProcessor(**kwargs)


class TestDocstoreIntegration:
    """Test IngestionPipeline with docstore for deduplication."""

    def test_pipeline_has_docstore(self) -> None:
        """DocumentProcessor pipeline should include docstore."""
        processor = create_test_processor()

        assert processor.pipeline.docstore is not None
        assert isinstance(processor.pipeline.docstore, SimpleDocumentStore)

    def test_pipeline_uses_upserts_strategy(self) -> None:
        """DocumentProcessor pipeline should use UPSERTS docstore strategy."""
        processor = create_test_processor()

        # Verify the docstore strategy is set to UPSERTS for deduplication
        assert processor.pipeline.docstore_strategy == DocstoreStrategy.UPSERTS

    def test_custom_docstore_is_used(self) -> None:
        """DocumentProcessor should use injected docstore if provided."""
        custom_docstore = SimpleDocumentStore()
        processor = create_test_processor(docstore=custom_docstore)

        assert processor.docstore is custom_docstore
        assert processor.pipeline.docstore is custom_docstore

    def test_default_docstore_created_if_not_provided(self) -> None:
        """DocumentProcessor should create SimpleDocumentStore if none provided."""
        processor = create_test_processor()

        assert processor.docstore is not None
        assert isinstance(processor.docstore, SimpleDocumentStore)

    def test_docstore_exposed_as_attribute(self) -> None:
        """DocumentProcessor should expose docstore as instance attribute."""
        processor = create_test_processor()

        # Verify docstore is accessible on the processor
        assert hasattr(processor, "docstore")
        assert processor.docstore is processor.pipeline.docstore


class TestDocstoreDeduplication:
    """Test deduplication behavior with docstore."""

    @pytest.mark.asyncio
    async def test_processor_handles_empty_pipeline_result(self) -> None:
        """Processor reports zero chunks when pipeline returns an empty result."""
        # Create processor with real docstore
        processor = create_test_processor()

        # Mock the pipeline's arun to track calls
        call_count = 0

        async def mock_arun(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call returns nodes, second returns empty (duplicate detected)
            if call_count == 1:
                return ["node1", "node2"]
            return []  # Duplicate detected, no new nodes

        processor.pipeline = AsyncMock()
        processor.pipeline.arun = mock_arun
        processor.pipeline.docstore = processor.docstore
        processor.pipeline.docstore_strategy = DocstoreStrategy.UPSERTS

        # Process same document twice
        test_file = Path("/watch/test.md")

        # Create a mock document that SimpleDirectoryReader would return
        mock_doc = Document(
            text="# Test\n\nContent here.",
            metadata={
                "file_path": str(test_file),
                "file_name": "test.md",
                "last_modified_date": "2009-02-13T23:31:30",
            },
        )
        mock_reader = MagicMock()
        mock_reader.load_data.return_value = [mock_doc]

        with patch(
            "crawl4r.processing.processor.SimpleDirectoryReader",
            return_value=mock_reader,
        ):
            with patch("pathlib.Path.exists", return_value=True):
                result1 = await processor.process_document(test_file)
                result2 = await processor.process_document(test_file)

        # Verify both calls were made
        assert call_count == 2

        # First should have chunks, second should be empty (duplicate)
        assert result1.chunks_processed == 2
        assert result2.chunks_processed == 0


class TestDocstorePersistence:
    """Test docstore persistence scenarios."""

    def test_docstore_can_be_shared_across_processors(self) -> None:
        """Multiple processors can share same docstore for session-scoped dedup."""
        shared_docstore = SimpleDocumentStore()

        processor1 = create_test_processor(docstore=shared_docstore)
        processor2 = create_test_processor(docstore=shared_docstore)

        assert processor1.docstore is processor2.docstore
        assert processor1.pipeline.docstore is processor2.pipeline.docstore

    def test_docstore_state_isolated_by_default(self) -> None:
        """Each processor creates isolated docstore by default."""
        processor1 = create_test_processor()
        processor2 = create_test_processor()

        assert processor1.docstore is not processor2.docstore


class TestMarkdownNodeParser:
    """Tests for MarkdownNodeParser integration."""

    def test_processor_uses_markdown_node_parser(self) -> None:
        """DocumentProcessor should use LlamaIndex MarkdownNodeParser instead of custom parser."""
        processor = create_test_processor()

        # Verify type
        assert isinstance(processor.node_parser, MarkdownNodeParser), (
            f"Expected MarkdownNodeParser, got {type(processor.node_parser).__name__}"
        )

    def test_markdown_node_parser_produces_nodes(self) -> None:
        """Verify MarkdownNodeParser produces nodes from markdown content."""
        processor = create_test_processor()

        test_doc = Document(
            text="# Title\n\nFirst paragraph.\n\n## Section\n\nSecond paragraph.",
            metadata={"filename": "test.md"},
        )

        nodes = processor.node_parser.get_nodes_from_documents([test_doc])

        # Verify nodes are produced
        assert len(nodes) > 0, "MarkdownNodeParser should produce at least one node"

        # Verify node has text content
        assert any(node.text.strip() for node in nodes), "Nodes should have text content"


@pytest.mark.asyncio
async def test_process_document_uses_pipeline(tmp_path):
    """Test that DocumentProcessor uses IngestionPipeline for document processing.

    Verifies:
    - IngestionPipeline is instantiated
    - pipeline.arun is called with the document content
    """
    # Setup mocks
    config = MagicMock()
    config.watch_folder = tmp_path
    config.collection_name = "test_collection"

    tei_client = MagicMock()
    vector_store = MagicMock()
    vector_store.client = MagicMock()  # Required for QdrantVectorStore initialization

    # Create a dummy file with test content
    doc_path = tmp_path / "test.md"
    doc_content = "# Test\n\nThis is test content."
    doc_path.write_text(doc_content)

    with patch("crawl4r.processing.processor.IngestionPipeline") as mock_pipeline_cls:
        pipeline_instance = mock_pipeline_cls.return_value
        pipeline_instance.arun = AsyncMock(return_value=[])

        # Create processor - this should construct the pipeline internally
        processor = DocumentProcessor(
            config=config,
            vector_store=vector_store,
            tei_client=tei_client,
        )

        # Verify IngestionPipeline was instantiated (processor builds it in __init__)
        assert mock_pipeline_cls.called, "IngestionPipeline should be instantiated"

        # Process the document
        await processor.process_document(doc_path)

        # Verify arun was called exactly once
        pipeline_instance.arun.assert_called_once()

        # Verify the call arguments - arun should receive Document nodes
        call_args = pipeline_instance.arun.call_args
        assert call_args is not None, "arun should have been called with arguments"

        # The first positional argument should be a list of documents/nodes
        if call_args.args:
            docs_arg = call_args.args[0]
            assert isinstance(docs_arg, list), "arun should receive a list of documents"
        elif call_args.kwargs and ("documents" in call_args.kwargs or "nodes" in call_args.kwargs):
            docs_kw = call_args.kwargs.get("documents") or call_args.kwargs.get("nodes")
            assert isinstance(docs_kw, list), "arun should receive a list of documents"
        else:
            pytest.fail(
                f"arun was called with unexpected argument structure: "
                f"args={call_args.args}, kwargs={call_args.kwargs}"
            )
