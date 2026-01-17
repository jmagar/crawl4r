# tests/unit/test_processor_pipeline.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crawl4r.processing.processor import DocumentProcessor


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
    vector_store_manager = MagicMock()
    chunker = MagicMock()

    # Create a dummy file with test content
    doc_path = tmp_path / "test.md"
    doc_content = "# Test\n\nThis is test content."
    doc_path.write_text(doc_content)

    with patch("crawl4r.processing.processor.IngestionPipeline") as mock_pipeline_cls:
        pipeline_instance = mock_pipeline_cls.return_value
        pipeline_instance.arun = AsyncMock(return_value=[])

        # Create processor - this should construct the pipeline internally
        processor = DocumentProcessor(config, tei_client, vector_store_manager, chunker)

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
            # Documents were passed as keyword argument
            pass
        else:
            pytest.fail(
                f"arun was called with unexpected argument structure: "
                f"args={call_args.args}, kwargs={call_args.kwargs}"
            )
