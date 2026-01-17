# tests/unit/test_processor_pipeline.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from crawl4r.processing.processor import DocumentProcessor
from llama_index.core.ingestion import IngestionPipeline

@pytest.mark.asyncio
async def test_process_document_uses_pipeline(tmp_path):
    # Setup mocks
    config = MagicMock()
    config.watch_folder = tmp_path
    config.collection_name = "test_collection" # Must be a string
    tei_client = MagicMock()
    vector_store_manager = MagicMock() # The legacy manager
    chunker = MagicMock()
    
    # Create a dummy file
    doc_path = tmp_path / "test.md"
    doc_path.write_text("# Test")
    
    with patch("crawl4r.processing.processor.IngestionPipeline") as MockPipeline:
        pipeline_instance = MockPipeline.return_value
        pipeline_instance.arun = AsyncMock(return_value=[])
        
        processor = DocumentProcessor(config, tei_client, vector_store_manager, chunker)
        
        # We need to manually inject the pipeline or ensure init creates it
        # Ideally, we pass it or the processor builds it.
        # For this refactor, let's assume processor builds it internally using the helpers.
        
        await processor.process_document(doc_path)
        
        # assert pipeline_instance.run.called or pipeline_instance.arun.called
        assert pipeline_instance.arun.called
