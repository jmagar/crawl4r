import pytest
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation import get_dispatcher
from crawl4r.processing.processor import DocumentProcessor
from crawl4r.core.config import Settings
from unittest.mock import MagicMock
from pathlib import Path
from llama_index.core.embeddings import BaseEmbedding

class MockEmbedding(BaseEmbedding):
    """Mock embedding model that satisfies Pydantic validation."""
    def _get_query_embedding(self, query): return []
    async def _aget_query_embedding(self, query): return []
    def _get_text_embedding(self, text): return []
    async def _aget_text_embedding(self, text): return []
    def _get_text_embeddings(self, texts): return []
    async def _aget_text_embeddings(self, texts): return []

# Mock Handler to capture events
class MockEventHandler(BaseEventHandler):
    events: list = []
    
    def handle(self, event, **kwargs):
        self.events.append(event)
        
    @classmethod
    def class_name(cls):
        return "MockEventHandler"

@pytest.mark.asyncio
async def test_processor_instrumentation(tmp_path):
    # Setup dispatcher and handler
    dispatcher = get_dispatcher("crawl4r")
    handler = MockEventHandler()
    handler.events = [] # Reset
    dispatcher.add_event_handler(handler)
    
    # Setup processor
    config = Settings(watch_folder=str(tmp_path))
    processor = DocumentProcessor(
        config=config,
        vector_store=MagicMock(),
        chunker=MagicMock(),
        tei_client=None,
        embed_model=MockEmbedding(model_name="mock")
    )
    
    # Create test file
    f = tmp_path / "test.md"
    f.write_text("# Test", encoding="utf-8")
    
    # Run processing
    await processor.process_document(f)
    
    # Assert events were fired
    assert len(handler.events) >= 2
    # Check for specific event types (using string check to avoid import circularity in test setup if needed)
    event_names = [e.__class__.__name__ for e in handler.events]
    assert "DocumentProcessingStartEvent" in event_names
    assert "DocumentProcessingEndEvent" in event_names
    
    # Clean up
    dispatcher.event_handlers.clear()
