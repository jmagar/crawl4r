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
    def __init__(self):
        super().__init__()
        self.events: list = []  # Instance attribute, not class attribute

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
    dispatcher.add_event_handler(handler)

    try:
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

        # Assert exactly 2 events were fired (start and end)
        assert len(handler.events) == 2, f"Expected 2 events, got {len(handler.events)}"
        event_names = [e.__class__.__name__ for e in handler.events]
        assert event_names[0] == "DocumentProcessingStartEvent"
        assert event_names[1] == "DocumentProcessingEndEvent"
    finally:
        # Clean up - remove handler from dispatcher
        dispatcher.event_handlers.clear()
