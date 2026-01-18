from unittest.mock import MagicMock

import pytest
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler

from crawl4r.core.config import Settings
from crawl4r.core.instrumentation import (
    ChunkingEndEvent,
    ChunkingStartEvent,
    DocumentProcessingEndEvent,
    DocumentProcessingStartEvent,
    EmbeddingBatchEvent,
    LoggingEventHandler,
    PipelineEndEvent,
    PipelineStartEvent,
    VectorStoreUpsertEvent,
    span,
    span_context,
)
from crawl4r.processing.processor import DocumentProcessor


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
    events: list = []  # Pydantic field declaration

    def model_post_init(self, __context) -> None:
        """Initialize events list after Pydantic model init."""
        # Use object.__setattr__ to bypass Pydantic's __setattr__
        object.__setattr__(self, "events", [])

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
        # Setup processor with correct constructor signature
        config = Settings(watch_folder=str(tmp_path))
        processor = DocumentProcessor(
            config=config,
            vector_store=MagicMock(),
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
        # Clean up - remove only the specific handler added in this test
        # Avoid clearing all handlers which would affect other tests
        if handler in dispatcher.event_handlers:
            dispatcher.event_handlers.remove(handler)


def test_custom_events_exist():
    """Test that all custom events are importable and have expected fields."""
    # Document events
    start = DocumentProcessingStartEvent(file_path="/test/path.md")
    assert start.file_path == "/test/path.md"

    end = DocumentProcessingEndEvent(file_path="/test/path.md", success=True)
    assert end.success is True
    assert end.error is None

    # Chunking events
    chunk_start = ChunkingStartEvent(file_path="/test.md", document_length=1000)
    assert chunk_start.document_length == 1000

    chunk_end = ChunkingEndEvent(file_path="/test.md", num_chunks=5, duration_ms=100.0)
    assert chunk_end.num_chunks == 5

    # Embedding event
    embed = EmbeddingBatchEvent(batch_size=32, duration_ms=50.0)
    assert embed.batch_size == 32

    # Vector store event
    upsert = VectorStoreUpsertEvent(
        collection_name="test", point_count=100, duration_ms=200.0
    )
    assert upsert.point_count == 100

    # Pipeline events
    pipeline_start = PipelineStartEvent(total_documents=10)
    assert pipeline_start.total_documents == 10

    pipeline_end = PipelineEndEvent(
        total_documents=10, successful=8, failed=2, duration_seconds=5.5
    )
    assert pipeline_end.failed == 2


def test_span_decorator_sync():
    """Test span decorator works with sync functions."""
    call_count = 0

    @span("test_sync_op")
    def sync_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    result = sync_function(5)
    assert result == 10
    assert call_count == 1


@pytest.mark.asyncio
async def test_span_decorator_async():
    """Test span decorator works with async functions."""
    call_count = 0

    @span("test_async_op")
    async def async_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    result = await async_function(5)
    assert result == 10
    assert call_count == 1


def test_span_context_manager():
    """Test span_context context manager."""
    with span_context("test_context", key="value") as ctx:
        ctx["extra"] = "data"

    assert ctx["name"] == "test_context"
    assert ctx["key"] == "value"
    assert ctx["extra"] == "data"
    assert "duration_ms" in ctx


def test_logging_event_handler():
    """Test LoggingEventHandler can be instantiated."""
    import logging

    handler = LoggingEventHandler(log_level=logging.INFO)
    assert handler.class_name() == "LoggingEventHandler"

    # Should handle events without error
    event = DocumentProcessingStartEvent(file_path="/test.md")
    handler.handle(event)  # Should not raise
