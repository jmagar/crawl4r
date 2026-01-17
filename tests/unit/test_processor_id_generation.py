import uuid
from unittest.mock import MagicMock

import pytest
from llama_index.core.embeddings import BaseEmbedding

from crawl4r.core.config import Settings
from crawl4r.processing.processor import DocumentProcessor


class MockEmbedding(BaseEmbedding):
    """Mock embedding model that satisfies Pydantic validation."""

    def _get_query_embedding(self, query):
        return []

    async def _aget_query_embedding(self, query):
        return []

    def _get_text_embedding(self, text):
        return []

    async def _aget_text_embedding(self, text):
        return []

    def _get_text_embeddings(self, texts):
        return []

    async def _aget_text_embeddings(self, texts):
        return []


@pytest.mark.asyncio
async def test_deterministic_document_ids(tmp_path):
    # Setup
    config = Settings(watch_folder=str(tmp_path))
    mock_vector_store = MagicMock()
    mock_chunker = MagicMock()

    processor = DocumentProcessor(
        config=config,
        vector_store=mock_vector_store,
        chunker=mock_chunker,
        tei_client=None,
        embed_model=MockEmbedding(model_name="mock"),
    )

    # Create dummy file
    test_file = tmp_path / "test.md"
    test_file.write_text("# Test", encoding="utf-8")

    # Process twice and verify both succeed
    result1 = await processor.process_document(test_file)
    result2 = await processor.process_document(test_file)
    assert result1.success
    assert result2.success

    # Verify deterministic ID generation using relative path from config
    rel_path = str(test_file.relative_to(config.watch_folder))
    id1 = processor._generate_document_id(rel_path)
    id2 = processor._generate_document_id(rel_path)
    assert id1 == id2, "Same relative path should produce same ID"

    # Verify uniqueness for different paths
    id3 = processor._generate_document_id("other.md")
    assert id1 != id3, "Different paths should produce different IDs"

    # Verify UUID5 format with NAMESPACE_URL
    expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, rel_path))
    assert id1 == expected_id, "ID should match UUID5 with NAMESPACE_URL"
