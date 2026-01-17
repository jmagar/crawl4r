import pytest
import hashlib
import uuid
from pathlib import Path
from crawl4r.processing.processor import DocumentProcessor
from crawl4r.core.config import Settings
from unittest.mock import MagicMock
from llama_index.core.embeddings import BaseEmbedding

class MockEmbedding(BaseEmbedding):
    """Mock embedding model that satisfies Pydantic validation."""
    def _get_query_embedding(self, query): return []
    async def _aget_query_embedding(self, query): return []
    def _get_text_embedding(self, text): return []
    async def _aget_text_embedding(self, text): return []
    def _get_text_embeddings(self, texts): return []
    async def _aget_text_embeddings(self, texts): return []

@pytest.mark.asyncio
async def test_deterministic_document_ids(tmp_path):
    # Setup
    config = Settings(watch_folder=str(tmp_path))
    # Mock dependencies to avoid strictly requiring real services
    mock_vector_store = MagicMock()
    mock_chunker = MagicMock()
    
    # Initialize processor with proper mock embedding
    processor = DocumentProcessor(
        config=config, 
        vector_store=mock_vector_store, 
        chunker=mock_chunker, 
        tei_client=None, 
        embed_model=MockEmbedding(model_name="mock")
    )
    
    # Create dummy file
    f1 = tmp_path / "test.md"
    f1.write_text("# Test", encoding="utf-8")
    
    # Process twice
    result1 = await processor.process_document(f1)
    result2 = await processor.process_document(f1)
    
    # Assert
    assert result1.success
    assert result2.success
    
    # Verify the document ID generation logic
    # We can inspect the private method if we can't easily capture the ID from result
    # (ProcessingResult doesn't include the ID, but we can verify the method exists and works)
    
    # Check if method exists
    assert hasattr(processor, "_generate_document_id")
    
    # Check determinism directly
    rel_path = "test.md"
    id1 = processor._generate_document_id(rel_path)
    id2 = processor._generate_document_id(rel_path)
    assert id1 == id2
    
    # Check uniqueness
    id3 = processor._generate_document_id("other.md")
    assert id1 != id3
    
    # Verify the specific ID format (SHA256 -> UUID)
    expected_hash = hashlib.sha256(rel_path.encode("utf-8")).digest()
    expected_id = str(uuid.UUID(bytes=expected_hash[:16]))
    assert id1 == expected_id
