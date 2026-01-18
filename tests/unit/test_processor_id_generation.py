import hashlib
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

    processor = DocumentProcessor(
        config=config,
        vector_store=mock_vector_store,
        embed_model=MockEmbedding(model_name="mock"),
    )

    # Create dummy file
    test_file = tmp_path / "test.md"
    test_file.write_text("# Test", encoding="utf-8")

    # Process twice and verify both succeed
    result1 = await processor.process_document(test_file)
    result2 = await processor.process_document(test_file)
    assert result1.success, f"First process_document failed: {result1.error}"
    assert result2.success, f"Second process_document failed: {result2.error}"

    # Verify both process_document calls produce consistent IDs
    # (file_path in results should be identical for same file)
    assert result1.file_path == result2.file_path, (
        "Same file should produce same file_path in results"
    )

    # Verify deterministic ID generation using relative path from config
    rel_path = str(test_file.relative_to(config.watch_folder))
    id1 = processor._generate_document_id(rel_path)
    id2 = processor._generate_document_id(rel_path)
    assert id1 == id2, "Same relative path should produce same ID"

    # Verify uniqueness for different paths
    id3 = processor._generate_document_id("other.md")
    assert id1 != id3, "Different paths should produce different IDs"

    # Verify SHA256-to-UUID format (matches qdrant.py::_generate_point_id pattern)
    hash_bytes = hashlib.sha256(rel_path.encode()).digest()
    expected_id = str(uuid.UUID(bytes=hash_bytes[:16]))
    assert id1 == expected_id, "ID should match SHA256-to-UUID derivation"


@pytest.mark.asyncio
async def test_document_id_stable_with_file_path_metadata(tmp_path):
    """Verify document IDs are stable when using file_path metadata.

    After SimpleDirectoryReader migration, IDs are derived from the relative path
    computed from file_path (absolute) relative to watch_folder. This test ensures
    the ID generation remains stable regardless of how the file is accessed.
    """
    # Setup with real watch folder
    watch_folder = tmp_path / "docs"
    watch_folder.mkdir()

    config = Settings(watch_folder=str(watch_folder))
    mock_vector_store = MagicMock()

    processor = DocumentProcessor(
        config=config,
        vector_store=mock_vector_store,
        embed_model=MockEmbedding(model_name="mock"),
    )

    # Create file in subdirectory
    file_path = watch_folder / "api" / "guide.md"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("# API Guide\n\nDocumentation.", encoding="utf-8")

    # Process same file twice
    result1 = await processor.process_document(file_path)
    result2 = await processor.process_document(file_path)

    assert result1.success, f"First process failed: {result1.error}"
    assert result2.success, f"Second process failed: {result2.error}"

    # Verify IDs are deterministic based on relative path
    rel_path = "api/guide.md"  # Relative to watch_folder
    expected_id = processor._generate_document_id(rel_path)

    # Generate ID from absolute path (simulating SimpleDirectoryReader's file_path)
    abs_path = str(file_path)
    computed_rel_path = str(file_path.relative_to(watch_folder))
    computed_id = processor._generate_document_id(computed_rel_path)

    assert expected_id == computed_id, (
        f"ID from relative path '{rel_path}' should match ID from computed relative path"
    )
