"""
End-to-end integration tests for Crawl4AIReader pipeline.

Tests integration between reader and chunker components.
Requires running service: Crawl4AI.

Note: Full integration with Qdrant/TEI would require metadata schema adapters
since Crawl4AIReader produces web-specific metadata (source_url, title, description)
while VectorStoreManager expects file-specific metadata (file_path_relative, filename).
This is intentional - the reader is a standalone LlamaIndex component.
"""

import pytest

from rag_ingestion.chunker import MarkdownChunker
from rag_ingestion.crawl4ai_reader import Crawl4AIReader
from rag_ingestion.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_reader_to_chunker() -> None:
    """
    Test E2E integration from Crawl4AIReader to MarkdownChunker.

    Verifies:
    - Reader crawls URL and returns Document with markdown text
    - Chunker splits Document text into semantic chunks
    - Chunks contain proper structure (chunk_text, chunk_index, section_path)

    This demonstrates the reader integrates with text-based downstream processing.
    """
    # Create reader (will validate Crawl4AI service health)
    try:
        reader = Crawl4AIReader(endpoint_url="http://localhost:52004")
    except ValueError as e:
        pytest.skip(f"Crawl4AI service unavailable: {e}")

    # Create chunker
    chunker = MarkdownChunker(
        chunk_size_tokens=512,
        chunk_overlap_percent=15,
    )

    # Crawl test URL
    test_url = "https://example.com"
    documents = await reader.aload_data([test_url])

    # Verify document created
    assert len(documents) == 1
    assert documents[0] is not None
    doc = documents[0]

    # Verify document has markdown content and web metadata
    assert len(doc.text) > 0
    assert doc.metadata["source"] == test_url
    assert doc.metadata["source_url"] == test_url
    assert doc.metadata["source_type"] == "web_crawl"
    assert "title" in doc.metadata

    logger.info(
        f"Crawled {test_url}: {len(doc.text)} chars, title='{doc.metadata['title']}'"
    )

    # Chunk the document using text-based API
    chunks = chunker.chunk(doc.text, filename=test_url)

    # Verify chunks created with proper structure
    assert len(chunks) > 0
    logger.info(f"Created {len(chunks)} chunks from crawled document")

    # Verify chunk structure
    for i, chunk in enumerate(chunks):
        # Required fields
        assert "chunk_text" in chunk
        assert "chunk_index" in chunk
        assert chunk["chunk_index"] == i
        assert len(chunk["chunk_text"]) > 0

        # Markdown-specific fields
        assert "section_path" in chunk
        assert "heading_level" in chunk

        logger.debug(
            f"Chunk {i}: {len(chunk['chunk_text'])} chars, "
            f"section='{chunk['section_path']}', level={chunk['heading_level']}"
        )

    # Verify integration preserves semantics
    # First chunk should start with content from crawled page
    assert len(chunks[0]["chunk_text"]) > 0

    logger.info(
        f"E2E test passed: {test_url} → {len(doc.text)} chars → {len(chunks)} chunks"
    )
