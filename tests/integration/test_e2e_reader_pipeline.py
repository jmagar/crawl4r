"""
End-to-end integration tests for Crawl4AIReader pipeline.

Tests full pipeline integration between reader, chunker, and vector store components.
Requires running services: Crawl4AI, TEI, Qdrant.
"""

import os

import pytest

from rag_ingestion.chunker import MarkdownChunker
from rag_ingestion.config import Settings
from rag_ingestion.crawl4ai_reader import Crawl4AIReader, Crawl4AIReaderConfig
from rag_ingestion.logger import get_logger
from rag_ingestion.tei_client import TEIClient
from rag_ingestion.vector_store import VectorStoreManager

# Get service endpoints from environment or use defaults
TEI_ENDPOINT = os.getenv("TEI_ENDPOINT", "http://localhost:52000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:52001")
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://localhost:52004")

logger = get_logger(__name__)


@pytest.fixture
def reader_config() -> Crawl4AIReaderConfig:
    """Provide Crawl4AIReader configuration for tests."""
    return Crawl4AIReaderConfig(
        base_url=CRAWL4AI_URL,
        timeout=30,
        max_retries=3,
        retry_delays=[1.0, 2.0, 4.0],
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=60,
        concurrency_limit=3,
    )


@pytest.fixture
def settings() -> Settings:
    """Provide Settings instance for tests."""
    return Settings(
        TEI_ENDPOINT=TEI_ENDPOINT,
        QDRANT_URL=QDRANT_URL,
    )


@pytest.fixture
def chunker() -> MarkdownChunker:
    """Provide MarkdownChunker instance for tests."""
    return MarkdownChunker(
        chunk_size_tokens=512,
        chunk_overlap_percent=15,
    )


@pytest.fixture
def tei_client(settings: Settings) -> TEIClient:
    """Provide TEI client instance for tests."""
    return TEIClient(settings=settings)


@pytest.fixture
def vector_store() -> VectorStoreManager:
    """Provide VectorStoreManager instance for tests."""
    return VectorStoreManager(
        qdrant_url=QDRANT_URL,
        collection_name="test_reader_pipeline",
        dimensions=1024,
    )


@pytest.mark.asyncio
async def test_e2e_reader_to_chunker(
    reader_config: Crawl4AIReaderConfig,
    chunker: MarkdownChunker,
) -> None:
    """
    Test E2E integration from Crawl4AIReader to MarkdownChunker.

    Verifies:
    - Reader crawls URL and returns Document
    - Chunker splits Document into chunks
    - Chunks contain proper metadata from reader
    """
    # Skip if Crawl4AI service unavailable
    try:
        reader = Crawl4AIReader(config=reader_config)
    except ValueError as e:
        pytest.skip(f"Crawl4AI service unavailable: {e}")

    # Crawl test URL
    test_url = "https://example.com"
    documents = await reader.aload_data([test_url])

    # Verify document created
    assert len(documents) == 1
    assert documents[0] is not None
    doc = documents[0]

    # Verify document has content and metadata
    assert len(doc.text) > 0
    assert doc.metadata["source"] == test_url
    assert doc.metadata["source_url"] == test_url
    assert "title" in doc.metadata

    # Chunk the document using chunker's text-based API
    chunks = chunker.chunk(doc.text, filename=test_url)

    # Verify chunks created
    assert len(chunks) > 0
    logger.info(f"Created {len(chunks)} chunks from document")

    # Verify chunks have proper structure
    for i, chunk in enumerate(chunks):
        assert "chunk_text" in chunk
        assert "chunk_index" in chunk
        assert chunk["chunk_index"] == i
        assert len(chunk["chunk_text"]) > 0
        assert "section_path" in chunk
        assert "heading_level" in chunk
        logger.debug(
            f"Chunk {i}: {len(chunk['chunk_text'])} chars, "
            f"section={chunk['section_path']}",
            extra={
                "chunk_index": i,
                "chunk_size": len(chunk["chunk_text"]),
                "section": chunk["section_path"],
            },
        )
