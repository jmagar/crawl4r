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
def tei_client() -> TEIClient:
    """Provide TEI client instance for tests."""
    return TEIClient(
        endpoint_url=TEI_ENDPOINT,
        dimensions=1024,
        timeout=30.0,
        max_retries=3,
    )


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


@pytest.mark.asyncio
async def test_e2e_reader_to_qdrant(
    reader_config: Crawl4AIReaderConfig,
    chunker: MarkdownChunker,
    tei_client: TEIClient,
    vector_store: VectorStoreManager,
) -> None:
    """
    Test E2E integration from Crawl4AIReader to Qdrant.

    Verifies:
    - Reader crawls URL and returns Document
    - Chunker splits Document into chunks
    - TEI generates embeddings for chunks
    - VectorStore stores vectors in Qdrant with metadata
    - source_url field is present in metadata (Issue #17)
    """
    # Skip if Crawl4AI service unavailable
    try:
        reader = Crawl4AIReader(config=reader_config)
    except ValueError as e:
        pytest.skip(f"Crawl4AI service unavailable: {e}")

    # Skip if TEI service unavailable
    try:
        await tei_client.health_check()
    except Exception as e:
        pytest.skip(f"TEI service unavailable: {e}")

    # Skip if Qdrant service unavailable
    try:
        await vector_store.create_collection()
    except Exception as e:
        pytest.skip(f"Qdrant service unavailable: {e}")

    # Crawl test URL
    test_url = "https://example.com"
    documents = await reader.aload_data([test_url])

    # Verify document created
    assert len(documents) == 1
    assert documents[0] is not None
    doc = documents[0]

    # Chunk the document
    chunks = chunker.chunk(doc.text, filename=test_url)
    assert len(chunks) > 0
    logger.info(f"Created {len(chunks)} chunks from document")

    # Generate embeddings for chunks
    chunk_texts = [chunk["chunk_text"] for chunk in chunks]
    embeddings = await tei_client.generate_embeddings(chunk_texts)
    assert len(embeddings) == len(chunks)
    logger.info(f"Generated {len(embeddings)} embeddings")

    # Store vectors in Qdrant with metadata from reader
    from rag_ingestion.vector_store import VectorMetadata

    vector_metadata_list = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        metadata = VectorMetadata(
            file_path_relative=test_url,
            file_path_absolute=test_url,
            source=doc.metadata["source"],
            source_url=doc.metadata["source_url"],  # Issue #17
            title=doc.metadata.get("title", ""),
            description=doc.metadata.get("description", ""),
            chunk_index=i,
            chunk_text=chunk["chunk_text"],
            section_path=chunk["section_path"],
            heading_level=chunk["heading_level"],
        )
        vector_metadata_list.append((embedding, metadata))

    # Upsert vectors to Qdrant
    await vector_store.upsert_vectors(vector_metadata_list)
    logger.info(f"Upserted {len(vector_metadata_list)} vectors to Qdrant")

    # Verify vectors stored with correct metadata
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    # Query by source_url (Issue #17)
    search_results = await vector_store.client.scroll(
        collection_name=vector_store.collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="source_url",
                    match=MatchValue(value=test_url),
                )
            ]
        ),
        limit=100,
    )

    points = search_results[0]
    assert len(points) == len(chunks)
    logger.info(f"Found {len(points)} vectors with source_url={test_url}")

    # Verify metadata fields
    for point in points:
        assert point.payload is not None
        assert point.payload["source"] == test_url
        assert point.payload["source_url"] == test_url  # Issue #17
        assert "title" in point.payload
        assert "chunk_index" in point.payload
        assert "chunk_text" in point.payload
        assert "section_path" in point.payload

    # Cleanup: delete test vectors
    await vector_store.delete_by_url(test_url)
    logger.info(f"Cleaned up test vectors for {test_url}")
