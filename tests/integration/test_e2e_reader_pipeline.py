"""
End-to-end integration tests for Crawl4AIReader pipeline.

Tests full pipeline integration between reader, chunker, and vector store components.
Requires running services: Crawl4AI, TEI, Qdrant.
"""

import pytest

from rag_ingestion.chunker import MarkdownChunker
from rag_ingestion.crawl4ai_reader import Crawl4AIReader, Crawl4AIReaderConfig
from rag_ingestion.embeddings import TEIClient, TEIClientConfig
from rag_ingestion.logger import get_logger
from rag_ingestion.vector_store import VectorStoreManager

logger = get_logger(__name__)


@pytest.fixture
def reader_config() -> Crawl4AIReaderConfig:
    """Provide Crawl4AIReader configuration for tests."""
    return Crawl4AIReaderConfig(
        base_url="http://localhost:52004",
        timeout=30,
        max_retries=3,
        retry_delays=[1.0, 2.0, 4.0],
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=60,
        concurrency_limit=3,
    )


@pytest.fixture
def chunker() -> MarkdownChunker:
    """Provide MarkdownChunker instance for tests."""
    return MarkdownChunker(
        chunk_size=512,
        chunk_overlap_percent=15,
    )


@pytest.fixture
def tei_client() -> TEIClient:
    """Provide TEI client instance for tests."""
    config = TEIClientConfig(
        endpoint_url="http://localhost:52000",
        timeout_seconds=30,
        max_retries=3,
    )
    return TEIClient(config)


@pytest.fixture
def vector_store(tei_client: TEIClient) -> VectorStoreManager:
    """Provide VectorStoreManager instance for tests."""
    return VectorStoreManager(
        qdrant_url="http://localhost:52001",
        collection_name="test_reader_pipeline",
        embeddings_client=tei_client,
    )
