"""End-to-end integration test for crawl -> embed -> Qdrant pipeline.

Validates the custom pipeline using Crawl4AIReader with a test-local metadata
adapter (no production code changes):
1. Crawl a URL with Crawl4AIReader
2. Chunk markdown content with MarkdownChunker
3. Generate embeddings via TEI
4. Upsert vectors into Qdrant via VectorStoreManager

This test requires Crawl4AI, TEI, and Qdrant services to be running.

Note: This test uses MarkdownChunker directly (not via DocumentProcessor) to test
the standalone chunker integration with web-crawled content. This is intentional
as it validates the chunker API independently of the document processing pipeline.
"""

import os
from collections.abc import AsyncIterator
from urllib.parse import urlparse

import httpx
import pytest
from qdrant_client import AsyncQdrantClient

from crawl4r.processing.chunker import MarkdownChunker
from crawl4r.readers.crawl4ai import Crawl4AIReader
from crawl4r.storage.tei import TEIClient
from crawl4r.storage.qdrant import VectorMetadata, VectorStoreManager

TEI_ENDPOINT = os.getenv("TEI_ENDPOINT", "http://localhost:52000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:52001")
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://localhost:52004")


@pytest.fixture(autouse=True)
async def check_services() -> None:
    """Skip test if Crawl4AI, TEI, or Qdrant are unavailable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{CRAWL4AI_URL}/health")
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip(f"Crawl4AI service not available at {CRAWL4AI_URL}")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{TEI_ENDPOINT}/health")
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip(f"TEI service not available at {TEI_ENDPOINT}")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{QDRANT_URL}/readyz")
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip(f"Qdrant service not available at {QDRANT_URL}")


@pytest.fixture
async def qdrant_client() -> AsyncIterator[AsyncQdrantClient]:
    """Async Qdrant client for verification queries."""
    client = AsyncQdrantClient(url=QDRANT_URL, timeout=30)
    yield client
    await client.close()


def _adapt_crawl_metadata(
    source_url: str,
    chunk: dict,
    chunk_index: int,
) -> VectorMetadata:
    """Adapt Crawl4AI metadata to vector store schema for testing."""
    parsed = urlparse(source_url)
    raw_path = parsed.path or "/"
    safe_path = raw_path if raw_path.startswith("/") else f"/{raw_path}"
    file_path_relative = f"web/{parsed.netloc}{safe_path}"
    filename = safe_path.rsplit("/", 1)[-1] or parsed.netloc

    metadata: VectorMetadata = {
        "file_path_relative": file_path_relative,
        "file_path_absolute": source_url,
        "filename": filename,
        "chunk_index": chunk_index,
        "chunk_text": chunk["chunk_text"],
        "section_path": chunk["section_path"],
        "heading_level": chunk["heading_level"],
        "source_url": source_url,
        "source_type": "web_crawl",
    }

    tags = chunk.get("tags")
    if tags:
        metadata["tags"] = tags

    return metadata


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_crawl_to_qdrant(
    test_collection: str,
    cleanup_fixture: None,
    qdrant_client: AsyncQdrantClient,
) -> None:
    """Test Crawl4AI -> chunking -> TEI -> Qdrant end-to-end."""
    reader = Crawl4AIReader(endpoint_url=CRAWL4AI_URL)
    chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
    tei_client = TEIClient(TEI_ENDPOINT)
    vector_store = VectorStoreManager(QDRANT_URL, test_collection)

    vector_store.ensure_collection()
    vector_store.ensure_payload_indexes()

    test_url = "https://example.com"
    documents = await reader.aload_data([test_url])

    assert len(documents) == 1
    assert documents[0] is not None
    document = documents[0]

    assert document.text
    assert document.metadata["source_url"] == test_url

    chunks = chunker.chunk(document.text, filename=test_url)
    assert chunks

    chunk_texts = [chunk["chunk_text"] for chunk in chunks]
    embeddings: list[list[float]] = []
    for i in range(0, len(chunk_texts), tei_client.batch_size_limit):
        batch = chunk_texts[i : i + tei_client.batch_size_limit]
        embeddings.extend(await tei_client.embed_batch(batch))

    vectors_with_metadata = []
    for chunk, embedding in zip(chunks, embeddings):
        metadata = _adapt_crawl_metadata(test_url, chunk, chunk["chunk_index"])
        vectors_with_metadata.append({"vector": embedding, "metadata": metadata})

    vector_store.upsert_vectors_batch(vectors_with_metadata)

    collection_info = await qdrant_client.get_collection(test_collection)
    assert collection_info.points_count == len(chunks)

    points, _ = await qdrant_client.scroll(
        collection_name=test_collection,
        limit=10,
    )
    assert points
    assert any(
        point.payload
        and point.payload.get("source_url") == test_url
        and point.payload.get("file_path_relative")
        for point in points
    )
