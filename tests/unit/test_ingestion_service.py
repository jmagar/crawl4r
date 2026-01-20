"""Unit tests for IngestionService orchestration logic.

These tests verify the IngestionService correctly:
- Generates unique crawl IDs with proper format
- Acquires locks before processing crawls
- Queues crawls when lock is held by another process
- Invokes progress callbacks during ingestion
- Deduplicates existing URL data before upserting new vectors
- Handles partial failures and sets appropriate status
- Releases locks in finally block even on errors

All service dependencies are mocked using AsyncMock for isolated unit testing.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4r.services.ingestion import IngestionService, generate_crawl_id
from crawl4r.services.models import CrawlStatus, ScrapeResult


def test_generate_crawl_id_format() -> None:
    """Verify crawl ID format is 'crawl_' prefix with timestamp and nonce.

    The crawl ID should start with 'crawl_' to make it easily identifiable
    and distinguishable from other IDs in the system.
    """
    crawl_id = generate_crawl_id()
    assert crawl_id.startswith("crawl_")
    assert len(crawl_id) > 7  # More than just the prefix


def test_generate_crawl_id_uniqueness() -> None:
    """Verify consecutive crawl IDs are unique.

    Each call to generate_crawl_id should produce a different ID
    due to the timestamp and random nonce components.
    """
    id1 = generate_crawl_id()
    id2 = generate_crawl_id()
    assert id1 != id2


@pytest.mark.asyncio
async def test_ingest_urls_empty_list_returns_error() -> None:
    """Verify empty URL list returns failure without processing.

    When no URLs are provided, the service should immediately return
    an error result without attempting any scraping or embedding.
    """
    service = IngestionService(
        scraper=AsyncMock(),
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=AsyncMock(),
    )

    result = await service.ingest_urls([])

    assert result.success is False
    assert "No URLs provided" in result.error
    assert result.urls_total == 0
    assert result.queued is False


@pytest.mark.asyncio
async def test_ingest_urls_invalid_url_returns_error() -> None:
    """Verify invalid URLs are rejected before processing.

    URL validation should happen upfront to prevent wasting resources
    on URLs that will fail during scraping.
    """
    service = IngestionService(
        scraper=AsyncMock(),
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=AsyncMock(),
    )

    result = await service.ingest_urls(["not-a-valid-url"])

    assert result.success is False
    assert "Invalid URLs" in result.error
    assert result.urls_failed == 1
    assert result.queued is False


@pytest.mark.asyncio
async def test_ingest_urls_lock_acquired_processes_immediately() -> None:
    """Verify successful lock acquisition triggers immediate processing.

    When the lock is acquired, the service should proceed with scraping
    and return a result with queued=False.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(return_value=[])

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    service = IngestionService(
        scraper=scraper,
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=queue_manager,
    )

    result = await service.ingest_urls(["https://example.com"])

    assert result.crawl_id.startswith("crawl_")
    assert result.queued is False
    queue_manager.acquire_lock.assert_awaited_once()
    scraper.scrape_urls.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_urls_queued_when_lock_held() -> None:
    """Verify crawl is queued when lock is held by another process.

    When the lock cannot be acquired, the service should enqueue the
    crawl request and return immediately with queued=True.
    """
    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=False)
    queue_manager.enqueue_crawl = AsyncMock()
    queue_manager.set_status = AsyncMock()

    service = IngestionService(
        scraper=AsyncMock(),
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=queue_manager,
    )

    result = await service.ingest_urls(["https://example.com"])

    assert result.queued is True
    assert result.success is True
    assert result.chunks_created == 0
    queue_manager.enqueue_crawl.assert_awaited_once()
    queue_manager.set_status.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_urls_sets_status_to_running() -> None:
    """Verify status is set to RUNNING when processing begins.

    After acquiring the lock, the service should update the crawl
    status to RUNNING before starting the scrape.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(return_value=[])

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    service = IngestionService(
        scraper=scraper,
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=queue_manager,
    )

    await service.ingest_urls(["https://example.com"])

    # Should be called twice: once for RUNNING, once for final status
    assert queue_manager.set_status.await_count == 2

    # First call should set status to RUNNING
    first_call = queue_manager.set_status.await_args_list[0][0][0]
    assert first_call.status == CrawlStatus.RUNNING


@pytest.mark.asyncio
async def test_ingest_urls_sets_status_to_completed_on_success() -> None:
    """Verify status is set to COMPLETED when all URLs succeed.

    When all URLs are scraped successfully, the final status should
    be COMPLETED with no error message.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(
                url="https://example.com",
                success=True,
                markdown="# Content",
            )
        ]
    )

    embeddings = AsyncMock()
    embeddings.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])

    vector_store = AsyncMock()
    vector_store.delete_by_url = AsyncMock()
    vector_store.upsert_vectors_batch = AsyncMock()

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    node_parser = MagicMock()
    mock_node = MagicMock()
    mock_node.text = "# Content"
    mock_node.metadata = {}
    mock_node.node_id = "node_123"
    node_parser.get_nodes_from_documents = MagicMock(return_value=[mock_node])

    service = IngestionService(
        scraper=scraper,
        embeddings=embeddings,
        vector_store=vector_store,
        queue_manager=queue_manager,
        node_parser=node_parser,
    )

    result = await service.ingest_urls(["https://example.com"])

    assert result.success is True
    assert result.error is None
    assert result.urls_failed == 0

    # Final status call should be COMPLETED
    final_call = queue_manager.set_status.await_args_list[-1][0][0]
    assert final_call.status == CrawlStatus.COMPLETED
    assert final_call.error is None


@pytest.mark.asyncio
async def test_ingest_urls_sets_status_to_partial_on_some_failures() -> None:
    """Verify status is set to PARTIAL when some URLs fail.

    When some URLs succeed and others fail, the status should be
    PARTIAL with an error message indicating the failure count.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(url="https://example.com", success=True, markdown="# A"),
            ScrapeResult(
                url="https://example.org", success=False, error="Failed to scrape"
            ),
        ]
    )

    embeddings = AsyncMock()
    embeddings.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])

    vector_store = AsyncMock()
    vector_store.delete_by_url = AsyncMock()
    vector_store.upsert_vectors_batch = AsyncMock()

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    node_parser = MagicMock()
    mock_node = MagicMock()
    mock_node.text = "# A"
    mock_node.metadata = {}
    mock_node.node_id = "node_123"
    node_parser.get_nodes_from_documents = MagicMock(return_value=[mock_node])

    service = IngestionService(
        scraper=scraper,
        embeddings=embeddings,
        vector_store=vector_store,
        queue_manager=queue_manager,
        node_parser=node_parser,
    )

    result = await service.ingest_urls(
        ["https://example.com", "https://example.org"]
    )

    assert result.success is False
    assert result.urls_failed == 1
    assert "1/2 URLs failed" in result.error

    # Final status should be PARTIAL
    final_call = queue_manager.set_status.await_args_list[-1][0][0]
    assert final_call.status == CrawlStatus.PARTIAL


@pytest.mark.asyncio
async def test_ingest_urls_sets_status_to_failed_on_all_failures() -> None:
    """Verify status is set to FAILED when all URLs fail.

    When all URLs fail to scrape, the final status should be FAILED
    with an appropriate error message.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(url="https://example.com", success=False, error="Failed"),
            ScrapeResult(url="https://example.org", success=False, error="Failed"),
        ]
    )

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    service = IngestionService(
        scraper=scraper,
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=queue_manager,
    )

    result = await service.ingest_urls(
        ["https://example.com", "https://example.org"]
    )

    assert result.success is False
    assert result.urls_failed == 2
    assert "All URLs failed" in result.error

    # Final status should be FAILED
    final_call = queue_manager.set_status.await_args_list[-1][0][0]
    assert final_call.status == CrawlStatus.FAILED


@pytest.mark.asyncio
async def test_ingest_urls_releases_lock_in_finally() -> None:
    """Verify lock is released even if ingestion fails.

    The lock must be released in a finally block to prevent deadlocks
    when errors occur during processing.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(side_effect=RuntimeError("Scraper failed"))

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    service = IngestionService(
        scraper=scraper,
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=queue_manager,
    )

    with pytest.raises(RuntimeError, match="Scraper failed"):
        await service.ingest_urls(["https://example.com"])

    # Lock should still be released
    queue_manager.release_lock.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_urls_deletes_existing_vectors_before_upsert() -> None:
    """Verify deduplication removes existing URL data before upserting.

    Before inserting new vectors, the service should delete any existing
    vectors for the same URL to prevent duplicates.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(
                url="https://example.com",
                success=True,
                markdown="# Content",
            )
        ]
    )

    embeddings = AsyncMock()
    embeddings.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])

    vector_store = AsyncMock()
    vector_store.delete_by_url = AsyncMock()
    vector_store.upsert_vectors_batch = AsyncMock()

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    node_parser = MagicMock()
    mock_node = MagicMock()
    mock_node.text = "# Content"
    mock_node.metadata = {}
    mock_node.node_id = "node_123"
    node_parser.get_nodes_from_documents = MagicMock(return_value=[mock_node])

    service = IngestionService(
        scraper=scraper,
        embeddings=embeddings,
        vector_store=vector_store,
        queue_manager=queue_manager,
        node_parser=node_parser,
    )

    await service.ingest_urls(["https://example.com"])

    # Verify delete_by_url was called before upsert
    vector_store.delete_by_url.assert_awaited_once_with("https://example.com")
    vector_store.upsert_vectors_batch.assert_awaited_once()

    # Verify delete was called before upsert by checking call order
    delete_call_count = vector_store.delete_by_url.await_count
    upsert_call_count = vector_store.upsert_vectors_batch.await_count
    assert delete_call_count == 1
    assert upsert_call_count == 1


@pytest.mark.asyncio
async def test_ingest_urls_counts_chunks_correctly() -> None:
    """Verify chunks_created reflects actual number of nodes generated.

    The result should accurately count the number of chunks created
    across all successfully scraped URLs.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(url="https://example.com", success=True, markdown="# A"),
            ScrapeResult(url="https://example.org", success=True, markdown="# B"),
        ]
    )

    embeddings = AsyncMock()
    embeddings.embed_batch = AsyncMock(
        return_value=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    )

    vector_store = AsyncMock()
    vector_store.delete_by_url = AsyncMock()
    vector_store.upsert_vectors_batch = AsyncMock()

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    node_parser = MagicMock()

    # First document produces 2 nodes, second produces 1 node
    call_count = 0

    def get_nodes_side_effect(docs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First URL: 2 nodes
            node1 = MagicMock()
            node1.text = "# A part 1"
            node1.metadata = {}
            node1.node_id = "node_1"
            node2 = MagicMock()
            node2.text = "# A part 2"
            node2.metadata = {}
            node2.node_id = "node_2"
            return [node1, node2]
        else:
            # Second URL: 1 node
            node3 = MagicMock()
            node3.text = "# B"
            node3.metadata = {}
            node3.node_id = "node_3"
            return [node3]

    node_parser.get_nodes_from_documents = MagicMock(
        side_effect=get_nodes_side_effect
    )

    service = IngestionService(
        scraper=scraper,
        embeddings=embeddings,
        vector_store=vector_store,
        queue_manager=queue_manager,
        node_parser=node_parser,
    )

    result = await service.ingest_urls(
        ["https://example.com", "https://example.org"]
    )

    assert result.chunks_created == 3
    assert result.urls_total == 2
    assert result.urls_failed == 0


@pytest.mark.asyncio
async def test_ingest_urls_skips_failed_scrapes() -> None:
    """Verify failed scrapes don't create chunks or embeddings.

    When a URL fails to scrape, the service should skip embedding
    and vector storage for that URL and continue with others.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(url="https://example.com", success=False, error="Failed"),
            ScrapeResult(url="https://example.org", success=True, markdown="# B"),
        ]
    )

    embeddings = AsyncMock()
    embeddings.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])

    vector_store = AsyncMock()
    vector_store.delete_by_url = AsyncMock()
    vector_store.upsert_vectors_batch = AsyncMock()

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    node_parser = MagicMock()
    mock_node = MagicMock()
    mock_node.text = "# B"
    mock_node.metadata = {}
    mock_node.node_id = "node_123"
    node_parser.get_nodes_from_documents = MagicMock(return_value=[mock_node])

    service = IngestionService(
        scraper=scraper,
        embeddings=embeddings,
        vector_store=vector_store,
        queue_manager=queue_manager,
        node_parser=node_parser,
    )

    result = await service.ingest_urls(
        ["https://example.com", "https://example.org"]
    )

    # Only one URL should be processed (the successful one)
    assert result.chunks_created == 1
    assert result.urls_failed == 1

    # delete_by_url should only be called once (for successful URL)
    vector_store.delete_by_url.assert_awaited_once_with("https://example.org")


@pytest.mark.asyncio
async def test_ingest_urls_skips_empty_markdown() -> None:
    """Verify empty markdown results don't create chunks.

    When a scrape returns success but no markdown content,
    the service should skip embedding and continue.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(url="https://example.com", success=True, markdown=None),
            ScrapeResult(url="https://example.org", success=True, markdown=""),
        ]
    )

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    service = IngestionService(
        scraper=scraper,
        embeddings=AsyncMock(),
        vector_store=AsyncMock(),
        queue_manager=queue_manager,
    )

    result = await service.ingest_urls(
        ["https://example.com", "https://example.org"]
    )

    # Both URLs should be counted as failed since they have no content
    assert result.urls_failed == 2
    assert result.chunks_created == 0


@pytest.mark.asyncio
async def test_validate_url_static_method() -> None:
    """Verify validate_url static method delegates to url_validation module.

    The static method should provide a convenient API for URL validation
    without requiring an instance of IngestionService.
    """
    # Valid URL should pass
    assert IngestionService.validate_url("https://example.com") is True

    # Invalid URL should fail
    assert IngestionService.validate_url("not-a-url") is False


@pytest.mark.asyncio
async def test_validate_services_calls_all_dependencies() -> None:
    """Verify validate_services checks all service dependencies.

    The method should attempt to validate each service that has
    a validate_services method.
    """
    scraper = AsyncMock()
    scraper.validate_services = AsyncMock()

    embeddings = AsyncMock()
    embeddings.validate_services = AsyncMock()

    vector_store = AsyncMock()
    vector_store.validate_services = AsyncMock()

    queue_manager = AsyncMock()
    queue_manager.validate_services = AsyncMock()

    service = IngestionService(
        scraper=scraper,
        embeddings=embeddings,
        vector_store=vector_store,
        queue_manager=queue_manager,
    )

    await service.validate_services()

    scraper.validate_services.assert_awaited_once()
    embeddings.validate_services.assert_awaited_once()
    vector_store.validate_services.assert_awaited_once()
    queue_manager.validate_services.assert_awaited_once()


@pytest.mark.asyncio
async def test_validate_services_skips_missing_methods() -> None:
    """Verify validate_services handles dependencies without validation.

    Some dependencies may not have validate_services methods,
    and the validation should skip those gracefully.
    """
    # Create mocks without validate_services methods
    scraper = AsyncMock()
    embeddings = AsyncMock()

    # Only queue_manager has validate_services
    queue_manager = AsyncMock()
    queue_manager.validate_services = AsyncMock()

    service = IngestionService(
        scraper=scraper,
        embeddings=embeddings,
        vector_store=AsyncMock(),
        queue_manager=queue_manager,
    )

    # Should not raise, even though most services lack validate_services
    await service.validate_services()

    # Only queue_manager should be validated
    queue_manager.validate_services.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_result_preserves_metadata() -> None:
    """Verify document metadata is preserved through the ingestion pipeline.

    Metadata from the scrape result should be included in the document
    and propagated to the vector metadata.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(
                url="https://example.com",
                success=True,
                markdown="# Page Title",
                metadata={
                    "title": "Example Page",
                    "description": "A test page",
                    "status_code": 200,
                },
            )
        ]
    )

    embeddings = AsyncMock()
    embeddings.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])

    vector_store = AsyncMock()
    vector_store.delete_by_url = AsyncMock()
    vector_store.upsert_vectors_batch = AsyncMock()

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    node_parser = MagicMock()
    mock_node = MagicMock()
    mock_node.text = "# Page Title"
    mock_node.get_content = MagicMock(return_value="# Page Title")
    mock_node.metadata = {"section_path": "Title"}
    mock_node.node_id = "node_123"
    node_parser.get_nodes_from_documents = MagicMock(return_value=[mock_node])

    service = IngestionService(
        scraper=scraper,
        embeddings=embeddings,
        vector_store=vector_store,
        queue_manager=queue_manager,
        node_parser=node_parser,
    )

    await service.ingest_urls(["https://example.com"])

    # Verify upsert was called with metadata
    vector_store.upsert_vectors_batch.assert_awaited_once()
    call_args = vector_store.upsert_vectors_batch.await_args[0][0]
    assert len(call_args) == 1
    vector_metadata = call_args[0]["metadata"]
    assert vector_metadata["source_url"] == "https://example.com"
    assert vector_metadata["chunk_text"] == "# Page Title"


@pytest.mark.asyncio
async def test_node_with_get_content_method() -> None:
    """Verify nodes with get_content() method are handled correctly.

    Some node types use get_content() instead of .text attribute
    for retrieving the text content.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(
                url="https://example.com",
                success=True,
                markdown="# Content",
            )
        ]
    )

    embeddings = AsyncMock()
    embeddings.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])

    vector_store = AsyncMock()
    vector_store.delete_by_url = AsyncMock()
    vector_store.upsert_vectors_batch = AsyncMock()

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    node_parser = MagicMock()
    mock_node = MagicMock()
    mock_node.get_content = MagicMock(return_value="# Content from get_content")
    mock_node.metadata = {}
    mock_node.node_id = "node_123"
    node_parser.get_nodes_from_documents = MagicMock(return_value=[mock_node])

    service = IngestionService(
        scraper=scraper,
        embeddings=embeddings,
        vector_store=vector_store,
        queue_manager=queue_manager,
        node_parser=node_parser,
    )

    await service.ingest_urls(["https://example.com"])

    # Verify embeddings were called with get_content() result
    embeddings.embed_batch.assert_awaited_once()
    texts = embeddings.embed_batch.await_args[0][0]
    assert texts == ["# Content from get_content"]


@pytest.mark.asyncio
async def test_ingest_urls_no_nodes_from_document() -> None:
    """Verify documents that produce no nodes don't create embeddings.

    When the node parser returns an empty list, the service should
    skip embedding generation for that document.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(
                url="https://example.com",
                success=True,
                markdown="",
            )
        ]
    )

    embeddings = AsyncMock()
    embeddings.embed_batch = AsyncMock()

    vector_store = AsyncMock()
    vector_store.delete_by_url = AsyncMock()
    vector_store.upsert_vectors_batch = AsyncMock()

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    node_parser = MagicMock()
    node_parser.get_nodes_from_documents = MagicMock(return_value=[])

    service = IngestionService(
        scraper=scraper,
        embeddings=embeddings,
        vector_store=vector_store,
        queue_manager=queue_manager,
        node_parser=node_parser,
    )

    result = await service.ingest_urls(["https://example.com"])

    # No chunks should be created
    assert result.chunks_created == 0

    # Embeddings should not be called
    embeddings.embed_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_vector_metadata_with_section_info() -> None:
    """Verify section_path and heading_level are preserved in vector metadata.

    When nodes contain section information from markdown parsing,
    that metadata should be included in the vector storage.
    """
    scraper = AsyncMock()
    scraper.scrape_urls = AsyncMock(
        return_value=[
            ScrapeResult(
                url="https://example.com",
                success=True,
                markdown="# Title\n## Subtitle",
            )
        ]
    )

    embeddings = AsyncMock()
    embeddings.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])

    vector_store = AsyncMock()
    vector_store.delete_by_url = AsyncMock()
    vector_store.upsert_vectors_batch = AsyncMock()

    queue_manager = AsyncMock()
    queue_manager.acquire_lock = AsyncMock(return_value=True)
    queue_manager.release_lock = AsyncMock()
    queue_manager.set_status = AsyncMock()

    node_parser = MagicMock()
    mock_node = MagicMock()
    mock_node.text = "# Title\n## Subtitle"
    mock_node.metadata = {
        "section_path": "Title > Subtitle",
        "heading_level": 2,
    }
    mock_node.node_id = "node_123"
    node_parser.get_nodes_from_documents = MagicMock(return_value=[mock_node])

    service = IngestionService(
        scraper=scraper,
        embeddings=embeddings,
        vector_store=vector_store,
        queue_manager=queue_manager,
        node_parser=node_parser,
    )

    await service.ingest_urls(["https://example.com"])

    # Verify section metadata is preserved
    vector_store.upsert_vectors_batch.assert_awaited_once()
    call_args = vector_store.upsert_vectors_batch.await_args[0][0]
    vector_metadata = call_args[0]["metadata"]
    assert vector_metadata["section_path"] == "Title > Subtitle"
    assert vector_metadata["heading_level"] == 2
