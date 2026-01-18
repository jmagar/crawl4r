"""Usage examples for Crawl4AIReader.

This file demonstrates various usage patterns for the Crawl4AIReader component,
including basic crawling, batch processing, error handling, deduplication, and
full pipeline integration.

Prerequisites:
    - Crawl4AI service running on http://localhost:52004
    - Qdrant running on http://localhost:52001 (for deduplication examples)
    - TEI running on http://localhost:52000 (for pipeline examples)

Run examples:
    python examples/crawl4ai_reader_usage.py
"""

import asyncio

from crawl4r.core.logger import get_logger
from crawl4r.readers.crawl4ai import Crawl4AIReader

logger = get_logger(__name__)


# =============================================================================
# Example 1: Basic Single URL Crawling
# =============================================================================


async def example_basic_single_url() -> None:
    """Example 1: Crawl a single URL with default configuration."""
    logger.info("=" * 80)
    logger.info("Example 1: Basic Single URL Crawling")
    logger.info("=" * 80)

    # Create reader with default configuration
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Crawl single URL (async)
    documents = await reader.aload_data(["https://example.com"])

    # Verify document created
    assert len(documents) == 1
    assert documents[0] is not None
    doc = documents[0]

    logger.info(f"Crawled URL: {doc.metadata['source']}")
    logger.info(f"Document ID: {doc.id_}")
    logger.info(f"Title: {doc.metadata['title']}")
    logger.info(f"Content length: {len(doc.text)} characters")
    logger.info(f"Status code: {doc.metadata['status_code']}")
    logger.info("")


# =============================================================================
# Example 2: Synchronous Wrapper
# =============================================================================


async def example_synchronous_wrapper() -> None:
    """Example 2: Use synchronous load_data wrapper for non-async code.

    Note: This example demonstrates the sync wrapper, but since we're already
    in an async context, we use aload_data directly. In a non-async script,
    you would use reader.load_data(urls) without await.
    """
    logger.info("=" * 80)
    logger.info("Example 2: Synchronous Wrapper")
    logger.info("=" * 80)

    # Create reader
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # In non-async code, you would use: reader.load_data(["https://example.org"])
    # But since we're in async context, use aload_data directly
    documents = await reader.aload_data(["https://example.org"])

    # Process documents
    assert len(documents) == 1
    assert documents[0] is not None

    logger.info(f"Crawled: {documents[0].metadata['source']}")
    logger.info(f"Title: {documents[0].metadata['title']}")
    logger.info("\nNote: In non-async code, use reader.load_data(urls) without await")
    logger.info("")


# =============================================================================
# Example 3: Batch Crawling Multiple URLs
# =============================================================================


async def example_batch_crawling() -> None:
    """Example 3: Crawl multiple URLs concurrently with batch processing."""
    logger.info("=" * 80)
    logger.info("Example 3: Batch Crawling Multiple URLs")
    logger.info("=" * 80)

    # Create reader with custom concurrency limit
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        max_concurrent_requests=3,  # Process 3 URLs at a time
    )

    # Crawl multiple URLs
    urls = [
        "https://example.com",
        "https://example.org",
        "https://example.net",
    ]

    documents = await reader.aload_data(urls)

    # Verify all documents created
    logger.info(f"Crawled {len(documents)} URLs")
    for i, doc in enumerate(documents):
        if doc is not None:
            logger.info(
                f"  [{i+1}] {doc.metadata['source']} - {doc.metadata['title']}"
            )
        else:
            logger.warning(f"  [{i+1}] {urls[i]} - FAILED")
    logger.info("")


# =============================================================================
# Example 4: Custom Configuration
# =============================================================================


async def example_custom_configuration() -> None:
    """Example 4: Use custom configuration for advanced settings."""
    logger.info("=" * 80)
    logger.info("Example 4: Custom Configuration")
    logger.info("=" * 80)

    # Create reader with custom configuration
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        max_concurrent_requests=10,
        max_retries=5,  # Retry up to 5 times
        timeout_seconds=120,  # 2 minute timeout
        fail_on_error=False,  # Return None instead of raising
    )

    # Crawl URL
    documents = await reader.aload_data(["https://example.com"])

    logger.info("Configuration:")
    logger.info(f"  Max concurrent requests: {reader.max_concurrent_requests}")
    logger.info(f"  Max retries: {reader.max_retries}")
    logger.info(f"  Timeout: {reader.timeout_seconds}s")
    logger.info(f"  Fail on error: {reader.fail_on_error}")
    source = documents[0].metadata["source"] if documents[0] else "FAILED"
    logger.info(f"Crawled: {source}")
    logger.info("")


# =============================================================================
# Example 5: Error Handling
# =============================================================================


async def example_error_handling() -> None:
    """Example 5: Demonstrate error handling with invalid URLs."""
    logger.info("=" * 80)
    logger.info("Example 5: Error Handling")
    logger.info("=" * 80)

    # Create reader with graceful error handling
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        fail_on_error=False,  # Don't raise exceptions
        max_retries=1,  # Minimize retries for demo
    )

    # Mix of valid and invalid URLs
    urls = [
        "https://example.com",  # Valid
        "http://invalid-domain-12345.example",  # Invalid
        "https://example.org",  # Valid
    ]

    documents = await reader.aload_data(urls)

    # Check results
    logger.info("Results:")
    for i, (url, doc) in enumerate(zip(urls, documents)):
        if doc is not None:
            logger.info(
                f"  [{i+1}] ✓ {url} - SUCCESS ({len(doc.text)} chars)"
            )
        else:
            logger.warning(f"  [{i+1}] ✗ {url} - FAILED")

    # Count successes
    successful = [doc for doc in documents if doc is not None]
    logger.info(f"\nSuccess rate: {len(successful)}/{len(urls)}")
    logger.info("")


# =============================================================================
# Example 6: Metadata Inspection
# =============================================================================


async def example_metadata_inspection() -> None:
    """Example 6: Inspect document metadata structure."""
    logger.info("=" * 80)
    logger.info("Example 6: Metadata Inspection")
    logger.info("=" * 80)

    # Create reader
    reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

    # Crawl URL
    documents = await reader.aload_data(["https://example.com"])
    assert documents[0] is not None
    doc = documents[0]

    # Display metadata
    logger.info("Document Metadata:")
    for key, value in doc.metadata.items():
        logger.info(f"  {key}: {value}")

    logger.info(f"\nDocument ID: {doc.id_}")
    logger.info(f"Content preview: {doc.text[:200]}...")
    logger.info("")


# =============================================================================
# Example 7: Deduplication Integration
# =============================================================================


async def example_deduplication() -> None:
    """Example 7: Automatic deduplication with VectorStoreManager."""
    logger.info("=" * 80)
    logger.info("Example 7: Deduplication Integration")
    logger.info("=" * 80)

    try:
        from crawl4r.storage.qdrant import VectorStoreManager

        # Setup vector store (requires Qdrant running)
        try:
            vector_store = VectorStoreManager(
                collection_name="web_crawl_demo",
                qdrant_url="http://localhost:52001",
            )

            # Ensure collection exists (create if needed)
            # This will handle the case where collection doesn't exist yet
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            client = QdrantClient(url="http://localhost:52001")
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if "web_crawl_demo" not in collection_names:
                logger.info("Creating collection 'web_crawl_demo'...")
                client.create_collection(
                    collection_name="web_crawl_demo",
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                )

            # Create reader with deduplication enabled
            reader = Crawl4AIReader(
                endpoint_url="http://localhost:52004",
                vector_store=vector_store,
                enable_deduplication=True,  # Auto-delete before crawling
            )

            # First crawl
            logger.info("First crawl (creating new data):")
            documents1 = await reader.aload_data(["https://example.com"])
            source1 = documents1[0].metadata["source"] if documents1[0] else "FAILED"
            logger.info(f"  Created document: {source1}")

            # Second crawl (will delete old data first)
            logger.info("\nSecond crawl (with deduplication):")
            logger.info("  Automatically deleting old data for URL...")
            documents2 = await reader.aload_data(["https://example.com"])
            source2 = documents2[0].metadata["source"] if documents2[0] else "FAILED"
            logger.info(f"  Re-created document: {source2}")

            logger.info("\nDeduplication ensures no duplicate vectors in Qdrant")
            logger.info("")

        except Exception as e:
            logger.warning(f"Qdrant not available - skipping example: {e}")
            logger.info("")

    except ImportError:
        logger.warning("VectorStoreManager not available - skipping example")
        logger.info("")


# =============================================================================
# Example 8: Pipeline Integration
# =============================================================================


async def example_pipeline_integration() -> None:
    """Example 8: Full pipeline integration with chunking."""
    logger.info("=" * 80)
    logger.info("Example 8: Pipeline Integration")
    logger.info("=" * 80)

    try:
        from llama_index.core.node_parser import MarkdownNodeParser
        from llama_index.core.schema import Document

        # Initialize components
        reader = Crawl4AIReader(endpoint_url="http://localhost:52004")
        node_parser = MarkdownNodeParser()

        # Crawl URL
        logger.info("Step 1: Crawling URL...")
        documents = await reader.aload_data(["https://example.com"])
        assert documents[0] is not None
        crawled_doc = documents[0]
        logger.info(f"  Crawled: {crawled_doc.metadata['source']}")
        logger.info(f"  Content: {len(crawled_doc.text)} characters")

        # Chunk document using MarkdownNodeParser
        logger.info("\nStep 2: Chunking document...")
        llama_doc = Document(
            text=crawled_doc.text,
            metadata={"source_url": crawled_doc.metadata["source_url"]}
        )
        nodes = node_parser.get_nodes_from_documents([llama_doc])
        logger.info(f"  Created: {len(nodes)} chunks")

        # Display first chunk
        if nodes:
            logger.info("\nFirst chunk preview:")
            logger.info(f"  Index: 0")
            logger.info(f"  Section: {nodes[0].metadata.get('header_path', 'N/A')}")
            logger.info(f"  Length: {len(nodes[0].get_content())} characters")
            logger.info(
                f"  Content: {nodes[0].get_content()[:100]}..."
            )

        logger.info(
            "\nNext steps: Generate embeddings and store in Qdrant"
        )
        logger.info("")

    except ImportError:
        logger.warning("MarkdownNodeParser not available - skipping example")
        logger.info("")


# =============================================================================
# Example 9: Circuit Breaker Behavior
# =============================================================================


async def example_circuit_breaker() -> None:
    """Example 9: Demonstrate circuit breaker with consecutive failures."""
    logger.info("=" * 80)
    logger.info("Example 9: Circuit Breaker Behavior")
    logger.info("=" * 80)

    # Create reader with fail_on_error=False to prevent early exit
    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        fail_on_error=False,
        max_retries=0,  # No retries for faster demo
    )

    # Trigger failures with invalid URLs (circuit breaker threshold is 5)
    logger.info("Triggering consecutive failures to open circuit breaker...")
    invalid_urls = [f"http://invalid-domain-{i}.example" for i in range(6)]
    results = await reader.aload_data(invalid_urls)

    # Check circuit breaker state
    logger.info(f"\nCircuit breaker state: {reader._circuit_breaker.state}")
    logger.info(
        f"Failure count: {reader._circuit_breaker.failure_count}"
    )
    logger.info(
        f"All requests failed: {all(r is None for r in results)}"
    )

    logger.info(
        "\nCircuit opened after 5 failures - prevents cascade"
    )
    logger.info("")


# =============================================================================
# Main: Run All Examples
# =============================================================================


async def main() -> None:
    """Run all usage examples."""
    logger.info("\n")
    logger.info("=" * 80)
    logger.info("Crawl4AIReader Usage Examples")
    logger.info("=" * 80)
    logger.info("")

    # Check if Crawl4AI service is available
    try:
        Crawl4AIReader(endpoint_url="http://localhost:52004")
        logger.info("✓ Crawl4AI service is available\n")
    except ValueError as e:
        logger.error(f"✗ Crawl4AI service unavailable: {e}")
        logger.error(
            "Please start the service: docker compose up -d crawl4ai"
        )
        return

    # Run examples
    await example_basic_single_url()
    await example_synchronous_wrapper()
    await example_batch_crawling()
    await example_custom_configuration()
    await example_error_handling()
    await example_metadata_inspection()
    await example_deduplication()
    await example_pipeline_integration()
    await example_circuit_breaker()

    logger.info("=" * 80)
    logger.info("All examples completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
