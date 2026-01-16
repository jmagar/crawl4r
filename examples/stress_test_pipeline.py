"""Stress test for full RAG pipeline with recursive crawling.

This script stress tests the complete pipeline by recursively crawling websites
and processing them through: Crawl → Chunk → Embed → Store.

Features:
- Recursive link following with depth control
- GPU-maximized concurrent processing
- Real-time performance monitoring
- Graceful shutdown on Ctrl+C
- Memory usage tracking

Prerequisites:
    - All services running (Crawl4AI, TEI, Qdrant)
    - GPU with sufficient memory
    - Sufficient system RAM (recommend 16GB+)

Usage:
    # Stress test with defaults (depth=2, max 100 URLs)
    python examples/stress_test_pipeline.py

    # Custom configuration
    python examples/stress_test_pipeline.py --depth 3 --max-urls 500 --concurrency 20

    # Monitor GPU during test
    watch -n 1 nvidia-smi
"""

import argparse
import asyncio
import signal
import time
from collections import deque
from typing import Any
from urllib.parse import urljoin, urlparse

import psutil

from rag_ingestion.chunker import MarkdownChunker
from rag_ingestion.crawl4ai_reader import Crawl4AIReader
from rag_ingestion.logger import get_logger
from rag_ingestion.tei_client import TEIClient
from rag_ingestion.vector_store import VectorStoreManager

logger = get_logger(__name__)

# Global shutdown flag
shutdown_requested = False


class StressTestStats:
    """Track performance statistics during stress test."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self.urls_crawled = 0
        self.urls_failed = 0
        self.documents_processed = 0
        self.chunks_created = 0
        self.embeddings_generated = 0
        self.vectors_stored = 0
        self.total_content_chars = 0
        self.peak_memory_mb = 0.0
        self.last_log_time = time.time()

    def log_progress(self) -> None:
        """Log current progress statistics."""
        elapsed = time.time() - self.start_time
        if elapsed < 1:
            return

        # Calculate rates
        urls_per_sec = self.urls_crawled / elapsed
        docs_per_sec = self.documents_processed / elapsed
        chunks_per_sec = self.chunks_created / elapsed
        embeddings_per_sec = self.embeddings_generated / elapsed

        # Memory usage
        process = psutil.Process()
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, current_memory_mb)

        logger.info("=" * 80)
        logger.info(f"STRESS TEST PROGRESS - Elapsed: {elapsed:.1f}s")
        logger.info("-" * 80)
        logger.info(
            f"URLs:       {self.urls_crawled} crawled, {self.urls_failed} failed"
        )
        logger.info(f"Documents:  {self.documents_processed} processed")
        logger.info(f"Chunks:     {self.chunks_created} created")
        logger.info(f"Embeddings: {self.embeddings_generated} generated")
        logger.info(f"Vectors:    {self.vectors_stored} stored in Qdrant")
        logger.info(f"Content:    {self.total_content_chars:,} total characters")
        logger.info("-" * 80)
        logger.info(f"Throughput: {urls_per_sec:.2f} URLs/s, {docs_per_sec:.2f} docs/s")
        logger.info(
            f"            {chunks_per_sec:.2f} chunks/s, "
            f"{embeddings_per_sec:.2f} emb/s"
        )
        logger.info("-" * 80)
        logger.info(
            f"Memory:     {current_memory_mb:.1f} MB current, "
            f"{self.peak_memory_mb:.1f} MB peak"
        )
        logger.info("=" * 80)

        self.last_log_time = time.time()

    def should_log(self, interval: float = 10.0) -> bool:
        """Check if enough time has passed to log progress."""
        return (time.time() - self.last_log_time) >= interval


class RecursiveCrawler:
    """Recursively crawl websites with depth control and link extraction."""

    def __init__(
        self,
        reader: Crawl4AIReader,
        max_depth: int = 2,
        max_urls: int = 100,
        same_domain_only: bool = True,
    ) -> None:
        self.reader = reader
        self.max_depth = max_depth
        self.max_urls = max_urls
        self.same_domain_only = same_domain_only
        self.visited: set[str] = set()
        self.queue: deque = deque()

    def extract_links(self, html_content: str, base_url: str) -> list[str]:
        """Extract links from HTML content.

        Note: This is a simplified extraction. In production, use BeautifulSoup
        or lxml for robust HTML parsing.
        """
        import re

        # Simple regex to find href attributes
        # Format: href="url" or href='url'
        pattern = r'href=["\']([^"\']+)["\']'
        matches = re.findall(pattern, html_content)

        links = []
        base_domain = urlparse(base_url).netloc

        for match in matches:
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, match)
            parsed = urlparse(absolute_url)

            # Filter: must be http/https, not visited, respect domain filter
            if parsed.scheme not in ("http", "https"):
                continue

            if absolute_url in self.visited:
                continue

            if self.same_domain_only and parsed.netloc != base_domain:
                continue

            # Avoid common non-content URLs
            if any(
                ext in absolute_url.lower()
                for ext in [".pdf", ".jpg", ".png", ".gif", ".css", ".js", ".zip"]
            ):
                continue

            links.append(absolute_url)

        return links

    async def crawl_recursive(self, seed_urls: list[str]) -> list[tuple[str, str]]:
        """Recursively crawl starting from seed URLs.

        Returns:
            List of (url, markdown_content) tuples
        """
        # Initialize queue with seed URLs at depth 0
        for url in seed_urls:
            self.queue.append((url, 0))
            self.visited.add(url)

        results = []

        while self.queue and len(results) < self.max_urls:
            if shutdown_requested:
                logger.warning("Shutdown requested, stopping crawl")
                break

            # Get next URL and its depth
            current_url, depth = self.queue.popleft()

            # Skip if we've hit max depth
            if depth > self.max_depth:
                continue

            # Crawl current URL
            documents = await self.reader.aload_data([current_url])

            if documents[0] is None:
                logger.warning(f"Failed to crawl {current_url}")
                continue

            doc = documents[0]
            results.append((current_url, doc.text))

            logger.info(
                f"Crawled [{len(results)}/{self.max_urls}] depth={depth}: {current_url}"
            )

            # Extract links for next depth level (if not at max depth)
            if depth < self.max_depth:
                # For markdown content, we need to extract links from it
                # Markdown links: [text](url)
                import re

                md_pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
                md_matches = re.findall(md_pattern, doc.text)
                links = [match[1] for match in md_matches]

                # Add new links to queue at next depth level
                for link in links:
                    absolute_url = urljoin(current_url, link)
                    if absolute_url not in self.visited:
                        self.visited.add(absolute_url)
                        self.queue.append((absolute_url, depth + 1))

                logger.debug(
                    f"Found {len(links)} links from {current_url}, "
                    f"queue size: {len(self.queue)}"
                )

        return results


async def stress_test_pipeline(
    seed_urls: list[str],
    max_depth: int = 2,
    max_urls: int = 100,
    crawl_concurrency: int = 10,
    embedding_batch_size: int = 32,
    collection_name: str = "stress_test",
) -> None:
    """Run full pipeline stress test.

    Args:
        seed_urls: Starting URLs for recursive crawl
        max_depth: Maximum link depth to follow
        max_urls: Maximum total URLs to crawl
        crawl_concurrency: Concurrent crawl requests
        embedding_batch_size: Batch size for embeddings
        collection_name: Qdrant collection name
    """
    stats = StressTestStats()

    logger.info("=" * 80)
    logger.info("STRESS TEST: Full RAG Pipeline")
    logger.info("=" * 80)
    logger.info(f"Seed URLs:          {seed_urls}")
    logger.info(f"Max depth:          {max_depth}")
    logger.info(f"Max URLs:           {max_urls}")
    logger.info(f"Crawl concurrency:  {crawl_concurrency}")
    logger.info(f"Embedding batch:    {embedding_batch_size}")
    logger.info(f"Collection:         {collection_name}")
    logger.info("=" * 80)

    # Initialize components
    logger.info("Initializing pipeline components...")

    reader = Crawl4AIReader(
        endpoint_url="http://localhost:52004",
        max_concurrent_requests=crawl_concurrency,
        fail_on_error=False,
    )

    chunker = MarkdownChunker(
        chunk_size_tokens=512,
        chunk_overlap_percent=15,
    )

    tei_client = TEIClient(endpoint_url="http://localhost:52000")

    vector_store = VectorStoreManager(
        collection_name=collection_name,
        qdrant_url="http://localhost:52001",
    )

    # Ensure collection exists
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(url="http://localhost:52001")
    try:
        client.get_collection(collection_name)
        logger.info(f"Using existing collection: {collection_name}")
    except Exception:
        logger.info(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    logger.info("✓ All components initialized")
    logger.info("")

    # Phase 1: Recursive crawling
    logger.info("=" * 80)
    logger.info("PHASE 1: Recursive Crawling")
    logger.info("=" * 80)

    crawler = RecursiveCrawler(
        reader=reader,
        max_depth=max_depth,
        max_urls=max_urls,
        same_domain_only=True,
    )

    crawled_docs = await crawler.crawl_recursive(seed_urls)
    stats.urls_crawled = len(crawled_docs)

    logger.info(f"✓ Crawled {len(crawled_docs)} URLs")
    logger.info("")

    # Phase 2: Chunking
    logger.info("=" * 80)
    logger.info("PHASE 2: Document Chunking")
    logger.info("=" * 80)

    all_chunks = []
    all_chunk_metadata = []

    for url, content in crawled_docs:
        if shutdown_requested:
            break

        stats.total_content_chars += len(content)

        chunks = chunker.chunk(content, filename=url)
        stats.chunks_created += len(chunks)
        stats.documents_processed += 1

        # Prepare chunks for embedding
        for chunk in chunks:
            all_chunks.append(chunk["chunk_text"])
            all_chunk_metadata.append(
                {
                    "source_url": url,
                    "chunk_index": chunk["chunk_index"],
                    "section_path": chunk["section_path"],
                    "heading_level": chunk["heading_level"],
                }
            )

        if stats.should_log():
            stats.log_progress()

    logger.info(
        f"✓ Created {len(all_chunks)} chunks from {len(crawled_docs)} documents"
    )
    logger.info("")

    # Phase 3: Embedding generation (batched for GPU efficiency)
    logger.info("=" * 80)
    logger.info("PHASE 3: Embedding Generation (GPU Maximized)")
    logger.info("=" * 80)

    all_vectors = []

    for i in range(0, len(all_chunks), embedding_batch_size):
        if shutdown_requested:
            break

        batch = all_chunks[i : i + embedding_batch_size]
        batch_start = time.time()

        vectors = await tei_client.embed_batch(batch)
        all_vectors.extend(vectors)

        stats.embeddings_generated += len(vectors)
        batch_elapsed = time.time() - batch_start
        embeddings_per_sec = len(vectors) / batch_elapsed if batch_elapsed > 0 else 0

        logger.info(
            f"Batch {i//embedding_batch_size + 1}: "
            f"{len(vectors)} embeddings in {batch_elapsed:.2f}s "
            f"({embeddings_per_sec:.1f} emb/s)"
        )

        if stats.should_log():
            stats.log_progress()

    logger.info(f"✓ Generated {len(all_vectors)} embeddings")
    logger.info("")

    # Phase 4: Vector storage (batched for Qdrant efficiency)
    logger.info("=" * 80)
    logger.info("PHASE 4: Vector Storage (Qdrant)")
    logger.info("=" * 80)

    storage_batch_size = 100
    for i in range(0, len(all_vectors), storage_batch_size):
        if shutdown_requested:
            break

        batch_vectors = all_vectors[i : i + storage_batch_size]
        batch_metadata = all_chunk_metadata[i : i + storage_batch_size]

        # Note: VectorStoreManager expects specific metadata schema
        # For stress test, we'll store minimal metadata using PointStruct
        points = [
            PointStruct(
                id=f"stress_{i+j}_{int(time.time())}",
                vector=vector,
                payload=metadata,
            )
            for j, (vector, metadata) in enumerate(zip(batch_vectors, batch_metadata))
        ]

        vector_store.client.upsert(
            collection_name=collection_name,
            points=points,
        )

        stats.vectors_stored += len(batch_vectors)

        logger.info(
            f"Stored batch {i//storage_batch_size + 1}: "
            f"{len(batch_vectors)} vectors "
            f"({stats.vectors_stored}/{len(all_vectors)} total)"
        )

        if stats.should_log():
            stats.log_progress()

    logger.info(f"✓ Stored {stats.vectors_stored} vectors in Qdrant")
    logger.info("")

    # Final statistics
    stats.log_progress()

    elapsed = time.time() - stats.start_time
    logger.info("")
    logger.info("=" * 80)
    logger.info("STRESS TEST COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total time:     {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")
    logger.info(f"URLs crawled:   {stats.urls_crawled}")
    logger.info(f"Chunks created: {stats.chunks_created}")
    logger.info(f"Vectors stored: {stats.vectors_stored}")
    logger.info(f"Peak memory:    {stats.peak_memory_mb:.1f} MB")
    logger.info("=" * 80)


def signal_handler(signum: int, frame: Any) -> None:
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    logger.warning("\n\nShutdown requested (Ctrl+C). Finishing current batch...")
    shutdown_requested = True


async def main() -> None:
    """Main entry point for stress test."""
    parser = argparse.ArgumentParser(
        description="Stress test full RAG pipeline with recursive crawling"
    )
    parser.add_argument(
        "--seed-urls",
        nargs="+",
        default=["https://en.wikipedia.org/wiki/Python_(programming_language)"],
        help="Seed URLs to start crawling from",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Maximum depth for recursive crawling (default: 2)",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=100,
        help="Maximum total URLs to crawl (default: 100)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrent crawl requests (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="stress_test",
        help="Qdrant collection name (default: stress_test)",
    )

    args = parser.parse_args()

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    try:
        await stress_test_pipeline(
            seed_urls=args.seed_urls,
            max_depth=args.depth,
            max_urls=args.max_urls,
            crawl_concurrency=args.concurrency,
            embedding_batch_size=args.batch_size,
            collection_name=args.collection,
        )
    except Exception as e:
        logger.error(f"Stress test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
