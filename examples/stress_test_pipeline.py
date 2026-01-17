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
    # Stress test with defaults (depth=2, max 100 URLs, 50 concurrent, 128 batch)
    python examples/stress_test_pipeline.py

    # Custom configuration
    python examples/stress_test_pipeline.py --depth 3 --max-urls 500 --concurrency 80 --batch-size 128

    # Monitor GPU during test
    watch -n 1 nvidia-smi
"""

import argparse
import asyncio
import signal
import time
import uuid
from collections import deque
from typing import Any
from urllib.parse import urljoin, urlparse

import psutil

from crawl4r.core.logger import get_logger
from crawl4r.processing.chunker import MarkdownChunker
from crawl4r.readers.crawl4ai import Crawl4AIReader
from crawl4r.storage.tei import TEIClient
from crawl4r.storage.qdrant import VectorStoreManager
from examples.monitor_resources import ResourceMonitor

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
        """Recursively crawl starting from seed URLs with concurrent processing.

        Returns:
            List of (url, markdown_content) tuples
        """
        # Initialize queue with seed URLs at depth 0
        for url in seed_urls:
            self.queue.append((url, 0))
            self.visited.add(url)

        results = []
        # Process URLs in concurrent batches for better throughput
        batch_size = 20  # Process 20 URLs concurrently (optimal)

        async def crawl_single_url(url: str, depth: int) -> tuple[str, str] | None:
            """Crawl a single URL and return (url, content) or None."""
            if depth > self.max_depth:
                return None

            documents = await self.reader.aload_data([url])
            if documents[0] is None:
                logger.warning(f"Failed to crawl {url}")
                return None

            doc = documents[0]
            logger.info(
                f"Crawled [{len(results) + 1}/{self.max_urls}] depth={depth}: {url}"
            )
            return (url, doc.text)

        while self.queue and len(results) < self.max_urls:
            if shutdown_requested:
                logger.warning("Shutdown requested, stopping crawl")
                break

            # Collect batch of URLs to process concurrently
            batch = []
            while self.queue and len(batch) < batch_size and len(results) + len(batch) < self.max_urls:
                current_url, depth = self.queue.popleft()
                batch.append((current_url, depth))

            # Process batch concurrently
            tasks = [crawl_single_url(url, depth) for url, depth in batch]
            crawl_results = await asyncio.gather(*tasks)

            # Process results and extract links from each successful crawl
            for result, (batch_url, batch_depth) in zip(crawl_results, batch):
                if result is None:
                    continue

                current_url, doc_text = result
                results.append((current_url, doc_text))

                # Extract links for next depth level (if not at max depth)
                if batch_depth < self.max_depth:
                    # For markdown content, we need to extract links from it
                    # Markdown links: [text](url)
                    import re

                    md_pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
                    md_matches = re.findall(md_pattern, doc_text)
                    links = [match[1] for match in md_matches]

                    # Filter and add new links to queue at next depth level
                    excluded_extensions = [
                        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
                        ".css", ".js", ".zip", ".tar", ".gz", ".bz2",
                        ".woff", ".woff2", ".ttf", ".eot", ".otf",
                        ".mp3", ".mp4", ".avi", ".mov", ".webm",
                        ".xml", ".json", ".csv"
                    ]

                    for link in links:
                        absolute_url = urljoin(current_url, link)

                        # Skip if already visited
                        if absolute_url in self.visited:
                            continue

                        # Skip non-HTTP(S) schemes
                        parsed = urlparse(absolute_url)
                        if parsed.scheme not in ("http", "https"):
                            continue

                        # Skip excluded file types
                        if any(ext in absolute_url.lower() for ext in excluded_extensions):
                            continue

                        # Skip if different domain (if same_domain_only enabled)
                        if self.same_domain_only:
                            base_domain = urlparse(current_url).netloc
                            if parsed.netloc != base_domain:
                                continue

                        self.visited.add(absolute_url)
                        self.queue.append((absolute_url, batch_depth + 1))

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
    embedding_batch_size: int = 128,
    parallel_batches: int = 2,
    collection_name: str = "stress_test",
    embed_only: bool = False,
    gpu_host: str | None = None,
) -> None:
    """Run full pipeline stress test.

    Args:
        seed_urls: Starting URLs for recursive crawl
        max_depth: Maximum link depth to follow
        max_urls: Maximum total URLs to crawl
        crawl_concurrency: Concurrent crawl requests
        embedding_batch_size: Batch size for embeddings
        parallel_batches: Number of concurrent embedding batches
        collection_name: Qdrant collection name
        embed_only: Skip vector storage (embedding testing only)
        gpu_host: SSH host for GPU monitoring (e.g., 'steamy-wsl')
    """
    stats = StressTestStats()

    # Initialize resource monitor
    monitor = ResourceMonitor(gpu_host=gpu_host, interval=2.0)
    await monitor.start()

    logger.info("=" * 80)
    logger.info("STRESS TEST: Full RAG Pipeline")
    logger.info("=" * 80)
    logger.info(f"Seed URLs:          {seed_urls}")
    logger.info(f"Max depth:          {max_depth}")
    logger.info(f"Max URLs:           {max_urls}")
    logger.info(f"Crawl concurrency:  {crawl_concurrency}")
    logger.info(f"Embedding batch:    {embedding_batch_size}")
    logger.info(f"Parallel batches:   {parallel_batches}")
    logger.info(f"Total per group:    {embedding_batch_size * parallel_batches}")
    logger.info(f"Collection:         {collection_name}")
    logger.info(f"Embed only:         {embed_only}")
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

    tei_client = TEIClient(
        endpoint_url="http://100.74.16.82:52000",  # RTX 4070 GPU machine
        batch_size_limit=embedding_batch_size,
    )

    vector_store = None
    if not embed_only:
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

    phase1_start = time.time()
    crawler = RecursiveCrawler(
        reader=reader,
        max_depth=max_depth,
        max_urls=max_urls,
        same_domain_only=True,
    )

    crawled_docs = await crawler.crawl_recursive(seed_urls)
    stats.urls_crawled = len(crawled_docs)
    phase1_elapsed = time.time() - phase1_start

    logger.info(f"✓ Crawled {len(crawled_docs)} URLs in {phase1_elapsed:.1f}s ({len(crawled_docs)/phase1_elapsed:.2f} URLs/s)")
    logger.info("")

    # Phase 2: Chunking
    logger.info("=" * 80)
    logger.info("PHASE 2: Document Chunking")
    logger.info("=" * 80)

    phase2_start = time.time()
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

    phase2_elapsed = time.time() - phase2_start
    logger.info(
        f"✓ Created {len(all_chunks)} chunks from {len(crawled_docs)} documents in {phase2_elapsed:.1f}s ({len(all_chunks)/phase2_elapsed:.1f} chunks/s)"
    )
    logger.info("")

    # Phase 3: Embedding generation (parallel batches for GPU saturation)
    logger.info("=" * 80)
    logger.info("PHASE 3: Embedding Generation (GPU Maximized)")
    logger.info("=" * 80)

    phase3_start = time.time()
    # Process batches with configurable parallelism
    max_concurrent_batches = parallel_batches

    async def process_embedding_batch(batch: list[str], batch_num: int) -> list[list[float]]:
        """Process a single embedding batch."""
        batch_start = time.time()
        vectors = await tei_client.embed_batch(batch)
        batch_elapsed = time.time() - batch_start
        embeddings_per_sec = len(vectors) / batch_elapsed if batch_elapsed > 0 else 0

        logger.info(
            f"Batch {batch_num}: "
            f"{len(vectors)} embeddings in {batch_elapsed:.2f}s "
            f"({embeddings_per_sec:.1f} emb/s)"
        )
        return vectors

    # Split chunks and metadata into batches
    batches = [
        all_chunks[i : i + embedding_batch_size]
        for i in range(0, len(all_chunks), embedding_batch_size)
    ]
    metadata_batches = [
        all_chunk_metadata[i : i + embedding_batch_size]
        for i in range(0, len(all_chunk_metadata), embedding_batch_size)
    ]

    # Process batches in concurrent groups
    for group_start in range(0, len(batches), max_concurrent_batches):
        if shutdown_requested:
            break

        group = batches[group_start : group_start + max_concurrent_batches]
        meta_group = metadata_batches[group_start : group_start + max_concurrent_batches]
        batch_nums = range(group_start + 1, group_start + len(group) + 1)

        # Execute concurrent batches
        tasks = [process_embedding_batch(batch, num) for batch, num in zip(group, batch_nums)]
        results = await asyncio.gather(*tasks)

        # Collect results and optionally store immediately to avoid extra memory
        for vectors, batch_metadata in zip(results, meta_group):
            stats.embeddings_generated += len(vectors)

            if embed_only:
                continue

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=metadata,
                )
                for vector, metadata in zip(vectors, batch_metadata)
            ]

            vector_store.client.upsert(
                collection_name=collection_name,
                points=points,
            )

            stats.vectors_stored += len(vectors)

        if stats.should_log():
            stats.log_progress()

    phase3_elapsed = time.time() - phase3_start
    logger.info(
        f"✓ Generated {stats.embeddings_generated} embeddings in "
        f"{phase3_elapsed:.1f}s ({stats.embeddings_generated/phase3_elapsed:.1f} emb/s avg)"
    )
    logger.info("")

    if not embed_only:
        # Phase 4: Vector storage already completed during embedding
        logger.info("=" * 80)
        logger.info("PHASE 4: Vector Storage (Qdrant)")
        logger.info("=" * 80)
        logger.info(
            f"✓ Stored {stats.vectors_stored} vectors in Qdrant "
            f"(storage overlapped with embedding)"
        )
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
    logger.info("")
    logger.info("PHASE BREAKDOWN:")
    logger.info("-" * 80)
    logger.info(f"Phase 1 (Crawl):    {phase1_elapsed:6.1f}s ({phase1_elapsed/elapsed*100:5.1f}%) - {len(crawled_docs)/phase1_elapsed:5.2f} URLs/s")
    logger.info(f"Phase 2 (Chunk):    {phase2_elapsed:6.1f}s ({phase2_elapsed/elapsed*100:5.1f}%) - {len(all_chunks)/phase2_elapsed:5.1f} chunks/s")
    logger.info(
        f"Phase 3 (Embed):    {phase3_elapsed:6.1f}s "
        f"({phase3_elapsed/elapsed*100:5.1f}%) - "
        f"{stats.embeddings_generated/phase3_elapsed:5.1f} emb/s"
    )
    if not embed_only:
        logger.info("Phase 4 (Store):    overlapped with embedding")
    logger.info("-" * 80)
    logger.info(f"Total:              {elapsed:6.1f}s (100.0%)")
    logger.info("=" * 80)

    # Stop monitoring and log resource summary
    await monitor.stop()
    resource_summary = monitor.get_summary()

    if resource_summary:
        logger.info("")
        logger.info("=" * 80)
        logger.info("RESOURCE USAGE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"CPU:        Avg {resource_summary['cpu_avg']:.1f}% | Max {resource_summary['cpu_max']:.1f}%")
        logger.info(f"Memory:     Avg {resource_summary['memory_avg_mb']:.0f}MB | Max {resource_summary['memory_max_mb']:.0f}MB")

        if "gpu_util_avg" in resource_summary:
            logger.info(f"GPU Util:   Avg {resource_summary['gpu_util_avg']:.1f}% | Max {resource_summary['gpu_util_max']:.1f}%")
            logger.info(f"GPU Memory: Avg {resource_summary['gpu_memory_avg_mb']:.0f}MB | Max {resource_summary['gpu_memory_max_mb']:.0f}MB")
            logger.info(f"GPU Temp:   Avg {resource_summary['gpu_temp_avg']:.1f}°C | Max {resource_summary['gpu_temp_max']:.1f}°C")
        logger.info("=" * 80)


def signal_handler(_signum: int, _frame: Any) -> None:
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
        default=30,
        help="Concurrent crawl requests (default: 30, optimal for Wikipedia)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size (default: 128, max: TEI_MAX_CLIENT_BATCH_SIZE)",
    )
    parser.add_argument(
        "--parallel-batches",
        type=int,
        default=2,
        help="Number of concurrent embedding batches (default: 2, reduce if OOM)",
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="Generate embeddings only (skip Qdrant storage for max GPU throughput)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="stress_test",
        help="Qdrant collection name (default: stress_test)",
    )
    parser.add_argument(
        "--gpu-host",
        type=str,
        default=None,
        help="SSH host for GPU monitoring (e.g., 'steamy-wsl')",
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
            parallel_batches=args.parallel_batches,
            collection_name=args.collection,
            embed_only=args.embed_only,
            gpu_host=args.gpu_host,
        )
    except Exception as e:
        logger.error(f"Stress test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
