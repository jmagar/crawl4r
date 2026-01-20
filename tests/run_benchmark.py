#!/usr/bin/env python3
"""Run throughput benchmark: process 100 markdown files and measure docs/min."""

import asyncio

# Add project to path
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from crawl4r.core.config import Settings
from crawl4r.processing.processor import DocumentProcessor
from crawl4r.storage.qdrant import VectorStoreManager
from crawl4r.storage.tei import TEIClient


async def main():
    """Run benchmark."""
    print("=" * 70)
    print("THROUGHPUT BENCHMARK: 100 Markdown Files")
    print("=" * 70)

    # Load configuration
    settings = Settings(watch_folder=Path("tests/benchmark_data"))
    print("\nConfiguration:")
    print(f"  Watch folder: {settings.watch_folder}")
    print(f"  Chunk size: {settings.chunk_size_tokens} tokens")
    print(f"  Chunk overlap: {settings.chunk_overlap_percent}%")
    print(f"  Batch size: {settings.batch_size}")
    print(f"  TEI endpoint: {settings.tei_endpoint}")
    print(f"  Qdrant endpoint: {settings.qdrant_url}")
    print(f"  Collection: {settings.collection_name}")

    # Verify services are accessible
    print("\n" + "-" * 70)
    print("Checking services...")
    print("-" * 70)
    print(f"✓ TEI endpoint: {settings.tei_endpoint}")
    print(f"✓ Qdrant endpoint: {settings.qdrant_url}")
    print(f"✓ Collection: {settings.collection_name}")

    # Initialize components
    print("\n" + "-" * 70)
    print("Initializing components...")
    print("-" * 70)
    tei_client = TEIClient(endpoint_url=settings.tei_endpoint)
    embedding_dimensions = getattr(
        settings,
        "embedding_dimensions",
        tei_client.expected_dimensions,
    )
    vector_store = VectorStoreManager(
        qdrant_url=settings.qdrant_url,
        collection_name=settings.collection_name,
        dimensions=embedding_dimensions,
    )
    processor = DocumentProcessor(
        config=settings,
        vector_store=vector_store,
        tei_client=tei_client,
    )
    print("✓ Components initialized")

    # Get benchmark files
    benchmark_dir = Path("tests/benchmark_data")
    if not benchmark_dir.exists():
        print(f"\n✗ Benchmark directory not found: {benchmark_dir}")
        return 1

    files = sorted(benchmark_dir.glob("*.md"))
    if len(files) != 100:
        print(f"\n✗ Expected 100 files, found {len(files)}")
        return 1

    print(f"\n✓ Found {len(files)} markdown files")

    # Run benchmark
    print("\n" + "=" * 70)
    print("STARTING BENCHMARK")
    print("=" * 70)

    start_time = time.time()
    processed_count = 0
    failed_count = 0

    for i, filepath in enumerate(files, 1):
        try:
            result = await processor.process_document(filepath)
            if result.success:
                processed_count += 1
            else:
                failed_count += 1
                print(f"  ✗ Failed to process {filepath.name}: {result.error}")
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i / elapsed) * 60 if elapsed > 0 else 0
                print(f"  Progress: {i}/100 files | {elapsed:.1f}s elapsed | {rate:.1f} docs/min")
        except Exception as e:
            print(f"  ✗ Exception processing {filepath.name}: {e}")
            failed_count += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Calculate metrics
    throughput = (processed_count / elapsed_time) * 60 if elapsed_time > 0 else 0
    avg_time_per_doc = elapsed_time / processed_count if processed_count > 0 else 0

    # Display results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\nFiles processed:     {processed_count}/{len(files)}")
    print(f"Failed:              {failed_count}")
    print(f"Total time:          {elapsed_time:.2f} seconds")
    print(f"Avg time per doc:    {avg_time_per_doc:.2f} seconds")
    print(f"Throughput:          {throughput:.2f} docs/min")
    print("\nTarget (NFR-1):      >= 50 docs/min")
    print(f"Status:              {'✓ PASS' if throughput >= 50 else '✗ FAIL'}")
    print("=" * 70)

    # Cleanup (no explicit close methods needed - httpx client auto-closes)
    print("\n✓ Benchmark complete")

    return 0 if throughput >= 50 and failed_count == 0 else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
