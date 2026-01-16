#!/usr/bin/env python3
"""Run latency benchmark: process single 2000-token markdown file and measure latency."""

import asyncio
import random

# Add project to path
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from crawl4r.processing.chunker import MarkdownChunker
from crawl4r.core.config import Settings
from crawl4r.processing.processor import DocumentProcessor
from crawl4r.storage.embeddings import TEIClient
from crawl4r.storage.vector_store import VectorStoreManager

# Sample content to generate 2000-token file
HEADINGS = [
    "Introduction", "Overview", "Background", "Context", "Summary",
    "Details", "Analysis", "Implementation", "Results", "Conclusion",
    "Methods", "Approach", "Strategy", "Framework", "Architecture",
    "Design", "Development", "Testing", "Deployment", "Maintenance"
]

PARAGRAPHS = [
    "This section explores the fundamental concepts and principles underlying the system. "
    "We examine various approaches and methodologies that have been developed over time. "
    "The analysis considers multiple perspectives and evaluates different strategies. "
    "Key factors include performance, scalability, maintainability, and reliability. ",

    "The implementation follows industry best practices and design patterns. "
    "We leverage modern technologies and frameworks to build robust solutions. "
    "Special attention is paid to code quality, testing, and documentation. "
    "The architecture supports extensibility and future enhancements. ",

    "Performance optimization is critical for delivering excellent user experience. "
    "We apply caching strategies, efficient algorithms, and resource management. "
    "Benchmarking and profiling help identify bottlenecks and areas for improvement. "
    "Continuous monitoring ensures the system meets performance targets. ",

    "Security considerations are integrated throughout the development lifecycle. "
    "We implement authentication, authorization, encryption, and input validation. "
    "Regular security audits and penetration testing identify vulnerabilities. "
    "Compliance with industry standards and regulations is maintained. ",

    "The testing strategy encompasses unit, integration, and end-to-end tests. "
    "Automated testing pipelines ensure code quality and prevent regressions. "
    "Test coverage metrics guide testing efforts and identify gaps. "
    "Continuous integration and deployment streamline the release process. ",
]

def generate_2000_token_content() -> str:
    """Generate markdown content with approximately 2000 tokens."""
    content = f"# Latency Benchmark Document {random.randint(1000, 9999)}\n\n"
    current_tokens = 10  # Approximate for title

    while current_tokens < 2000:
        # Add heading (level 2 or 3)
        level = random.choice(["##", "###"])
        heading = random.choice(HEADINGS)
        content += f"\n{level} {heading}\n\n"
        current_tokens += 5

        # Add 2-4 paragraphs
        num_paragraphs = random.randint(2, 4)
        for _ in range(num_paragraphs):
            paragraph = random.choice(PARAGRAPHS)
            # Repeat sentences to reach target length
            repeats = max(1, (2000 - current_tokens) // 50)
            repeats = min(repeats, 5)  # Cap repetitions
            content += paragraph * repeats + "\n\n"
            current_tokens += len(paragraph.split()) * repeats

            if current_tokens >= 2000:
                break

    return content

async def main():
    """Run latency benchmark."""
    print("=" * 70)
    print("LATENCY BENCHMARK: Single 2000-Token Markdown File")
    print("=" * 70)

    # Load configuration
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        settings = Settings(watch_folder=temp_dir)

        print("\nConfiguration:")
        print(f"  Watch folder: {settings.watch_folder}")
        print(f"  Chunk size: {settings.chunk_size_tokens} tokens")
        print(f"  Chunk overlap: {settings.chunk_overlap_percent}%")
        print(f"  Batch size: {settings.batch_size}")
        print(f"  TEI endpoint: {settings.tei_endpoint}")
        print(f"  Qdrant endpoint: {settings.qdrant_url}")
        print(f"  Collection: {settings.collection_name}")

        # Initialize components
        print("\n" + "-" * 70)
        print("Initializing components...")
        print("-" * 70)
        tei_client = TEIClient(endpoint_url=settings.tei_endpoint)
        vector_store = VectorStoreManager(
            qdrant_url=settings.qdrant_url,
            collection_name=settings.collection_name,
            dimensions=1024,  # Qwen3-Embedding-0.6B dimension
        )
        chunker = MarkdownChunker()
        processor = DocumentProcessor(
            config=settings,
            tei_client=tei_client,
            vector_store=vector_store,
            chunker=chunker,
        )
        print("✓ Components initialized")

        # Clear Qdrant collection for clean measurement
        print("\n" + "-" * 70)
        print("Clearing Qdrant collection for clean measurement...")
        print("-" * 70)
        try:
            # Delete collection if it exists
            if vector_store.client.collection_exists(vector_store.collection_name):
                vector_store.client.delete_collection(vector_store.collection_name)
            # Recreate collection
            vector_store.ensure_collection()
            print("✓ Collection cleared and recreated")
        except Exception as e:
            print(f"✗ Failed to clear collection: {e}")
            return 1

        # Generate test file
        print("\n" + "-" * 70)
        print("Generating 2000-token test file...")
        print("-" * 70)
        content = generate_2000_token_content()
        actual_tokens = len(content.split())
        test_file = temp_dir / "latency_test.md"

        # Measure from file creation to processing complete
        print(f"✓ Generated content: ~{actual_tokens} tokens")
        print("\n" + "=" * 70)
        print("STARTING LATENCY MEASUREMENT")
        print("=" * 70)

        # Start timer BEFORE writing file
        start_time = time.perf_counter()

        # Write file
        test_file.write_text(content)

        # Process file
        try:
            result = await processor.process_document(test_file)
            end_time = time.perf_counter()

            latency = end_time - start_time

            # Display results
            print("\n" + "=" * 70)
            print("LATENCY BENCHMARK RESULTS")
            print("=" * 70)
            print(f"\nFile:                {test_file.name}")
            print(f"Token count:         ~{actual_tokens} tokens")
            print(f"Processing success:  {'✓ Yes' if result.success else '✗ No'}")
            if not result.success:
                print(f"Error:               {result.error}")
            print(f"Chunks processed:    {result.chunks_processed}")
            print(f"Latency:             {latency:.3f} seconds")
            print("\nTarget (NFR-2):      < 5.0 seconds")
            print(f"Status:              {'✓ PASS' if latency < 5.0 and result.success else '✗ FAIL'}")

            if latency < 5.0 and result.success:
                margin = ((5.0 - latency) / 5.0) * 100
                print(f"Margin:              {margin:.1f}% under target")

            print("=" * 70)

            return 0 if latency < 5.0 and result.success else 1

        except Exception as e:
            end_time = time.perf_counter()
            latency = end_time - start_time

            print("\n" + "=" * 70)
            print("LATENCY BENCHMARK RESULTS")
            print("=" * 70)
            print(f"\n✗ Exception during processing: {e}")
            print(f"Latency:             {latency:.3f} seconds")
            print("Status:              ✗ FAIL")
            print("=" * 70)

            return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
