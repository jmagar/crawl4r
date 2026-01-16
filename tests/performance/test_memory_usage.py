#!/usr/bin/env python3
"""
Memory usage validation test (V38)
Generates 1000 test files and monitors memory usage during processing
"""
import asyncio
import os
import sys
import time
from pathlib import Path

import httpx
import psutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crawl4r.core.config import Settings


async def generate_test_files(watch_dir: Path, count: int = 1000) -> list[Path]:
    """Generate test markdown files with varying sizes."""
    print(f"Generating {count} test files in {watch_dir}...")

    # Clear existing test files
    if watch_dir.exists():
        for file in watch_dir.glob("test_mem_*.md"):
            file.unlink()
    else:
        watch_dir.mkdir(parents=True, exist_ok=True)

    files = []

    # Generate files with varying sizes
    sizes = [
        ("small", 500),    # ~500 words - 30% of files
        ("medium", 2000),  # ~2000 words - 50% of files
        ("large", 5000),   # ~5000 words - 20% of files
    ]

    size_distribution = [
        ("small", 300),
        ("medium", 500),
        ("large", 200),
    ]

    file_num = 0
    for size_type, num_files in size_distribution:
        word_count = next(wc for st, wc in sizes if st == size_type)

        for i in range(num_files):
            file_path = watch_dir / f"test_mem_{file_num:04d}_{size_type}.md"

            # Generate content with headings and paragraphs
            content_lines = [
                f"# Test Document {file_num}",
                "",
                f"This is test document number {file_num} for memory validation.",
                "",
                "## Introduction",
                "",
            ]

            # Add paragraphs
            words_added = 20  # From header
            paragraph_num = 0

            while words_added < word_count:
                # Add section every 5 paragraphs
                if paragraph_num % 5 == 0 and paragraph_num > 0:
                    content_lines.extend([
                        "",
                        f"## Section {paragraph_num // 5}",
                        "",
                    ])

                # Generate paragraph with ~100 words
                paragraph = f"Paragraph {paragraph_num}: " + " ".join([
                    f"word{j}" for j in range(100)
                ])
                content_lines.append(paragraph)
                content_lines.append("")

                words_added += 100
                paragraph_num += 1

            content_lines.extend([
                "",
                "## Conclusion",
                "",
                f"This concludes test document {file_num}.",
            ])

            file_path.write_text("\n".join(content_lines))
            files.append(file_path)
            file_num += 1

    print(f"Generated {len(files)} test files")
    return files


async def clear_qdrant_collection(settings: Settings):
    """Clear the Qdrant collection for clean test."""
    print("Clearing Qdrant collection...")

    # Use localhost URL for direct access
    qdrant_url = "http://localhost:52001"
    collection_name = settings.collection_name

    async with httpx.AsyncClient() as client:
        # Try to delete collection
        try:
            response = await client.delete(
                f"{qdrant_url}/collections/{collection_name}"
            )
            print(f"Collection delete response: {response.status_code}")
        except Exception as e:
            print(f"Collection delete error (may not exist): {e}")

        # Recreate collection
        await asyncio.sleep(1)

        try:
            response = await client.put(
                f"{qdrant_url}/collections/{collection_name}",
                json={
                    "vectors": {
                        "size": 1024,
                        "distance": "Cosine"
                    }
                }
            )
            print(f"Collection create response: {response.status_code}")
        except Exception as e:
            print(f"Collection create error: {e}")

    print("Qdrant collection cleared")


def get_process_memory_mb(pid: int) -> float:
    """Get memory usage of a process in MB."""
    try:
        process = psutil.Process(pid)
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert to MB
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


async def monitor_memory(pid: int, interval: float = 1.0, duration: float = 300.0):
    """Monitor memory usage of a process."""
    start_time = time.time()
    peak_memory_mb = 0.0
    samples = []

    print(f"\nMonitoring memory for PID {pid}...")
    print(f"{'Time (s)':<10} {'Memory (MB)':<15} {'Peak (MB)':<15}")
    print("-" * 40)

    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        current_memory = get_process_memory_mb(pid)

        if current_memory > 0:
            peak_memory_mb = max(peak_memory_mb, current_memory)
            samples.append({
                "time": elapsed,
                "memory_mb": current_memory
            })

            print(f"{elapsed:<10.1f} {current_memory:<15.2f} {peak_memory_mb:<15.2f}")

        await asyncio.sleep(interval)

    return {
        "peak_memory_mb": peak_memory_mb,
        "samples": samples
    }


async def main():
    """Run memory validation test."""
    print("=" * 60)
    print("Memory Usage Validation Test (V38)")
    print("=" * 60)
    print()

    # Get settings - use default watch folder with localhost endpoints
    settings = Settings(
        watch_folder="/home/jmagar/workspace/crawl4r/data/watched_folder",
        tei_endpoint="http://localhost:52000",
        qdrant_url="http://localhost:52001"
    )
    watch_dir = Path(settings.watch_folder)

    # Step 1: Generate test files
    await generate_test_files(watch_dir, count=1000)

    # Step 2: Clear Qdrant collection
    await clear_qdrant_collection(settings)

    # Step 3: Start pipeline
    print("\n" + "=" * 60)
    print("Starting RAG ingestion pipeline...")
    print("=" * 60)
    print()

    # Import and start pipeline
    from crawl4r.cli.main import main as pipeline_main

    # Start pipeline in background
    pipeline_task = asyncio.create_task(pipeline_main())

    # Wait for pipeline to initialize
    await asyncio.sleep(5)

    # Find pipeline process
    pipeline_pid = os.getpid()
    print(f"Pipeline PID: {pipeline_pid}")

    # Step 4: Monitor memory usage
    print("\n" + "=" * 60)
    print("Monitoring memory usage (5 minutes)...")
    print("=" * 60)

    try:
        # Monitor for 5 minutes or until pipeline completes
        memory_stats = await asyncio.wait_for(
            monitor_memory(pipeline_pid, interval=1.0, duration=300.0),
            timeout=310.0
        )
    except asyncio.TimeoutError:
        print("\nMonitoring timed out")
        memory_stats = {"peak_memory_mb": 0.0, "samples": []}

    # Step 5: Report results
    print("\n" + "=" * 60)
    print("Memory Usage Results")
    print("=" * 60)

    peak_mb = memory_stats["peak_memory_mb"]
    peak_gb = peak_mb / 1024
    target_gb = 4.0

    print(f"Peak memory usage: {peak_mb:.2f} MB ({peak_gb:.3f} GB)")
    print(f"Target: < {target_gb} GB")

    if peak_gb < target_gb:
        print(f"✓ PASS: Memory usage ({peak_gb:.3f} GB) < {target_gb} GB")
        result = "PASS"
    else:
        print(f"✗ FAIL: Memory usage ({peak_gb:.3f} GB) >= {target_gb} GB")
        result = "FAIL"

    # Calculate statistics
    if memory_stats["samples"]:
        samples_mb = [s["memory_mb"] for s in memory_stats["samples"]]
        avg_mb = sum(samples_mb) / len(samples_mb)
        avg_gb = avg_mb / 1024

        print(f"Average memory: {avg_mb:.2f} MB ({avg_gb:.3f} GB)")
        print(f"Samples collected: {len(samples_mb)}")

    # Cancel pipeline task
    pipeline_task.cancel()
    try:
        await pipeline_task
    except asyncio.CancelledError:
        pass

    print("\n" + "=" * 60)
    print(f"Test Result: {result}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result == "PASS" else 1)
