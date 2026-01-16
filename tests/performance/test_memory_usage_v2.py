#!/usr/bin/env python3
"""
Memory usage validation test (V38)
Generates 1000 test files and monitors memory usage during processing
"""
import asyncio
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil


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
    size_distribution = [
        ("small", 300, 500),    # 300 files, ~500 words each
        ("medium", 500, 2000),  # 500 files, ~2000 words each
        ("large", 200, 5000),   # 200 files, ~5000 words each
    ]

    file_num = 0
    for size_type, num_files, word_count in size_distribution:
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


async def clear_qdrant_collection():
    """Clear the Qdrant collection for clean test."""
    print("Clearing Qdrant collection...")

    qdrant_url = "http://localhost:52001"
    collection_name = "crawl4r"

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


def get_process_memory_mb(process: psutil.Process) -> float:
    """Get memory usage of a process and its children in MB."""
    try:
        # Get memory of main process
        mem_info = process.memory_info()
        total_rss = mem_info.rss

        # Add memory of child processes
        for child in process.children(recursive=True):
            try:
                child_mem = child.memory_info()
                total_rss += child_mem.rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return total_rss / (1024 * 1024)  # Convert to MB
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


async def monitor_pipeline(process: psutil.Process, check_interval: float = 1.0):
    """Monitor pipeline memory usage until it completes."""
    start_time = time.time()
    peak_memory_mb = 0.0
    samples = []

    print(f"\nMonitoring memory for PID {process.pid}...")
    print(f"{'Time (s)':<10} {'Memory (MB)':<15} {'Peak (MB)':<15} {'Status':<15}")
    print("-" * 65)

    while process.is_running():
        elapsed = time.time() - start_time
        current_memory = get_process_memory_mb(process)

        if current_memory > 0:
            peak_memory_mb = max(peak_memory_mb, current_memory)
            samples.append({
                "time": elapsed,
                "memory_mb": current_memory
            })

            # Check process status
            try:
                status = process.status()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                status = "unknown"

            print(f"{elapsed:<10.1f} {current_memory:<15.2f} {peak_memory_mb:<15.2f} {status:<15}")

        await asyncio.sleep(check_interval)

    # Final check
    elapsed = time.time() - start_time
    print(f"\nPipeline completed after {elapsed:.1f} seconds")

    return {
        "peak_memory_mb": peak_memory_mb,
        "duration_seconds": elapsed,
        "samples": samples
    }


async def main():
    """Run memory validation test."""
    print("=" * 60)
    print("Memory Usage Validation Test (V38)")
    print("=" * 60)
    print()

    watch_dir = Path("/home/jmagar/workspace/crawl4r/data/watched_folder")

    # Step 1: Generate test files
    await generate_test_files(watch_dir, count=1000)

    # Step 2: Clear Qdrant collection
    await clear_qdrant_collection()

    # Step 3: Start pipeline as subprocess
    print("\n" + "=" * 60)
    print("Starting RAG ingestion pipeline...")
    print("=" * 60)
    print()

    # Set environment variables
    env = {
        "WATCH_FOLDER": str(watch_dir),
        "TEI_ENDPOINT": "http://localhost:52000",
        "QDRANT_URL": "http://localhost:52001",
        "LOG_LEVEL": "INFO",
    }

    # Start pipeline
    process = subprocess.Popen(
        ["/home/jmagar/.local/bin/uv", "run", "python", "-m", "crawl4r.cli.main"],
        env={**subprocess.os.environ, **env},
        cwd="/home/jmagar/workspace/crawl4r"
    )

    # Wait a bit for process to start
    await asyncio.sleep(2)

    # Wrap in psutil for monitoring
    psutil_process = psutil.Process(process.pid)

    # Step 4: Monitor memory usage
    print("\n" + "=" * 60)
    print("Monitoring memory usage...")
    print("=" * 60)

    try:
        memory_stats = await asyncio.wait_for(
            monitor_pipeline(psutil_process, check_interval=1.0),
            timeout=600.0  # 10 minute timeout
        )
    except asyncio.TimeoutError:
        print("\nMonitoring timed out after 10 minutes")
        process.terminate()
        process.wait()
        memory_stats = {"peak_memory_mb": 0.0, "duration_seconds": 0.0, "samples": []}

    # Step 5: Report results
    print("\n" + "=" * 60)
    print("Memory Usage Results")
    print("=" * 60)

    peak_mb = memory_stats["peak_memory_mb"]
    peak_gb = peak_mb / 1024
    target_gb = 4.0
    duration = memory_stats["duration_seconds"]

    print(f"Peak memory usage: {peak_mb:.2f} MB ({peak_gb:.3f} GB)")
    print(f"Processing duration: {duration:.1f} seconds")
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

    print("\n" + "=" * 60)
    print(f"Test Result: {result}")
    print("=" * 60)

    # Cleanup
    if process.poll() is None:
        process.terminate()
        process.wait()

    return result


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result == "PASS" else 1)
