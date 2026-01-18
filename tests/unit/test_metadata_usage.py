"""Verify source code uses MetadataKeys constants instead of hardcoded strings.

These tests verify that runtime code uses MetadataKeys constants rather than
hardcoded string literals. Docstrings and TypedDict field definitions are
excluded from these checks since they serve documentation purposes.
"""
import re
from pathlib import Path


def _count_runtime_usages(source: str, key: str) -> int:
    """Count runtime usages of a hardcoded metadata key string.

    Excludes docstring examples (lines starting with ... or >>>),
    comments, and TypedDict field definitions.

    Args:
        source: Source code content
        key: Metadata key to search for (e.g., "file_path_relative")

    Returns:
        Count of runtime usages that should use MetadataKeys constant
    """
    count = 0
    # Patterns that indicate runtime code usage (not docstrings/comments)
    runtime_patterns = [
        rf'metadata\["{key}"\]',  # metadata["file_path_relative"]
        rf'\.get\("{key}"',  # .get("file_path_relative"
        rf'= \["{key}"',  # = ["file_path_relative"
        rf'_delete_by_filter\("{key}"',  # _delete_by_filter("file_path_relative"
        rf'"{key}":',  # "file_path_relative": in dict literal (non-docstring)
    ]

    for line in source.split("\n"):
        # Skip docstring example lines
        if ">>>" in line or "..." in line:
            continue
        # Skip comment lines
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Check for runtime patterns
        for pattern in runtime_patterns:
            if re.search(pattern, line):
                count += 1
                break

    return count


def test_qdrant_imports_metadata_keys():
    """Verify qdrant.py imports MetadataKeys."""
    source = Path("crawl4r/storage/qdrant.py").read_text()
    assert "from crawl4r.core.metadata import MetadataKeys" in source, (
        "qdrant.py must import MetadataKeys"
    )


def test_qdrant_no_hardcoded_file_path():
    """Verify qdrant.py doesn't use hardcoded 'file_path' in runtime code (uses MetadataKeys)."""
    source = Path("crawl4r/storage/qdrant.py").read_text()
    count = _count_runtime_usages(source, "file_path")
    assert count == 0, f"Found {count} runtime hardcoded 'file_path' in qdrant.py (should use MetadataKeys)"


def test_processor_imports_metadata_keys():
    """Verify processor.py imports MetadataKeys."""
    source = Path("crawl4r/processing/processor.py").read_text()
    assert "from crawl4r.core.metadata import MetadataKeys" in source, (
        "processor.py must import MetadataKeys"
    )


def test_processor_no_hardcoded_file_path():
    """Verify processor.py doesn't use hardcoded 'file_path' in runtime code (uses MetadataKeys)."""
    source = Path("crawl4r/processing/processor.py").read_text()
    count = _count_runtime_usages(source, "file_path")
    assert count == 0, f"Found {count} runtime hardcoded 'file_path' in processor.py (should use MetadataKeys)"


def test_recovery_imports_metadata_keys():
    """Verify recovery.py imports MetadataKeys."""
    source = Path("crawl4r/resilience/recovery.py").read_text()
    assert "from crawl4r.core.metadata import MetadataKeys" in source, (
        "recovery.py must import MetadataKeys"
    )


def test_recovery_no_hardcoded_file_path():
    """Verify recovery.py doesn't use hardcoded 'file_path' in runtime code (uses MetadataKeys)."""
    source = Path("crawl4r/resilience/recovery.py").read_text()
    count = _count_runtime_usages(source, "file_path")
    assert count == 0, f"Found {count} runtime hardcoded 'file_path' in recovery.py (should use MetadataKeys)"


def test_failed_docs_no_longer_imports_metadata_keys():
    """Verify failed_docs.py no longer needs MetadataKeys (uses literal keys)."""
    source = Path("crawl4r/resilience/failed_docs.py").read_text()
    # After migration, failed_docs.py uses plain string keys in TypedDict
    # It no longer needs MetadataKeys import
    assert "from crawl4r.core.metadata import MetadataKeys" not in source, (
        "failed_docs.py should not import MetadataKeys after migration"
    )
