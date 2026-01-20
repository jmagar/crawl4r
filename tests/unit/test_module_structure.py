"""Test module reorganization - verify new import paths work."""


def test_core_modules_importable():
    """Test core submodule imports."""
    from crawl4r.core.config import Settings
    from crawl4r.core.logger import get_logger
    from crawl4r.core.quality import QualityVerifier

    assert Settings is not None
    assert get_logger is not None
    assert QualityVerifier is not None


def test_readers_modules_importable():
    """Test readers submodule imports."""
    from crawl4r.readers.crawl4ai import Crawl4AIReader
    from crawl4r.readers.file_watcher import FileWatcher

    assert Crawl4AIReader is not None
    assert FileWatcher is not None


def test_processing_modules_importable():
    """Test processing submodule imports."""
    from llama_index.core.node_parser import MarkdownNodeParser

    from crawl4r.processing.processor import DocumentProcessor

    assert MarkdownNodeParser is not None
    assert DocumentProcessor is not None


def test_storage_modules_importable():
    """Test storage submodule imports."""
    from crawl4r.storage.qdrant import VectorStoreManager
    from crawl4r.storage.tei import TEIClient

    assert TEIClient is not None
    assert VectorStoreManager is not None


def test_resilience_modules_importable():
    """Test resilience submodule imports."""
    from crawl4r.resilience.circuit_breaker import CircuitBreaker
    from crawl4r.resilience.failed_docs import FailedDocLogger
    from crawl4r.resilience.recovery import StateRecovery

    assert CircuitBreaker is not None
    assert FailedDocLogger is not None
    assert StateRecovery is not None


def test_cli_modules_importable():
    """Test CLI submodule imports."""
    from crawl4r.cli.main import main

    assert main is not None


def test_api_modules_exist():
    """Test API submodule structure exists."""
    import crawl4r.api

    assert crawl4r.api is not None


def test_no_markdown_chunker_references_in_docs() -> None:
    """Verify no MarkdownChunker references remain in documentation."""
    doc_files = ["README.md", "CLAUDE.md", "ENHANCEMENTS.md"]
    matches = []

    for filename in doc_files:
        path = filename
        try:
            with open(path, encoding="utf-8") as handle:
                content = handle.read()
        except FileNotFoundError:
            continue
        if "MarkdownChunker" in content:
            matches.append(filename)

    assert not matches, f"MarkdownChunker still referenced in docs: {', '.join(matches)}"
