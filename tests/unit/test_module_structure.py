"""Test module reorganization - verify new import paths work."""
import pytest


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
    from crawl4r.processing.chunker import MarkdownChunker
    from crawl4r.processing.processor import DocumentProcessor

    assert MarkdownChunker is not None
    assert DocumentProcessor is not None


def test_storage_modules_importable():
    """Test storage submodule imports."""
    from crawl4r.storage.embeddings import TEIClient
    from crawl4r.storage.vector_store import VectorStoreManager

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
