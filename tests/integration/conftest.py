"""Integration test fixtures and configuration.

Provides pytest fixtures for integration tests that interact with real services
(TEI, Qdrant). Includes test configuration overrides, unique collection naming,
and automatic cleanup after tests complete.

Example:
    >>> @pytest.mark.integration
    >>> async def test_real_service(test_config, test_collection):
    ...     # test_config has test endpoints
    ...     # test_collection is unique per test
    ...     pass
"""

import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from rag_ingestion.config import Settings


@pytest.fixture
def test_config(tmp_path: Path) -> Settings:
    """Test configuration with overridden endpoints.

    Creates a Settings instance with test-specific values:
    - Uses localhost endpoints for TEI and Qdrant
    - Points to a temporary watch folder
    - Uses environment variables or defaults for ports

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Settings instance configured for integration tests

    Example:
        >>> def test_example(test_config):
        ...     assert "localhost" in test_config.tei_endpoint
        ...     assert test_config.watch_folder.exists()
    """
    return Settings(
        watch_folder=tmp_path,
        tei_endpoint="http://localhost:52000",
        qdrant_url="http://localhost:52001",
        collection_name="test_collection",
        chunk_size_tokens=512,
        chunk_overlap_percent=15,
        max_concurrent_docs=5,
        queue_max_size=100,
        batch_size=10,
        log_level="DEBUG",
    )


@pytest.fixture
def test_collection() -> str:
    """Generate unique collection name per test.

    Creates a UUID-based collection name to ensure test isolation.
    Each test gets its own collection that won't conflict with others
    running in parallel.

    Returns:
        Unique collection name with 'test_' prefix

    Example:
        >>> def test_example(test_collection):
        ...     # test_collection = "test_12345678-1234-1234-1234-123456789abc"
        ...     assert test_collection.startswith("test_")
    """
    return f"test_{uuid.uuid4().hex[:16]}"


@pytest.fixture
async def cleanup_fixture(
    test_collection: str,
) -> AsyncGenerator[None, None]:
    """Delete test collection after test completes.

    Ensures Qdrant collections created during tests are cleaned up,
    even if the test fails. Connects to Qdrant using the test URL
    and deletes the collection in the teardown phase.

    Args:
        test_collection: Unique collection name from test_collection fixture

    Yields:
        None (setup phase has no value to yield)

    Example:
        >>> @pytest.mark.integration
        >>> async def test_example(test_collection, cleanup_fixture):
        ...     # Create collection and run test
        ...     pass
        ...     # Collection automatically deleted after test
    """
    # Setup phase - nothing to do
    yield

    # Teardown phase - delete test collection
    client = AsyncQdrantClient(url="http://localhost:52001")
    try:
        # Check if collection exists before trying to delete
        collections = await client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if test_collection in collection_names:
            await client.delete_collection(collection_name=test_collection)
    finally:
        await client.close()


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers for integration tests.

    Adds the @pytest.mark.integration marker to pytest's recognized markers.
    This allows filtering integration tests with `pytest -m integration`.

    Args:
        config: pytest configuration object

    Example:
        Run only integration tests:
        $ pytest -m integration

        Run only unit tests (exclude integration):
        $ pytest -m "not integration"
    """
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require real services)",
    )
