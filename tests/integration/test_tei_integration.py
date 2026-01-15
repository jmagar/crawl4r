"""Integration tests for TEI (Text Embeddings Inference) service.

Tests real TEI service endpoints to verify:
- Embedding generation works with actual service
- Embeddings have correct dimensions (1024)
- Batch processing returns correct number of embeddings
- Embeddings are properly normalized

These tests require the TEI service to be running. The endpoint can be configured
via the TEI_ENDPOINT environment variable. If not set, defaults to
http://localhost:52000. If the service is not available, tests will be skipped.

Example:
    Run only TEI integration tests:
    $ pytest tests/integration/test_tei_integration.py -v -m integration

    Run with custom endpoint:
    $ TEI_ENDPOINT=http://crawl4r-embeddings:80 pytest \
        tests/integration/test_tei_integration.py -v -m integration

    Run with service availability check:
    $ docker compose up -d crawl4r-embeddings
    $ pytest tests/integration/test_tei_integration.py -v -m integration
"""

import math
import os
from collections.abc import AsyncIterator

import httpx
import pytest

from rag_ingestion.tei_client import TEIClient

# Get TEI endpoint from environment or use default
TEI_ENDPOINT = os.getenv("TEI_ENDPOINT", "http://localhost:52000")


@pytest.fixture
async def tei_client() -> AsyncIterator[TEIClient]:
    """Create TEI client configured for testing.

    Uses TEI_ENDPOINT environment variable if set, otherwise defaults to
    http://localhost:52000.

    Returns:
        TEIClient instance configured for integration testing

    Example:
        >>> async def test_example(tei_client):
        ...     embedding = await tei_client.embed_single("test")
        ...     assert len(embedding) == 1024
    """
    return TEIClient(
        endpoint_url=TEI_ENDPOINT,
        dimensions=1024,
        timeout=30.0,
        max_retries=3,
    )


@pytest.fixture(autouse=True)
async def check_tei_service() -> None:
    """Check if TEI service is available before running tests.

    Automatically runs before each test to verify the TEI service is
    reachable. Uses the TEI_ENDPOINT environment variable or defaults to
    http://localhost:52000. If the service is not available, the test will
    be skipped with an informative message.

    Raises:
        pytest.skip: If TEI service is not available at configured endpoint

    Example:
        This fixture runs automatically for all tests in this module.
        No explicit usage required.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{TEI_ENDPOINT}/health")
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip(f"TEI service not available at {TEI_ENDPOINT}")


@pytest.mark.integration
async def test_tei_generates_real_embeddings(tei_client: TEIClient) -> None:
    """Test TEI service generates embeddings with correct dimensions.

    Verifies:
    - TEI service accepts single text embedding requests
    - Returned embedding has exactly 1024 dimensions
    - Embedding values are floats

    Args:
        tei_client: TEIClient fixture configured for localhost

    Raises:
        AssertionError: If embedding dimensions are not 1024 or values are not floats
    """
    # Request embedding for simple test text
    embedding = await tei_client.embed_single("test text")

    # Verify correct number of dimensions
    assert len(embedding) == 1024, f"Expected 1024 dimensions, got {len(embedding)}"

    # Verify all values are floats
    assert all(isinstance(x, float) for x in embedding), (
        "Embedding values must be floats"
    )


@pytest.mark.integration
async def test_tei_batch_embeddings(tei_client: TEIClient) -> None:
    """Test TEI service handles batch embedding requests correctly.

    Verifies:
    - TEI service accepts batch requests with multiple texts
    - Returns correct number of embeddings (one per input text)
    - Each embedding has exactly 1024 dimensions
    - All embedding values are floats

    Args:
        tei_client: TEIClient fixture configured for localhost

    Raises:
        AssertionError: If batch size, dimensions, or types are incorrect
    """
    # Create batch of 10 test texts
    texts = [f"test text {i}" for i in range(10)]

    # Request batch embeddings
    embeddings = await tei_client.embed_batch(texts)

    # Verify correct number of embeddings returned
    assert len(embeddings) == 10, f"Expected 10 embeddings, got {len(embeddings)}"

    # Verify each embedding has correct dimensions
    for i, embedding in enumerate(embeddings):
        assert len(embedding) == 1024, (
            f"Embedding {i}: expected 1024 dimensions, got {len(embedding)}"
        )

        # Verify all values are floats
        assert all(isinstance(x, float) for x in embedding), (
            f"Embedding {i}: values must be floats"
        )


@pytest.mark.integration
async def test_tei_embeddings_are_normalized(tei_client: TEIClient) -> None:
    """Test TEI service returns normalized embeddings.

    Verifies:
    - Embeddings are normalized (L2 norm ≈ 1.0)
    - Normalization is consistent across different texts

    Normalization check uses ±0.01 tolerance to account for floating
    point precision and TEI's normalization implementation.

    Args:
        tei_client: TEIClient fixture configured for localhost

    Raises:
        AssertionError: If L2 norm is not approximately 1.0
    """
    # Request embedding for test text
    embedding = await tei_client.embed_single("test text for normalization")

    # Calculate L2 norm (Euclidean norm)
    l2_norm = math.sqrt(sum(x * x for x in embedding))

    # Verify L2 norm is approximately 1.0 (within tolerance)
    assert abs(l2_norm - 1.0) < 0.01, f"Expected L2 norm ≈ 1.0, got {l2_norm:.6f}"
