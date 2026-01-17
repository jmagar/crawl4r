"""Shared pytest fixtures for unit/performance tests.

Re-exports common integration fixtures so performance tests can access
shared utilities like memory tracking and file generators.
"""

import os

import httpx
import pytest

from tests.integration.conftest import (  # type: ignore[import-not-found]
    cleanup_fixture,
    generate_n_files,
    memory_tracker,
    performance_timer,
    test_collection,
)

TEI_ENDPOINT = os.getenv("TEI_ENDPOINT", "http://localhost:52000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:52001")


@pytest.fixture
async def require_tei_service() -> None:
    """Skip tests when TEI service is unavailable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{TEI_ENDPOINT}/health")
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip(f"TEI service not available at {TEI_ENDPOINT}")


@pytest.fixture
async def require_qdrant_service() -> None:
    """Skip tests when Qdrant service is unavailable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{QDRANT_URL}/readyz")
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip(f"Qdrant service not available at {QDRANT_URL}")


__all__ = [
    "cleanup_fixture",
    "generate_n_files",
    "memory_tracker",
    "performance_timer",
    "require_tei_service",
    "require_qdrant_service",
    "test_collection",
]
