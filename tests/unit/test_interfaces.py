"""Tests for crawl4r.core.interfaces module."""

from typing import Any

import pytest

from crawl4r.core.interfaces import VectorStoreProtocol


class MockVectorStore:
    """Mock implementation of VectorStoreProtocol for testing."""

    async def get_collection_info(self) -> dict[str, Any]:
        """Mock get_collection_info."""
        return {"vector_size": 1024, "distance": "Cosine"}

    async def scroll(self) -> list[dict[str, Any]]:
        """Mock scroll."""
        return [{"id": "1", "payload": {"file_path": "/test.md"}}]


def test_vector_store_protocol_has_required_methods() -> None:
    """Verify VectorStoreProtocol defines expected methods.

    Ensures:
    - Protocol has get_collection_info method
    - Protocol has scroll method
    """
    assert hasattr(VectorStoreProtocol, "get_collection_info")
    assert hasattr(VectorStoreProtocol, "scroll")


@pytest.mark.asyncio
async def test_mock_implements_vector_store_protocol() -> None:
    """Verify mock implementation satisfies protocol.

    Ensures:
    - Mock can be used as VectorStoreProtocol
    - Required methods are callable
    """
    store: VectorStoreProtocol = MockVectorStore()

    info = await store.get_collection_info()
    assert "vector_size" in info

    points = await store.scroll()
    assert isinstance(points, list)
