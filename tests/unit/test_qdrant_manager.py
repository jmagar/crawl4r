from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import crawl4r.storage.qdrant as qdrant_module
from crawl4r.storage.qdrant import VectorStoreManager


@pytest.fixture
def vector_store_manager():
    async_client = MagicMock()
    async_client.collection_exists = AsyncMock(return_value=True)
    async_client.scroll = AsyncMock(return_value=([], None))
    async_client.query_points = AsyncMock()
    async_client.create_payload_index = AsyncMock()
    async_client.delete = AsyncMock()
    async_client.close = AsyncMock()

    sync_client = MagicMock()

    with (
        patch("crawl4r.storage.qdrant.AsyncQdrantClient", return_value=async_client),
        patch("crawl4r.storage.qdrant.QdrantClient", return_value=sync_client),
    ):
        yield VectorStoreManager(
            qdrant_url="http://localhost:6333",
            collection_name="test-collection",
        )


@pytest.mark.asyncio
async def test_retry_with_backoff_returns_value(vector_store_manager):
    async def operation():
        return "ok"

    result = await vector_store_manager._retry_with_backoff(operation, max_retries=1)

    assert result == "ok"


@pytest.mark.asyncio
async def test_search_similar_uses_retry_result(vector_store_manager):
    fake_point = SimpleNamespace(id="point-1", score=0.99, payload={"key": "value"})
    vector_store_manager._retry_with_backoff = AsyncMock(
        return_value=SimpleNamespace(points=[fake_point])
    )

    query_vector = [0.0] * vector_store_manager.dimensions
    results = await vector_store_manager.search_similar(query_vector, top_k=1)

    assert results == [{"id": "point-1", "score": 0.99, "key": "value"}]


@pytest.mark.asyncio
async def test_delete_by_filter_resets_ids_between_retries(vector_store_manager):
    async def scroll_side_effect(*args, **kwargs):
        points = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
        return points, None

    vector_store_manager.client.scroll = AsyncMock(side_effect=scroll_side_effect)
    vector_store_manager.client.delete = AsyncMock()

    call_count = {"count": 0}

    async def retry_stub(operation, max_retries=qdrant_module.MAX_RETRIES):
        call_count["count"] += 1
        if call_count["count"] == 1:
            await operation()
            return await operation()
        return await operation()

    vector_store_manager._retry_with_backoff = retry_stub

    deleted = await vector_store_manager._delete_by_filter("file_path", "/tmp/file.md")

    assert deleted == 2


@pytest.mark.asyncio
async def test_scroll_uses_retry_helper(vector_store_manager):
    vector_store_manager._retry_with_backoff = AsyncMock(return_value=([], None))

    await vector_store_manager.scroll(limit=10)

    vector_store_manager._retry_with_backoff.assert_called()


@pytest.mark.asyncio
async def test_close_closes_clients(vector_store_manager):
    vector_store_manager.sync_client.close = MagicMock()

    await vector_store_manager.close()

    vector_store_manager.client.close.assert_awaited_once()
    vector_store_manager.sync_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_async_context_manager_closes_clients(vector_store_manager):
    vector_store_manager.sync_client.close = MagicMock()

    async with vector_store_manager as manager:
        assert manager is vector_store_manager

    vector_store_manager.client.close.assert_awaited_once()
    vector_store_manager.sync_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_payload_index_conflict_is_ignored(vector_store_manager, monkeypatch):
    class FakeApiException(Exception):
        def __init__(self, status: int):
            super().__init__("conflict")
            self.status = status

    monkeypatch.setattr(qdrant_module, "ApiException", FakeApiException)

    async def raise_conflict(*args, **kwargs):
        raise FakeApiException(status=409)

    vector_store_manager.client.create_payload_index = AsyncMock(
        side_effect=raise_conflict
    )

    await vector_store_manager.ensure_payload_indexes()

    assert vector_store_manager.client.create_payload_index.call_count > 0
