# tests/unit/test_llama_embeddings.py
from unittest.mock import AsyncMock, MagicMock

import pytest

from crawl4r.storage.embeddings import TEIClient
from crawl4r.storage.llama_embeddings import TEIEmbedding


@pytest.fixture
def mock_tei_client():
    client = MagicMock(spec=TEIClient)
    client.embed_single = AsyncMock(return_value=[0.1] * 1024)
    client.embed_batch = AsyncMock(return_value=[[0.1] * 1024, [0.1] * 1024])
    return client

def test_tei_embedding_init(mock_tei_client):
    embed_model = TEIEmbedding(endpoint_url="http://mock:80")
    # We mock the internal client creation or we can check attributes
    assert embed_model.model_name == "TEI"

@pytest.mark.asyncio
async def test_aget_query_embedding(mock_tei_client):
    embed_model = TEIEmbedding(endpoint_url="http://mock:80")
    # Inject mock client
    embed_model._client = mock_tei_client

    embedding = await embed_model._aget_query_embedding("test")
    assert len(embedding) == 1024
    mock_tei_client.embed_single.assert_called_with("test")

@pytest.mark.asyncio
async def test_aget_text_embeddings(mock_tei_client):
    embed_model = TEIEmbedding(endpoint_url="http://mock:80")
    embed_model._client = mock_tei_client

    embeddings = await embed_model._aget_text_embeddings(["test1", "test2"])
    assert len(embeddings) == 2
    mock_tei_client.embed_batch.assert_called_with(["test1", "test2"])
