# tests/unit/test_llama_embeddings.py
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from crawl4r.storage.llama_embeddings import TEIEmbedding
from crawl4r.storage.tei import TEIClient


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


def test_tei_embedding_accepts_embed_batch_size():
    """TEIEmbedding should accept and store embed_batch_size parameter."""
    embed_model = TEIEmbedding(
        endpoint_url="http://mock:80",
        embed_batch_size=50,
    )
    assert embed_model.embed_batch_size == 50


def test_tei_embedding_defaults_embed_batch_size():
    """TEIEmbedding should default embed_batch_size to 10."""
    embed_model = TEIEmbedding(endpoint_url="http://mock:80")
    assert embed_model.embed_batch_size == 10


def test_tei_embedding_validates_embed_batch_size_range():
    """TEIEmbedding should validate embed_batch_size is 1-2048.

    LlamaIndex BaseEmbedding uses Pydantic validation which raises
    ValidationError (subclass of ValueError) with Pydantic-formatted messages.
    """
    # Too low (Pydantic validates > 0)
    with pytest.raises(ValidationError, match="greater than 0"):
        TEIEmbedding(endpoint_url="http://mock:80", embed_batch_size=0)

    # Too high (Pydantic validates <= 2048)
    with pytest.raises(ValidationError, match="less than or equal to 2048"):
        TEIEmbedding(endpoint_url="http://mock:80", embed_batch_size=3000)

    # Valid boundary values
    embed_model_1 = TEIEmbedding(endpoint_url="http://mock:80", embed_batch_size=1)
    assert embed_model_1.embed_batch_size == 1

    embed_model_2048 = TEIEmbedding(endpoint_url="http://mock:80", embed_batch_size=2048)
    assert embed_model_2048.embed_batch_size == 2048

def test_tei_embedding_class_name():
    # Verify strict serialization requirement
    assert TEIEmbedding.class_name() == "TEIEmbedding"
