import asyncio
from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

from crawl4r.storage.embeddings import TEIClient


class TEIEmbedding(BaseEmbedding):
    """LlamaIndex wrapper for TEIClient with circuit breaker support."""

    _client: TEIClient = PrivateAttr()

    def __init__(
        self,
        endpoint_url: str | None = None,
        timeout: float = 30.0,
        client: TEIClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="TEI", **kwargs)
        if client:
            self._client = client
        elif endpoint_url:
            self._client = TEIClient(endpoint_url=endpoint_url, timeout=timeout)
        else:
            raise ValueError("Must provide either endpoint_url or client")

    def _get_query_embedding(self, query: str) -> list[float]:
        return asyncio.run(self._client.embed_single(query))

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._client.embed_single(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return asyncio.run(self._client.embed_single(text))

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await self._client.embed_single(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(self._client.embed_batch(texts))

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await self._client.embed_batch(texts)
