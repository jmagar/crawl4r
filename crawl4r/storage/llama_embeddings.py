import asyncio
import atexit
import concurrent.futures
import threading
from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

from crawl4r.storage.embeddings import TEIClient

# Shared executor for running coroutines from sync context.
# Lazily initialized on first use to avoid overhead if never needed.
_shared_executor: concurrent.futures.ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_shared_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the shared ThreadPoolExecutor."""
    global _shared_executor
    with _executor_lock:
        if _shared_executor is None:
            _shared_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            atexit.register(_shutdown_executor)
    return _shared_executor


def _shutdown_executor() -> None:
    """Shutdown the shared executor at process exit."""
    global _shared_executor
    if _shared_executor is not None:
        _shared_executor.shutdown(wait=False)
        _shared_executor = None


def _run_sync(coro: Any) -> Any:
    """Run async coroutine from sync context, handling existing event loops.

    WARNING: Running coroutines in a separate thread may break event-loop-bound
    resources (e.g., aiohttp connection pools, async context managers). If the
    underlying async client (like TEIClient) uses such resources, prefer running
    in the main event loop or provide a dedicated synchronous wrapper that doesn't
    share state with async code paths.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - safe to use asyncio.run
        return asyncio.run(coro)

    # Loop is running - execute in shared thread pool to avoid RuntimeError
    executor = _get_shared_executor()
    return executor.submit(asyncio.run, coro).result()


class TEIEmbedding(BaseEmbedding):
    """LlamaIndex wrapper for TEIClient with circuit breaker support."""

    _client: TEIClient = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "TEIEmbedding"

    def __init__(
        self,
        endpoint_url: str | None = None,
        timeout: float = 30.0,
        client: TEIClient | None = None,
        embed_batch_size: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize TEIEmbedding with TEI client and batch configuration.

        Args:
            endpoint_url: TEI service endpoint URL.
            timeout: HTTP request timeout in seconds.
            client: Pre-configured TEIClient (alternative to endpoint_url).
            embed_batch_size: Batch size for embedding calls (default: 10, max: 2048).
                Controls how many texts are sent to TEI in a single request when
                using get_text_embedding_batch(). LlamaIndex validates this to be
                between 1 and 2048.
            **kwargs: Additional BaseEmbedding parameters.

        Raises:
            ValidationError: If embed_batch_size not in range 1-2048.
            ValueError: If neither endpoint_url nor client provided.
        """
        super().__init__(
            model_name="TEI",
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        if client:
            self._client = client
        elif endpoint_url:
            self._client = TEIClient(endpoint_url=endpoint_url, timeout=timeout)
        else:
            raise ValueError("Must provide either endpoint_url or client")

    def _get_query_embedding(self, query: str) -> list[float]:
        return _run_sync(self._client.embed_single(query))

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._client.embed_single(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return _run_sync(self._client.embed_single(text))

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await self._client.embed_single(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return _run_sync(self._client.embed_batch(texts))

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await self._client.embed_batch(texts)
