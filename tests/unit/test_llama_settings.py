from typing import Callable
from unittest.mock import MagicMock

from llama_index.core import Settings as LlamaSettings

from crawl4r.core.config import Settings
from crawl4r.core.llama_settings import configure_llama_settings
from crawl4r.storage.llama_embeddings import TEIEmbedding


def test_configure_llama_settings_sets_globals() -> None:
    app_settings = Settings(
        watch_folder="/tmp",
        chunk_size_tokens=512,
        chunk_overlap_percent=15,
        tei_endpoint="http://tei:80",
        tei_model_name="Qwen/Qwen3-Embedding-0.6B",
    )

    # Mock tokenizer factory to avoid downloads
    def tokenizer_factory(_: str) -> Callable[[str], list[int]]:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text: [1] * len(text)
        return mock_tokenizer

    configure_llama_settings(
        app_settings=app_settings,
        tokenizer_factory=tokenizer_factory,
    )

    assert isinstance(LlamaSettings.embed_model, TEIEmbedding)
    assert LlamaSettings.chunk_size == 512
    # 15% of 512 is 76.8, int() truncates to 76
    assert LlamaSettings.chunk_overlap == 76
    assert callable(LlamaSettings.tokenizer)


def test_configure_llama_settings_uses_provided_embed_model() -> None:
    app_settings = Settings(
        watch_folder="/tmp",
        tei_endpoint="http://tei:80",
        tei_model_name="Qwen/Qwen3-Embedding-0.6B",
    )
    embed_model = TEIEmbedding(endpoint_url="http://tei:80")

    # Mock tokenizer factory
    def tokenizer_factory(_: str) -> Callable[[str], list[int]]:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text: [1] * len(text)
        return mock_tokenizer

    configure_llama_settings(
        app_settings=app_settings, 
        embed_model=embed_model,
        tokenizer_factory=tokenizer_factory
    )

    assert LlamaSettings.embed_model is embed_model
