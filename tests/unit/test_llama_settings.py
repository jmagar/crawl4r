from typing import Any
from unittest.mock import MagicMock

import pytest
from llama_index.core import Settings as LlamaSettings

from crawl4r.core.config import Settings
from crawl4r.core.llama_settings import configure_llama_settings
from crawl4r.storage.llama_embeddings import TEIEmbedding


@pytest.fixture
def mock_tokenizer_factory():
    """Factory that returns a mock tokenizer to avoid network downloads."""
    def factory(_: str) -> Any:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text: [1] * len(text)
        return mock_tokenizer
    return factory


# Capture LlamaSettings defaults at import time for restoration
# LlamaSettings does not provide a public reset() API, so we capture defaults
# at import and restore them after each test to prevent cross-test pollution.
_LLAMA_SETTINGS_DEFAULTS = {
    "chunk_size": LlamaSettings.chunk_size,
    "chunk_overlap": LlamaSettings.chunk_overlap,
}


@pytest.fixture
def reset_llama_settings():
    """Reset LlamaSettings globals after each test to prevent pollution.

    Note: LlamaSettings does not provide a public reset() API, so we must
    manipulate private attributes (_embed_model, _tokenizer) and restore
    chunk_size/chunk_overlap from cached defaults captured at import time.
    """
    yield
    # Clean up global settings after test using private attributes
    # (no public API available for resetting these)
    LlamaSettings._embed_model = None
    LlamaSettings._tokenizer = None
    # Restore defaults from cached values (not hardcoded)
    LlamaSettings.chunk_size = _LLAMA_SETTINGS_DEFAULTS["chunk_size"]
    LlamaSettings.chunk_overlap = _LLAMA_SETTINGS_DEFAULTS["chunk_overlap"]


def test_configure_llama_settings_sets_globals(
    mock_tokenizer_factory, reset_llama_settings
) -> None:
    app_settings = Settings(
        watch_folder="/tmp",
        chunk_size_tokens=512,
        chunk_overlap_percent=15,
        tei_endpoint="http://tei:80",
        tei_model_name="Qwen/Qwen3-Embedding-0.6B",
    )

    configure_llama_settings(
        app_settings=app_settings,
        tokenizer_factory=mock_tokenizer_factory,
    )

    assert isinstance(LlamaSettings.embed_model, TEIEmbedding)
    assert LlamaSettings.chunk_size == 512
    # 15% of 512 is 76.8, int() truncates to 76
    assert LlamaSettings.chunk_overlap == 76
    assert callable(LlamaSettings.tokenizer)


def test_configure_llama_settings_uses_provided_embed_model(
    mock_tokenizer_factory, reset_llama_settings
) -> None:
    app_settings = Settings(
        watch_folder="/tmp",
        tei_endpoint="http://tei:80",
        tei_model_name="Qwen/Qwen3-Embedding-0.6B",
    )
    embed_model = TEIEmbedding(endpoint_url="http://tei:80")

    configure_llama_settings(
        app_settings=app_settings,
        embed_model=embed_model,
        tokenizer_factory=mock_tokenizer_factory,
    )

    assert LlamaSettings.embed_model is embed_model
