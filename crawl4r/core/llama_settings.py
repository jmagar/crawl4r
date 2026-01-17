"""LlamaIndex Settings bridge for Crawl4r configuration."""

from collections.abc import Callable
from typing import Any

from llama_index.core import Settings as LlamaSettings
from transformers import AutoTokenizer

from crawl4r.core.config import Settings
from crawl4r.storage.llama_embeddings import TEIEmbedding


def configure_llama_settings(
    app_settings: Settings,
    embed_model: TEIEmbedding | None = None,
    tokenizer_factory: Callable[[str], Any] | None = None,
) -> None:
    """Configure LlamaIndex Settings from application config.

    Args:
        app_settings: Crawl4r application settings
        embed_model: Optional pre-configured TEIEmbedding instance
        tokenizer_factory: Optional factory for producing a tokenizer
    """
    model_name = app_settings.tei_model_name

    if embed_model is None:
        embed_model = TEIEmbedding(endpoint_url=app_settings.tei_endpoint)

    if tokenizer_factory is None:
        tokenizer_factory = AutoTokenizer.from_pretrained

    # Configure Global Settings
    LlamaSettings.embed_model = embed_model
    LlamaSettings.chunk_size = app_settings.chunk_size_tokens
    LlamaSettings.chunk_overlap = int(
        app_settings.chunk_size_tokens * (app_settings.chunk_overlap_percent / 100)
    )

    try:
        tokenizer = tokenizer_factory(model_name)
        LlamaSettings.tokenizer = tokenizer.encode
    except Exception as e:
        # Fallback or log warning if tokenizer download fails (e.g. offline)
        print(f"Warning: Failed to load tokenizer '{model_name}': {e}")
