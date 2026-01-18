"""LlamaIndex Settings bridge for Crawl4r configuration."""

import logging
from collections.abc import Callable
from typing import Any

from llama_index.core import Settings as LlamaSettings
from transformers import AutoTokenizer

from crawl4r.core.config import Settings
from crawl4r.storage.llama_embeddings import TEIEmbedding

logger = logging.getLogger(__name__)


def configure_llama_settings(
    app_settings: Settings,
    embed_model: TEIEmbedding | None = None,
    tokenizer_factory: Callable[[str], Any] | None = None,
) -> None:
    """Configure LlamaIndex Settings from application config.

    WARNING: This function mutates global LlamaIndex Settings state. It modifies
    the module-level LlamaSettings singleton, affecting all LlamaIndex operations
    in the application. Call once at startup before any LlamaIndex operations.

    The following global settings are modified:
    - LlamaSettings.embed_model: Set to provided embed_model or a new TEIEmbedding
    - LlamaSettings.chunk_size: Set from app_settings.chunk_size_tokens
    - LlamaSettings.chunk_overlap: Computed from chunk_size and overlap_percent
    - LlamaSettings.tokenizer: Set from tokenizer_factory (if successful)

    Args:
        app_settings: Crawl4r application settings containing TEI endpoint,
            model name, chunk size, and overlap configuration.
        embed_model: Optional pre-configured TEIEmbedding instance. If None,
            creates a new instance using app_settings.tei_endpoint.
        tokenizer_factory: Optional factory for producing a tokenizer. If None,
            uses AutoTokenizer.from_pretrained. Factory is called with model name.

    Side Effects:
        Modifies global LlamaSettings state. Tests should restore original values
        or use isolation patterns to avoid cross-test contamination.
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
        # Log error with full details for debugging
        logger.error(
            "Failed to load tokenizer '%s': %s. "
            "LlamaSettings.tokenizer will use default (may cause inconsistent "
            "chunking).",
            model_name,
            e,
            exc_info=True,
        )
        # Set a deterministic fallback: simple whitespace-based tokenizer
        # This ensures predictable behavior rather than relying on LlamaIndex defaults
        def fallback_tokenizer(text: str) -> list[int]:
            """Simple fallback tokenizer using whitespace splitting."""
            return list(range(len(text.split())))

        LlamaSettings.tokenizer = fallback_tokenizer
        logger.warning(
            "Using fallback whitespace tokenizer. Chunk boundaries may differ from "
            "production behavior with the actual '%s' tokenizer.",
            model_name,
        )
