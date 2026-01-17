from collections.abc import Sequence
from typing import Any

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, TextNode

from crawl4r.processing.chunker import MarkdownChunker


class CustomMarkdownNodeParser(NodeParser):
    """Node parser that uses MarkdownChunker logic."""

    _chunker: MarkdownChunker = PrivateAttr()

    def __init__(self, chunker: MarkdownChunker | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._chunker = chunker or MarkdownChunker()

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> list[BaseNode]:
        out_nodes = []
        for node in nodes:
            # Get filename from metadata or fallback
            filename = node.metadata.get("filename", "unknown.md")

            # Use existing chunker
            chunks = self._chunker.chunk(node.get_content(), filename=filename)

            for chunk in chunks:
                # Merge original metadata with chunk metadata
                metadata = node.metadata.copy()
                metadata.update(
                    {
                        "chunk_index": chunk["chunk_index"],
                        "section_path": chunk["section_path"],
                        "heading_level": chunk["heading_level"],
                    }
                )
                if chunk["tags"]:
                    metadata["tags"] = chunk["tags"]

                # Create TextNode
                text_node = TextNode(
                    text=chunk["chunk_text"],
                    metadata=metadata,
                    excluded_embed_metadata_keys=[
                        "chunk_index",
                        "heading_level",
                        "filename",
                        "tags",
                    ],
                    excluded_llm_metadata_keys=["chunk_index", "heading_level"],
                )
                out_nodes.append(text_node)

        return out_nodes
