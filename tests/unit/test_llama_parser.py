# tests/unit/test_llama_parser.py
from llama_index.core.schema import Document, TextNode

from crawl4r.processing.chunker import MarkdownChunker
from crawl4r.processing.llama_parser import CustomMarkdownNodeParser


def test_parser_nodes():
    text = "# Title\n\nSection 1 content"
    doc = Document(text=text, metadata={"filename": "test.md"})

    # We use the existing chunker logic
    chunker = MarkdownChunker()
    parser = CustomMarkdownNodeParser(chunker=chunker)

    nodes = parser.get_nodes_from_documents([doc])

    assert len(nodes) > 0
    assert nodes[0].text == "# Title\n\nSection 1 content" # Depending on chunk size
    assert nodes[0].metadata["filename"] == "test.md"
    # Verify custom metadata from chunker
    assert "section_path" in nodes[0].metadata


def test_parse_nodes_accepts_show_progress():
    """_parse_nodes should accept show_progress for API compatibility."""
    chunker = MarkdownChunker(chunk_size_tokens=512)
    parser = CustomMarkdownNodeParser(chunker=chunker)

    node = TextNode(text="# Test\n\nContent", metadata={"filename": "test.md"})

    # Should not raise - show_progress is accepted but unused
    result = parser._parse_nodes([node], show_progress=True)
    assert len(result) >= 1
