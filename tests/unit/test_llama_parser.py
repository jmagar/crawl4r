# tests/unit/test_llama_parser.py
"""Tests for MarkdownNodeParser usage in the processing pipeline.

This module tests the LlamaIndex MarkdownNodeParser that replaces the
custom MarkdownChunker-based implementation. The old CustomMarkdownNodeParser
has been deprecated in favor of the standard LlamaIndex parser.
"""
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document, TextNode


def test_parser_nodes():
    """Test MarkdownNodeParser creates nodes from markdown documents."""
    text = "# Title\n\nSection 1 content"
    doc = Document(text=text, metadata={"filename": "test.md"})

    # Use LlamaIndex's built-in MarkdownNodeParser
    parser = MarkdownNodeParser()

    nodes = parser.get_nodes_from_documents([doc])

    assert len(nodes) > 0
    # MarkdownNodeParser preserves the document content
    assert "Title" in nodes[0].text or "Section 1" in nodes[0].text
    # Metadata is preserved
    assert nodes[0].metadata["filename"] == "test.md"


def test_parser_handles_complex_markdown():
    """Test MarkdownNodeParser handles complex markdown structure."""
    text = """# Main Title

Introduction paragraph.

## Section 1

Section 1 content.

### Subsection 1.1

Subsection content with **bold** and *italic*.

## Section 2

More content here.
"""
    doc = Document(text=text, metadata={"source": "complex.md"})

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents([doc])

    # Should produce multiple nodes for the different sections
    assert len(nodes) >= 1
    # All nodes should preserve source metadata
    for node in nodes:
        assert node.metadata.get("source") == "complex.md"


def test_parser_accepts_text_nodes():
    """Test MarkdownNodeParser can process TextNode inputs."""
    parser = MarkdownNodeParser()

    node = TextNode(text="# Test\n\nContent here", metadata={"filename": "test.md"})

    # Should not raise - parser handles TextNode input
    result = parser._parse_nodes([node], show_progress=False)
    assert len(result) >= 1
