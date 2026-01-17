# tests/unit/test_llama_parser.py
from llama_index.core.schema import Document
from crawl4r.processing.llama_parser import CustomMarkdownNodeParser
from crawl4r.processing.chunker import MarkdownChunker

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
