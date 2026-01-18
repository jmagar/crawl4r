"""
End-to-end integration tests for Crawl4AIReader pipeline.

Tests integration between reader and node parser components.
Requires running service: Crawl4AI.

Note: Full integration with Qdrant/TEI would require metadata schema adapters
since Crawl4AIReader produces web-specific metadata (source_url, title, description)
while VectorStoreManager expects file-specific metadata (file_path_relative, filename).
This is intentional - the reader is a standalone LlamaIndex component.

Note: This test uses MarkdownNodeParser directly (not via DocumentProcessor) to validate
the LlamaIndex node parsing API independently with web-crawled content. This tests the
standalone node parsing integration path separate from the document processing pipeline.
"""

import pytest
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document

from crawl4r.core.logger import get_logger
from crawl4r.core.metadata import MetadataKeys
from crawl4r.readers.crawl4ai import Crawl4AIReader

logger = get_logger(__name__)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_reader_to_node_parser() -> None:
    """
    Test E2E integration from Crawl4AIReader to MarkdownNodeParser.

    Verifies:
    - Reader crawls URL and returns Document with markdown text
    - Node parser splits Document text into semantic nodes
    - Nodes contain proper content and metadata

    This demonstrates the reader integrates with LlamaIndex node-based processing.
    """
    # Create reader (will validate Crawl4AI service health)
    try:
        reader = Crawl4AIReader(endpoint_url="http://localhost:52004")
    except ValueError as e:
        pytest.skip(f"Crawl4AI service unavailable: {e}")

    # Create node parser
    node_parser = MarkdownNodeParser()

    # Crawl test URL
    test_url = "https://example.com"
    documents = await reader.aload_data([test_url])

    # Verify document created
    assert len(documents) == 1
    assert documents[0] is not None
    doc = documents[0]

    # Verify document has markdown content and web metadata
    assert len(doc.text) > 0
    assert doc.metadata["source"] == test_url
    assert doc.metadata[MetadataKeys.SOURCE_URL] == test_url
    assert doc.metadata[MetadataKeys.SOURCE_TYPE] == "web_crawl"
    assert MetadataKeys.TITLE in doc.metadata

    logger.info(
        f"Crawled {test_url}: {len(doc.text)} chars, title='{doc.metadata[MetadataKeys.TITLE]}'"
    )

    # Parse the document into nodes using LlamaIndex MarkdownNodeParser
    llama_doc = Document(text=doc.text, metadata={"filename": test_url})
    nodes = node_parser.get_nodes_from_documents([llama_doc])

    # Verify nodes created
    assert len(nodes) > 0
    logger.info(f"Created {len(nodes)} nodes from crawled document")

    # Verify node structure
    for i, node in enumerate(nodes):
        # Nodes have content accessible via get_content()
        content = node.get_content()
        assert len(content) > 0

        # Nodes have a unique ID
        assert node.node_id is not None

        logger.debug(
            f"Node {i}: {len(content)} chars, "
            f"id='{node.node_id[:8]}...'"
        )

    # Verify integration preserves semantics
    # First node should have content from crawled page
    assert len(nodes[0].get_content()) > 0

    logger.info(
        f"E2E test passed: {test_url} → {len(doc.text)} chars → {len(nodes)} nodes"
    )
