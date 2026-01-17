# LlamaIndex Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Refactor the ingestion pipeline to use native LlamaIndex abstractions (`IngestionPipeline`, `BaseEmbedding`, `NodeParser`, `QdrantVectorStore`) while preserving existing resilience and functionality.

**Architecture:** 
- Wrap `TEIClient` in `TEIEmbedding` (LlamaIndex compatible).
- Wrap `MarkdownChunker` in `CustomMarkdownNodeParser` (LlamaIndex compatible).
- Use `IngestionPipeline` for the processing flow (Chunk -> Embed -> Upsert).
- Retain `VectorStoreManager` for collection management and cleanup operations.

**Tech Stack:** Python, LlamaIndex, Qdrant, TEI.

---

### Task 1: TEIEmbedding Wrapper

**Files:**
- Create: `crawl4r/storage/llama_embeddings.py`
- Create: `tests/unit/test_llama_embeddings.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_llama_embeddings.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from crawl4r.storage.tei import TEIClient
from crawl4r.storage.llama_embeddings import TEIEmbedding

@pytest.fixture
def mock_tei_client():
    client = MagicMock(spec=TEIClient)
    client.embed_single = AsyncMock(return_value=[0.1] * 1024)
    # Return two embeddings to match test input of ["test1", "test2"]
    client.embed_batch = AsyncMock(return_value=[[0.1] * 1024, [0.1] * 1024])
    return client

def test_tei_embedding_init(mock_tei_client):
    embed_model = TEIEmbedding(endpoint_url="http://mock:80")
    # We mock the internal client creation or we can check attributes
    assert embed_model.model_name == "TEI"

@pytest.mark.asyncio
async def test_aget_query_embedding(mock_tei_client):
    embed_model = TEIEmbedding(endpoint_url="http://mock:80")
    # Inject mock client
    embed_model._client = mock_tei_client
    
    embedding = await embed_model._aget_query_embedding("test")
    assert len(embedding) == 1024
    mock_tei_client.embed_single.assert_called_with("test")

@pytest.mark.asyncio
async def test_aget_text_embeddings(mock_tei_client):
    embed_model = TEIEmbedding(endpoint_url="http://mock:80")
    embed_model._client = mock_tei_client
    
    embeddings = await embed_model._aget_text_embeddings(["test1", "test2"])
    assert len(embeddings) == 2
    mock_tei_client.embed_batch.assert_called_with(["test1", "test2"])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_llama_embeddings.py`
Expected: FAIL (ModuleNotFoundError or ImportError)

**Step 3: Write minimal implementation**

```python
# crawl4r/storage/llama_embeddings.py
from typing import Any, List
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from crawl4r.storage.tei import TEIClient
import asyncio

class TEIEmbedding(BaseEmbedding):
    """LlamaIndex wrapper for TEIClient with circuit breaker support."""
    
    _client: TEIClient = PrivateAttr()

    def __init__(
        self,
        endpoint_url: str,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="TEI", **kwargs)
        self._client = TEIClient(endpoint_url=endpoint_url, timeout=timeout)

    def _get_query_embedding(self, query: str) -> List[float]:
        return asyncio.run(self._client.embed_single(query))

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._client.embed_single(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return asyncio.run(self._client.embed_single(text))

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._client.embed_single(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return asyncio.run(self._client.embed_batch(texts))
    
    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._client.embed_batch(texts)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_llama_embeddings.py`
Expected: PASS

**Step 5: Commit**

```bash
git add crawl4r/storage/llama_embeddings.py tests/unit/test_llama_embeddings.py
git commit -m "feat(storage): add TEIEmbedding wrapper for LlamaIndex"
```

---

### Task 2: CustomMarkdownNodeParser

**Files:**
- Create: `crawl4r/processing/llama_parser.py`
- Create: `tests/unit/test_llama_parser.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_llama_parser.py`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# crawl4r/processing/llama_parser.py
from typing import List, Sequence, Any
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, TextNode, Document
from llama_index.core.bridge.pydantic import PrivateAttr
from crawl4r.processing.chunker import MarkdownChunker

class CustomMarkdownNodeParser(NodeParser):
    """Node parser that uses MarkdownChunker logic."""
    
    _chunker: MarkdownChunker = PrivateAttr()

    def __init__(self, chunker: MarkdownChunker = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._chunker = chunker or MarkdownChunker()

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        out_nodes = []
        for node in nodes:
            # Get filename from metadata or fallback
            filename = node.metadata.get("filename", "unknown.md")
            
            # Use existing chunker
            chunks = self._chunker.chunk(node.get_content(), filename=filename)
            
            for chunk in chunks:
                # Merge original metadata with chunk metadata
                metadata = node.metadata.copy()
                metadata.update({
                    "chunk_index": chunk["chunk_index"],
                    "section_path": chunk["section_path"],
                    "heading_level": chunk["heading_level"],
                })
                if chunk["tags"]:
                    metadata["tags"] = chunk["tags"]

                # Create TextNode
                text_node = TextNode(
                    text=chunk["chunk_text"],
                    metadata=metadata,
                    excluded_embed_metadata_keys=["chunk_index", "heading_level", "filename", "tags"],
                    excluded_llm_metadata_keys=["chunk_index", "heading_level"],
                )
                out_nodes.append(text_node)
                
        return out_nodes
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_llama_parser.py`
Expected: PASS

**Step 5: Commit**

```bash
git add crawl4r/processing/llama_parser.py tests/unit/test_llama_parser.py
git commit -m "feat(processing): add CustomMarkdownNodeParser"
```

---

### Task 3: Refactor DocumentProcessor to use IngestionPipeline

**Files:**
- Modify: `crawl4r/processing/processor.py`
- Modify: `tests/unit/test_processor.py` (to adapt to new dependencies)

**Step 1: Write the failing test**

Modify `tests/unit/test_processor.py` to assert that `IngestionPipeline` is used. Or simply run existing processor tests which should fail after we modify the processor (if we change logic).

Better: Create a temporary test `tests/unit/test_processor_pipeline.py` to verify the new integration logic before replacing the old one.

```python
# tests/unit/test_processor_pipeline.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from crawl4r.processing.processor import DocumentProcessor
from llama_index.core.ingestion import IngestionPipeline

@pytest.mark.asyncio
async def test_process_document_uses_pipeline(tmp_path):
    # Setup mocks
    config = MagicMock()
    config.watch_folder = tmp_path
    tei_client = MagicMock()
    vector_store_manager = MagicMock() # The legacy manager
    chunker = MagicMock()
    
    # Create a dummy file
    doc_path = tmp_path / "test.md"
    doc_path.write_text("# Test")
    
    with patch("crawl4r.processing.processor.IngestionPipeline") as MockPipeline:
        pipeline_instance = MockPipeline.return_value
        pipeline_instance.arun = AsyncMock(return_value=[])
        
        processor = DocumentProcessor(config, tei_client, vector_store_manager, chunker)
        
        # We need to manually inject the pipeline or ensure init creates it
        # Ideally, we pass it or the processor builds it.
        # For this refactor, let's assume processor builds it internally using the helpers.
        
        await processor.process_document(doc_path)
        
        # assert pipeline_instance.run.called or pipeline_instance.arun.called
        assert pipeline_instance.arun.called
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_processor_pipeline.py`
Expected: FAIL (Pipeline not yet integrated)

**Step 3: Write implementation**

Modify `crawl4r/processing/processor.py`:

1. Import `IngestionPipeline`, `QdrantVectorStore`, `Document`.
2. Import `TEIEmbedding` and `CustomMarkdownNodeParser`.
3. In `__init__`, initialize `IngestionPipeline`.
   - Use `vector_store_manager.client` (QdrantClient) to init `QdrantVectorStore`.
   - Setup transformations: `[CustomMarkdownNodeParser(chunker), TEIEmbedding(tei_client)]`.
4. In `process_document`, replace manual logic with:
   - Load text.
   - Create `Document` object.
   - Run `await self.pipeline.arun(documents=[doc])`.
5. Map results back to `ProcessingResult`.

```python
# Key parts of modification in crawl4r/processing/processor.py

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.qdrant import QdrantVectorStore
from crawl4r.storage.llama_embeddings import TEIEmbedding
from crawl4r.processing.llama_parser import CustomMarkdownNodeParser

class DocumentProcessor:
    def __init__(self, config, tei_client, vector_store, chunker):
        self.config = config
        self.tei_client = tei_client
        self.vector_store = vector_store
        self.chunker = chunker
        
        # Initialize LlamaIndex components
        self.embed_model = TEIEmbedding(
            endpoint_url=config.tei_endpoint,
            timeout=30.0 # From config if available
        )
        self.node_parser = CustomMarkdownNodeParser(chunker=chunker)
        
        # Initialize QdrantVectorStore using the existing client
        self.llama_vector_store = QdrantVectorStore(
            client=vector_store.client,
            collection_name=config.collection_name
        )
        
        self.pipeline = IngestionPipeline(
            transformations=[self.node_parser, self.embed_model],
            vector_store=self.llama_vector_store,
        )

    async def process_document(self, file_path: Path) -> ProcessingResult:
        start_time = time.time()
        try:
            # ... validation logic ...
            content = await self._load_markdown_file(file_path)
            
            # Metadata construction (similar to before, for the Document)
            stat = file_path.stat()
            metadata = {
                "file_path_relative": str(file_path.relative_to(self.config.watch_folder)),
                # ... other metadata ...
            }
            
            doc = Document(text=content, metadata=metadata)
            
            # Run pipeline
            nodes = await self.pipeline.arun(documents=[doc])
            
            return ProcessingResult(
                success=True,
                chunks_processed=len(nodes),
                # ...
            )
        except Exception as e:
            # ... error handling ...
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_processor_pipeline.py`
Expected: PASS

**Step 5: Commit**

```bash
git add crawl4r/processing/processor.py tests/unit/test_processor_pipeline.py
git commit -m "refactor(processing): integrate IngestionPipeline"
```

---

### Task 4: Integration Verification

**Files:**
- Test: `tests/integration/test_e2e_pipeline.py` (Run existing tests)

**Step 1: Run existing integration tests**

Run: `pytest tests/integration/test_e2e_pipeline.py`
Expected: PASS (if refactor is correct, functionality should be preserved)

**Step 2: Fix any regressions**

If tests fail, debug and fix in `processor.py`, `llama_parser.py`, or `llama_embeddings.py`.

**Step 3: Commit fixes**

```bash
git commit -am "fix(integration): resolve regressions from llamaindex refactor"
```

```