# Crawl4r Enhancement Recommendations

A comprehensive analysis of potential enhancements and new features for the Crawl4r RAG ingestion pipeline, based on LlamaIndex capabilities and modern RAG best practices.

**Generated:** 2026-01-17
**Current Version:** 0.1.0
**Analysis Scope:** LlamaIndex framework, workflows, and cloud services

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Priority 1: High-Impact Enhancements](#priority-1-high-impact-enhancements)
3. [Priority 2: Advanced Retrieval Features](#priority-2-advanced-retrieval-features)
4. [Priority 3: Query & Response Layer](#priority-3-query--response-layer)
5. [Priority 4: Observability & Evaluation](#priority-4-observability--evaluation)
6. [Priority 5: Advanced Processing](#priority-5-advanced-processing)
7. [Priority 6: Agent & Tool Integration](#priority-6-agent--tool-integration)
8. [Priority 7: Workflow Orchestration](#priority-7-workflow-orchestration)
9. [Priority 8: Cloud & Enterprise Features](#priority-8-cloud--enterprise-features)
10. [Implementation Roadmap](#implementation-roadmap)
11. [Appendix: Current Architecture](#appendix-current-architecture)

---

## Executive Summary

Crawl4r currently implements a solid foundation for RAG ingestion with:
- ✅ Custom `Crawl4AIReader` for web crawling
- ✅ `FileWatcher` for local markdown monitoring
- ✅ `MarkdownNodeParser` (LlamaIndex built-in) for markdown parsing
- ✅ `TEIClient` for custom embedding provider
- ✅ `VectorStoreManager` for Qdrant operations
- ✅ `IngestionPipeline` integration
- ✅ Circuit breaker pattern for fault tolerance
- ✅ State recovery for restart resilience

**Key Enhancement Opportunities:**

| Category | Impact | Effort | Priority |
|----------|--------|--------|----------|
| Hybrid Search (Dense + Sparse) | High | Medium | P1 |
| Reranking Pipeline | High | Low | P1 |
| Metadata Extraction | High | Medium | P1 |
| Query Engine Layer | Critical | Medium | P2 |
| Observability/Tracing | High | Low | P2 |
| Knowledge Graphs | Medium | High | P3 |
| MCP Server Exposure | Medium | Medium | P3 |
| Workflows Migration | Medium | High | P4 |

---

## Priority 1: High-Impact Enhancements

### 1.1 Hybrid Search with Sparse Embeddings

**Current State:** Dense embeddings only (Qwen3-Embedding-0.6B, 1024 dims)

**Enhancement:** Add sparse embedding support for hybrid retrieval combining semantic search with keyword matching.

**LlamaIndex Support:**
- Qdrant native hybrid search (`llama-index-vector-stores-qdrant`)
- BM42 sparse embeddings (Qdrant's lightweight approach)
- SPLADE sparse embeddings
- Reciprocal Rank Fusion for combining results

**Implementation:**

```python
# Example: Hybrid retrieval with Qdrant
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import models

# Enable sparse vectors in collection
vector_store = QdrantVectorStore(
    client=client,
    collection_name="crawl4r_hybrid",
    enable_hybrid=True,
    sparse_doc_fn=lambda doc: generate_sparse_vector(doc),  # BM42 or SPLADE
    sparse_query_fn=lambda query: generate_sparse_vector(query),
)
```

**Benefits:**
- 15-25% improvement in retrieval accuracy for keyword-heavy queries
- Better handling of technical terms, code snippets, proper nouns
- Fallback when semantic similarity fails

**Files to Modify:**
- `crawl4r.storage.qdrant.py` - Add hybrid vector support
- `crawl4r/storage/sparse_embeddings.py` - New file for sparse embedding generation
- `crawl4r/core/config.py` - Add hybrid search configuration

---

### 1.2 Reranking Pipeline

**Current State:** No reranking - uses raw similarity scores

**Enhancement:** Add post-retrieval reranking to improve result quality.

**LlamaIndex Support:**
- `CohereRerank` - Cohere Rerank API
- `SentenceTransformerRerank` - Local cross-encoder models
- `ColBERTRerank` - Late interaction models
- `LLMRerank` - LLM-based reranking
- `JinaRerank` - Jina AI reranker

**Implementation:**

```python
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank

# Local reranker (no API calls)
local_reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=10,
)

# Or Cohere for higher quality
cohere_reranker = CohereRerank(
    api_key=os.environ["COHERE_API_KEY"],
    top_n=5,
)
```

**Benefits:**
- 10-30% improvement in retrieval precision
- Better handling of nuanced queries
- Can use smaller initial retrieval set (faster) then rerank

**Files to Create:**
- `crawl4r/retrieval/reranker.py` - Reranking abstractions
- `crawl4r/retrieval/__init__.py` - Retrieval module

---

### 1.3 Automatic Metadata Extraction

**Current State:** Basic metadata (source_url, title, chunk_index, section_path)

**Enhancement:** LLM-powered metadata extraction during ingestion.

**LlamaIndex Support:**
- `TitleExtractor` - Document title extraction
- `SummaryExtractor` - Section/document summaries
- `QuestionsAnsweredExtractor` - Questions the chunk can answer
- `KeywordExtractor` - Key terms extraction
- `EntityExtractor` - Named entity recognition
- Custom extractors via `BaseExtractor`

**Implementation:**

```python
from llama_index.core.extractors import (
    TitleExtractor,
    SummaryExtractor,
    KeywordExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.extractors.entity import EntityExtractor

# Add to ingestion pipeline
pipeline = IngestionPipeline(
    transformations=[
        CustomMarkdownNodeParser(),
        TitleExtractor(nodes=5, llm=local_llm),
        KeywordExtractor(keywords=10, llm=local_llm),
        QuestionsAnsweredExtractor(questions=3, llm=local_llm),
        EntityExtractor(prediction_threshold=0.5),  # Uses local NER model
        TEIEmbedding(),
    ],
    vector_store=vector_store,
)
```

**Benefits:**
- Richer metadata for filtering and routing
- Better retrieval through metadata matching
- Enables question-based retrieval strategies

**Files to Modify:**
- `crawl4r/processing/processor.py` - Add metadata extractors to pipeline
- `crawl4r/processing/extractors.py` - Custom extractors (e.g., CodeExtractor, URLExtractor)

---

### 1.4 Ingestion Pipeline Caching

**Current State:** No transformation caching - reprocesses all documents

**Enhancement:** Leverage LlamaIndex's built-in caching for ingestion pipelines.

**LlamaIndex Support:**
- `IngestionCache` with multiple backends
- `RedisCache` - Remote distributed cache
- `MongoDBCache` - Document-based cache
- Local file-based cache

**Implementation:**

```python
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache

# Use Redis for distributed caching (already have Redis in stack)
ingest_cache = IngestionCache(
    cache=RedisCache.from_host_and_port(
        host="crawl4r-cache",
        port=6379,
    ),
    collection="crawl4r_ingestion_cache",
)

pipeline = IngestionPipeline(
    transformations=[...],
    cache=ingest_cache,
)
```

**Benefits:**
- Skip reprocessing unchanged documents
- Faster incremental updates
- Reduced TEI API calls (cost/latency savings)

**Files to Modify:**
- `crawl4r/processing/processor.py` - Add cache configuration
- `crawl4r/core/config.py` - Add cache settings

---

## Priority 2: Advanced Retrieval Features

### 2.1 Query Transformations

**Current State:** Direct query to vector store

**Enhancement:** Transform queries before retrieval for better results.

**LlamaIndex Support:**
- `HyDEQueryTransform` - Hypothetical Document Embeddings
- `StepDecomposeQueryTransform` - Break complex queries into steps
- `MultiStepQueryEngine` - Iterative query refinement

**Implementation:**

```python
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

# HyDE: Generate hypothetical answer, then search for similar content
hyde_transform = HyDEQueryTransform(llm=local_llm, include_original=True)
query_engine = TransformQueryEngine(
    base_query_engine,
    query_transform=hyde_transform,
)
```

**Benefits:**
- Better handling of vague or abstract queries
- Improved recall for complex questions
- Enables query expansion/reformulation

---

### 2.2 Auto-Merging Retriever

**Current State:** Flat chunk retrieval

**Enhancement:** Hierarchical retrieval with automatic parent merging.

**LlamaIndex Support:**
- `AutoMergingRetriever` - Merges leaf nodes into parent nodes
- `HierarchicalNodeParser` - Creates parent-child node relationships
- Configurable merge thresholds

**Implementation:**

```python
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.retrievers import AutoMergingRetriever

# Create hierarchical nodes
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128],  # Parent → Child → Leaf
)

# Auto-merge during retrieval
retriever = AutoMergingRetriever(
    vector_retriever,
    storage_context,
    simple_ratio_thresh=0.5,  # Merge if >50% of parent's children retrieved
)
```

**Benefits:**
- Better context preservation for long documents
- Reduces "lost in the middle" problem
- More coherent retrieved passages

---

### 2.3 Sentence Window Retrieval

**Current State:** Fixed-size chunks with overlap

**Enhancement:** Retrieve small chunks, return with surrounding context.

**LlamaIndex Support:**
- `SentenceWindowNodeParser` - Single sentence nodes with window metadata
- `MetadataReplacementPostProcessor` - Replace node with windowed context

**Implementation:**

```python
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# Parse into sentence windows
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,  # 3 sentences on each side
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# At query time, expand to full window
postprocessor = MetadataReplacementPostProcessor(
    target_metadata_key="window",
)
```

**Benefits:**
- More precise retrieval (sentence-level)
- Full context available for response synthesis
- Better handling of fine-grained queries

---

### 2.4 Fusion Retrieval

**Current State:** Single retrieval strategy

**Enhancement:** Combine multiple retrieval strategies with result fusion.

**LlamaIndex Support:**
- `QueryFusionRetriever` - Multiple sub-retrievers with fusion
- Reciprocal Rank Fusion (RRF)
- Relative Score Fusion
- Distribution-Based Score Fusion

**Implementation:**

```python
from llama_index.core.retrievers import QueryFusionRetriever

# Combine vector, keyword, and knowledge graph retrieval
fusion_retriever = QueryFusionRetriever(
    retrievers=[
        vector_retriever,
        bm25_retriever,
        kg_retriever,
    ],
    similarity_top_k=10,
    num_queries=4,  # Generate 4 query variations
    mode="reciprocal_rerank",
    use_async=True,
)
```

**Benefits:**
- Combines strengths of different retrieval methods
- More robust to query variations
- Better recall across diverse document types

---

## Priority 3: Query & Response Layer

### 3.1 Query Engine Implementation

**Current State:** Ingestion only - no query interface

**Enhancement:** Full query engine for RAG responses.

**LlamaIndex Support:**
- `VectorStoreIndex.as_query_engine()` - Basic query engine
- `RetrieverQueryEngine` - Custom retriever + response synthesis
- Multiple response modes (refine, compact, tree_summarize)

**Implementation:**

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

# Create query engine from existing vector store
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker, similarity_filter],
    response_mode="compact",  # Combine chunks before LLM call
)

# Or with custom retriever
retriever = fusion_retriever
synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
    llm=local_llm,
)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
)
```

**Files to Create:**
- `crawl4r/query/__init__.py`
- `crawl4r/query/engine.py` - Query engine implementations
- `crawl4r/query/synthesizer.py` - Response synthesis configurations

---

### 3.2 Chat Engine with Memory

**Current State:** No conversational interface

**Enhancement:** Multi-turn chat with conversation memory.

**LlamaIndex Support:**
- `ContextChatEngine` - Chat with retrieved context
- `CondensePlusContextChatEngine` - Condense follow-ups into standalone queries
- `ChatMemoryBuffer` - Conversation history management
- Multiple chat stores (Redis, MongoDB, Simple)

**Implementation:**

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.storage.chat_store.redis import RedisChatStore

# Use Redis for distributed chat history
chat_store = RedisChatStore(redis_url="redis://crawl4r-cache:6379")
memory = ChatMemoryBuffer.from_defaults(
    token_limit=4096,
    chat_store=chat_store,
    chat_store_key="user_session_123",
)

chat_engine = CondensePlusContextChatEngine.from_defaults(
    retriever=fusion_retriever,
    memory=memory,
    llm=local_llm,
    context_prompt="Use the following context to answer...",
)

# Multi-turn conversation
response = chat_engine.chat("What is Crawl4r?")
response = chat_engine.chat("How does it handle failures?")  # Uses context
```

**Files to Create:**
- `crawl4r/chat/__init__.py`
- `crawl4r/chat/engine.py` - Chat engine implementations
- `crawl4r/chat/memory.py` - Memory management

---

### 3.3 Streaming Responses

**Current State:** No streaming support

**Enhancement:** Stream tokens as they're generated.

**LlamaIndex Support:**
- `query_engine.query(..., streaming=True)` - Streaming query
- `chat_engine.stream_chat()` - Streaming chat
- Async streaming with `astream_chat()`

**Implementation:**

```python
# Streaming query
streaming_response = query_engine.query("Explain the pipeline", streaming=True)
for text in streaming_response.response_gen:
    print(text, end="", flush=True)

# Streaming chat
streaming_response = chat_engine.stream_chat("Tell me about chunking")
for token in streaming_response.response_gen:
    yield token  # For FastAPI StreamingResponse
```

**Files to Modify:**
- `crawl4r/api/routes/query.py` - Add streaming endpoints
- `crawl4r/api/routes/chat.py` - Add streaming chat endpoints

---

## Priority 4: Observability & Evaluation

### 4.1 Tracing & Instrumentation

**Current State:** Basic structured logging

**Enhancement:** Full observability with tracing.

**LlamaIndex Support:**
- Built-in callback system
- OpenTelemetry integration
- Langfuse integration
- Arize Phoenix integration
- Custom `CallbackManager`

**Implementation:**

```python
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings

# Debug handler for development
debug_handler = LlamaDebugHandler(print_trace_on_end=True)

# OpenTelemetry for production
from opentelemetry import trace
from llama_index.callbacks.opentelemetry import OpenTelemetryCallbackHandler

otel_handler = OpenTelemetryCallbackHandler(
    tracer=trace.get_tracer("crawl4r"),
)

Settings.callback_manager = CallbackManager([debug_handler, otel_handler])
```

**Benefits:**
- Debug retrieval and synthesis steps
- Measure latency per component
- Track token usage and costs

**Files to Create:**
- `crawl4r/observability/__init__.py`
- `crawl4r/observability/tracing.py` - Tracing configuration
- `crawl4r/observability/callbacks.py` - Custom callbacks

---

### 4.2 RAG Evaluation Framework

**Current State:** No evaluation metrics

**Enhancement:** Automated RAG quality evaluation.

**LlamaIndex Support:**
- `FaithfulnessEvaluator` - Checks if response is grounded in context
- `RelevancyEvaluator` - Checks context relevance to query
- `CorrectnessEvaluator` - Checks response correctness
- `BatchEvalRunner` - Run evaluations at scale
- DeepEval integration

**Implementation:**

```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner,
)

# Create evaluators
faithfulness = FaithfulnessEvaluator(llm=eval_llm)
relevancy = RelevancyEvaluator(llm=eval_llm)

# Run batch evaluation
runner = BatchEvalRunner(
    evaluators={
        "faithfulness": faithfulness,
        "relevancy": relevancy,
    },
    workers=4,
)

eval_results = await runner.aevaluate_queries(
    query_engine,
    queries=test_queries,
    reference_answers=expected_answers,  # Optional
)
```

**Files to Create:**
- `crawl4r/evaluation/__init__.py`
- `crawl4r/evaluation/evaluators.py` - Evaluation implementations
- `crawl4r/evaluation/datasets.py` - Test dataset management

---

### 4.3 Cost Tracking

**Current State:** No cost visibility

**Enhancement:** Track LLM and embedding API costs.

**LlamaIndex Support:**
- Token counting callbacks
- Cost calculation per model
- Usage aggregation

**Implementation:**

```python
from llama_index.core.callbacks import TokenCountingHandler
import tiktoken

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-4").encode,
)

# After queries
print(f"Embedding tokens: {token_counter.total_embedding_token_count}")
print(f"LLM prompt tokens: {token_counter.prompt_llm_token_count}")
print(f"LLM completion tokens: {token_counter.completion_llm_token_count}")
```

---

## Priority 5: Advanced Processing

### 5.1 Semantic Chunking

**Current State:** Heading-based chunking with fixed overlap

**Enhancement:** Semantically-aware chunk boundaries.

**LlamaIndex Support:**
- `SemanticSplitterNodeParser` - Split on semantic boundaries
- Uses embeddings to detect topic shifts
- Configurable breakpoint thresholds

**Implementation:**

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser

semantic_splitter = SemanticSplitterNodeParser(
    buffer_size=1,  # Sentences to group for embedding
    breakpoint_percentile_threshold=95,  # Higher = fewer splits
    embed_model=embed_model,
)
```

**Benefits:**
- More coherent chunks
- Better topic separation
- Reduced mid-sentence/mid-thought breaks

---

### 5.2 Multimodal Support

**Current State:** Text-only processing

**Enhancement:** Process images, tables, and code within documents.

**LlamaIndex Support:**
- `ImageNode` - Image content nodes
- Multi-modal LLMs (GPT-4V, Claude 3, Gemini)
- Image embedding models
- `SimpleDirectoryReader` with image support

**Implementation:**

```python
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

# For documents with images
mm_llm = OpenAIMultiModal(model="gpt-4-vision-preview")

# Extract image descriptions during ingestion
class ImageDescriptionExtractor(BaseExtractor):
    async def aextract(self, nodes):
        metadata_list = []
        for node in nodes:
            if isinstance(node, ImageNode):
                description = await mm_llm.acomplete(
                    prompt="Describe this image:",
                    image_documents=[node],
                )
                metadata_list.append({"image_description": description.text})
            else:
                metadata_list.append({})
        return metadata_list
```

**Files to Create:**
- `crawl4r/processing/multimodal.py` - Multimodal processing
- `crawl4r/readers/image_reader.py` - Image content reader

---

### 5.3 Code-Aware Processing

**Current State:** Code treated as plain text

**Enhancement:** Specialized handling for code blocks.

**Implementation:**

```python
from llama_index.core.node_parser import CodeSplitter

# Language-aware code splitting
code_splitter = CodeSplitter(
    language="python",
    chunk_lines=40,
    chunk_lines_overlap=15,
    max_chars=1500,
)

# Custom metadata for code blocks
class CodeMetadataExtractor(BaseExtractor):
    async def aextract(self, nodes):
        for node in nodes:
            if "```" in node.text:
                # Extract language, function names, imports
                node.metadata["contains_code"] = True
                node.metadata["languages"] = detect_languages(node.text)
```

---

### 5.4 Parallel Processing Enhancement

**Current State:** Basic multiprocessing support

**Enhancement:** Optimized parallel processing with async batching.

**LlamaIndex Support:**
- `IngestionPipeline.run(num_workers=N)` - Parallel processing
- Async transformations
- Batch processing optimizations

**Implementation:**

```python
# Parallel ingestion with worker pool
pipeline = IngestionPipeline(
    transformations=[...],
    vector_store=vector_store,
)

# Process with 8 workers
nodes = pipeline.run(
    documents=documents,
    num_workers=8,
    show_progress=True,
)

# Or async with batching
nodes = await pipeline.arun(
    documents=documents,
    batch_size=100,
)
```

---

## Priority 6: Agent & Tool Integration

### 6.1 RAG as Agent Tool

**Current State:** Pipeline-only, no agent integration

**Enhancement:** Expose RAG as tools for LLM agents.

**LlamaIndex Support:**
- `QueryEngineTool` - Wrap query engine as tool
- `RetrieverTool` - Wrap retriever as tool
- `FunctionTool` - Custom function tools
- Agent frameworks (ReAct, Function Calling)

**Implementation:**

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

# Create tool from query engine
crawl4r_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="crawl4r_search",
        description="Search through crawled web documents and local files. "
                    "Use for questions about documentation, articles, or technical content.",
    ),
)

# Create agent with tool
agent = ReActAgent.from_tools(
    tools=[crawl4r_tool, other_tools...],
    llm=local_llm,
    verbose=True,
)

response = agent.chat("Find information about circuit breakers in the docs")
```

**Files to Create:**
- `crawl4r/agents/__init__.py`
- `crawl4r/agents/tools.py` - Tool definitions
- `crawl4r/agents/react.py` - ReAct agent configuration

---

### 6.2 MCP Server Exposure

**Current State:** No MCP support

**Enhancement:** Expose Crawl4r as an MCP server for use with Claude, IDEs, etc.

**LlamaIndex Support:**
- `workflow_as_mcp()` - Convert workflow to MCP server
- `tool_as_mcp()` - Convert tool to MCP
- FastMCP integration

**Implementation:**

```python
from llama_index.tools.mcp import tool_as_mcp, McpToolSpec
from fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("crawl4r")

# Expose query tool
@mcp.tool()
async def search_documents(query: str, top_k: int = 5) -> str:
    """Search through crawled documents."""
    response = await query_engine.aquery(query)
    return response.response

# Expose ingestion tool
@mcp.tool()
async def crawl_url(url: str) -> str:
    """Crawl a URL and add to knowledge base."""
    docs = await reader.aload_data([url])
    await pipeline.arun(documents=docs)
    return f"Successfully crawled {url}"

# Run server
mcp.run()
```

**Benefits:**
- Use Crawl4r from Claude Desktop
- IDE integration (VS Code, Cursor)
- Interoperability with other MCP clients

**Files to Create:**
- `crawl4r/mcp/__init__.py`
- `crawl4r/mcp/server.py` - MCP server implementation
- `crawl4r/mcp/tools.py` - MCP tool definitions

---

### 6.3 Multi-Document Agents

**Current State:** Flat document retrieval

**Enhancement:** Per-document agents with routing.

**LlamaIndex Support:**
- `ObjectIndex` - Index of tools/agents
- `RouterQueryEngine` - Route queries to appropriate engine
- `SubQuestionQueryEngine` - Break into sub-questions

**Implementation:**

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

# Create per-source tools
tools = [
    QueryEngineTool.from_defaults(
        query_engine=docs_engine,
        description="Documentation and guides",
    ),
    QueryEngineTool.from_defaults(
        query_engine=blog_engine,
        description="Blog posts and articles",
    ),
    QueryEngineTool.from_defaults(
        query_engine=code_engine,
        description="Code examples and repositories",
    ),
]

# Sub-question decomposition
sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=tools,
    llm=local_llm,
)
```

---

## Priority 7: Workflow Orchestration

### 7.1 LlamaIndex Workflows Migration

**Current State:** Custom async orchestration in `cli/main.py`

**Enhancement:** Migrate to LlamaIndex Workflows for better orchestration.

**LlamaIndex Support:**
- Event-driven workflow engine
- `@step` decorators for handlers
- Built-in state management
- Parallel execution (fan-in/fan-out)
- Human-in-the-loop patterns
- Retry and error handling

**Implementation:**

```python
from llama_index.core.workflow import Workflow, Event, StartEvent, StopEvent, step

class CrawlEvent(Event):
    urls: list[str]

class ChunkEvent(Event):
    documents: list[Document]

class EmbedEvent(Event):
    nodes: list[TextNode]

class IngestionWorkflow(Workflow):
    @step
    async def crawl(self, ev: StartEvent) -> CrawlEvent:
        urls = ev.urls
        documents = await self.reader.aload_data(urls)
        return CrawlEvent(documents=documents)

    @step
    async def chunk(self, ev: CrawlEvent) -> EmbedEvent:
        nodes = self.node_parser.get_nodes_from_documents(ev.documents)
        return EmbedEvent(nodes=nodes)

    @step
    async def embed_and_store(self, ev: EmbedEvent) -> StopEvent:
        # Batch embed and store
        await self.vector_store.async_add(ev.nodes)
        return StopEvent(result={"nodes_processed": len(ev.nodes)})

# Run workflow
workflow = IngestionWorkflow()
result = await workflow.run(urls=["https://example.com"])
```

**Benefits:**
- Cleaner orchestration logic
- Built-in retry/error handling
- Visual workflow debugging
- Easier testing and modification

**Files to Create:**
- `crawl4r/workflows/__init__.py`
- `crawl4r/workflows/ingestion.py` - Ingestion workflow
- `crawl4r/workflows/query.py` - Query workflow
- `crawl4r/workflows/events.py` - Custom events

---

### 7.2 Durable Workflows

**Current State:** State recovery via Qdrant queries

**Enhancement:** Persistent workflow state for long-running operations.

**LlamaIndex Support:**
- `DurableWorkflow` - Checkpoint-based persistence
- Resume from failures
- Distributed execution support

**Implementation:**

```python
from llama_index.core.workflow import DurableWorkflow

class DurableIngestionWorkflow(DurableWorkflow):
    @step(checkpoint=True)  # Save state after this step
    async def crawl(self, ev: StartEvent) -> CrawlEvent:
        ...

    @step(checkpoint=True)
    async def chunk(self, ev: CrawlEvent) -> EmbedEvent:
        ...

# Resume from checkpoint after failure
workflow = DurableIngestionWorkflow(checkpoint_dir="./checkpoints")
result = await workflow.run(resume=True)
```

---

### 7.3 Human-in-the-Loop

**Current State:** Fully automated

**Enhancement:** Add human approval steps for quality control.

**LlamaIndex Support:**
- `HumanInputEvent` - Pause for human input
- `InputRequiredEvent` - Request user decision
- Integration with messaging systems

**Implementation:**

```python
from llama_index.core.workflow import InputRequiredEvent

class QualityCheckWorkflow(Workflow):
    @step
    async def check_quality(self, ev: ChunkEvent) -> InputRequiredEvent | EmbedEvent:
        low_quality = [n for n in ev.nodes if len(n.text) < 50]
        if low_quality:
            return InputRequiredEvent(
                prefix="Low quality chunks detected",
                data={"chunks": low_quality},
            )
        return EmbedEvent(nodes=ev.nodes)
```

---

## Priority 8: Cloud & Enterprise Features

### 8.1 LlamaParse Integration

**Current State:** Custom markdown parsing

**Enhancement:** Use LlamaParse for complex document formats.

**LlamaIndex Support:**
- PDF parsing with layout understanding
- Table extraction
- Image extraction
- Code block detection
- Multi-modal parsing

**Implementation:**

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    result_type="markdown",
    parsing_instruction="Extract all code blocks with language tags",
    use_vendor_multimodal_model=True,
)

documents = await parser.aload_data("./complex_document.pdf")
```

**Use Cases:**
- PDF documentation
- Scanned documents
- Complex tables and figures

---

### 8.2 LlamaCloud Index

**Current State:** Self-hosted Qdrant

**Enhancement:** Optional LlamaCloud managed index for scaling.

**LlamaIndex Support:**
- Managed vector store
- Automatic optimization
- Built-in retrieval pipelines

**Implementation:**

```python
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

# Create managed index
index = LlamaCloudIndex.from_documents(
    documents,
    name="crawl4r-production",
    project_name="crawl4r",
)

# Query with managed retrieval
query_engine = index.as_query_engine()
```

---

### 8.3 Multi-Tenancy

**Current State:** Single collection

**Enhancement:** Tenant isolation for multi-user deployments.

**Implementation:**

```python
# Tenant-specific collections
class TenantVectorStore:
    def __init__(self, tenant_id: str):
        self.collection_name = f"crawl4r_{tenant_id}"
        self.vector_store = VectorStoreManager(
            collection_name=self.collection_name,
            qdrant_url=settings.QDRANT_URL,
        )

    async def query(self, query: str, filters: dict = None):
        # Add tenant filter automatically
        tenant_filter = {"tenant_id": self.tenant_id}
        combined_filters = {**tenant_filter, **(filters or {})}
        return await self.vector_store.query(query, filters=combined_filters)
```

---

### 8.4 Rate Limiting & Quotas

**Current State:** Basic circuit breaker

**Enhancement:** Per-tenant rate limiting and usage quotas.

**Implementation:**

```python
from redis import Redis
from datetime import timedelta

class RateLimiter:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def check_limit(self, tenant_id: str, limit: int = 1000) -> bool:
        key = f"rate_limit:{tenant_id}:{datetime.now().strftime('%Y-%m-%d')}"
        current = await self.redis.incr(key)
        if current == 1:
            await self.redis.expire(key, timedelta(days=1))
        return current <= limit
```

---

## Implementation Roadmap

### Phase 1: Foundation (2-3 weeks)
1. **Query Engine Layer** - Enable RAG responses
2. **Reranking Pipeline** - Improve retrieval quality
3. **Ingestion Caching** - Reduce reprocessing
4. **Basic Tracing** - Observability foundation

### Phase 2: Advanced Retrieval (2-3 weeks)
1. **Hybrid Search** - Dense + sparse embeddings
2. **Metadata Extraction** - Richer document metadata
3. **Fusion Retrieval** - Multiple retrieval strategies
4. **Auto-Merging** - Hierarchical retrieval

### Phase 3: Conversational (2 weeks)
1. **Chat Engine** - Multi-turn conversations
2. **Streaming** - Token-by-token responses
3. **Memory Management** - Conversation persistence

### Phase 4: Agents & Tools (2 weeks)
1. **RAG Tools** - Agent-compatible tools
2. **MCP Server** - External integration
3. **Multi-Document Agents** - Source routing

### Phase 5: Enterprise (3-4 weeks)
1. **Workflows Migration** - Better orchestration
2. **Evaluation Framework** - Quality metrics
3. **Multi-Tenancy** - User isolation
4. **LlamaCloud Integration** - Optional managed services

---

## Appendix: Current Architecture

### Existing Components

```
crawl4r/
├── core/
│   ├── config.py          # Pydantic Settings (✓ complete)
│   ├── logger.py          # Structured logging (✓ complete)
│   └── quality.py         # Startup validation (✓ complete)
├── readers/
│   ├── crawl4ai.py        # Web crawler reader (✓ complete)
│   └── file_watcher.py    # File system monitor (✓ complete)
├── processing/
│   └── processor.py       # Pipeline orchestration using MarkdownNodeParser (✓ complete)
├── storage/
│   ├── embeddings.py      # TEI client (✓ complete)
│   ├── llama_embeddings.py# BaseEmbedding wrapper (✓ complete)
│   └── vector_store.py    # Qdrant manager (✓ complete)
├── resilience/
│   ├── circuit_breaker.py # Fault tolerance (✓ complete)
│   ├── failed_docs.py     # Failure logging (✓ complete)
│   └── recovery.py        # State recovery (✓ complete)
├── api/
│   └── app.py             # FastAPI skeleton (⚠ partial)
└── cli/
    └── main.py            # Entry point (✓ complete)
```

### Proposed New Components

```
crawl4r/
├── query/                 # NEW: Query layer
│   ├── engine.py          # Query engine implementations
│   ├── synthesizer.py     # Response synthesis
│   └── transformers.py    # Query transformations
├── chat/                  # NEW: Chat layer
│   ├── engine.py          # Chat engine implementations
│   └── memory.py          # Conversation memory
├── retrieval/             # NEW: Advanced retrieval
│   ├── reranker.py        # Reranking implementations
│   ├── fusion.py          # Fusion retrieval
│   └── hybrid.py          # Hybrid search
├── agents/                # NEW: Agent integration
│   ├── tools.py           # RAG as tools
│   └── react.py           # ReAct agent
├── mcp/                   # NEW: MCP server
│   ├── server.py          # MCP server implementation
│   └── tools.py           # MCP tool definitions
├── workflows/             # NEW: Workflow orchestration
│   ├── ingestion.py       # Ingestion workflow
│   └── query.py           # Query workflow
├── observability/         # NEW: Observability
│   ├── tracing.py         # Distributed tracing
│   └── callbacks.py       # Custom callbacks
└── evaluation/            # NEW: Evaluation
    ├── evaluators.py      # RAG evaluators
    └── datasets.py        # Test datasets
```

---

## References

- [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/)
- [LlamaIndex Workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/)
- [LlamaCloud Services](https://cloud.llamaindex.ai/)
- [Qdrant Hybrid Search](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
