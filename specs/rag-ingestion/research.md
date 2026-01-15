---
spec: rag-ingestion
phase: research
created: 2026-01-14T00:00:00Z
---

# Research: RAG Ingestion Pipeline

## Executive Summary

A RAG ingestion pipeline using LlamaIndex Python, HuggingFace Text Embeddings Inference (TEI) with Qwen3-Embedding-0.6B, and Qdrant is highly feasible. All components have mature Python integrations and well-documented APIs. The proposed stack leverages industry best practices: LlamaIndex for document orchestration, TEI for high-performance embedding generation with custom 1024-dimensional outputs, Qdrant for efficient vector storage, and Python's watchdog library for reliable file monitoring. The primary technical considerations are proper debouncing of file watch events, optimal chunking strategies for markdown documents, and custom embedding integration between TEI and LlamaIndex.

## External Research

### LlamaIndex Python Library

**Architecture & Best Practices**:
- LlamaIndex is a data framework for building LLM applications with support for data connectors, structured indices, and advanced retrieval interfaces
- **Event-Driven Architecture (v0.11+)**: The framework introduced Workflows, an event-driven, async-first architecture that replaced Query Pipelines for building complex RAG systems
- **Modular Design**: The library follows a "single responsibility" principle with support for nested workflows for complex applications
- **Package Structure**: Namespaced packages distinguish core functionality (`llama-index-core`) from integrations (`llama-index-vector-stores-qdrant`, `llama-index-embeddings-*`)
- **Recent Improvements (2024)**: 42% reduction in core package size, enhanced observability with instrumentation features, and property graph support

**Document Ingestion Patterns**:
- `SimpleDirectoryReader` is the primary document loader that automatically selects readers based on file extensions
- Native support for markdown files (.md) with automatic detection
- Supports recursive directory scanning with file type filtering via `required_exts` parameter
- Parallel processing available for loading many files
- Example usage:
  ```python
  from llama_index.core import SimpleDirectoryReader
  reader = SimpleDirectoryReader(
      input_dir="./data",
      required_exts=[".md"],
      recursive=True
  )
  documents = reader.load_data()
  ```

**Sources**:
- [LlamaIndex Python Documentation](https://docs.llamaindex.ai/)
- [Real Python LlamaIndex Guide](https://realpython.com/llamaindex-examples/)
- [LlamaIndex v0.11 Release](https://www.llamaindex.ai/blog/introducing-llamaindex-0-11)
- [SimpleDirectoryReader Documentation](https://developers.llamaindex.ai/python/framework/module_guides/loading/simpledirectoryreader/)

### HuggingFace Text Embeddings Inference (TEI)

**Setup & Configuration**:
- TEI is a high-performance toolkit for deploying and serving open-source text embedding models
- Supports streamlined deployment with small Docker images and rapid boot times for serverless
- Key optimizations: Flash Attention, Candle framework, cuBLASLt, token-based dynamic batching
- Recommended deployment with Docker (GPU or CPU):
  ```bash
  docker run --gpus all -p 8080:80 -v $PWD/data:/data --pull always \
    ghcr.io/huggingface/text-embeddings-inference:1.8 \
    --model-id Qwen/Qwen3-Embedding-0.6B
  ```

**Configuration Parameters**:
- `--max-concurrent-requests`: Default 128, can be lowered to 64 for better back pressure handling
- `--quantize`: Quantization method (defaults to model's config.json setting)
- Volume mounting recommended to avoid re-downloading model weights on each run

**API Endpoints**:
- `/embed` - Primary embedding endpoint (HTTP POST)
- `/v1/embeddings` - OpenAI-compatible endpoint
- `/rerank` - For cross-encoder re-ranking models
- `/predict` - For sequence classification

**Inference Methods**:
1. **cURL**: Direct HTTP requests
2. **HuggingFace InferenceClient**: Recommended Python SDK
3. **OpenAI SDK**: Compatible with OpenAI client libraries

**Sources**:
- [TEI GitHub Repository](https://github.com/huggingface/text-embeddings-inference)
- [TEI Documentation](https://huggingface.co/docs/text-embeddings-inference/en/index)
- [TEI Quick Tour](https://huggingface.co/docs/text-embeddings-inference/en/quick_tour)

### Qwen3-Embedding-0.6B Model

**Model Specifications**:
- **Parameters**: 0.6 billion (600M)
- **Context Length**: 32,768 tokens
- **Embedding Dimensions**: Flexible from 32 to 1024 (MRL - Multi-Representation Learning)
- **Default Dimension**: 1024 (matches requirements)
- **Languages**: 100+ languages supported
- **Architecture**: 28 layers with instruction-aware capabilities

**Performance Benchmarks**:
- MTEB Multilingual: 64.33 mean score
- MTEB English v2: 70.70 mean score
- C-MTEB (Chinese): 66.33 mean score
- Top-performing in 0.6B parameter class

**Key Features**:
- Multi-Representation Learning (MRL) allows custom output dimensions without retraining
- Instruction-aware embeddings (queries benefit from task-specific instructions)
- Supports Flash Attention 2 for better performance and memory efficiency
- Requires `transformers >= 4.51.0` to avoid KeyError issues

**Usage with TEI**:
```bash
# GPU deployment with 1024 dimensions
docker run --gpus all -p 8080:80 -v hf_cache:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:1.8 \
  --model-id Qwen/Qwen3-Embedding-0.6B
```

**Custom Dimensions via API**:
TEI supports a `dimensions` parameter in embedding requests:
```bash
curl http://localhost:8080/v1/embeddings \
  -X POST \
  -d '{"input":["Your text here"], "dimensions":1024}' \
  -H 'Content-Type: application/json'
```

**Sources**:
- [Qwen3-Embedding-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Qwen3 Embedding Blog Post](https://qwenlm.github.io/blog/qwen3-embedding/)
- [TEI API Documentation](https://huggingface.github.io/text-embeddings-inference/)

### Qdrant Vector Database

**Python Client Setup**:
```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Deployment options:
# 1. In-memory (development)
client = QdrantClient(":memory:")

# 2. Local persistent storage
client = QdrantClient(path="path/to/db")

# 3. Docker/remote server
client = QdrantClient(url="http://localhost:6333")
```

**Collection Setup for 1024-Dim Vectors**:
```python
client.create_collection(
    collection_name="markdown_embeddings",
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE
    )
)
```

**Distance Metrics**:
- `Distance.COSINE` - Cosine similarity (most common for embeddings)
- `Distance.DOT` - Dot product
- `Distance.EUCLID` - Euclidean distance

**Inserting Vectors**:
```python
from qdrant_client.models import PointStruct

client.upsert(
    collection_name="markdown_embeddings",
    points=[
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={"filename": "doc.md", "chunk_index": 0}
        )
        for idx, embedding in enumerate(embeddings)
    ]
)
```

**Integration with LlamaIndex**:
```python
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

client = qdrant_client.QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="markdown_embeddings"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build index from documents
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
```

**Advanced Features**:
- Hybrid search combining sparse (BM25) and dense vectors
- Multi-tenancy support with filtering
- Distributed deployment options
- Payload filtering for metadata queries

**Sources**:
- [Qdrant Python Client Documentation](https://python-client.qdrant.tech/)
- [Qdrant GitHub](https://github.com/qdrant/qdrant-client)
- [LlamaIndex Qdrant Integration](https://developers.llamaindex.ai/python/examples/vector_stores/qdrantindexdemo/)
- [Qdrant LlamaIndex Documentation](https://qdrant.tech/documentation/frameworks/llama-index/)

### File Watching with Python Watchdog

**Library Overview**:
- Watchdog is the standard Python library for monitoring filesystem events
- Cross-platform support (Windows, macOS, Linux)
- Supports creation, deletion, modification, and movement events
- Lightweight and versatile solution

**Core Components**:
1. **Observer**: Monitors filesystem changes
2. **Event Handler**: Defines actions for specific events
3. **Path to Monitor**: Directory to watch

**Markdown File Monitoring**:
```python
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

class MarkdownHandler(PatternMatchingEventHandler):
    patterns = ["*.md"]
    ignore_directories = True

    def on_created(self, event):
        print(f"New markdown file: {event.src_path}")
        # Trigger ingestion

    def on_modified(self, event):
        print(f"Modified markdown file: {event.src_path}")
        # Re-ingest document

observer = Observer()
handler = MarkdownHandler()
observer.schedule(handler, path="./watched_folder", recursive=True)
observer.start()
```

**Best Practices**:

1. **Debouncing**: Prevent processing rapid duplicate events
   - Use `threading.Timer` to delay processing
   - Track last event time per file (1-second threshold common)
   - Built-in support in `watchmedo` CLI tool with `--debounce-interval`

2. **Rate Limiting**: Throttle event handling to prevent overwhelming downstream systems
   - Disable processing until throttle time passes
   - Useful when editors perform multiple operations (auto-format, auto-save)

3. **Avoid Infinite Loops**: Use ignore patterns to prevent watching output directories

4. **Editor Compatibility**: Vim users need special configuration as on-modified events may not trigger by default

**Command-line Tool**:
```bash
# Watch for markdown files only
watchmedo log \
  --patterns='**/*.md' \
  --ignore-directories \
  --recursive \
  --verbose \
  /path/to/watch
```

**Sources**:
- [Watchdog PyPI](https://pypi.org/project/watchdog/)
- [Watchdog GitHub](https://github.com/gorakhargosh/watchdog)
- [Rate-Limiting with Watchdog (Medium)](https://medium.com/@RampantLions/smarter-file-watching-in-python-rate-limiting-and-change-history-with-watchdog-2114e45e7774)
- [Mastering File System Monitoring (DEV)](https://dev.to/devasservice/mastering-file-system-monitoring-with-watchdog-in-python-483c)

### RAG Chunking Strategies

**Best Practices for Markdown Documents (2024)**:

1. **Markdown-Aware Chunking** (Recommended):
   - Split by headings (#, ##, ###) to capture semantic sections
   - Preserves document hierarchy and context
   - Markdown is the most suitable format for chunking due to clear structure

2. **Recursive Character Text Splitter**:
   - Start with 400-512 tokens with 10-20% overlap
   - Uses semantic separators: "\n\n", "\n", ". "
   - Preserves markdown headers and structure
   - Implemented in LlamaIndex as `MarkdownNodeParser`

3. **Semantic Chunking**:
   - Context-aware splitting based on punctuation and paragraph breaks
   - Utilizes markdown/HTML tags for natural boundaries
   - Maintains semantic coherence within chunks

4. **Page-Level Chunking**:
   - NVIDIA 2024 benchmarks: 0.648 accuracy with page-level chunks
   - Good balance for long-form documents

**Chunk Size Recommendations**:
- **Factoid queries**: 256-512 tokens optimal
- **Analytical queries**: 1024+ tokens needed
- **Balance**: Large enough for context, small enough for efficiency
- **Overlap**: 10-20% overlap between chunks maintains continuity

**Metadata Handling**:
- Generate summaries reflecting content AND hierarchical header context
- Preserve document structure information (section path, heading level)
- Include file metadata (filename, modification time, chunk index)
- Ensure each chunk retains full context through metadata

**Key Considerations**:
- No universal strategy - depends on document structure and query types
- Experiment with different approaches
- Monitor real-world retrieval performance
- Refine strategies as application evolves

**Sources**:
- [Chunking for RAG Best Practices (Unstructured)](https://unstructured.io/blog/chunking-for-rag-best-practices)
- [Breaking Up is Hard to Do (Stack Overflow)](https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/)
- [Databricks Chunking Guide](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)
- [Best Chunking Strategies 2025 (Firecrawl)](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)

### RAG Ingestion Pipeline Error Handling

**Core Error Handling Strategies**:

1. **Comprehensive Logging**:
   - Implement logging at each pipeline stage
   - Track errors during parsing, embedding, and storage
   - Helps identify upstream data quality issues

2. **Advanced Error-Handling Mechanisms**:
   - Automatic retries with exponential backoff
   - Failover strategies for service unavailability
   - Alerting systems for critical failures
   - Minimizes disruptions from ingestion lags or retrieval failures

3. **Robust Retry Logic**:
   - Automated retries for transient failures
   - Circuit breaker pattern for persistent failures
   - Ensures data integrity

**Architectural Best Practices**:

4. **Error Isolation**:
   - Document-level or chunk-level processing isolation
   - If one document/chunk fails, others continue
   - Prevents cascade failures

5. **Asynchronous Processing**:
   - Async workflows for document chunking, embedding generation, vector storage
   - Enhances system responsiveness
   - Allows parallel processing and better resource utilization

6. **Data Preservation & Traceability**:
   - Store raw source data in target table
   - Ensures data preservation, traceability, and auditing
   - Enables reprocessing if needed

**Monitoring & Operations**:

7. **Comprehensive Monitoring**:
   - Monitor each pipeline stage
   - Track processing times, error rates, throughput
   - Audit trails for regulatory compliance

8. **Scalability Considerations**:
   - Horizontal scaling to handle varying document volumes
   - Queue-based architectures for buffering
   - Load balancing across processing workers

**Sources**:
- [Databricks RAG Pipeline Guide](https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag)
- [RAG in Production (Coralogix)](https://coralogix.com/ai-blog/rag-in-production-deployment-strategies-and-practical-considerations/)
- [Building Production-Ready RAG Systems (Medium)](https://medium.com/@meeran03/building-production-ready-rag-systems-best-practices-and-latest-tools-581cae9518e7)
- [Complete RAG Pipeline Guide (DHiWise)](https://www.dhiwise.com/post/build-rag-pipeline-guide)

## Codebase Analysis

### Project Structure

**Current State**:
- This appears to be a new Python project with no existing codebase
- Project directory: `/home/jmagar/code/llama`
- Only specification files exist in `./specs/rag-ingestion/`
- No existing Python files, requirements.txt, or pyproject.toml found

**Observations**:
- Clean slate allows implementation of best practices from the start
- No legacy patterns or technical debt to work around
- Need to establish project structure, dependency management, and quality tooling

### Existing Patterns

**Not Applicable**: No existing code patterns found.

**Recommendations for Initial Structure**:
```
llama/
├── pyproject.toml          # Modern Python dependency management
├── requirements.txt        # Or use poetry/pipenv
├── src/
│   └── rag_ingestion/
│       ├── __init__.py
│       ├── file_watcher.py    # Watchdog implementation
│       ├── embeddings.py      # TEI client wrapper
│       ├── ingestion.py       # LlamaIndex document processing
│       └── vector_store.py    # Qdrant client wrapper
├── tests/
│   └── test_*.py
├── data/
│   └── watched_folder/        # Markdown files to watch
├── docker-compose.yml         # TEI + Qdrant services
└── README.md
```

### Dependencies to Install

**Core Dependencies**:
```
llama-index-core>=0.11.0
llama-index-vector-stores-qdrant
llama-index-readers-file
qdrant-client
watchdog
huggingface-hub  # For InferenceClient
```

**Optional/Development**:
```
pytest
pytest-asyncio
python-dotenv
pydantic  # For configuration management
```

### Constraints

**None Identified**: No existing technical constraints or limitations in the codebase.

## Related Specs

**No Related Specs Found**: This is currently the only specification in the project.

**Future Considerations**:
- Query/retrieval spec will likely need to integrate with this ingestion pipeline
- API/interface spec for exposing RAG capabilities
- Monitoring/observability spec for production deployment

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| **Technical Viability** | High | All components have mature Python integrations with well-documented APIs. LlamaIndex officially supports Qdrant, TEI is production-ready, and watchdog is the standard for file monitoring. |
| **Effort Estimate** | M (Medium) | Core implementation: 3-5 days. Includes TEI setup, LlamaIndex integration with custom embeddings, Qdrant configuration, watchdog implementation with debouncing, error handling, and basic testing. Does not include advanced features like distributed processing or complex retry logic. |
| **Risk Level** | Low-Medium | **Low risks**: Well-established libraries, clear documentation. **Medium risks**: Custom TEI integration with LlamaIndex (may need custom embedding class), proper debouncing implementation critical for stability, determining optimal chunking strategy requires experimentation. |
| **Performance** | High | TEI provides optimized inference, Qdrant is built for high-performance vector search, 1024-dim vectors are standard size. Expected throughput: 100+ documents/minute on modest hardware. |
| **Maintainability** | High | Modular architecture with clear separation of concerns, standard Python tooling, active community support for all libraries. |
| **Scalability** | High | All components support horizontal scaling. TEI can run multiple instances behind load balancer, Qdrant supports distributed clusters, watchdog can be replaced with queue-based system for very high volumes. |

## Recommendations for Requirements

### Architecture Approach

1. **Microservices/Containerized Design**:
   - Deploy TEI in Docker container for embedding generation
   - Deploy Qdrant in Docker container for vector storage
   - Python application connects to both services
   - Use docker-compose for local development

2. **Modular Component Design**:
   - **File Watcher Module**: Watchdog-based monitoring with debouncing (1-second threshold)
   - **Document Processor**: LlamaIndex SimpleDirectoryReader with markdown filtering
   - **Chunking Module**: Markdown-aware chunking with 512-token chunks and 15% overlap
   - **Embedding Module**: Custom TEI client wrapper (OpenAI-compatible API)
   - **Vector Store Module**: Qdrant client wrapper with upsert operations

3. **Event-Driven Processing**:
   - File events trigger async processing pipeline
   - Queue-based buffering for high-volume scenarios
   - Parallel processing of independent documents

### Integration Strategy

**LlamaIndex + TEI Custom Embedding**:
Since LlamaIndex doesn't have native TEI integration, implement custom embedding class:

```python
from llama_index.core.embeddings import BaseEmbedding
from huggingface_hub import InferenceClient

class TEIEmbedding(BaseEmbedding):
    def __init__(self, endpoint_url: str, dimensions: int = 1024):
        self.client = InferenceClient(model=endpoint_url)
        self.dimensions = dimensions
        super().__init__()

    def _get_embedding(self, text: str) -> List[float]:
        return self.client.feature_extraction(text)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text) for text in texts]
```

**Alternative**: Use OpenAI-compatible endpoint with LlamaIndex's OpenAI embedding class.

### Configuration Management

1. **Environment Variables**:
   - `TEI_ENDPOINT_URL`: TEI service URL (default: http://localhost:8080)
   - `QDRANT_URL`: Qdrant service URL (default: http://localhost:6333)
   - `WATCH_FOLDER`: Path to monitor (default: ./data/watched_folder)
   - `EMBEDDING_DIMENSIONS`: Vector dimensions (default: 1024)
   - `CHUNK_SIZE`: Tokens per chunk (default: 512)
   - `CHUNK_OVERLAP`: Overlap percentage (default: 15)

2. **Configuration File** (optional):
   - Use Pydantic BaseSettings for type-safe configuration
   - Support .env files for local development

### Error Handling Strategy

1. **File Processing Errors**:
   - Log failed files to error queue
   - Continue processing remaining files
   - Retry failed files with exponential backoff

2. **TEI Service Errors**:
   - Implement circuit breaker pattern
   - Fallback to queuing documents for retry
   - Alert on sustained failures

3. **Qdrant Storage Errors**:
   - Retry with exponential backoff (3 attempts)
   - Log failed upserts for manual review
   - Maintain transaction log for recovery

4. **Watchdog Errors**:
   - Implement supervisor/restart mechanism
   - Log observer crashes
   - Alert on observer failures

### Testing Strategy

1. **Unit Tests**:
   - Test chunking logic with sample markdown
   - Mock TEI/Qdrant clients for isolated testing
   - Test debouncing logic with simulated events

2. **Integration Tests**:
   - Test full pipeline with test TEI/Qdrant instances
   - Verify embedding dimensions (1024)
   - Validate vector storage and retrieval

3. **End-to-End Tests**:
   - Create test markdown files
   - Verify automatic ingestion
   - Query Qdrant to validate stored vectors

### Performance Optimization

1. **Batch Processing**:
   - Accumulate file events over short window (2-5 seconds)
   - Process multiple files in single batch
   - Reduces TEI API calls and Qdrant upserts

2. **Async Processing**:
   - Use asyncio for concurrent document processing
   - Parallel embedding generation for multiple chunks
   - Non-blocking Qdrant upserts

3. **Caching**:
   - Cache document hashes to detect actual changes
   - Skip re-processing unchanged files
   - Store processing metadata in Qdrant payload

### Monitoring & Observability

1. **Metrics to Track**:
   - Documents processed per minute
   - Average embedding generation time
   - Qdrant upsert latency
   - Error rates by component
   - Queue depth (if using queues)

2. **Logging**:
   - Structured logging (JSON format)
   - Log levels: DEBUG for development, INFO for production
   - Include correlation IDs for tracing documents through pipeline

3. **Health Checks**:
   - TEI service health endpoint monitoring
   - Qdrant connection status
   - Watchdog observer status

## Open Questions

1. **Chunking Strategy Validation**:
   - Should we experiment with multiple chunking strategies (markdown-aware vs. recursive) and compare retrieval quality?
   - What is the expected document structure (technical docs, articles, notes)?

2. **Initial Document Loading**:
   - Should the system process existing markdown files on startup, or only monitor for new/changed files?
   - How should we handle bulk ingestion of existing documents?

3. **Document Updates**:
   - When a markdown file is modified, should we delete old vectors and re-ingest, or use versioning?
   - How should we handle document deletions - clean up vectors in Qdrant automatically?

4. **Performance Requirements**:
   - What is the expected volume of markdown files?
   - What is the acceptable processing latency (seconds, minutes)?
   - Are there concurrent user requirements while ingestion is running?

5. **Deployment Environment**:
   - Will this run on local development machines, servers, or cloud?
   - GPU availability for TEI (significant performance difference vs. CPU)?
   - Storage requirements for Qdrant (expected total vector count)?

6. **Metadata Requirements**:
   - What metadata should be stored with vectors (filename, modification date, author, tags)?
   - Do we need to support filtering/searching by metadata?

7. **Quality Verification**:
   - How should we verify that embeddings are generated correctly?
   - Should we implement quality checks (embedding dimension validation, vector normalization)?

## Quality Commands

Since this is a new Python project with no existing infrastructure, quality commands cannot be discovered from the codebase. The following commands are recommended for the project:

**Recommended Quality Commands**:

| Type | Command | Notes |
|------|---------|-------|
| Lint | `ruff check .` or `flake8 src/` | Ruff is faster, modern alternative |
| Format | `ruff format .` or `black src/` | Consistent code formatting |
| TypeCheck | `ty src/` | Static type checking |
| Unit Test | `pytest tests/` | Run all tests |
| Test Coverage | `pytest --cov=src tests/` | With coverage report |
| Integration Test | `pytest tests/integration/` | Separate integration tests |

**Installation**:
```bash
pip install ruff ty pytest pytest-cov pytest-asyncio
```

**Note**: These commands should be added to a Makefile or package.json scripts section once the project structure is established.

## Sources

### LlamaIndex
- [LlamaIndex Python Documentation](https://docs.llamaindex.ai/)
- [LlamaIndex in Python: A RAG Guide With Examples – Real Python](https://realpython.com/llamaindex-examples/)
- [GitHub - run-llama/llama_index](https://github.com/run-llama/llama_index)
- [Introducing LlamaIndex 0.11](https://www.llamaindex.ai/blog/introducing-llamaindex-0-11)
- [SimpleDirectoryReader Documentation](https://developers.llamaindex.ai/python/framework/module_guides/loading/simpledirectoryreader/)
- [Qdrant Vector Store | LlamaIndex](https://developers.llamaindex.ai/python/examples/vector_stores/qdrantindexdemo/)

### HuggingFace TEI
- [Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference/en/index)
- [GitHub - huggingface/text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)
- [TEI Quick Tour](https://huggingface.co/docs/text-embeddings-inference/en/quick_tour)
- [Text Embeddings Inference API](https://huggingface.github.io/text-embeddings-inference/)

### Qwen3 Model
- [Qwen/Qwen3-Embedding-0.6B · Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Qwen3 Embedding Blog Post](https://qwenlm.github.io/blog/qwen3-embedding/)
- [GitHub - QwenLM/Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)

### Qdrant
- [Qdrant Python Client Documentation](https://python-client.qdrant.tech/)
- [GitHub - qdrant/qdrant-client](https://github.com/qdrant/qdrant-client)
- [Qdrant Local Quickstart](https://qdrant.tech/documentation/quickstart/)
- [LlamaIndex - Qdrant](https://qdrant.tech/documentation/frameworks/llama-index/)

### Watchdog
- [watchdog · PyPI](https://pypi.org/project/watchdog/)
- [GitHub - gorakhargosh/watchdog](https://github.com/gorakhargosh/watchdog)
- [Smarter File Watching in Python (Medium)](https://medium.com/@RampantLions/smarter-file-watching-in-python-rate-limiting-and-change-history-with-watchdog-2114e45e7774)
- [Mastering File System Monitoring with Watchdog](https://dev.to/devasservice/mastering-file-system-monitoring-with-watchdog-in-python-483c)

### RAG Best Practices
- [Chunking for RAG: best practices | Unstructured](https://unstructured.io/blog/chunking-for-rag-best-practices)
- [Breaking up is hard to do: Chunking in RAG applications - Stack Overflow](https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/)
- [Databricks Chunking Guide](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)
- [Best Chunking Strategies for RAG in 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- [RAG in Production (Coralogix)](https://coralogix.com/ai-blog/rag-in-production-deployment-strategies-and-practical-considerations/)
- [Building Production-Ready RAG Systems (Medium)](https://medium.com/@meeran03/building-production-ready-rag-systems-best-practices-and-latest-tools-581cae9518e7)
- [Complete Guide to Building a Robust RAG Pipeline 2025](https://www.dhiwise.com/post/build-rag-pipeline-guide)
