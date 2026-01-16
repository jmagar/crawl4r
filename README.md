# Crawl4r RAG Ingestion Pipeline

An automated document processing system that monitors markdown files, generates embeddings using HuggingFace TEI with Qwen3-Embedding-0.6B, and stores vectors in Qdrant for retrieval-augmented generation (RAG) applications.

## Features

- **Automated File Monitoring**: Watches a folder for markdown file changes (create/modify/delete) with 1-second debouncing
- **Intelligent Chunking**: Markdown-aware chunking with heading-based splitting (512 tokens, 15% overlap)
- **High-Performance Embeddings**: GPU-accelerated TEI service with 1024-dimensional Qwen3 embeddings
- **Robust Error Handling**: Circuit breaker pattern, exponential backoff retries, and failed document logging
- **Idempotent Operations**: Deterministic point IDs (SHA256) enable safe re-ingestion without duplicates
- **State Recovery**: Automatically detects deleted files and removes stale vectors on startup
- **Quality Assurance**: Validates embedding dimensions, metadata structure, and service health
- **Async-First Architecture**: Non-blocking I/O with asyncio for high throughput
- **Comprehensive Testing**: 97%+ test coverage with unit and integration tests
- **Web Crawling**: LlamaIndex-compatible reader for crawling and indexing web pages

## Web Crawling with Crawl4AIReader

The `Crawl4AIReader` component enables web page ingestion into your RAG pipeline using the Crawl4AI service.

### Quick Start

```python
from rag_ingestion.crawl4ai_reader import Crawl4AIReader

# Initialize reader
reader = Crawl4AIReader(endpoint_url="http://localhost:52004")

# Crawl single URL
documents = await reader.aload_data(["https://docs.example.com"])

# Batch crawl multiple URLs
urls = ["https://example.com/page1", "https://example.com/page2"]
documents = await reader.aload_data(urls)
```

### Key Features

- **LlamaIndex Integration**: Drop-in replacement for file-based readers
- **Automatic Deduplication**: Removes existing URL data before re-crawling
- **Circuit Breaker**: Prevents cascade failures with automatic recovery
- **Retry Logic**: Exponential backoff for transient errors (timeout, network, 5xx)
- **Deterministic IDs**: SHA256-based UUID generation for idempotent operations
- **Rich Metadata**: Includes source_url, title, description, status_code, crawl_timestamp
- **Concurrent Crawling**: Configurable concurrency limit (default: 5)

### Configuration

```python
from rag_ingestion.crawl4ai_reader import Crawl4AIReader, Crawl4AIReaderConfig

config = Crawl4AIReaderConfig(
    endpoint_url="http://localhost:52004",
    max_concurrent_requests=10,
    max_retries=3,
    timeout=60.0,
    fail_on_error=False  # Return None for failed URLs
)

reader = Crawl4AIReader(**config.model_dump())
```

### Integration Example

```python
from rag_ingestion.crawl4ai_reader import Crawl4AIReader
from rag_ingestion.chunker import MarkdownChunker
from rag_ingestion.vector_store import VectorStoreManager

# Initialize components
reader = Crawl4AIReader(endpoint_url="http://localhost:52004")
chunker = MarkdownChunker(chunk_size_tokens=512, chunk_overlap_percent=15)
vector_store = VectorStoreManager(collection_name="web_docs", qdrant_url="http://localhost:52001")

# Crawl and index
documents = await reader.aload_data(["https://docs.example.com"])
for doc in documents:
    if doc:
        chunks = chunker.chunk(doc.text, filename=doc.metadata["source_url"])
        # Continue with embedding and storage...
```

See `CLAUDE.md` for complete usage documentation and advanced features.

## Prerequisites

### Hardware
- **GPU Required**: NVIDIA GPU with CUDA support for TEI and Qdrant GPU acceleration
- **RAM**: Minimum 8GB, recommended 16GB+
- **Disk**: 10GB+ for models and data

### Software
- **Docker**: Version 20.10+ with Docker Compose v2
- **NVIDIA Container Toolkit**: For GPU access in containers
- **Python**: 3.10 or higher (for development)
- **uv**: Fast Python package installer (install via `curl -LsSf https://astral.sh/uv/install.sh | sh`)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/crawl4r.git
cd crawl4r
```

### 2. Create Docker Network

```bash
docker network create crawl4r
```

### 3. Configure Environment

Copy the example environment file and configure required variables:

```bash
cp .env.example .env
```

Edit `.env` and set the following **required** variables:

```bash
# REQUIRED: Directory to watch for markdown files
WATCH_FOLDER=/path/to/your/markdown/files

# OPTIONAL: Override defaults if needed (see Configuration section)
TEI_HTTP_PORT=52000
QDRANT_HTTP_PORT=52001
QDRANT_GRPC_PORT=52002
POSTGRES_PORT=53432
REDIS_PORT=53379
CRAWL4AI_PORT=52004
```

### 4. Start Infrastructure Services

```bash
docker compose up -d
```

This starts:
- **crawl4r-embeddings**: TEI service on port 52000
- **crawl4r-vectors**: Qdrant vector database on ports 52001 (HTTP) and 52002 (gRPC)
- **crawl4r-db**: PostgreSQL database on port 53432
- **crawl4r-cache**: Redis cache on port 53379
- **crawl4ai**: Web crawling service on port 52004

Wait 30-60 seconds for services to initialize, then verify health:

```bash
curl http://localhost:52000/health  # TEI
curl http://localhost:52001/readyz   # Qdrant
```

### 5. Install Python Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install production dependencies
uv sync

# Install development dependencies (for testing)
uv sync --group dev
```

### 6. Run the Pipeline

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the ingestion pipeline
python -m rag_ingestion.main
```

The pipeline will:
1. Validate service health (TEI, Qdrant)
2. Perform state recovery (detect deleted files)
3. Batch process existing markdown files
4. Start real-time file monitoring

## Configuration

### Environment Variables

All configuration is managed via `.env` file. Copy `.env.example` and customize as needed.

#### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `WATCH_FOLDER` | Directory to monitor for markdown files | `/home/user/documents` |

#### Service Endpoints

| Variable | Description | Default |
|----------|-------------|---------|
| `TEI_ENDPOINT` | TEI embeddings service URL | `http://crawl4r-embeddings:80` |
| `QDRANT_URL` | Qdrant vector database URL | `http://crawl4r-vectors:6333` |
| `COLLECTION_NAME` | Qdrant collection name | `crawl4r` |

#### Chunking Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CHUNK_SIZE_TOKENS` | Target tokens per chunk | `512` |
| `CHUNK_OVERLAP_PERCENT` | Overlap percentage (0-50) | `15` (77 tokens) |

#### Performance Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_CONCURRENT_DOCS` | Maximum concurrent document processing | `10` |
| `QUEUE_MAX_SIZE` | Maximum queue size before backpressure | `1000` |
| `BATCH_SIZE` | Embedding batch size | `32` |

#### Logging

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `INFO` |
| `FAILED_DOCS_LOG` | Path to failed documents log file | `./failed_documents.jsonl` |

#### Docker Service Ports

| Variable | Description | Default |
|----------|-------------|---------|
| `TEI_HTTP_PORT` | TEI service HTTP port | `52000` |
| `QDRANT_HTTP_PORT` | Qdrant HTTP API port | `52001` |
| `QDRANT_GRPC_PORT` | Qdrant gRPC API port | `52002` |
| `POSTGRES_PORT` | PostgreSQL database port | `53432` |
| `POSTGRES_PASSWORD` | PostgreSQL password | (required, no default) |
| `REDIS_PORT` | Redis cache port | `53379` |
| `CRAWL4AI_PORT` | Crawl4AI service port | `52004` |

#### TEI Performance Tuning (Optional)

| Variable | Description | Default |
|----------|-------------|---------|
| `TEI_EMBEDDING_MODEL` | HuggingFace model ID | `Qwen/Qwen3-Embedding-0.6B` |
| `TEI_MAX_CONCURRENT_REQUESTS` | Max concurrent TEI requests | `128` |
| `TEI_MAX_BATCH_TOKENS` | Max tokens per batch | `131072` |
| `TEI_MAX_BATCH_REQUESTS` | Max requests per batch | `32` |
| `TEI_MAX_CLIENT_BATCH_SIZE` | Max client batch size | `128` |
| `TEI_POOLING` | Token pooling strategy | `last-token` |
| `TEI_TOKENIZATION_WORKERS` | Tokenization worker threads | `8` |

## Usage

### Running the Pipeline

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with default configuration
python -m rag_ingestion.main

# Run with custom log level
LOG_LEVEL=DEBUG python -m rag_ingestion.main
```

### Pipeline Behavior

1. **Startup Phase**:
   - Validates TEI and Qdrant service health
   - Checks embedding dimensions (must be 1024)
   - Performs state recovery (removes vectors for deleted files)

2. **Batch Processing Phase**:
   - Processes all existing `.md` files in `WATCH_FOLDER`
   - Generates chunks with heading-based splitting
   - Creates embeddings and stores in Qdrant

3. **Monitoring Phase**:
   - Watches for file system events (create, modify, delete)
   - Debounces rapid changes (1-second delay)
   - Updates vectors incrementally

### Stopping the Pipeline

Press `Ctrl+C` to gracefully shutdown. The pipeline will:
- Complete in-flight processing
- Close file watchers
- Flush logs

### Monitoring

View logs in real-time:

```bash
# Pipeline logs (stdout)
tail -f logs/rag-ingestion.log

# Failed documents (if any)
tail -f failed_documents.jsonl
```

Check Qdrant collection status:

```bash
curl http://localhost:52001/collections/crawl4r
```

## Testing

### Run All Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests with coverage
pytest --cov=rag_ingestion --cov-report=term --cov-report=html:.cache/htmlcov

# Coverage report will be in .cache/htmlcov/index.html
```

### Run Specific Tests

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only (requires services running)
pytest tests/integration/

# Specific test file
pytest tests/unit/test_chunker.py

# Specific test function
pytest tests/unit/test_chunker.py::test_chunk_by_heading_hierarchy
```

### Test Requirements

Integration tests require Docker services to be running:

```bash
docker compose up -d
pytest tests/integration/
```

### Quality Checks

```bash
# Linting with Ruff
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Type checking with ty
ty check rag_ingestion/

# Format code
ruff format .
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                   RAG Ingestion Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │   Startup    │───▶│    State     │───▶│   Batch     │  │
│  │  Validation  │    │   Recovery   │    │  Processor  │  │
│  └──────────────┘    └──────────────┘    └─────────────┘  │
│                                                  │          │
│                                                  ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │     File     │───▶│  Processing  │───▶│  Document   │  │
│  │   Watcher    │    │    Queue     │    │  Processor  │  │
│  └──────────────┘    └──────────────┘    └─────────────┘  │
│                                                  │          │
│                          ┌───────────────────────┘          │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │     TEI      │◀───│   Chunker    │    │   Vector    │  │
│  │    Client    │    │  (Markdown)  │    │    Store    │  │
│  └──────────────┘    └──────────────┘    └─────────────┘  │
│         │                                        │          │
│         └────────────────┬───────────────────────┘          │
│                          ▼                                  │
│                   ┌──────────────┐                         │
│                   │   Quality    │                         │
│                   │   Verifier   │                         │
│                   └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │         External Services              │
         ├────────────────────────────────────────┤
         │  TEI (Port 52000)  │ GPU-accelerated   │
         │  Qdrant (52001)    │ Vector database   │
         │  PostgreSQL (53432)│ Metadata store    │
         │  Redis (53379)     │ Cache/queue       │
         └────────────────────────────────────────┘
```

### Key Modules

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `main.py` | Entry point and orchestration | Startup validation, batch processing, monitoring loop |
| `config.py` | Configuration management | Pydantic settings, .env loading, validation |
| `tei_client.py` | TEI embeddings client | Circuit breaker, batch processing, retry logic |
| `vector_store.py` | Qdrant operations | Upsert, delete, search, collection management |
| `chunker.py` | Document chunking | Markdown-aware, heading-based splitting |
| `file_watcher.py` | File system monitoring | Watchdog integration, debouncing, event handling |
| `processor.py` | Document processing | Chunking, embedding, storage orchestration |
| `quality.py` | Quality validation | Service health, dimension checks, metadata validation |
| `circuit_breaker.py` | Fault tolerance | Circuit breaker pattern for service failures |
| `recovery.py` | State recovery | Detect deleted files, clean stale vectors |
| `failed_docs.py` | Error logging | JSONL logging of failed document processing |

## Troubleshooting

### Services Won't Start

**Problem**: Docker services fail health checks

**Solutions**:
```bash
# Check logs
docker compose logs crawl4r-embeddings
docker compose logs crawl4r-vectors

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Restart services
docker compose down
docker compose up -d
```

### TEI Connection Errors

**Problem**: `TEI service unavailable` errors

**Solutions**:
```bash
# Verify TEI is running
curl http://localhost:52000/health

# Check TEI logs
docker compose logs crawl4r-embeddings

# Restart TEI
docker compose restart crawl4r-embeddings
```

### Qdrant Connection Errors

**Problem**: `Failed to connect to Qdrant` errors

**Solutions**:
```bash
# Verify Qdrant is running
curl http://localhost:52001/readyz

# Check Qdrant logs
docker compose logs crawl4r-vectors

# Restart Qdrant
docker compose restart crawl4r-vectors
```

### Dimension Mismatch Errors

**Problem**: `Expected 1024 dimensions, got X`

**Solutions**:
- Ensure TEI model is `Qwen/Qwen3-Embedding-0.6B` (produces 1024-dim vectors)
- Verify `TEI_EMBEDDING_MODEL` in `.env`
- Recreate Qdrant collection with correct dimensions

### File Not Processing

**Problem**: Markdown files not being ingested

**Solutions**:
```bash
# Check file permissions
ls -la /path/to/watch/folder

# Verify WATCH_FOLDER in .env
grep WATCH_FOLDER .env

# Check pipeline logs
tail -f logs/rag-ingestion.log

# Verify file extension is .md
```

### Out of Memory Errors

**Problem**: Pipeline crashes with OOM errors

**Solutions**:
- Reduce `MAX_CONCURRENT_DOCS` in `.env`
- Reduce `BATCH_SIZE` in `.env`
- Increase Docker memory limit
- Monitor GPU memory: `nvidia-smi`

### Failed Documents

**Problem**: Documents failing to process

**Solutions**:
```bash
# Check failed documents log
cat failed_documents.jsonl | jq

# Common issues:
# - Invalid markdown syntax
# - Extremely large files (>10MB)
# - Special characters in filenames
# - Permission denied errors
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest --cov=rag_ingestion`)
4. Ensure linting passes (`ruff check .`)
5. Commit changes (`git commit -m 'feat: add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Support

- **Documentation**: See `CLAUDE.md` for development guidelines
- **Issues**: Report bugs via GitHub Issues
- **Specifications**: See `specs/rag-ingestion/` for detailed design docs

## Acknowledgments

- **LlamaIndex**: Document orchestration framework
- **HuggingFace TEI**: High-performance embedding inference
- **Qdrant**: GPU-accelerated vector database
- **Qwen3**: State-of-the-art embedding model
