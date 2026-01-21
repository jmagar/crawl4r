# Crawl4r - Web Crawling & RAG Ingestion Pipeline

A comprehensive web crawling and document processing system with RAG (Retrieval-Augmented Generation) capabilities. Features an intuitive CLI for web scraping, URL discovery, structured data extraction, screenshots, and automated vector storage.

## Features

### Web Crawling & Processing
- **Web Scraping**: Extract clean markdown from web pages using Crawl4AI
- **URL Discovery**: Recursive link mapping with depth control and same-domain filtering
- **Structured Extraction**: LLM-powered data extraction with JSON schema validation
- **Screenshot Capture**: Full-page and viewport screenshots with wait/selector support
- **Automated Ingestion**: Vector storage with embeddings for RAG applications

### RAG Pipeline
- **Automated File Monitoring**: Watches folders for markdown file changes with debouncing
- **Intelligent Chunking**: Markdown-aware chunking with heading-based splitting (512 tokens, 15% overlap)
- **High-Performance Embeddings**: GPU-accelerated TEI service with 1024-dimensional Qwen3 embeddings
- **Robust Error Handling**: Circuit breaker pattern, exponential backoff retries, and failed document logging
- **Idempotent Operations**: Deterministic point IDs (SHA256) enable safe re-ingestion without duplicates
- **State Recovery**: Automatically detects deleted files and removes stale vectors
- **Quality Assurance**: Validates embedding dimensions, metadata structure, and service health
- **Comprehensive Testing**: 87%+ test coverage with 752 passing tests

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/crawl4r.git
cd crawl4r

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Create Docker network
docker network create crawl4r

# Configure environment
cp .env.example .env
# Edit .env and set required variables

# Start services
docker compose up -d
```

### Basic Usage

```bash
# Scrape a single URL
crawl4r scrape https://example.com

# Crawl and ingest into vector store
crawl4r crawl https://docs.example.com --depth 2

# Discover all links on a page
crawl4r map https://example.com --depth 1

# Extract structured data
crawl4r extract https://example.com --schema schema.json

# Capture screenshot
crawl4r screenshot https://example.com --full-page -o page.png

# Check crawl status
crawl4r status

# Watch folder for changes
crawl4r watch --folder /path/to/docs
```

## CLI Commands

### `scrape` - Extract Markdown from URLs

Scrape one or more URLs and output clean markdown content.

```bash
crawl4r scrape [OPTIONS] [URLS...]
```

**Options:**
- `-f, --file PATH` - Read URLs from file (one per line)
- `-o, --output PATH` - Write output to file (single URL) or directory (multiple URLs)
- `-c, --concurrent INTEGER` - Max concurrent requests (default: 5)

**Examples:**

```bash
# Scrape single URL to stdout
crawl4r scrape https://example.com

# Scrape multiple URLs
crawl4r scrape https://example.com https://example.org

# Scrape URLs from file
crawl4r scrape -f urls.txt

# Save output to file
crawl4r scrape https://example.com -o page.md

# Save multiple URLs to directory
crawl4r scrape https://example.com https://example.org -o output/

# Increase concurrency
crawl4r scrape -f urls.txt --concurrent 10
```

**Output Format:**
- Single URL: Prints markdown to stdout or saves to file
- Multiple URLs: Creates separate files in output directory (e.g., `example.com.md`)

---

### `crawl` - Ingest URLs into Vector Store

Crawl URLs, generate embeddings, and store in Qdrant for RAG applications.

```bash
crawl4r crawl [OPTIONS] [URLS...]
```

**Options:**
- `-f, --file PATH` - Read URLs from file (one per line, max 1MB)
- `-d, --depth INTEGER` - Crawl depth for link discovery (default: 1)

**Examples:**

```bash
# Crawl single URL (depth 1 link discovery)
crawl4r crawl https://docs.example.com

# Crawl with increased depth
crawl4r crawl https://example.com --depth 3

# Crawl URLs from file
crawl4r crawl -f urls.txt

# Shallow crawl (no link discovery)
crawl4r crawl https://example.com --depth 0
```

**What Happens:**
1. Scrapes URLs and discovered links (based on depth)
2. Chunks markdown into nodes using LlamaIndex MarkdownNodeParser
3. Generates embeddings using TEI service
4. Stores vectors in Qdrant with metadata
5. Tracks status in Redis (use `status` command to monitor)

**Background Processing:**
- Crawls run asynchronously in background queue
- Returns `crawl_id` for status tracking
- Use `crawl4r status <crawl_id>` to check progress

---

### `status` - View Crawl Status

Monitor crawl job status and history.

```bash
crawl4r status [OPTIONS] [CRAWL_ID]
```

**Options:**
- `--list` - Show recent crawl history
- `--active` - Show only active/running crawls

**Examples:**

```bash
# Show recent crawl history (default)
crawl4r status

# List all recent crawls
crawl4r status --list

# Show active crawls only
crawl4r status --active

# Get specific crawl status
crawl4r status crawl_abc123def456
```

**Status Fields:**
- **Crawl ID**: Unique identifier
- **Status**: QUEUED, RUNNING, COMPLETED, PARTIAL, FAILED
- **URLs**: Total and successful counts
- **Chunks**: Number of document chunks created
- **Started/Finished**: Timestamps
- **Error**: Error message (if failed)

---

### `map` - Discover URLs from Page

Extract and discover all links from a web page.

```bash
crawl4r map [OPTIONS] URL
```

**Options:**
- `-d, --depth INTEGER` - Max crawl depth (default: 0 = single page)
- `--same-domain` / `--external` - Filter to same-domain links only (default: same-domain)
- `-o, --output PATH` - Save URLs to file

**Examples:**

```bash
# Get all links from page
crawl4r map https://example.com

# Recursive discovery (depth 2)
crawl4r map https://example.com --depth 2

# Include external links
crawl4r map https://example.com --external

# Save to file
crawl4r map https://example.com --depth 1 -o urls.txt
```

**Output Format:**
```
Discovered URLs:
- https://example.com/about
- https://example.com/contact
- https://example.com/blog
```

---

### `extract` - Extract Structured Data

Use LLMs to extract structured data from web pages with schema validation.

```bash
crawl4r extract [OPTIONS] URL
```

**Options:**
- `--schema TEXT` - JSON schema file path OR inline JSON schema string
- `--prompt TEXT` - Natural language extraction prompt (alternative to schema)
- `-o, --output PATH` - Save extracted JSON to file
- `--provider TEXT` - LLM provider (e.g., `ollama/llama3`, `openai/gpt-4o-mini`)

**Examples:**

```bash
# Extract with JSON schema file
crawl4r extract https://example.com --schema schema.json

# Extract with inline schema
crawl4r extract https://example.com --schema '{"type":"object","properties":{"title":{"type":"string"}}}'

# Extract with natural language prompt
crawl4r extract https://example.com --prompt "Extract the article title, author, and date"

# Save output
crawl4r extract https://example.com --schema schema.json -o output.json

# Use specific LLM provider
crawl4r extract https://example.com --schema schema.json --provider ollama/llama3
```

**Schema Example:**
```json
{
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "author": {"type": "string"},
    "published_date": {"type": "string"},
    "tags": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["title"]
}
```

---

### `screenshot` - Capture Page Screenshots

Capture full-page or viewport screenshots of web pages.

```bash
crawl4r screenshot [OPTIONS] URL
```

**Options:**
- `-o, --output PATH` - Output file path (auto-named if omitted)
- `-f, --full-page` - Capture entire page (scrollable content)
- `-w, --wait FLOAT` - Wait seconds before capture (default: 0.0)
- `-s, --selector TEXT` - Wait for CSS selector to appear
- `--width INTEGER` - Viewport width in pixels
- `--height INTEGER` - Viewport height in pixels

**Examples:**

```bash
# Basic screenshot
crawl4r screenshot https://example.com

# Full-page screenshot
crawl4r screenshot https://example.com --full-page

# Save to specific file
crawl4r screenshot https://example.com -o page.png

# Wait for page load
crawl4r screenshot https://example.com --wait 2.0

# Wait for element
crawl4r screenshot https://example.com --selector "#main-content"

# Custom viewport
crawl4r screenshot https://example.com --width 1920 --height 1080

# Combined options
crawl4r screenshot https://example.com --full-page --wait 1.5 -o screenshot.png
```

---

### `watch` - Monitor Directory for Changes

Automatically monitor a directory for markdown file changes and process them.

```bash
crawl4r watch [OPTIONS]
```

**Options:**
- `--folder PATH` - Override watch folder from settings (default: `WATCH_FOLDER` env var)

**Examples:**

```bash
# Watch default folder from .env
crawl4r watch

# Watch specific folder
crawl4r watch --folder /path/to/docs
```

**What Happens:**
1. **Startup Recovery**: Processes all existing files and detects deletions
2. **Continuous Monitoring**: Watches for create/modify/delete events
3. **Auto-Processing**:
   - **Create/Modify**: Generates embeddings and stores vectors
   - **Delete**: Removes associated vectors from Qdrant
4. **Debouncing**: 1-second delay prevents duplicate processing

**Use Cases:**
- Documentation sites (update docs → auto-reindex)
- Knowledge bases (add notes → auto-embed)
- Content management (edit files → instant vector updates)

---

## Prerequisites

### Hardware
- **GPU Required**: NVIDIA GPU with CUDA support for TEI and Qdrant GPU acceleration
- **RAM**: Minimum 8GB, recommended 16GB+
- **Disk**: 10GB+ for models and data

### Software
- **Docker**: Version 20.10+ with Docker Compose v2
- **NVIDIA Container Toolkit**: For GPU access in containers
- **Python**: 3.10 or higher
- **uv**: Fast Python package installer

## Services

The system uses 5 containerized services:

| Service | Port | Purpose |
|---------|------|---------|
| **crawl4ai** | 52004 | Web crawling service (Playwright + Crawl4AI) |
| **crawl4r-embeddings** (TEI) | 52000 | GPU-accelerated embeddings (Qwen3-Embedding-0.6B) |
| **crawl4r-vectors** (Qdrant) | 52001/52002 | Vector database (HTTP/gRPC) |
| **crawl4r-db** (PostgreSQL) | 53432 | Metadata storage |
| **crawl4r-cache** (Redis) | 53379 | Job queue and caching |

All services connect via the `crawl4r` Docker network.

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

#### Required
```bash
POSTGRES_PASSWORD=your_secure_password
```

#### Service Endpoints (defaults shown)
```bash
CRAWL4AI_PORT=52004
TEI_HTTP_PORT=52000
QDRANT_HTTP_PORT=52001
QDRANT_GRPC_PORT=52002
POSTGRES_PORT=53432
REDIS_PORT=53379
```

#### Watch Folder (for `watch` command)
```bash
WATCH_FOLDER=/path/to/your/markdown/files
```

#### Chunking Settings
```bash
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_PERCENT=15
```

#### Performance Tuning
```bash
MAX_CONCURRENT_DOCS=10
BATCH_SIZE=32
LOG_LEVEL=INFO
```

See `.env.example` for complete configuration options.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=crawl4r --cov-report=term

# Unit tests only
pytest tests/unit/

# Integration tests (requires services)
pytest tests/integration/ -m integration

# Specific command tests
pytest tests/unit/test_cli_commands.py -v
```

**Test Coverage:** 87.40% overall (752 passing tests)

## Development

### Code Quality

```bash
# Linting
ruff check .

# Auto-fix
ruff check . --fix

# Type checking
ty check crawl4r/

# Format code
ruff format .
```

### Service Management

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f crawl4ai
docker compose logs -f crawl4r-embeddings

# Restart service
docker compose restart crawl4ai
```

## Troubleshooting

### Services Won't Start

```bash
# Check logs
docker compose logs

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Recreate services
docker compose down
docker compose up -d --force-recreate
```

### Crawl4AI Connection Errors

```bash
# Verify service
curl http://localhost:52004/health

# Check logs
docker compose logs crawl4ai

# Restart
docker compose restart crawl4ai
```

### TEI/Qdrant Connection Errors

```bash
# Verify TEI
curl http://localhost:52000/health

# Verify Qdrant
curl http://localhost:52001/readyz

# Check network
docker network inspect crawl4r
```

## License

[MIT License](LICENSE)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Ensure linting passes (`ruff check .`)
5. Commit changes (`git commit -m 'feat: add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Documentation

- **Development**: See `CLAUDE.md` for development guidelines
- **Specifications**: See `specs/` for detailed design docs
- **Architecture**: See architecture diagrams in `docs/`

## Acknowledgments

- **Crawl4AI**: Web crawling infrastructure
- **LlamaIndex**: Document orchestration framework
- **HuggingFace TEI**: High-performance embedding inference
- **Qdrant**: GPU-accelerated vector database
- **Qwen3**: State-of-the-art embedding model
