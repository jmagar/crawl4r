---
spec: llamaindex-crawl4ai-reader
phase: research
created: 2026-01-15T00:00:00Z
---

# Research: LlamaIndex Crawl4AI Reader

## Executive Summary

Creating a LlamaIndex reader for Crawl4AI is **highly feasible** with well-established patterns. The reader should inherit from `BasePydanticReader` for serialization support, implement async operations via `aload_data()`, and leverage Crawl4AI's Docker service at port 52004. The implementation aligns well with existing RAG pipeline patterns and can reuse the project's circuit breaker, retry logic, and structured logging infrastructure. Estimated effort is **Small (2-3 days)** with **Low risk** due to proven patterns and comprehensive API documentation.

**Key Findings**:
- Crawl4AI provides rich REST API with `/crawl`, `/crawl/stream`, `/crawl/job` endpoints
- Response includes markdown, links, metadata, extracted_content suitable for LlamaIndex Documents
- Existing codebase has reusable circuit breaker, httpx client patterns, and async infrastructure
- LlamaIndex `BasePydanticReader` provides clear contract for custom readers

## External Research

### Best Practices for LlamaIndex Reader Implementation

#### Base Class Selection

LlamaIndex provides two base classes for custom readers:

1. **`BaseReader`** - Standard base class for data loaders
   - Simple contract: implement `load_data()` method returning `List[Document]`
   - No serialization support
   - Used by basic readers like TrafilaturaWebReader

2. **`BasePydanticReader`** - Serializable data loader with Pydantic (RECOMMENDED)
   - Extends Pydantic BaseModel for automatic serialization
   - Properties: `is_remote` (bool, default: false) and `class_name` (str, default: "base_component")
   - Used by modern readers: SimpleWebPageReader, S3Reader, AgentQLWebReader
   - Compatible with ReaderConfig and IngestionPipeline
   - **Recommendation**: Use `BasePydanticReader` for this implementation

**Source**: [BasePydanticReader Documentation](https://docs.llamaindex.ai/en/v0.10.19/api/llama_index.core.readers.base.BasePydanticReader.html)

#### Required Methods

All LlamaIndex readers must implement:

1. **`load_data(*args, **kwargs) -> List[Document]`** - Synchronous data loading
2. **`aload_data(*args, **kwargs) -> List[Document]`** (optional but recommended) - Async data loading
3. **`lazy_load_data(*args, **kwargs) -> Iterator[Document]`** (optional) - Streaming/lazy loading

**Best Practice**: Implement both `load_data()` and `aload_data()` for flexibility. Use async by default for HTTP-based readers.

**Source**: [BaseReader API Reference](https://docs.llamaindex.ai/en/v0.10.20/api/llama_index.core.readers.base.BaseReader.html)

#### Document Structure

LlamaIndex `Document` objects have the following structure:

```python
Document(
    text="<main content>",           # Required: primary text content
    metadata={                         # Optional: arbitrary key-value pairs
        "source": "url",
        "title": "Page Title",
        "author": "Author Name",
        # ... any custom fields
    },
    id_="<unique_id>",                # Optional: deterministic ID
    excluded_llm_metadata_keys=[],    # Keys to hide from LLM
    excluded_embed_metadata_keys=[],  # Keys to hide from embeddings
)
```

**Metadata Best Practices**:
- Default metadata format: `"{key}: {value}"` with separator between pairs
- Vector DB constraints: keys must be strings, values must be flat (str, float, int)
- Control metadata visibility: use `excluded_llm_metadata_keys` and `excluded_embed_metadata_keys`
- Include source URL, title, description, timestamps, and custom fields

**Source**: [LlamaIndex Document Management](https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management/)

#### HTTP-Based Reader Patterns

Existing LlamaIndex web readers demonstrate several patterns:

**SimpleWebPageReader** (reference implementation):
```python
class SimpleWebPageReader(BasePydanticReader):
    is_remote: bool = True
    html_to_text: bool = False
    _timeout: int = 60
    _fail_on_error: bool = False

    def load_data(self, urls: List[str]) -> List[Document]:
        # Uses requests.get() with timeout
        # Applies html2text if enabled
        # Extracts metadata via custom function
        # Returns List[Document] with UUID identifiers
```

**Key Patterns Observed**:
- Set `is_remote = True` for HTTP-based readers
- Accept list of URLs or single URL
- Use configurable timeouts (default: 60s)
- Provide error handling flags (`fail_on_error`)
- Apply content transformations (HTML to text, markdown)
- Extract rich metadata from responses
- Generate deterministic or UUID-based document IDs

**Source**: [SimpleWebPageReader Source Code](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-web/llama_index/readers/web/simple_web/base.py)

### Crawl4AI API Capabilities and Response Format

#### Available Endpoints

The Crawl4AI Docker service (port 52004) provides these REST endpoints:

1. **`POST /crawl`** - Standard synchronous crawling
   - Returns complete result immediately
   - Suitable for single URLs with quick response times
   - Timeout: configurable via crawler_params

2. **`POST /crawl/stream`** - Streaming results via NDJSON
   - Real-time incremental results
   - Set Accept header: `application/x-ndjson`
   - Suitable for monitoring progress on long crawls

3. **`POST /crawl/job`** - Asynchronous job submission with webhooks
   - Returns task_id immediately
   - Supports webhook notifications (v0.7.6+)
   - Eliminates polling overhead
   - Best for batch processing

4. **`GET /job/{task_id}`** - Check job status and retrieve results
   - Poll this endpoint after submitting async job
   - Returns status: "pending", "processing", "completed", "failed"
   - Full result available when status = "completed"

5. **`GET /health`** - Health check endpoint
   - Verify service availability before crawling

**Recommendation**: Use **`POST /crawl`** for simplicity, **`POST /crawl/job`** for production workloads with webhooks.

**Sources**:
- [Crawl4AI Self-Hosting Guide](https://docs.crawl4ai.com/core/self-hosting/)
- [Crawl4AI v0.7.6 Release Notes](https://docs.crawl4ai.com/blog/releases/0.7.6/)

#### Request Parameters

The `/crawl` endpoint accepts JSON body:

```json
{
  "url": "https://example.com",
  "crawler_params": {
    "cache_mode": "BYPASS",           // CacheMode.BYPASS | CacheMode.ENABLED
    "word_count_threshold": 10,       // Min words per content block
    "extraction_strategy": "markdown", // or "css", "llm", "xpath"
    "chunking_strategy": "regex",      // Chunking method
    "css_selector": ".main-content",   // CSS selector for content
    "excluded_tags": ["nav", "footer"], // Tags to exclude
    "exclude_external_links": true,    // Filter external links
    "exclude_social_media_links": true, // Filter social media
    "screenshot": false,               // Capture screenshot
    "pdf": false,                      // Generate PDF
    "wait_until": "networkidle",       // Wait condition
    "timeout": 30000,                  // Timeout in milliseconds
    "user_agent": "Mozilla/5.0...",    // Custom user agent
    "headers": {},                     // Custom HTTP headers
    "proxy": null,                     // Proxy configuration
    "session_id": null,                // Browser session reuse
    "js_code": null,                   // JavaScript to execute
    "verbose": false                   // Detailed logging
  }
}
```

**Key Parameters for RAG Use Cases**:
- `cache_mode: "BYPASS"` - Always fetch fresh content
- `word_count_threshold: 10` - Filter low-quality content blocks
- `extraction_strategy: "markdown"` - Get LLM-friendly format
- `exclude_external_links: true` - Focus on internal content
- `timeout: 30000` - 30 second timeout (adjust as needed)

**Source**: [Crawl4AI Browser & Crawler Parameters](https://docs.crawl4ai.com/api/parameters/)

#### Response Structure (CrawlResult)

The `/crawl` endpoint returns a `CrawlResult` object with comprehensive fields:

**Basic Information**:
- `url` (str) - Final URL after redirects
- `success` (bool) - Whether crawl completed without major errors
- `status_code` (Optional[int]) - HTTP response code (200, 404, etc.)
- `error_message` (Optional[str]) - Failure description if success=False
- `session_id` (Optional[str]) - Browser context ID for session reuse

**Content Fields** (Primary data for LlamaIndex):
- `html` (str) - Original unmodified HTML
- `cleaned_html` (Optional[str]) - Sanitized HTML (scripts/styles removed)
- `fit_html` (Optional[str]) - Preprocessed HTML optimized for extraction
- `markdown` (Optional[Union[str, MarkdownGenerationResult]]) - **PRIMARY FIELD for RAG**
  - `raw_markdown` - Direct HTML-to-markdown conversion
  - `fit_markdown` - Filtered markdown (removes clutter: menus, ads, footers)
  - `markdown_with_citations` - Markdown with numbered citations
  - `references_markdown` - Reference block for citations
- `extracted_content` (Optional[str]) - Structured output from extraction strategy (JSON string)

**Media & Links**:
- `media` (Dict[str, List[Dict]]) - Images, videos, audio with metadata
- `links` (Dict[str, List[Dict]]) - Internal/external links
  - `links["internal"]` - Same-domain links
  - `links["external"]` - Cross-domain links

**Additional Captures**:
- `metadata` (Optional[dict]) - Page-level metadata (title, description, OG tags)
- `screenshot` (Optional[str]) - Base64-encoded PNG
- `pdf` (Optional[bytes]) - Raw PDF bytes
- `downloaded_files` (Optional[List[str]]) - Local paths for downloads

**Recommendation for LlamaIndex Integration**:
1. Use `markdown.fit_markdown` as primary text content (best signal-to-noise ratio)
2. Extract metadata from `metadata` field (title, description)
3. Store `url` and `links` in document metadata
4. Use `status_code` and `success` for error handling

**Sources**:
- [Crawl4AI CrawlResult Documentation](https://docs.crawl4ai.com/api/crawl-result/)
- [Crawl4AI Markdown Generation](https://docs.crawl4ai.com/core/markdown-generation/)
- [Crawl4AI Link & Media Extraction](https://docs.crawl4ai.com/core/link-media/)

### Prior Art: Existing Web Readers in LlamaIndex

#### Service-Based Readers (Similar Architecture)

**FireCrawlWebReader**:
- Converts URLs to markdown via Firecrawl.dev API
- Supports multiple modes: scrape, crawl, map, search, extract
- Returns `List[Document]` with markdown content
- **Similarity**: External HTTP service for web content extraction

**AgentQLWebReader**:
- Uses AgentQL API for intelligent scraping
- Natural language queries for data extraction
- Inherits from `BasePydanticReader`
- **Similarity**: API-based web content retrieval with structured output

**ZenRowsWebReader**:
- Universal scraper using ZenRows API
- JavaScript rendering, anti-bot bypass, premium proxies
- Multiple output formats: markdown, plaintext, PDF
- **Similarity**: External service handling complex web scraping

**Key Takeaways**:
- External service-based readers are common in LlamaIndex ecosystem
- API-based approach is proven pattern
- Markdown output format is standard for RAG use cases
- Rich metadata extraction is expected

**Source**: [LlamaIndex Web Readers](https://developers.llamaindex.ai/python/framework-api-reference/readers/web/)

#### Standard HTTP Readers (Simpler Architecture)

**SimpleWebPageReader**:
- Direct HTTP requests with `requests` library
- Optional HTML-to-text conversion via `html2text`
- Configurable timeout and error handling
- Metadata extraction via custom function

**AsyncWebPageReader**:
- Concurrent HTTP requests with configurable limits
- Async/await for non-blocking I/O
- Optional HTML conversion
- **Recommendation**: Model async implementation after this reader

**TrafilaturaWebReader**:
- Uses Trafilatura library for main content extraction
- Handles boilerplate removal automatically
- Formatting options for output

**Key Takeaways**:
- Async implementation is standard for HTTP-based readers
- HTML-to-text conversion is optional feature
- Error handling should be configurable
- Metadata extraction should be flexible

**Source**: [LlamaIndex Web Readers PyPI](https://pypi.org/project/llama-index-readers-web/)

### Pitfalls to Avoid

#### Common Mistakes from Community

1. **Blocking I/O in Async Readers**
   - Problem: Using synchronous HTTP libraries in async methods
   - Solution: Use `httpx.AsyncClient` instead of `requests`
   - Example from existing code: Project already uses `httpx` in TEI client

2. **Insufficient Error Handling**
   - Problem: Unhandled HTTP errors crash entire pipeline
   - Solution: Wrap HTTP calls in try/except, provide `fail_on_error` flag
   - Example: `SimpleWebPageReader._fail_on_error` pattern

3. **Missing Circuit Breaker for External Services**
   - Problem: Cascading failures when service is down
   - Solution: Implement circuit breaker pattern (project already has this!)
   - Reference: `/home/jmagar/workspace/crawl4r/rag_ingestion/circuit_breaker.py`

4. **Lack of Retry Logic**
   - Problem: Transient network errors fail permanently
   - Solution: Exponential backoff with configurable retries
   - Example: Project's TEI client has retry implementation

5. **Ignoring Metadata in Document Creation**
   - Problem: Loss of important context (source URL, timestamp, author)
   - Solution: Always populate Document.metadata with rich fields
   - Best Practice: Include source, title, description, crawl_timestamp

6. **Not Validating Service Health Before Operations**
   - Problem: Batch operations fail midway due to service unavailability
   - Solution: Implement health check validation on startup
   - Example: Project's quality.py module validates services

7. **Missing Deterministic Document IDs**
   - Problem: Duplicate documents on re-ingestion
   - Solution: Generate deterministic UUID from URL (SHA256-derived, matches existing vector_store pattern)
   - Example: Project uses SHA256 → UUID pattern for file_path + chunk_index

8. **Timeout Misconfiguration**
   - Problem: Long-running crawls timeout prematurely
   - Solution: Make timeout configurable, default to 60-90 seconds
   - Note: Crawl4AI timeout is in milliseconds (30000 = 30s)

**Sources**: Community discussions from LlamaIndex GitHub issues and reader implementations

## Codebase Analysis

### Existing Patterns Available for Reuse

#### Circuit Breaker Pattern

**File**: `/home/jmagar/workspace/crawl4r/rag_ingestion/circuit_breaker.py`

**Features**:
- Three states: CLOSED, OPEN, HALF_OPEN
- Configurable failure threshold (default: 5)
- Automatic reset timeout (default: 60s)
- Async-safe with asyncio.Lock
- Method: `async def call(func: Callable) -> T`

**Usage Pattern**:
```python
cb = CircuitBreaker(failure_threshold=5, reset_timeout=60.0)
result = await cb.call(lambda: crawl4ai_request(url))
```

**Benefit**: Can wrap Crawl4AI API calls to prevent cascading failures

#### Structured Logging

**File**: `/home/jmagar/workspace/crawl4r/rag_ingestion/logger.py`

**Features**:
- JSON-formatted structured logs
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Standardized field names
- Performance metrics tracking

**Usage Pattern**:
```python
from rag_ingestion.logger import setup_logger
logger = setup_logger("crawl4ai_reader", log_level="INFO")
logger.info("crawling_started", url=url, timeout=30)
```

**Benefit**: Consistent logging across pipeline components

#### Configuration Management

**File**: `/home/jmagar/workspace/crawl4r/rag_ingestion/config.py`

**Features**:
- Pydantic BaseSettings for validation
- Environment variable support
- Type-safe field access
- Comprehensive field validators

**Usage Pattern**:
```python
class Crawl4AIReaderConfig(BaseSettings):
    crawl4ai_endpoint: str = "http://localhost:52004"
    timeout_seconds: int = 60
    fail_on_error: bool = False
```

**Benefit**: Can extend existing Settings class or create reader-specific config

#### Async HTTP Client Pattern (TEI Client)

**File**: `/home/jmagar/workspace/crawl4r/rag_ingestion/tei_client.py`

**Features**:
- Uses `httpx.AsyncClient` for async HTTP requests
- Exponential backoff retry logic
- Circuit breaker integration
- Health check validation
- Batch processing support

**Key Methods**:
- `async def health_check() -> bool`
- `async def embed_batch(texts: List[str]) -> List[List[float]]`

**Usage Pattern**:
```python
async with httpx.AsyncClient(timeout=60.0) as client:
    response = await client.post(
        f"{endpoint}/crawl",
        json={"url": url, "crawler_params": params}
    )
    response.raise_for_status()
    return response.json()
```

**Benefit**: Can directly adapt httpx patterns from TEI client

### Dependencies Already Available

From `pyproject.toml`:
- ✅ `llama-index-core>=0.14.0` - Core LlamaIndex library
- ✅ `pydantic>=2.0.0` - For BasePydanticReader
- ✅ `httpx` (implicit via llama-index-core) - Async HTTP client
- ❌ **Missing**: No explicit `httpx` dependency (should add)

**Recommendation**: Add `httpx>=0.24.0` to dependencies

### Constraints and Limitations

#### Crawl4AI Service Constraints

1. **Port Configuration**: Service runs on port 52004 (mapped from 11235)
   - Health endpoint: `http://localhost:52004/health`
   - API endpoint: `http://localhost:52004/crawl`
   - Must use high port (52000+) per project standards

2. **Docker Dependency**: Crawl4AI runs as Docker container
   - Requires Docker network: `crawl4r`
   - Service name: `crawl4ai`
   - Health check interval: 30s

3. **Browser Resource Usage**: Crawl4AI uses headless browser
   - Shared memory: 2GB (`shm_size: "2gb"`)
   - Can be resource-intensive for concurrent requests
   - Recommendation: Limit concurrent crawls (default: 5-10)

4. **Response Size**: Web pages can be large
   - Markdown responses can exceed 100KB
   - Recommendation: Implement streaming for large results
   - Consider chunking long documents

#### Integration with Existing Pipeline

1. **Async-First Architecture**: Pipeline uses asyncio throughout
   - Reader MUST implement `aload_data()` for async compatibility
   - Cannot use blocking I/O without breaking pipeline

2. **LlamaIndex Document Chunking**: Pipeline uses custom chunker
   - File: `/home/jmagar/workspace/crawl4r/rag_ingestion/chunker.py`
   - Strategy: Markdown-aware heading-based splitting
   - Chunk size: 512 tokens, 15% overlap
   - Reader should return full documents (chunking happens downstream)

3. **Vector Store Integration**: Qdrant expects specific metadata
   - Required fields: `file_path_relative`, `file_path_absolute`
   - Optional fields: `section_path`, `source_type`, `ingestion_timestamp`
   - Reader should populate `source_type: "web_crawl"`

4. **Circuit Breaker Integration**: TEI and Qdrant use circuit breakers
   - Reader should use same pattern for consistency
   - Failure threshold: 5 consecutive failures
   - Reset timeout: 60 seconds

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | **High** | Clear API contract, proven reader patterns, reusable infrastructure |
| Effort Estimate | **Small (S)** | 2-3 days: 1 day implementation, 0.5 days testing, 0.5 days integration |
| Risk Level | **Low** | Well-documented API, existing circuit breaker, no novel patterns required |
| Integration Complexity | **Low** | Aligns perfectly with existing async pipeline, minimal changes needed |
| Maintenance Burden | **Low** | Crawl4AI has stable API, minimal dependencies, clear error handling |

### Technical Viability: High

**Strengths**:
- Crawl4AI provides comprehensive REST API with clear documentation
- LlamaIndex `BasePydanticReader` provides proven contract
- Project has reusable circuit breaker, logging, and async HTTP patterns
- Response format (markdown) maps directly to LlamaIndex Document requirements

**Challenges** (all mitigable):
- Need to add `httpx` dependency (trivial)
- Async job polling requires timeout/retry logic (can reuse TEI client pattern)
- Large response handling (can implement streaming if needed)

### Effort Estimate: Small (2-3 days)

**Day 1: Core Implementation**
- Create `Crawl4AIReader` class extending `BasePydanticReader`
- Implement `__init__` with configuration (endpoint, timeout, error handling)
- Implement `aload_data(urls: List[str])` with httpx async client
- Implement `load_data(urls: List[str])` as sync wrapper
- Integrate circuit breaker pattern

**Day 2: Testing & Error Handling**
- Unit tests for reader instantiation and configuration
- Integration tests with real Crawl4AI service
- Error handling tests (timeouts, 404s, service down)
- Circuit breaker integration tests
- Health check validation

**Day 3: Documentation & Integration**
- Add docstrings and usage examples
- Update pyproject.toml dependencies
- Integration with existing pipeline
- Performance benchmarking (crawl times, throughput)

### Risk Level: Low

**Mitigating Factors**:
1. **Proven Patterns**: SimpleWebPageReader, AsyncWebPageReader provide reference implementations
2. **Existing Infrastructure**: Circuit breaker, logger, httpx client already implemented
3. **Stable API**: Crawl4AI v0.7.x has mature, documented REST API
4. **Comprehensive Testing**: Project has pytest setup with async support

**Potential Risks & Mitigations**:

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Crawl4AI service unavailable | High | Low | Circuit breaker prevents cascading failures, health check on startup |
| Large response timeouts | Medium | Medium | Configurable timeout (60-90s), async job endpoint for long crawls |
| Rate limiting / resource exhaustion | Medium | Medium | Limit concurrent requests (5-10), implement backoff |
| Markdown parsing issues | Low | Low | Use `fit_markdown` field (pre-filtered), fallback to `raw_markdown` |

## Recommendations for Requirements

Based on research findings, the following requirements are recommended for the implementation:

### Functional Requirements

1. **FR-1: Reader Initialization**
   - Accept `crawl4ai_endpoint` URL (default: `http://localhost:52004`)
   - Accept `timeout_seconds` (default: 60)
   - Accept `fail_on_error` flag (default: False)
   - Accept `max_concurrent_requests` (default: 5)
   - Validate endpoint health on initialization

2. **FR-2: Synchronous URL Loading**
   - Implement `load_data(urls: List[str]) -> List[Document]`
   - Support single URL: `load_data(["https://example.com"])`
   - Support multiple URLs: `load_data(["url1", "url2", ...])`
   - Return one Document per successfully crawled URL

3. **FR-3: Asynchronous URL Loading** (PRIMARY)
   - Implement `aload_data(urls: List[str]) -> List[Document]`
   - Use `httpx.AsyncClient` for non-blocking I/O
   - Process URLs concurrently with configurable limit
   - Return Documents in same order as input URLs

4. **FR-4: Document Structure**
   - Document.text = `result.markdown.fit_markdown` (primary content)
   - Document.metadata = {
       - `source`: URL
       - `title`: page title from metadata
       - `description`: page description
       - `status_code`: HTTP status
       - `crawl_timestamp`: ISO8601 timestamp
       - `internal_links_count`: len(result.links["internal"])
       - `external_links_count`: len(result.links["external"])
       - `source_type`: "web_crawl"
       - `source_url`: url (indexed in Qdrant for deduplication)
     }
   - Document.id_ = deterministic UUID (SHA256-derived) from URL

5. **FR-5: Error Handling**
   - Wrap HTTP calls in circuit breaker
   - Log errors with structured logging
   - If `fail_on_error=True`: raise exception on first failure
   - If `fail_on_error=False`: skip failed URLs, continue processing
   - Return empty list if all URLs fail

6. **FR-6: Circuit Breaker Integration**
   - Use existing `CircuitBreaker` class
   - Failure threshold: 5 consecutive failures
   - Reset timeout: 60 seconds
   - State: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)

### Non-Functional Requirements

1. **NFR-1: Performance**
   - Concurrent crawling with configurable limit (default: 5)
   - Timeout per request: 60 seconds
   - Throughput target: 50-100 URLs/minute (depends on page complexity)

2. **NFR-2: Reliability**
   - Exponential backoff retry: [1s, 2s, 4s] for transient errors
   - Circuit breaker prevents cascading failures
   - Health check validation before batch operations

3. **NFR-3: Observability**
   - Structured logging for all operations (info, warning, error)
   - Log fields: url, duration_ms, status_code, success, error_message
   - Integration with existing logger module

4. **NFR-4: Maintainability**
   - Type hints on all methods
   - Google-style docstrings
   - Unit test coverage: 85%+
   - Integration tests with real Crawl4AI service

5. **NFR-5: Compatibility**
   - Compatible with LlamaIndex IngestionPipeline
   - Compatible with existing RAG pipeline components
   - Python 3.10+ requirement

### Configuration Requirements

Add to existing `Settings` class in `config.py`:

```python
# Crawl4AI Reader Configuration
crawl4ai_endpoint: str = "http://localhost:52004"
crawl4ai_timeout_seconds: int = 60
crawl4ai_max_concurrent_requests: int = 5
crawl4ai_fail_on_error: bool = False
```

### Dependency Requirements

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing dependencies
    "httpx>=0.24.0",  # Async HTTP client
]
```

## Open Questions

1. **Chunking Strategy**: Should the reader return full-page documents (recommended) or pre-chunked sections?
   - **Recommendation**: Return full documents, let downstream chunker handle splitting
   - **Rationale**: Existing pipeline has sophisticated markdown-aware chunker

2. **Async Job Endpoint**: Should we support `/crawl/job` for long-running crawls?
   - **Recommendation**: Start with synchronous `/crawl`, add async job support in v2
   - **Rationale**: Simpler implementation, most pages crawl in <60s

3. **Webhook Support**: Should we implement webhook listener for async jobs?
   - **Recommendation**: Not for MVP, consider for production workloads
   - **Rationale**: Requires additional infrastructure (webhook server)

4. **Streaming Support**: Should we support `/crawl/stream` endpoint?
   - **Recommendation**: Not for MVP, evaluate based on performance needs
   - **Rationale**: Adds complexity, most use cases satisfied by standard endpoint

5. **Metadata Extraction**: Which metadata fields are most valuable for RAG?
   - **Recommendation**: Start with core fields (source, title, description, timestamp)
   - **Rationale**: Can extend metadata schema based on usage patterns

6. **Link Extraction**: Should we crawl linked pages recursively?
   - **Recommendation**: Not for MVP, reader should handle explicit URLs only
   - **Rationale**: Recursive crawling is separate concern (crawler vs. reader)

7. **Cache Mode**: Should we enable Crawl4AI's caching?
   - **Recommendation**: Default to `cache_mode: "BYPASS"` for fresh content
   - **Rationale**: RAG pipelines typically want current data

8. **Content Filtering**: Should we apply word count threshold or CSS selectors?
   - **Recommendation**: Use Crawl4AI's defaults (`word_count_threshold: 10`)
   - **Rationale**: Crawl4AI's filtering is well-tuned, avoid premature optimization

## Related Specs

### Spec: rag-ingestion

**Location**: `/home/jmagar/workspace/crawl4r/specs/rag-ingestion/`

**Relationship**: **High - Direct Integration**

**Original Goal**: Build RAG ingestion pipeline that monitors folder for markdown files, generates embeddings, stores in Qdrant

**Overlap Areas**:
- **Document Processing**: Both specs deal with ingesting documents into Qdrant
- **LlamaIndex Integration**: Both use LlamaIndex for orchestration
- **Async Architecture**: Both require async/await for non-blocking I/O
- **Metadata Management**: Both need to populate document metadata for vector store
- **Circuit Breaker Pattern**: Both use circuit breaker for external service failures

**Integration Points**:
1. Crawl4AIReader will feed Documents into existing pipeline
2. Documents from reader will be chunked by existing chunker (512 tokens, 15% overlap)
3. TEI client will generate embeddings for crawled content
4. Vector store will handle upsert with deterministic UUIDs (SHA256-derived)
5. Vector store will handle automatic deduplication (delete old versions before inserting new)

**mayNeedUpdate**: **true** (UPDATED)

**Rationale**: The new reader is additive to the pipeline, but requires **two minor additions to vector_store.py**:
1. Add `source_url` to PAYLOAD_INDEXES for fast deduplication queries
2. Add `delete_by_url(source_url: str)` method (mirrors existing `delete_by_file()` pattern)

These changes are non-breaking and follow existing patterns. The file watcher and other existing components are unaffected.

## Quality Commands

Based on analysis of `pyproject.toml`, the project uses the following quality commands:

| Type | Command | Source |
|------|---------|--------|
| Lint | `ruff check .` | pyproject.toml [tool.ruff] |
| Lint (fix) | `ruff check . --fix` | pyproject.toml [tool.ruff] |
| TypeCheck | `ty check rag_ingestion/` | pyproject.toml [tool.ty] |
| Unit Test | `pytest tests/unit/` | pyproject.toml [tool.pytest.ini_options] |
| Integration Test | `pytest tests/integration/` | pyproject.toml [tool.pytest.ini_options] |
| Test (all) | `pytest` | pyproject.toml [tool.pytest.ini_options] |
| Coverage | `pytest --cov=rag_ingestion --cov-report=term` | pyproject.toml [tool.coverage] |
| Build | Not found | N/A |

**Local CI Simulation**:
```bash
ruff check . && ty check rag_ingestion/ && pytest --cov=rag_ingestion --cov-report=term
```

**Notes**:
- No CI workflow files found (`.github/workflows/`)
- No Makefile found
- No package.json (Python project only)
- Quality commands extracted from pyproject.toml tool configurations
- Test paths: `tests/unit/` and `tests/integration/`
- Cache directory: `.cache/` for pytest and ruff

## Sources

### LlamaIndex Documentation
- [BasePydanticReader - LlamaIndex](https://docs.llamaindex.ai/en/v0.10.19/api/llama_index.core.readers.base.BasePydanticReader.html)
- [BaseReader - LlamaIndex](https://docs.llamaindex.ai/en/v0.10.20/api/llama_index.core.readers.base.BaseReader.html)
- [LlamaIndex Document Management](https://docs.llamaindex.ai/en/stable/module_guides/indexing/document_management/)
- [LlamaIndex Web Readers](https://developers.llamaindex.ai/python/framework-api-reference/readers/web/)
- [SimpleDirectoryReader](https://developers.llamaindex.ai/python/framework/module_guides/loading/simpledirectoryreader/)
- [S3 Reader](https://developers.llamaindex.ai/python/framework-api-reference/readers/s3/)

### Crawl4AI Documentation
- [Docker Deployment - Crawl4AI](https://docs.crawl4ai.com/core/docker-deployment/)
- [Self-Hosting Guide - Crawl4AI](https://docs.crawl4ai.com/core/self-hosting/)
- [CrawlResult Documentation](https://docs.crawl4ai.com/api/crawl-result/)
- [Browser & Crawler Parameters](https://docs.crawl4ai.com/api/parameters/)
- [Markdown Generation](https://docs.crawl4ai.com/core/markdown-generation/)
- [Link & Media Extraction](https://docs.crawl4ai.com/core/link-media/)
- [Crawl4AI v0.7.6 Release Notes](https://docs.crawl4ai.com/blog/releases/0.7.6/)

### GitHub Repositories
- [SimpleWebPageReader Source Code](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-web/llama_index/readers/web/simple_web/base.py)
- [Crawl4AI GitHub Repository](https://github.com/unclecode/crawl4ai)

### Python Libraries
- [httpx-retries PyPI](https://will-ockmore.github.io/httpx-retries/)
- [PyBreaker - Circuit Breaker](https://github.com/danielfm/pybreaker)
- [llama-index-readers-web PyPI](https://pypi.org/project/llama-index-readers-web/)

### Project Files
- `/home/jmagar/workspace/crawl4r/rag_ingestion/circuit_breaker.py` - Circuit breaker implementation
- `/home/jmagar/workspace/crawl4r/rag_ingestion/tei_client.py` - Async HTTP client pattern
- `/home/jmagar/workspace/crawl4r/rag_ingestion/config.py` - Configuration management
- `/home/jmagar/workspace/crawl4r/rag_ingestion/logger.py` - Structured logging
- `/home/jmagar/workspace/crawl4r/pyproject.toml` - Project dependencies and tool configuration
- `/home/jmagar/workspace/crawl4r/docker-compose.yaml` - Crawl4AI service configuration
