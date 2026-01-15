---
spec: llamaindex-crawl4ai-reader
phase: requirements
created: 2026-01-15T00:00:00Z
---

# Requirements: LlamaIndex Crawl4AI Reader

## Goal

Create a production-ready LlamaIndex reader that integrates with the existing Crawl4AI Docker service to fetch web content as markdown-formatted documents. This reader will enable the RAG ingestion pipeline to process web pages alongside local markdown files, extending the system's data sources with minimal architectural changes. All development will follow Test-Driven Development (TDD) methodology with 85%+ test coverage.

## User Stories

### US-1: Configure Reader with Service Endpoint
**As a** pipeline developer
**I want to** configure the Crawl4AI reader with custom endpoint and timeout settings
**So that** I can adapt the reader to different deployment environments and performance requirements

**Acceptance Criteria:**
- [ ] AC-1.1: Reader accepts `crawl4ai_endpoint` URL parameter (default: `http://localhost:52004`)
- [ ] AC-1.2: Reader accepts `timeout_seconds` parameter (default: 60, range: 10-300)
- [ ] AC-1.3: Reader accepts `fail_on_error` boolean flag (default: False)
- [ ] AC-1.4: Reader accepts `max_concurrent_requests` parameter (default: 5, range: 1-20)
- [ ] AC-1.5: Reader validates endpoint health on initialization via `/health` endpoint
- [ ] AC-1.6: Reader raises `ValueError` if endpoint is unreachable during validation
- [ ] AC-1.7: Configuration parameters are exposed as Pydantic fields with type validation

**TDD Checklist:**
- [ ] Write test for default configuration instantiation (RED)
- [ ] Write test for custom endpoint configuration (RED)
- [ ] Write test for health check validation on init (RED)
- [ ] Write test for validation failure on unreachable endpoint (RED)
- [ ] Implement minimal configuration class (GREEN)
- [ ] Refactor configuration into Settings integration (REFACTOR)

### US-2: Load Single Web Page Synchronously
**As a** RAG pipeline user
**I want to** load a single URL's content as a LlamaIndex Document
**So that** I can process web pages through the existing ingestion pipeline

**Acceptance Criteria:**
- [ ] AC-2.1: `load_data(urls: List[str])` method accepts single URL in list format
- [ ] AC-2.2: Method returns `List[Document]` with exactly one document for successful crawl
- [ ] AC-2.3: Document.text contains `markdown.fit_markdown` content from Crawl4AI response
- [ ] AC-2.4: Document.metadata includes: `source`, `source_url`, `title`, `description`, `status_code`, `crawl_timestamp`, `source_type="web_crawl"`
- [ ] AC-2.5: Document.id_ is deterministic UUID (SHA256-derived) of URL
- [ ] AC-2.6: Method raises exception if `fail_on_error=True` and crawl fails
- [ ] AC-2.7: Method returns empty list if `fail_on_error=False` and crawl fails
- [ ] AC-2.8: Method logs crawl start, duration, and result with structured logging

**TDD Checklist:**
- [ ] Write test for single URL crawl success (RED)
- [ ] Write test for Document structure validation (RED)
- [ ] Write test for deterministic ID generation (RED)
- [ ] Write test for fail_on_error=True behavior (RED)
- [ ] Write test for fail_on_error=False behavior (RED)
- [ ] Implement minimal load_data method (GREEN)
- [ ] Refactor error handling and logging (REFACTOR)

### US-3: Load Multiple Web Pages Asynchronously
**As a** RAG pipeline developer
**I want to** load multiple URLs concurrently with async/await
**So that** I can efficiently process batches of web pages without blocking

**Acceptance Criteria:**
- [ ] AC-3.1: `aload_data(urls: List[str])` method accepts list of multiple URLs
- [ ] AC-3.2: Method uses `httpx.AsyncClient` for non-blocking HTTP requests
- [ ] AC-3.3: Method processes URLs concurrently with limit from `max_concurrent_requests`
- [ ] AC-3.4: Method returns `List[Document | None]` in same order as input URLs (preserves order even with failures)
- [ ] AC-3.5: Method includes `None` for failed URLs when `fail_on_error=False`, raises exception when `fail_on_error=True`
- [ ] AC-3.6: Method handles partial failures gracefully (some URLs succeed, some fail)
- [ ] AC-3.7: Method respects timeout for each individual URL request
- [ ] AC-3.8: Method logs batch statistics: total URLs, successes, failures, duration

**TDD Checklist:**
- [ ] Write test for multiple URLs concurrent processing (RED)
- [ ] Write test for concurrency limit enforcement (RED)
- [ ] Write test for document ordering preservation (RED)
- [ ] Write test for partial failure handling (RED)
- [ ] Write test for timeout per-request (RED)
- [ ] Implement minimal aload_data with asyncio.gather (GREEN)
- [ ] Refactor with asyncio.Semaphore for concurrency control (REFACTOR)

### US-4: Integrate Circuit Breaker for Fault Tolerance
**As a** pipeline operator
**I want to** automatically stop making requests to Crawl4AI when the service is failing
**So that** I prevent cascading failures and allow the service time to recover

**Acceptance Criteria:**
- [ ] AC-4.1: Reader uses existing `CircuitBreaker` class from `rag_ingestion.circuit_breaker`
- [ ] AC-4.2: Circuit breaker wraps all HTTP calls to Crawl4AI service
- [ ] AC-4.3: Circuit opens after 5 consecutive failures (project standard)
- [ ] AC-4.4: Circuit remains open for 60 seconds before attempting recovery (project standard)
- [ ] AC-4.5: Circuit transitions to HALF_OPEN for test call after reset timeout
- [ ] AC-4.6: Circuit closes on successful test call, resetting failure counter
- [ ] AC-4.7: Failed calls during OPEN state raise `CircuitBreakerError` with clear message
- [ ] AC-4.8: Circuit state transitions are logged with structured logging

**TDD Checklist:**
- [ ] Write test for circuit breaker initialization (RED)
- [ ] Write test for circuit opening after threshold (RED)
- [ ] Write test for circuit remaining open during timeout (RED)
- [ ] Write test for circuit recovery transition (RED)
- [ ] Write test for CircuitBreakerError during OPEN state (RED)
- [ ] Implement circuit breaker wrapper around HTTP calls (GREEN)
- [ ] Refactor error messages and state logging (REFACTOR)

### US-5: Enrich Documents with Comprehensive Metadata
**As a** RAG system user
**I want to** receive documents with rich metadata from crawled pages
**So that** I can filter, sort, and contextualize search results effectively

**Acceptance Criteria:**
- [ ] AC-5.1: Document.metadata includes `source`: original URL
- [ ] AC-5.2: Document.metadata includes `source_url`: original URL (indexed in Qdrant for deduplication)
- [ ] AC-5.3: Document.metadata includes `title`: page title from Crawl4AI metadata
- [ ] AC-5.4: Document.metadata includes `description`: page description from Crawl4AI metadata
- [ ] AC-5.5: Document.metadata includes `status_code`: HTTP status code
- [ ] AC-5.6: Document.metadata includes `crawl_timestamp`: ISO8601 timestamp
- [ ] AC-5.7: Document.metadata includes `internal_links_count`: count of same-domain links
- [ ] AC-5.8: Document.metadata includes `external_links_count`: count of cross-domain links
- [ ] AC-5.9: Document.metadata includes `source_type`: fixed value "web_crawl"
- [ ] AC-5.10: All metadata values are flat types (str, int, float) compatible with Qdrant
- [ ] AC-5.11: Missing metadata fields default to empty string or 0, never None

**TDD Checklist:**
- [ ] Write test for complete metadata structure (RED)
- [ ] Write test for metadata type validation (RED)
- [ ] Write test for missing metadata field defaults (RED)
- [ ] Write test for link count calculation (RED)
- [ ] Implement metadata extraction from CrawlResult (GREEN)
- [ ] Refactor metadata builder into helper function (REFACTOR)

### US-6: Handle Crawl Failures Gracefully
**As a** pipeline operator
**I want to** continue processing valid URLs when some URLs fail to crawl
**So that** batch operations don't fail completely due to individual errors

**Acceptance Criteria:**
- [ ] AC-6.1: Reader catches and logs HTTP errors (4xx, 5xx) without crashing
- [ ] AC-6.2: Reader catches and logs network errors (timeouts, DNS failures) without crashing
- [ ] AC-6.3: Reader catches and logs Crawl4AI service errors (success=False) without crashing
- [ ] AC-6.4: Error logs include: URL, error type, error message, timestamp
- [ ] AC-6.5: When `fail_on_error=False`, failed URLs are skipped and processing continues
- [ ] AC-6.6: When `fail_on_error=True`, first error raises exception with context
- [ ] AC-6.7: Batch operations return partial results for successful URLs
- [ ] AC-6.8: All errors are logged with structured logging at ERROR level

**TDD Checklist:**
- [ ] Write test for HTTP 404 error handling (RED)
- [ ] Write test for HTTP 500 error handling (RED)
- [ ] Write test for network timeout handling (RED)
- [ ] Write test for DNS resolution failure (RED)
- [ ] Write test for Crawl4AI service error (success=False) (RED)
- [ ] Write test for partial batch success (RED)
- [ ] Implement error handling with try/except blocks (GREEN)
- [ ] Refactor error categorization and logging (REFACTOR)

### US-7: Implement Exponential Backoff Retry Logic
**As a** pipeline operator
**I want to** automatically retry failed requests with increasing delays
**So that** transient network issues don't cause permanent failures

**Acceptance Criteria:**
- [ ] AC-7.1: Reader retries failed requests up to 3 times (configurable)
- [ ] AC-7.2: Retry delays follow exponential backoff: [1s, 2s, 4s]
- [ ] AC-7.3: Only transient errors trigger retries (5xx, timeouts, network errors)
- [ ] AC-7.4: Client errors (4xx) do not trigger retries
- [ ] AC-7.5: Circuit breaker errors do not trigger retries
- [ ] AC-7.6: Each retry attempt is logged with attempt number and delay
- [ ] AC-7.7: After max retries, error is raised or logged based on `fail_on_error`
- [ ] AC-7.8: Retry logic respects per-request timeout (doesn't extend it)

**TDD Checklist:**
- [ ] Write test for successful retry after transient error (RED)
- [ ] Write test for retry count limit enforcement (RED)
- [ ] Write test for exponential backoff timing (RED)
- [ ] Write test for no retry on 4xx errors (RED)
- [ ] Write test for no retry on circuit breaker errors (RED)
- [ ] Implement retry decorator or helper function (GREEN)
- [ ] Refactor retry logic into reusable utility (REFACTOR)

### US-8: Automatic Deduplication on Re-Crawl
**As a** RAG system operator
**I want to** automatically remove old versions of crawled URLs before ingesting new versions
**So that** I don't accumulate duplicate documents and waste storage on outdated content

**Acceptance Criteria:**
- [ ] AC-8.1: Reader deletes existing documents with matching `source_url` before crawling
- [ ] AC-8.2: Deduplication queries Qdrant using indexed `source_url` field for fast lookups
- [ ] AC-8.3: Deduplication is enabled by default (`enable_deduplication=True`)
- [ ] AC-8.4: Deduplication can be disabled via configuration parameter
- [ ] AC-8.5: Reader accepts optional `vector_store` parameter for deduplication integration
- [ ] AC-8.6: If `vector_store` is None, deduplication is skipped gracefully
- [ ] AC-8.7: Deleted vector count is logged with structured logging
- [ ] AC-8.8: VectorStoreManager.delete_by_url() method mirrors delete_by_file() pattern
- [ ] AC-8.9: `source_url` is added to PAYLOAD_INDEXES for efficient Qdrant queries
- [ ] AC-8.10: Deduplication matches file watcher pattern: delete → process

**TDD Checklist:**
- [ ] Write test for deduplication enabled by default (RED)
- [ ] Write test for deduplication with vector_store=None (RED)
- [ ] Write test for delete_by_url() call before crawl (RED)
- [ ] Write test for deduplication logging (RED)
- [ ] Write test for VectorStoreManager.delete_by_url() method (RED)
- [ ] Implement _deduplicate_url() method (GREEN)
- [ ] Implement VectorStoreManager.delete_by_url() (GREEN)
- [ ] Refactor deduplication into aload_data() integration (REFACTOR)

## Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | LlamaIndex Integration | High | Inherit from `BasePydanticReader`, implement `load_data()` and `aload_data()`, compatible with `IngestionPipeline` |
| FR-2 | Async HTTP Client | High | Use `httpx.AsyncClient` for all HTTP requests, support concurrent requests with semaphore |
| FR-3 | Document Generation | High | Generate LlamaIndex `Document` objects with markdown text and comprehensive metadata |
| FR-4 | Deterministic IDs | High | Generate deterministic UUID (SHA256-derived) of URL as Document.id_ for idempotent ingestion |
| FR-5 | Crawl4AI API Integration | High | Call `POST /crawl` endpoint with JSON body, parse `CrawlResult` response |
| FR-6 | Markdown Content Extraction | High | Extract `markdown.fit_markdown` as primary text content, fallback to `raw_markdown` |
| FR-7 | Metadata Enrichment | Medium | Extract title, description, status_code, links from CrawlResult |
| FR-8 | Error Handling | High | Handle HTTP errors, network errors, service errors gracefully |
| FR-9 | Circuit Breaker | High | Integrate existing `CircuitBreaker` to prevent cascading failures |
| FR-10 | Retry Logic | Medium | Implement exponential backoff retry for transient errors |
| FR-11 | Structured Logging | Medium | Log all operations with structured JSON logging using existing logger |
| FR-12 | Configuration Management | Medium | Extend `Settings` class with Crawl4AI-specific fields |
| FR-13 | Health Check Validation | Medium | Validate Crawl4AI service health on reader initialization |
| FR-14 | Concurrency Control | Medium | Limit concurrent requests using `asyncio.Semaphore` |
| FR-15 | Timeout Management | Medium | Enforce per-request timeout, configurable via settings |
| FR-16 | Batch Processing | Low | Support multiple URLs in single `aload_data()` call |
| FR-17 | Partial Failure Handling | Medium | Return partial results when some URLs fail in batch |
| FR-18 | Type Safety | Medium | Full type hints on all methods, pass `ty check` without errors |
| FR-19 | Automatic Deduplication | High | Delete old documents before re-crawling URLs (matches file watcher pattern) |
| FR-20 | Source URL Indexing | Medium | Add source_url to Qdrant payload indexes for fast deduplication queries |

## Non-Functional Requirements

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-1 | Test Coverage | Line coverage | 85%+ measured by pytest-cov |
| NFR-2 | TDD Compliance | Development workflow | All features written test-first (RED-GREEN-REFACTOR) |
| NFR-3 | Performance | Throughput | 50-100 URLs/minute for average web pages |
| NFR-4 | Concurrency | Max concurrent requests | 5 (default), configurable 1-20 |
| NFR-5 | Timeout | Per-request timeout | 60 seconds (default), configurable 10-300 |
| NFR-6 | Reliability | Circuit breaker threshold | 5 consecutive failures before opening |
| NFR-7 | Reliability | Circuit breaker reset | 60 seconds before attempting recovery |
| NFR-8 | Retry Strategy | Max retry attempts | 3 attempts with exponential backoff [1s, 2s, 4s] |
| NFR-9 | Type Checking | Type hint coverage | 100% of public API, pass `ty check` |
| NFR-10 | Code Quality | Linting | Zero errors from `ruff check .` |
| NFR-11 | Documentation | Docstring coverage | 100% of public classes/methods with Google-style docstrings |
| NFR-12 | Compatibility | LlamaIndex version | Compatible with llama-index-core>=0.14.0 |
| NFR-13 | Compatibility | Python version | Python 3.10+ required |
| NFR-14 | Integration | Pipeline compatibility | Works with existing async RAG pipeline |
| NFR-15 | Observability | Structured logging | All operations logged with JSON format |

## TDD Methodology Requirements

### Test-First Development Workflow

**MANDATORY for all implementation:**

1. **RED Phase**: Write failing test that defines desired behavior
   - Test must fail for the right reason (functionality not implemented)
   - Test must be clear, specific, and verify single behavior
   - Test must use realistic input data and mock external dependencies

2. **GREEN Phase**: Write minimal code to pass test
   - Implement only what's needed to make test pass
   - No premature optimization or extra features
   - Verify test passes before moving to refactor

3. **REFACTOR Phase**: Improve code while keeping tests green
   - Extract duplicated logic into helper functions
   - Improve naming and code organization
   - Verify all tests still pass after changes

### Test Organization

**Unit Tests** (`tests/unit/test_crawl4ai_reader.py`):
- Test reader configuration and initialization
- Test Document generation logic
- Test metadata extraction
- Test error handling logic
- Test deterministic ID generation
- Mock all HTTP calls with `httpx.MockTransport`
- Mock CircuitBreaker to test state transitions

**Integration Tests** (`tests/integration/test_crawl4ai_reader_integration.py`):
- Test real HTTP calls to Crawl4AI service (requires service running)
- Test end-to-end document loading from live URLs
- Test circuit breaker integration with real failures
- Test retry logic with real network conditions
- Test concurrency with multiple URLs
- Use pytest fixtures for service availability checks

### Test Quality Standards

- [ ] All tests follow Arrange-Act-Assert pattern
- [ ] All tests are independent and isolated
- [ ] All tests are deterministic (no flaky tests)
- [ ] All tests have descriptive names: `test_<method>_<scenario>_<expected_result>`
- [ ] All tests use pytest fixtures for setup/teardown
- [ ] All integration tests check service health before running (skip if unavailable)
- [ ] All mocks are verified with `assert_called_once_with()` or similar
- [ ] All assertions include descriptive messages

### Coverage Requirements

**Minimum Coverage Thresholds:**
- Overall: 85%
- Critical paths (load_data, aload_data): 95%
- Error handling: 90%
- Configuration/initialization: 90%

**Excluded from Coverage:**
- Type stubs and protocols
- Deprecation warnings
- Debug-only code paths

## Glossary

- **BasePydanticReader**: LlamaIndex base class for serializable data loaders with Pydantic support
- **Circuit Breaker**: Design pattern that prevents cascading failures by temporarily blocking calls to failing services
- **Crawl4AI**: Self-hosted web crawling service that extracts markdown and metadata from web pages
- **CrawlResult**: Response object from Crawl4AI containing HTML, markdown, metadata, and links
- **Document**: LlamaIndex data structure containing text content, metadata, and unique identifier
- **Exponential Backoff**: Retry strategy with increasing delays between attempts (1s, 2s, 4s, etc.)
- **fit_markdown**: Pre-filtered markdown from Crawl4AI with high signal-to-noise ratio (removes nav, ads, footers)
- **Idempotent**: Operation that produces same result when called multiple times with same input
- **Semaphore**: Asyncio primitive for limiting concurrent operations
- **Structured Logging**: JSON-formatted logging with consistent field names for machine parsing
- **TDD**: Test-Driven Development methodology (RED-GREEN-REFACTOR cycle)
- **TEI**: Text Embeddings Inference, HuggingFace service for generating vector embeddings

## Out of Scope

**This specification covers the basic LlamaIndex Crawl4AI reader (v1).** Advanced features listed below are deferred to a separate specification for Advanced Reader (v2).

### Deferred to Advanced Reader v2

The following features require significant additional complexity and will be designed/implemented in a separate specification:

- **Recursive Link Crawling**: Following internal links to discover and crawl additional pages automatically (requires URL frontier, visited tracking, deduplication, depth limits)
- **Async Job Endpoint**: Using `/crawl/job` endpoint for long-running crawls with polling and job status management
- **Webhook Listener**: Implementing webhook server/endpoint for async job notifications (requires additional infrastructure)
- **Streaming Support**: Using `/crawl/stream` endpoint with NDJSON response handling (requires streaming parser)
- **Extended Metadata**: Extracting additional fields like author, keywords, language, Open Graph tags, Twitter Card metadata (beyond core 9 fields)
- **Content Filtering Options**: Exposing custom CSS selectors, excluded tags, configurable word_count_threshold (beyond Crawl4AI defaults)
- **Configurable Cache Modes**: Supporting cache modes beyond BYPASS (ENABLED, REFRESH, etc.)

### Permanently Out of Scope

The following are not planned for any version:

- **JavaScript Execution**: Will not expose Crawl4AI's `js_code` parameter (advanced feature, security concern)
- **PDF/Screenshot Generation**: Will not request PDF or screenshot outputs (not needed for RAG)
- **Session Management**: Will not reuse browser sessions across requests (adds state complexity)
- **Custom User Agents**: Will not expose user agent configuration (use Crawl4AI defaults)
- **Proxy Configuration**: Will not expose proxy settings (deployment concern, not reader concern)
- **Link Graph Extraction**: Will not build link graph structures (simple link counts sufficient)
- **Lazy Loading**: Will not implement `lazy_load_data()` (async batch loading sufficient)
- **Pagination Support**: Will not handle paginated content automatically (caller responsibility)
- **Rate Limiting**: Will not implement API rate limiting (handled by circuit breaker and concurrency limit)

## Dependencies

### External Dependencies

1. **Crawl4AI Service**: Docker container must be running on configured endpoint
   - Service: `crawl4ai` on Docker network `crawl4r`
   - Port: 52004 (mapped from container port 11235)
   - Health check: `GET /health` must return 200 OK
   - Version: v0.7.x or later

2. **Python Dependencies**:
   - `llama-index-core>=0.14.0` - Already in project
   - `pydantic>=2.0.0` - Already in project
   - `httpx==0.28.1` - **MUST ADD** as first implementation task to pyproject.toml dependencies
   - `pytest>=7.0.0` - Already in project (dev dependency)
   - `pytest-asyncio>=0.21.0` - Already in project (dev dependency)
   - `pytest-cov>=4.0.0` - Already in project (dev dependency)

3. **Project Modules**:
   - `rag_ingestion.circuit_breaker.CircuitBreaker` - Existing circuit breaker implementation
   - `rag_ingestion.logger.setup_logger` - Existing structured logging
   - `rag_ingestion.config.Settings` - Will extend with Crawl4AI configuration

### Integration Points

1. **RAG Ingestion Pipeline**: Reader outputs will be consumed by existing pipeline components
   - Documents must be compatible with `rag_ingestion.chunker` (512 tokens, 15% overlap)
   - Metadata must include `source_type="web_crawl"` for Qdrant filtering
   - Document IDs must be deterministic (SHA256-derived UUID)

2. **Vector Store**: Documents will be embedded and stored in Qdrant
   - Metadata must use flat types (str, int, float) only
   - Required metadata fields: `source`, `source_url`, `source_type`, `ingestion_timestamp`
   - `source_url` must be indexed in Qdrant for fast deduplication queries

3. **Embeddings Service**: Chunked content will be processed by TEI
   - No special integration needed (handled by existing pipeline)

## Success Criteria

The implementation will be considered successful when:

1. **Functionality**: All 8 user stories are implemented with passing acceptance criteria
2. **Test Coverage**: Overall test coverage is 85%+ with no critical paths below 90%
3. **TDD Compliance**: All features implemented using RED-GREEN-REFACTOR cycle with documented test-first approach
4. **Quality Gates**: All checks pass:
   - `ruff check .` (zero errors)
   - `ty check rag_ingestion/` (zero errors)
   - `pytest --cov=rag_ingestion --cov-report=term` (85%+ coverage, all tests pass)
5. **Integration**: Reader successfully loads documents into existing pipeline
   - Documents are chunked by existing chunker
   - Documents are embedded by TEI
   - Documents are stored in Qdrant with correct metadata
6. **Performance**: Reader achieves 50+ URLs/minute throughput on test dataset
7. **Reliability**: Circuit breaker correctly prevents cascading failures during service outages
8. **Documentation**: All public API has Google-style docstrings with examples
9. **Configuration**: Settings are exposed in `config.py` with validation
10. **Observability**: All operations produce structured logs parseable by log aggregators

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Crawl4AI service unavailable | High | Low | Circuit breaker prevents cascading failures; health check on startup; clear error messages |
| Large response timeouts | Medium | Medium | Configurable timeout (60-90s); retry logic for transient failures; concurrency limit prevents resource exhaustion |
| Network instability | Medium | Medium | Exponential backoff retry [1s, 2s, 4s]; circuit breaker for persistent failures; structured logging for debugging |
| Markdown parsing inconsistencies | Low | Low | Use pre-filtered `fit_markdown` field; fallback to `raw_markdown`; comprehensive test coverage with real examples |
| Memory exhaustion from large batches | Medium | Low | Concurrency limit (default: 5); per-request timeout; circuit breaker prevents runaway failures |
| Incompatibility with LlamaIndex updates | Low | Low | Pin llama-index-core version in pyproject.toml; integration tests catch breaking changes |
| Flaky integration tests | Low | Medium | Skip tests if service unavailable; use deterministic mock data for unit tests; retry logic for transient failures |
| TDD workflow resistance | Low | Low | Clear TDD requirements and examples; pair programming for TDD adoption; automated coverage checks in CI |
| Insufficient test coverage | Medium | Low | Mandatory 85%+ coverage; pre-commit hooks for coverage check; detailed coverage reporting |
| Metadata type mismatches | Low | Medium | Pydantic validation for all metadata; explicit type conversion; comprehensive unit tests |

## Verification Plan

### Phase 1: Unit Testing (TDD RED-GREEN-REFACTOR)
- [ ] Write failing tests for configuration validation
- [ ] Write failing tests for Document generation
- [ ] Write failing tests for metadata extraction
- [ ] Write failing tests for error handling
- [ ] Write failing tests for circuit breaker integration
- [ ] Write failing tests for retry logic
- [ ] Implement minimal code to pass all tests
- [ ] Refactor with confidence (tests stay green)
- [ ] Verify 85%+ coverage with `pytest --cov`

### Phase 2: Integration Testing
- [ ] Test real HTTP calls to Crawl4AI service
- [ ] Test end-to-end document loading from live URLs
- [ ] Test circuit breaker behavior with simulated service failures
- [ ] Test retry logic with simulated network issues
- [ ] Test concurrent processing with multiple URLs
- [ ] Test partial failure scenarios
- [ ] Test timeout enforcement
- [ ] Verify structured logging output

### Phase 3: Pipeline Integration
- [ ] Load documents into existing RAG pipeline
- [ ] Verify chunking with markdown-aware splitter
- [ ] Verify embedding generation with TEI
- [ ] Verify storage in Qdrant with correct metadata
- [ ] Verify deterministic UUID generation enables idempotent upsert
- [ ] Verify automatic deduplication on re-crawl (old versions deleted)
- [ ] Verify source_url field is indexed and queryable in Qdrant
- [ ] Verify metadata fields are queryable in Qdrant
- [ ] Run end-to-end ingestion test with 10+ URLs

### Phase 4: Performance Validation
- [ ] Benchmark single URL crawl time
- [ ] Benchmark concurrent URL processing (5, 10, 20 URLs)
- [ ] Verify throughput target (50+ URLs/minute)
- [ ] Profile memory usage during batch processing
- [ ] Test circuit breaker recovery time
- [ ] Test retry delay accuracy

### Phase 5: Documentation Review
- [ ] Verify all public API has docstrings
- [ ] Verify docstrings include Args, Returns, Raises sections
- [ ] Verify docstrings include usage examples
- [ ] Verify configuration is documented in config.py
- [ ] Verify README includes usage examples
- [ ] Verify integration guide is added to CLAUDE.md
