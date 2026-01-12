# Research: RAG Pipeline Technical Decisions

**Feature**: 001-rag-pipeline
**Date**: 2025-01-11
**Researcher**: Implementation Planning Agent
**Status**: Completed

---

## Executive Summary

This document consolidates research findings for building a production-grade RAG pipeline with hybrid semantic and keyword search. All technology choices align with constitutional requirements for self-hosted infrastructure, test-driven development, and resilience-first architecture.

**Key Decisions**:
- Hybrid search: Vector (Qdrant) + FTS (PostgreSQL) fused via Reciprocal Rank Fusion (RRF)
- Embeddings: TEI with Qwen3 0.6B model (1024 dimensions)
- Web crawling: Crawl4AI for JavaScript-rendered content extraction
- Background processing: ARQ for async job queues
- Resilience: Circuit breaker + rate limiter + retry logic on all external calls
- API: FastAPI with Pydantic validation, OpenAPI contracts

---

## 1. Hybrid Search Architecture

### Decision: Reciprocal Rank Fusion (RRF)

**Rationale**: RRF combines vector and keyword search results without requiring relevance score normalization, making it robust to different scoring scales.

**Formula**:
```
RRF_score(d) = Σ(1 / (k + rank_vector(d))) * w_v + Σ(1 / (k + rank_keyword(d))) * w_k
where:
  k = 60 (RRF constant, configurable)
  w_v = vector weight (default 1.0)
  w_k = keyword weight (default 1.0)
```

**Best Practices**:
- Retrieve top 20-50 results from each retriever before fusion
- Allow per-query weight tuning (e.g., boost keyword for exact matches)
- Cache query embeddings (TTL: 24h) to avoid regeneration
- Use reranking (modernbert) for top 10-20 fused results when precision critical

**Alternatives Considered**:
- **Simple concatenation**: Rejected due to score scale mismatches (cosine vs BM25)
- **Weighted averaging**: Rejected due to need for score normalization
- **Learned fusion model**: Rejected for MVP (adds training complexity)

**References**:
- Cormack et al. (2009): "Reciprocal Rank Fusion outperforms the best known automatic evaluation measures"
- Benham & Culpepper (2017): RRF robust to score distribution differences

---

## 2. Vector Search Layer (Qdrant)

### Decision: Qdrant 1.9+ with INT8 Quantization

**Rationale**: Self-hosted vector database with best-in-class performance, quantization support for memory efficiency, and payload filtering for multi-tenant queries.

**Configuration**:
```python
VectorParams(
    size=1024,                    # Qwen3 0.6B embedding dimension
    distance=Distance.COSINE,     # Normalized vectors (dot product ≈ cosine)
    on_disk=True                  # Store vectors on disk (SSD recommended)
)
HnswConfig(
    m=16,                         # Connections per node (balance speed/memory)
    ef_construct=100,             # Build accuracy (higher = better recall)
)
ScalarQuantization(
    type=ScalarType.INT8,         # 4x memory reduction (fp32 → int8)
    quantile=0.99,                # Clip outliers at 99th percentile
    always_ram=True               # Keep quantized vectors in RAM
)
```

**Best Practices**:
- Index payload fields used in filtering (collection_id, domain, tags, source)
- Use `ef` (search-time accuracy) = 128 for production queries
- Batch upserts (100-500 points) to reduce network overhead
- Monitor memory usage; enable on-disk storage if RAM < 2x dataset size
- Set `full_scan_threshold` to switch to brute force for small result sets

**Performance Expectations**:
- p95 latency: < 50ms for 100k vectors (indexed)
- Memory: ~1.5GB per 1M vectors (with INT8 quantization)
- Throughput: 1000+ qps (with quantization + HNSW)

**Alternatives Considered**:
- **Milvus**: Rejected (more complex deployment, Go-based vs Rust-based Qdrant)
- **Weaviate**: Rejected (GraphQL API adds complexity)
- **FAISS**: Rejected (library, not service; requires custom wrapper)

**References**:
- Qdrant documentation: https://qdrant.tech/documentation/
- INT8 quantization benchmarks show <5% recall loss with 4x memory savings

---

## 3. Full-Text Search (PostgreSQL FTS)

### Decision: PostgreSQL 15+ with tsvector + GIN Index

**Rationale**: Built-in full-text search eliminates need for Elasticsearch/Solr, reducing operational complexity. English stemming and stop word removal built-in.

**Schema Design**:
```sql
-- Generated tsvector column for automatic indexing
fts_vector TSVECTOR GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
    setweight(to_tsvector('english', content), 'B')
) STORED

CREATE INDEX idx_documents_fts ON documents USING GIN(fts_vector);
```

**Query Pattern**:
```sql
SELECT
    id,
    ts_rank_cd(fts_vector, query) AS rank
FROM documents,
     to_tsquery('english', 'neural & networks') AS query
WHERE fts_vector @@ query
ORDER BY rank DESC
LIMIT 50;
```

**Best Practices**:
- Use `ts_rank_cd()` (cover density ranking) for multi-term queries
- Prefix search with `'term':*` for partial matching (autocomplete)
- Weight title ('A') higher than body ('B') in relevance
- Monitor GIN index bloat; VACUUM ANALYZE regularly
- Consider separate tsvector for chunks (chunk-level search)

**Performance Expectations**:
- p95 latency: < 20ms for FTS queries on 100k documents
- Index size: ~30% of text corpus size
- Concurrent queries: 100+ qps (with connection pooling)

**Alternatives Considered**:
- **Elasticsearch**: Rejected (operational overhead, not self-hosted in spirit)
- **Meilisearch**: Rejected (newer, less mature than PostgreSQL FTS)
- **Tantivy**: Rejected (Rust library, requires custom integration)

**References**:
- PostgreSQL FTS documentation: https://www.postgresql.org/docs/current/textsearch.html
- ts_rank_cd algorithm: cover density ranking for phrase proximity

---

## 4. Embedding Generation (TEI)

### Decision: Text Embeddings Inference (TEI) with Qwen3 0.6B

**Rationale**: Hugging Face's optimized inference server provides low-latency embedding generation with batching support. Qwen3 0.6B balances quality and speed.

**Configuration**:
```bash
docker run --gpus all \
  -p 52010:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id Qwen/Qwen3-0.6B-Embedding \
  --max-batch-size 128 \
  --max-client-batch-size 32
```

**API Usage**:
```python
async def embed_batch(texts: list[str]) -> list[list[float]]:
    response = await httpx_client.post(
        "http://tei:80/embed",
        json={"inputs": texts},
        timeout=30.0
    )
    return response.json()
```

**Best Practices**:
- Batch requests (32-128 texts) to amortize inference overhead
- Timeout: 30s for large batches (adaptive based on batch size)
- Cache embeddings by content SHA256 (TTL: 7 days)
- Circuit breaker threshold: 5 consecutive failures
- Monitor GPU utilization; scale horizontally if bottleneck

**Performance Expectations**:
- Latency: < 50ms per chunk (batched at 32)
- Throughput: 500+ embeddings/second (GPU)
- Dimension: 1024 (normalized to unit length)

**Alternatives Considered**:
- **OpenAI embeddings**: Rejected (cloud dependency, cost)
- **Sentence-Transformers directly**: Rejected (prefer dedicated inference service)
- **Ollama embeddings**: Rejected (slower than TEI, less optimized)

**References**:
- TEI repository: https://github.com/huggingface/text-embeddings-inference
- Qwen3 0.6B: https://huggingface.co/Qwen/Qwen3-0.6B-Embedding

---

## 5. Web Crawling (Crawl4AI)

### Decision: Crawl4AI 0.4+ for JavaScript-Rendered Content

**Rationale**: Built on Playwright, supports headless browser rendering, LLM extraction strategies, and markdown conversion. Self-hosted Docker deployment.

**Features Used**:
- **JavaScript execution**: Handles SPA frameworks (React, Vue, Angular)
- **Markdown extraction**: Clean, readable text extraction
- **Chunking strategies**: Sliding window, semantic boundary detection
- **Screenshot capture**: Debug rendering issues
- **Robots.txt compliance**: Built-in respect for crawling policies

**API Usage**:
```python
async def crawl_url(url: str, config: CrawlConfig) -> CrawlResult:
    response = await httpx_client.post(
        "http://crawl4ai:11235/crawl",
        json={
            "urls": [url],
            "extraction_strategy": config.extraction_strategy,
            "chunking_strategy": config.chunking_strategy,
            "chunk_size": config.chunk_size,
            "wait_until": "networkidle",
            "timeout": config.page_timeout_ms
        },
        timeout=config.page_timeout_ms / 1000 + 10  # Add 10s buffer
    )
    return CrawlResult(**response.json())
```

**Best Practices**:
- Rate limit: 1 request/second per domain (configurable)
- Timeout: 30s page load (fail fast for slow sites)
- Retry: 3 attempts with exponential backoff (2s, 4s, 8s)
- User-Agent: "crawl4r/1.0 (+https://github.com/yourorg/crawl4r)"
- Respect robots.txt: Check before crawling (cache 7 days)
- Block private IPs: Prevent SSRF attacks

**Performance Expectations**:
- Throughput: 10+ pages/second sustained
- Memory: ~500MB per concurrent browser instance
- Latency: 2-5s per page (JS rendering + extraction)

**Alternatives Considered**:
- **Scrapy**: Rejected (no JS rendering)
- **Playwright directly**: Rejected (prefer managed service)
- **Puppeteer**: Rejected (Chromium-only, Crawl4AI uses Playwright for multi-browser)

**References**:
- Crawl4AI documentation: https://crawl4ai.com/docs
- Playwright for web scraping: https://playwright.dev/

---

## 6. Background Job Processing (ARQ)

### Decision: ARQ 0.26+ (Redis-Backed Async Queue)

**Rationale**: Lightweight async job queue built on Redis, native async/await support, perfect for FastAPI integration.

**Configuration**:
```python
# Worker configuration
WorkerSettings(
    redis_settings=RedisSettings(host="redis", port=6379),
    queue_name="crawl4r:queue:{priority}",  # Separate queues per priority
    max_jobs=10,                             # Concurrent jobs per worker
    job_timeout=300,                         # 5 minutes
    keep_result=3600,                        # Keep results 1 hour
)
```

**Job Definition**:
```python
async def crawl_job(ctx: dict, job_id: str, url: str, config_id: str):
    """Crawl a URL and store the document."""
    crawler = ctx["crawler"]
    document_store = ctx["document_store"]

    result = await crawler.crawl_url(url, config)
    document = await document_store.save(result.to_document())

    # Enqueue embedding job
    await ctx["arq_pool"].enqueue_job(
        "embedding_job",
        document_id=str(document.id)
    )

    return {"document_id": str(document.id), "chunks": len(document.chunks)}
```

**Best Practices**:
- Three priority queues: `high`, `normal`, `low`
- Worker pool: 2-3 processes per container (CPU-bound tasks)
- Idempotency: Use Redis locks to prevent duplicate processing
- Dead letter queue: Retry failed jobs max 3 times, then move to DLQ
- Health checks: Monitor queue depth (alert if > 1000 pending)

**Performance Expectations**:
- Throughput: 100+ jobs/second (depending on job complexity)
- Latency: < 1s job dispatch overhead
- Reliability: Guaranteed execution with retries

**Alternatives Considered**:
- **Celery**: Rejected (heavyweight, complex config)
- **RQ**: Rejected (synchronous, not async-native)
- **Dramatiq**: Rejected (less integration with FastAPI ecosystem)

**References**:
- ARQ documentation: https://arq-docs.helpmanual.io/
- ARQ vs Celery comparison: Async-native design, simpler config

---

## 7. Resilience Patterns

### 7.1 Circuit Breaker

**Decision**: Redis-Backed Circuit Breaker with CLOSED/OPEN/HALF_OPEN States

**Implementation**:
```python
class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(self, redis: Redis, service_name: str, threshold: int = 5, timeout: int = 300):
        self.redis = redis
        self.service_name = service_name
        self.threshold = threshold        # Failures before opening
        self.timeout = timeout            # Seconds before half-open
        self.key = f"crawl4r:circuit:{service_name}"

    async def call(self, func: Callable, *args, **kwargs):
        state = await self._get_state()

        if state == "OPEN":
            if await self._should_attempt_reset():
                state = "HALF_OPEN"
            else:
                raise CircuitOpenError(f"{self.service_name} circuit is open")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
```

**Best Practices**:
- Per-service circuits (e.g., `circuit:qdrant`, `circuit:tei`, `circuit:crawl4ai`)
- Threshold: 5 consecutive failures (configurable per service)
- Timeout: 300s (5 minutes) before half-open retry
- Monitor state changes (log transitions, alert on OPEN)
- Fallback: Return cached results when circuit open

**References**:
- Martin Fowler: Circuit Breaker pattern
- Netflix Hystrix design principles

### 7.2 Rate Limiting

**Decision**: Token Bucket Algorithm with Redis

**Implementation**:
```lua
-- Redis Lua script for atomic token bucket
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local tokens = tonumber(redis.call('GET', key) or capacity)
local last_refill = tonumber(redis.call('GET', key .. ':last') or now)

-- Refill tokens based on elapsed time
local elapsed = now - last_refill
local refill_amount = elapsed * refill_rate
tokens = math.min(capacity, tokens + refill_amount)

if tokens >= cost then
    tokens = tokens - cost
    redis.call('SET', key, tokens)
    redis.call('SET', key .. ':last', now)
    redis.call('EXPIRE', key, 60)
    redis.call('EXPIRE', key .. ':last', 60)
    return 1  -- Allowed
else
    return 0  -- Rate limited
end
```

**Best Practices**:
- Domain-level limits: 1 request/second per domain (crawling)
- API-key limits: 60 requests/minute per key (API access)
- Global limit: 1000 requests/second (DDoS protection)
- Headers: `X-RateLimit-Remaining`, `X-RateLimit-Reset`
- HTTP 429: Return `Retry-After` header when rate limited

**References**:
- Token bucket algorithm: https://en.wikipedia.org/wiki/Token_bucket
- Redis rate limiting patterns: https://redis.io/topics/rate-limiting

### 7.3 Retry Logic

**Decision**: Exponential Backoff with Jitter

**Implementation**:
```python
async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (httpx.RequestError,)
):
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            if attempt == max_retries:
                raise

            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)  # 10% jitter
            await asyncio.sleep(delay + jitter)
```

**Best Practices**:
- Retry transient failures (network errors, 5xx responses)
- Don't retry client errors (4xx except 429)
- Max retries: 3 (configurable per service)
- Base delay: 1s (doubles: 1s, 2s, 4s)
- Jitter: 10% of delay (prevent thundering herd)
- Timeout: Independent timeout per retry attempt

**References**:
- AWS exponential backoff and jitter: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/

---

## 8. API Design (FastAPI)

### Decision: FastAPI 0.115+ with OpenAPI 3.1 Contracts

**Rationale**: Async-native, automatic OpenAPI generation, Pydantic validation, excellent performance.

**Best Practices**:

#### 8.1 API Versioning
```python
# URL-based versioning (preferred for backward compatibility)
app.include_router(api_v1_router, prefix="/api/v1")
```

#### 8.2 Request Validation
```python
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=100)
    filters: SearchFilters = Field(default_factory=SearchFilters)

    @validator("query")
    def normalize_query(cls, v):
        return v.strip().lower()
```

#### 8.3 Response Models
```python
class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int
    has_more: bool
```

#### 8.4 Error Handling
```python
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation failed", "details": exc.errors()}
    )
```

#### 8.5 Dependency Injection
```python
async def get_document_store() -> DocumentStore:
    return PostgresDocumentStore(db_pool)

@app.post("/api/v1/documents")
async def create_document(
    req: DocumentUploadRequest,
    doc_store: DocumentStore = Depends(get_document_store)
):
    return await doc_store.save(req.to_document())
```

**Performance Expectations**:
- Throughput: 10,000+ req/s (hello world benchmark)
- Latency: < 1ms framework overhead (p95)
- Concurrency: 1000+ concurrent requests (with async workers)

**References**:
- FastAPI documentation: https://fastapi.tiangolo.com/
- Pydantic performance: https://docs.pydantic.dev/latest/concepts/performance/

---

## 9. Security

### 9.1 URL Validation (SSRF Prevention)

**Decision**: Whitelist Approach with IP Blocklist

**Implementation**:
```python
BLOCKED_IP_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),     # Loopback
    ipaddress.ip_network("10.0.0.0/8"),      # Private
    ipaddress.ip_network("172.16.0.0/12"),   # Private
    ipaddress.ip_network("192.168.0.0/16"),  # Private
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local
]

def validate_url(url: str) -> ValidatedURL:
    parsed = urlparse(url)

    # Resolve hostname to IP
    ip = socket.gethostbyname(parsed.hostname)

    # Check if IP is in blocked ranges
    for blocked_range in BLOCKED_IP_RANGES:
        if ipaddress.ip_address(ip) in blocked_range:
            raise ValueError(f"IP {ip} is in blocked range")

    return ValidatedURL(normalized=normalize_url(url), is_valid=True)
```

**Best Practices**:
- Validate ALL user-provided URLs before crawling
- Block private IP ranges (RFC 1918)
- Block metadata endpoints (169.254.169.254)
- Use DNS resolution before HTTP request
- Timeout DNS lookups (5s max)

**References**:
- OWASP SSRF prevention: https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html

### 9.2 Robots.txt Compliance

**Decision**: Cache robots.txt for 7 Days

**Implementation**:
```python
async def check_robots_txt(url: str) -> bool:
    domain = urlparse(url).netloc
    robots_url = f"https://{domain}/robots.txt"

    # Check cache
    cached = await redis.get(f"crawl4r:robots:{domain}")
    if cached:
        return _is_allowed(url, cached)

    # Fetch and cache
    try:
        response = await httpx_client.get(robots_url, timeout=5.0)
        robots_content = response.text
        await redis.setex(f"crawl4r:robots:{domain}", 7 * 24 * 3600, robots_content)
        return _is_allowed(url, robots_content)
    except httpx.RequestError:
        # Assume allowed if robots.txt unavailable
        return True
```

**Best Practices**:
- Cache robots.txt for 7 days (reduce fetches)
- Timeout: 5s for robots.txt fetch
- Fallback: Allow crawling if robots.txt unavailable
- User-Agent: Match crawler's UA in robots.txt parsing
- Respect `Crawl-delay` directive (if present)

**References**:
- Robots.txt specification: https://www.robotstxt.org/

### 9.3 Webhook Signing (HMAC-SHA256)

**Decision**: Sign Webhook Payloads with Shared Secret

**Implementation**:
```python
import hmac
import hashlib

def sign_webhook(payload: dict, secret: str) -> str:
    """Generate HMAC-SHA256 signature for webhook payload."""
    payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
    signature = hmac.new(
        secret.encode('utf-8'),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()
    return signature

# Send webhook with signature
headers = {
    "X-Webhook-Signature": sign_webhook(payload, webhook.secret),
    "X-Webhook-Event": event_type,
    "X-Webhook-ID": str(delivery_id)
}
```

**Best Practices**:
- Use HMAC-SHA256 (not MD5 or SHA1)
- Include timestamp in payload (prevent replay)
- Verify signature on receiver side
- Rotate secrets periodically (90 days)
- Log signature mismatches (potential security issue)

**References**:
- GitHub webhook security: https://docs.github.com/en/webhooks/using-webhooks/validating-webhook-deliveries

---

## 10. Caching Strategy

### Decision: Multi-Layer Cache with Redis

**Cache Layers**:

1. **Crawl Result Cache** (TTL: 24h)
   - Key: `crawl4r:cache:crawl:{url_sha256[:32]}`
   - Value: Compressed JSON (gzip for > 10KB)
   - Eviction: LRU when memory > 80%

2. **Embedding Cache** (TTL: 7 days)
   - Key: `crawl4r:cache:embed:{content_sha256[:32]}`
   - Value: JSON array of floats
   - Dedup: Same content generates same embedding

3. **Query Result Cache** (TTL: 1 hour)
   - Key: `crawl4r:cache:query:{query_hash[:32]}`
   - Value: JSON search results
   - Invalidation: On document update/delete

4. **Query Embedding Cache** (TTL: 24h)
   - Key: `crawl4r:cache:query_embed:{query_sha256[:32]}`
   - Value: JSON embedding vector
   - Hit rate: ~70% (same queries repeated)

**Best Practices**:
- Compress large values (> 10KB) with gzip
- Use SHA256 hashes for cache keys (deterministic)
- Monitor hit rates (target > 60%)
- Set max memory policy: `allkeys-lru`
- Use Redis pipelining for bulk cache operations

**Performance Impact**:
- Cache hit: < 1ms (Redis latency)
- Cache miss + regenerate: 50-200ms (depends on operation)
- Expected hit rate: 60-80% (steady state)

**References**:
- Redis caching best practices: https://redis.io/docs/manual/client-side-caching/
- HTTP caching with Redis: https://www.nginx.com/blog/benefits-of-microcaching-nginx/

---

## 11. Observability

### 11.1 Structured Logging

**Decision**: JSON Structured Logs with Correlation IDs

**Implementation**:
```python
import structlog

logger = structlog.get_logger()

# Configure logger
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

# Usage
logger.info(
    "search_completed",
    query=query,
    results_count=len(results),
    latency_ms=elapsed,
    correlation_id=correlation_id
)
```

**Best Practices**:
- Include correlation ID in all logs (trace requests)
- Log at appropriate levels (DEBUG/INFO/WARNING/ERROR/CRITICAL)
- Sanitize sensitive data (API keys, URLs with auth)
- Timestamp in ISO 8601 format (UTC)
- Include request/response IDs for API calls

**References**:
- Structlog documentation: https://www.structlog.org/

### 11.2 Health Checks

**Endpoints**:
```python
@app.get("/health")
async def health():
    """Liveness probe (is server running?)"""
    return {"status": "healthy"}

@app.get("/health/ready")
async def readiness():
    """Readiness probe (are dependencies healthy?)"""
    checks = {
        "postgres": await check_postgres(),
        "redis": await check_redis(),
        "qdrant": await check_qdrant(),
        "tei": await check_tei(),
        "crawl4ai": await check_crawl4ai()
    }

    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all_healthy else "degraded", "checks": checks}
    )
```

**Best Practices**:
- Separate liveness (server) and readiness (dependencies)
- Timeout health checks (5s max per dependency)
- Return 503 when not ready (signals load balancer to remove)
- Include version info in response
- Monitor health check failures (alert if > 5%)

**References**:
- Kubernetes health checks: https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/

### 11.3 Metrics

**Key Metrics**:
- `search_latency_ms` (histogram): p50, p95, p99
- `crawl_success_rate` (counter): successful vs failed crawls
- `queue_depth` (gauge): pending jobs per priority queue
- `cache_hit_rate` (gauge): hits / (hits + misses)
- `embedding_generation_ms` (histogram): TEI latency
- `circuit_breaker_state` (gauge): CLOSED=0, OPEN=1, HALF_OPEN=0.5

**Exposure**:
```python
# Prometheus format (if using Prometheus)
from prometheus_client import Counter, Histogram, Gauge

search_latency = Histogram(
    "search_latency_seconds",
    "Search query latency",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)
```

**References**:
- Prometheus best practices: https://prometheus.io/docs/practices/naming/

---

## 12. Testing Strategy

### 12.1 Test Pyramid

**Distribution**:
- **Unit tests**: 70% (fast, isolated, mocked dependencies)
- **Integration tests**: 25% (real dependencies, test containers)
- **Contract tests**: 5% (OpenAPI spec validation)

### 12.2 Test-Driven Development (TDD)

**Workflow**:
1. **RED**: Write failing test defining expected behavior
2. **GREEN**: Write minimal code to pass test
3. **REFACTOR**: Improve code while keeping tests green

**Example**:
```python
# 1. RED: Write failing test
def test_rrf_fusion_combines_results():
    vector_results = [("doc1", 0.9), ("doc2", 0.8)]
    keyword_results = [("doc2", 0.95), ("doc3", 0.85)]

    fused = rrf_fusion(vector_results, keyword_results, k=60)

    assert fused[0][0] == "doc2"  # doc2 appears in both (highest RRF score)
    assert len(fused) == 3         # All unique docs included

# 2. GREEN: Implement RRF fusion
def rrf_fusion(vector_results, keyword_results, k=60):
    scores = defaultdict(float)

    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] += 1 / (k + rank)

    for rank, (doc_id, _) in enumerate(keyword_results):
        scores[doc_id] += 1 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# 3. REFACTOR: Extract constants, add type hints
```

### 12.3 Test Coverage Goals

- **Overall**: 85%+ coverage
- **Critical paths**: 100% (search, crawl, auth)
- **Utility functions**: 90%+
- **Integration tests**: All external service interactions

**Tool**: `pytest-cov` for coverage reporting

```bash
pytest --cov=app --cov-report=html --cov-report=term-missing
```

**References**:
- pytest documentation: https://docs.pytest.org/
- TDD by example (Kent Beck)

---

## 13. Deployment Strategy

### 13.1 Docker Compose Multi-Service Architecture

**Services**:
```yaml
services:
  api:
    build: .
    ports:
      - "52003:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:5432/crawl4r
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
      - TEI_URL=http://tei:80
      - CRAWL4AI_URL=http://crawl4ai:11235
    depends_on:
      - postgres
      - redis
      - qdrant
      - tei
      - crawl4ai
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  worker:
    build: .
    command: arq app.workers.WorkerSettings
    environment: [same as api]
    depends_on: [same as api]

  postgres:
    image: postgres:15-alpine
    ports:
      - "53432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=crawl4r
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}

  redis:
    image: redis:7-alpine
    ports:
      - "53379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru

  qdrant:
    image: qdrant/qdrant:v1.9
    ports:
      - "52002:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  tei:
    image: ghcr.io/huggingface/text-embeddings-inference:latest
    ports:
      - "52010:80"
    volumes:
      - tei_data:/data
    environment:
      - MODEL_ID=Qwen/Qwen3-0.6B-Embedding

  crawl4ai:
    image: unclecode/crawl4ai:latest
    ports:
      - "52001:11235"
    shm_size: 2gb
```

### 13.2 Environment Configuration

**Required Variables**:
```bash
# Database
DATABASE_URL=postgresql://postgres:password@localhost:53432/crawl4r

# Redis
REDIS_URL=redis://localhost:53379

# External Services
QDRANT_URL=http://localhost:52002
TEI_URL=http://localhost:52010
CRAWL4AI_URL=http://localhost:52001

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=<generated-secret>
ALLOWED_ORIGINS=http://localhost:3000,https://app.example.com

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

**References**:
- Docker Compose documentation: https://docs.docker.com/compose/
- 12-Factor App: https://12factor.net/

---

## 14. Performance Tuning

### 14.1 Database Optimization

**Connection Pooling**:
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,           # Max connections
    max_overflow=10,        # Burst connections
    pool_pre_ping=True,     # Check connection health
    echo=False              # Disable SQL logging (production)
)

async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)
```

**Index Strategy**:
- GIN index on tsvector (FTS)
- B-tree indexes on foreign keys, filter columns
- Partial indexes for common queries (e.g., `WHERE deleted_at IS NULL`)

**Query Optimization**:
- Use `SELECT` with explicit columns (avoid `SELECT *`)
- Batch inserts with `INSERT ... VALUES (...), (...)`
- Use `EXPLAIN ANALYZE` to identify slow queries

### 14.2 Redis Optimization

**Memory Management**:
```
maxmemory 2gb
maxmemory-policy allkeys-lru
```

**Persistence**:
```
# Disable RDB for cache-only use
save ""

# AOF for durability (if needed)
appendonly yes
appendfsync everysec
```

**Connection Pooling**:
```python
redis_pool = redis.asyncio.ConnectionPool(
    host="localhost",
    port=53379,
    max_connections=50,
    decode_responses=True
)
```

### 14.3 Qdrant Optimization

**HNSW Tuning**:
- `m=16`: Good balance (increase to 32 for higher recall)
- `ef_construct=100`: Good build speed (increase to 200 for better recall)
- `ef=128`: Good search accuracy (increase to 256 for highest recall)

**Quantization**:
- INT8 quantization: 4x memory reduction, < 5% recall loss
- Always keep quantized vectors in RAM for speed

**Batch Operations**:
- Upsert in batches of 100-500 points
- Use async client for non-blocking operations

**References**:
- Qdrant performance tuning: https://qdrant.tech/documentation/guides/tuning/

---

## 15. Summary of Decisions

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| API Framework | FastAPI | 0.115+ | Async-native, Pydantic validation, OpenAPI |
| Validation | Pydantic | 2.x | Type-safe schemas, settings management |
| Database | PostgreSQL | 15+ | FTS support, JSON columns, reliability |
| Vector DB | Qdrant | 1.9+ | Self-hosted, quantization, performance |
| Cache/Queue | Redis | 7+ | In-memory speed, pub/sub, job queues |
| Embeddings | TEI + Qwen3 | 0.6B | Low latency, batching, self-hosted |
| Crawler | Crawl4AI | 0.4+ | JS rendering, markdown extraction |
| Job Queue | ARQ | 0.26+ | Async-native, Redis-backed, simple |
| HTTP Client | httpx | 0.28+ | Async support, connection pooling |
| Testing | pytest | latest | Async support, fixtures, plugins |

**Total Stack**: 10 core technologies, all self-hosted, all async-compatible, all production-ready.

---

## 16. Open Questions (None)

All technical decisions resolved through research. No clarifications needed.

---

## 17. Next Steps

**Phase 1 Deliverables**:
1. **data-model.md**: Entity definitions, relationships, validation rules
2. **contracts/**: OpenAPI specifications for all API endpoints
3. **quickstart.md**: Local development setup guide
4. **Agent context update**: Add technology choices to CLAUDE.md

**Ready to proceed**: ✅ All research complete, no blockers.
