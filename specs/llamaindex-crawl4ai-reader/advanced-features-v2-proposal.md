# Advanced Crawl4AI Reader (v2) - Feature Proposal

**Status:** Planning
**Created:** 2026-01-15
**Dependencies:** Basic Crawl4AI Reader (v1) must be completed and validated

## Overview

This document outlines advanced features deferred from the basic LlamaIndex Crawl4AI reader (v1) specification. These features require significant additional complexity and infrastructure, warranting a separate specification and implementation phase.

**Estimated Effort:** 4-5 days
**Prerequisite:** Basic reader (v1) must be deployed, tested, and validated in production

---

## Deferred Features

### 1. Async Job Endpoint (`/crawl/job`)

**Problem:** Some web pages take >60 seconds to crawl (JavaScript-heavy SPAs, large content). Synchronous `/crawl` endpoint times out.

**Solution:** Use Crawl4AI's `/crawl/job` endpoint for long-running crawls:
- Submit job, receive job ID
- Poll `/crawl/job/{job_id}/status` for completion
- Retrieve results when status == "completed"
- Handle failures when status == "failed"

**Implementation Challenges:**
- Job state management (pending, running, completed, failed)
- Polling interval optimization (start 1s, backoff to 5s)
- Timeout handling (max job duration: 5 minutes?)
- Concurrent job tracking (many URLs → many job IDs)
- Error recovery (job lost, service restart mid-job)

**API Changes:**
```python
class Crawl4AIReader:
    use_async_jobs: bool = False  # New parameter
    max_job_duration: int = 300  # 5 minutes max
    job_poll_interval: float = 1.0  # Start with 1s
    job_poll_backoff: float = 1.5  # Multiply interval by 1.5 each poll
```

**Estimated Effort:** 2 days

---

### 2. Webhook Listener

**Problem:** Polling for job completion wastes resources. Crawl4AI supports webhook notifications when jobs complete.

**Solution:** Implement webhook server/endpoint to receive notifications:
- Start webhook server on configurable port (e.g., 53100)
- Register webhook URL with Crawl4AI when submitting jobs
- Receive POST notifications when jobs complete
- Match job ID to original URL request
- Return results to waiting async caller

**Implementation Challenges:**
- Webhook server infrastructure (FastAPI app? aiohttp server?)
- Port management and availability checking
- Webhook authentication/validation (prevent spoofing)
- Request-response correlation (job ID → original caller)
- Webhook delivery failures (retry logic on Crawl4AI side?)
- Cleanup of expired job registrations
- Testing without real Crawl4AI webhooks

**API Changes:**
```python
class Crawl4AIReader:
    enable_webhooks: bool = False  # New parameter
    webhook_port: int = 53100  # Default port
    webhook_path: str = "/webhook/crawl4ai"  # Endpoint path
    webhook_timeout: int = 600  # 10 minutes max wait
```

**Estimated Effort:** 1.5 days

---

### 3. Recursive Link Crawling

**Problem:** Users want to crawl entire website sections, not just explicit URL lists. Following links manually is tedious.

**Solution:** Implement recursive crawling with configurable depth:
- Extract internal links from crawled pages
- Follow links up to max depth
- Track visited URLs to prevent loops
- Deduplicate URLs (normalize, canonicalize)
- Respect crawl scope (same domain only? path prefix?)

**Implementation Challenges:**
- URL frontier management (queue of URLs to visit)
- Visited URL tracking (in-memory set? persistent store?)
- URL normalization (http vs https, www vs non-www, trailing slash, query params)
- Deduplication strategy (exact match? content hash?)
- Depth tracking per URL
- Exponential URL growth (limit total crawls? time budget?)
- Cycle detection (A→B→A link loops)
- Politeness (delay between requests to same domain)
- Scope enforcement (stay within subdomain? path?)

**API Changes:**
```python
class Crawl4AIReader:
    enable_recursive_crawl: bool = False  # New parameter
    max_crawl_depth: int = 2  # Default: 2 levels deep
    max_total_urls: int = 100  # Safety limit
    same_domain_only: bool = True  # Don't leave domain
    url_path_prefix: str | None = None  # e.g., "/docs/" to stay in docs
    crawl_delay: float = 1.0  # Seconds between requests to same domain
```

**Estimated Effort:** 2 days

---

### 4. Streaming Endpoint (`/crawl/stream`)

**Problem:** No visibility into long-running crawls. User can't monitor progress or cancel stuck requests.

**Solution:** Use Crawl4AI's `/crawl/stream` endpoint with NDJSON streaming:
- Receive progress updates as NDJSON lines
- Parse each line as JSON event
- Handle event types: progress, log, result, error
- Surface progress to caller (optional callback?)

**Implementation Challenges:**
- NDJSON parsing (newline-delimited JSON)
- Streaming HTTP response handling (AsyncClient streaming)
- Event dispatching (progress, log, result, error events)
- Progress callback API design
- Error handling mid-stream
- Incomplete streams on connection loss
- Testing streaming without real Crawl4AI

**API Changes:**
```python
from typing import Callable

class Crawl4AIReader:
    enable_streaming: bool = False  # New parameter
    progress_callback: Callable[[dict], None] | None = None  # Optional callback
```

**Estimated Effort:** 1 day

---

### 5. Extended Metadata Extraction

**Problem:** Basic metadata (title, description, link counts) insufficient for some use cases. Users want author, keywords, language, Open Graph tags.

**Solution:** Extract additional metadata fields from Crawl4AI response:
- Author: `metadata.author`
- Keywords: `metadata.keywords` (list → comma-separated string)
- Language: `metadata.language` or detect from content
- Open Graph: `metadata.og.title`, `metadata.og.image`, `metadata.og.type`, etc.
- Twitter Card: `metadata.twitter.card`, `metadata.twitter.site`, etc.

**Implementation Challenges:**
- Flattening nested metadata for Qdrant (og.title → og_title)
- Handling missing fields gracefully
- List-to-string conversion (keywords: ["python", "ai"] → "python, ai")
- Field naming conventions (camelCase vs snake_case)
- Configurable field selection (expose all? subset? user choice?)
- Metadata size limits (Qdrant payloads)

**API Changes:**
```python
class Crawl4AIReader:
    metadata_fields: list[str] = ["core"]  # Options: core, author, keywords, language, opengraph, twitter
    # or:
    include_author: bool = False
    include_keywords: bool = False
    include_language: bool = False
    include_opengraph: bool = False
    include_twitter: bool = False
```

**Estimated Effort:** 0.5 days

---

### 6. Content Filtering Options

**Problem:** Crawl4AI defaults work for most cases, but some users need custom filtering (e.g., extract only specific sections).

**Solution:** Expose Crawl4AI's content filtering parameters:
- `word_count_threshold`: Minimum word count per element (default: 10)
- `css_selector`: Extract only matching elements (e.g., "article.main-content")
- `excluded_tags`: Remove specific HTML tags (e.g., ["nav", "footer"])

**Implementation Challenges:**
- CSS selector validation (malformed selectors → errors)
- Tag list validation
- Documenting parameter effects
- Testing with real HTML fixtures
- Balancing flexibility vs. complexity

**API Changes:**
```python
class Crawl4AIReader:
    word_count_threshold: int = 10  # Default from Crawl4AI
    css_selector: str | None = None  # Optional: extract only matching elements
    excluded_tags: list[str] = []  # Optional: remove specific tags
```

**Estimated Effort:** 0.5 days

---

### 7. Configurable Cache Modes

**Problem:** Some use cases benefit from caching (development, repeated crawls of static content). Current implementation always bypasses cache.

**Solution:** Expose Crawl4AI's cache_mode parameter:
- `BYPASS`: Always fetch fresh (current behavior)
- `ENABLED`: Use cache if available
- `REFRESH`: Fetch fresh and update cache
- `READ_ONLY`: Use cache, don't update

**Implementation Challenges:**
- Documenting cache behavior and invalidation
- Cache key understanding (URL-based? hash-based?)
- Cache size management (Crawl4AI responsibility)
- Cache invalidation strategy
- Testing cached vs fresh content

**API Changes:**
```python
from enum import Enum

class CacheMode(str, Enum):
    BYPASS = "BYPASS"
    ENABLED = "ENABLED"
    REFRESH = "REFRESH"
    READ_ONLY = "READ_ONLY"

class Crawl4AIReader:
    cache_mode: CacheMode = CacheMode.BYPASS  # Default: always fresh
```

**Estimated Effort:** 0.5 days

---

## Integration Strategy

### Backward Compatibility

Advanced reader (v2) must remain compatible with basic reader (v1) API:
- All new parameters must be optional with defaults matching v1 behavior
- `load_data()` and `aload_data()` signatures unchanged
- Document structure unchanged
- Metadata keys additive (new fields, not replacing)

### Migration Path

Users can upgrade from v1 to v2 without code changes:
```python
# v1 code works with v2
reader = Crawl4AIReader(endpoint_url="http://localhost:52004")
docs = reader.load_data(["https://example.com"])

# v2 features opt-in
reader = Crawl4AIReader(
    endpoint_url="http://localhost:52004",
    enable_recursive_crawl=True,  # New in v2
    max_crawl_depth=2,  # New in v2
    enable_webhooks=True,  # New in v2
)
docs = reader.load_data(["https://example.com"])
```

### Testing Strategy

- Unit tests for all new features
- Integration tests with real Crawl4AI service
- Backward compatibility tests (v1 API on v2 implementation)
- Performance benchmarks (v2 overhead vs v1)
- Webhook infrastructure testing (mock server)
- Recursive crawl limits enforcement (prevent runaway crawls)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Webhook port conflicts** | Medium | High | Port availability checking, configurable port, fallback to polling |
| **Runaway recursive crawls** | High | High | Strict depth and total URL limits, time budgets, circuit breaker |
| **Job polling inefficiency** | Medium | Medium | Adaptive backoff, webhook alternative, job batching |
| **Increased complexity** | High | Medium | Thorough testing, phased rollout, feature flags, comprehensive docs |
| **Crawl4AI API changes** | Low | High | Version pinning, API contract tests, fallback to v1 endpoints |
| **Memory usage from URL frontier** | Medium | Medium | Persistent storage option, memory limits, URL deduplication |

---

## Success Criteria

Advanced reader (v2) is successful when:

1. **Async Jobs**: Pages with 60-300s crawl time complete successfully
2. **Webhooks**: Webhook delivery reduces polling overhead by 90%+
3. **Recursive Crawling**: Can crawl 100-page documentation site in single call
4. **Streaming**: Progress visible for >30s crawls
5. **Extended Metadata**: Author/keywords extracted when available
6. **Content Filtering**: Custom CSS selectors reduce noise by 50%+
7. **Cache Modes**: Cached reads 10x faster than fresh crawls
8. **Backward Compatibility**: All v1 tests pass on v2 implementation
9. **Performance**: v2 overhead <5% when advanced features disabled
10. **Coverage**: 85%+ test coverage maintained

---

## Implementation Phases

### Phase 1: Foundation (Day 1)
- [ ] Add async job endpoint support
- [ ] Implement job polling with backoff
- [ ] Add job state management
- [ ] Write unit tests for job handling
- [ ] Write integration tests with real jobs

### Phase 2: Webhooks (Day 2)
- [ ] Implement webhook server (FastAPI or aiohttp)
- [ ] Add webhook registration with Crawl4AI
- [ ] Implement request-response correlation
- [ ] Add webhook authentication/validation
- [ ] Write webhook integration tests

### Phase 3: Recursive Crawling (Day 3)
- [ ] Implement URL frontier and visited tracking
- [ ] Add URL normalization and deduplication
- [ ] Implement depth and scope enforcement
- [ ] Add politeness delays
- [ ] Write recursive crawl tests

### Phase 4: Enhancements (Day 4)
- [ ] Add streaming endpoint support
- [ ] Implement extended metadata extraction
- [ ] Add content filtering options
- [ ] Add configurable cache modes
- [ ] Write comprehensive integration tests

### Phase 5: Polish (Day 5)
- [ ] Backward compatibility validation
- [ ] Performance benchmarking vs v1
- [ ] Documentation updates
- [ ] Migration guide
- [ ] Final testing and bug fixes

---

## Next Steps

1. **Complete and validate basic reader (v1)** - Must be production-ready
2. **Gather user feedback** - Validate which features are most needed
3. **Prioritize features** - May split into v2a and v2b if needed
4. **Create full specification** - Run Ralph Specum for advanced reader
5. **Begin implementation** - Follow TDD methodology with 85%+ coverage

---

## Notes

- This is a **proposal document**, not a full specification
- Actual implementation may add/remove features based on user needs
- Timeline estimates assume v1 patterns reused where possible
- Testing overhead significant due to infrastructure requirements (webhooks, recursive crawls)
- Consider feature flags for gradual rollout and A/B testing
