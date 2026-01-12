<!--
Sync Impact Report:
- Version change: 0.0.0 → 1.0.0 (initial ratification)
- Added sections: 7 Core Principles, Technology Constraints, Quality Gates, Governance
- Templates requiring updates:
  - ✅ plan-template.md (Constitution Check section compatible)
  - ✅ spec-template.md (Requirements section compatible)
  - ✅ tasks-template.md (Task categorization compatible)
- Follow-up TODOs: None
-->

# Crawl4r Constitution

A RAG (Retrieval-Augmented Generation) pipeline for intelligent web crawling, document processing, and hybrid semantic search.

## Core Principles

### I. Test-Driven Development (NON-NEGOTIABLE)

All features and bug fixes MUST follow the TDD cycle:

- **RED**: Write failing tests first that define expected behavior
- **GREEN**: Write minimal code to make tests pass
- **REFACTOR**: Improve code while keeping tests green

No implementation code may be written before tests exist. Test coverage MUST exceed 85%. Integration tests MUST cover all external service interactions (Qdrant, Redis, PostgreSQL, TEI, Crawl4AI).

### II. API-First Design

Every feature starts with its API contract:

- OpenAPI/Swagger specification MUST precede implementation
- Pydantic models define all request/response schemas
- All endpoints follow RESTful conventions with `/api/v1/` prefix
- Pagination MUST use `PaginatedResponse[T]` for list endpoints
- Batch operations MUST be provided for bulk processing
- Rate limiting and circuit breakers protect all external-facing endpoints

### III. Hybrid Search Excellence

The core value proposition is intelligent retrieval:

- Vector search (Qdrant, 1024 dimensions, cosine distance) for semantic similarity
- Full-text search (PostgreSQL FTS, tsvector) for keyword matching
- Reciprocal Rank Fusion (RRF) MUST combine results with configurable weights
- Reranking with `gte-reranker-modernbert-base` for relevance optimization
- Source filtering (`crawl`, `upload`, `api`) MUST be supported in all search queries
- Embedding model versioning MUST track which model generated each vector

### IV. Resilience-First Architecture

All external service calls MUST implement resilience patterns:

- **Circuit Breakers**: Redis-backed state (CLOSED/OPEN/HALF_OPEN) with configurable thresholds
- **Rate Limiters**: Token bucket algorithm with Lua scripts for atomic operations
- **Retry Logic**: Exponential backoff with jitter for transient failures
- **Graceful Degradation**: Fallback to cached results or partial responses
- **Health Checks**: `/health` endpoints for all services with dependency status
- **Timeouts**: Explicit timeouts on all external calls (no unbounded waits)

### V. Self-Hosted Infrastructure

No cloud provider dependencies:

- PostgreSQL for relational data (self-hosted)
- Redis for caching, rate limiting, job queues, circuit breaker state
- Qdrant for vector storage and similarity search
- TEI (Text Embeddings Inference) for embedding generation
- Crawl4AI for web content extraction
- All services deployable via Docker Compose
- Port assignments MUST use 52000+ range to avoid conflicts

### VI. Security by Default

Security is built-in, not bolted-on:

- URL validation with `tldextract` for domain extraction
- `robots.txt` compliance checking with caching
- HMAC-SHA256 signing for all webhooks (`X-Webhook-Signature` header)
- Input sanitization for all user-provided content
- SQL injection prevention via parameterized queries (SQLAlchemy)
- No credentials in code, logs, or documentation
- Rate limiting on all public endpoints

### VII. Observability

All operations MUST be observable:

- Structured logging with correlation IDs across service boundaries
- Request/response logging for debugging (sanitized of sensitive data)
- Metrics for: crawl success/failure rates, search latencies, queue depths
- Health endpoints exposing dependency status
- Error tracking with stack traces and context
- Cache hit/miss ratios for performance tuning

## Technology Constraints

### Required Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| API Framework | FastAPI | 0.115+ | REST API with async support |
| Validation | Pydantic | 2.x | Schema validation and settings |
| Database | PostgreSQL | 15+ | Document storage, FTS |
| Vector DB | Qdrant | 1.9+ | Semantic search (port 52002) |
| Cache/Queue | Redis | 7+ | Caching, rate limiting, jobs |
| Embeddings | TEI | latest | Vector generation (port 52010) |
| Crawler | Crawl4AI | 0.4+ | Web content extraction (port 52001) |
| Job Queue | ARQ | 0.26+ | Background task processing |
| HTTP Client | httpx | 0.28+ | Async HTTP requests |

### Prohibited Technologies

- Cloud-managed databases (Supabase, Neon, PlanetScale)
- Cloud providers (AWS, GCP, Azure)
- Synchronous HTTP clients (requests library)
- ORM magic (lazy loading, implicit queries)
- Global state or singletons (use dependency injection)

## Quality Gates

### Pre-Merge Requirements

All pull requests MUST pass:

1. **Unit Tests**: 85%+ coverage, all passing
2. **Integration Tests**: External service mocks or test containers
3. **Type Checking**: `mypy --strict` with no errors
4. **Linting**: `ruff check` and `ruff format` with no issues
5. **API Contract**: OpenAPI spec matches implementation
6. **Documentation**: Docstrings on all public functions (Google style)

### Performance Requirements

- Search latency: p95 < 200ms for hybrid search
- Crawl throughput: 10+ pages/second sustained
- Embedding generation: < 50ms per chunk (batched)
- API response: p95 < 100ms for CRUD operations
- Database queries: No N+1 patterns, all queries indexed

### Constitution Compliance Checklist

For every implementation plan, verify:

- [ ] Tests written before implementation code
- [ ] API contract defined in OpenAPI spec
- [ ] Circuit breaker wraps external service calls
- [ ] Rate limiting configured for public endpoints
- [ ] Health check includes new dependencies
- [ ] Structured logging with correlation IDs
- [ ] No cloud provider dependencies introduced
- [ ] Port assignments in 52000+ range

## Governance

### Amendment Process

1. Propose change with rationale in PR description
2. Update constitution version following SemVer:
   - **MAJOR**: Principle removal or incompatible redefinition
   - **MINOR**: New principle or significant expansion
   - **PATCH**: Clarifications, typos, non-semantic changes
3. Update `LAST_AMENDED_DATE` to change date
4. All dependent templates MUST be reviewed for compatibility

### Compliance Enforcement

- All PRs MUST include "Constitution Check" in plan documents
- Code reviewers MUST verify TDD evidence (test commits before implementation)
- CI pipeline MUST enforce type checking and test coverage gates
- Architectural decisions MUST reference relevant principles

### Exceptions

Exceptions to principles require:

1. Documented justification in PR description
2. Explicit reviewer approval
3. Time-boxed validity (exception expires after N sprints)
4. Migration plan to return to compliance

**Version**: 1.0.0 | **Ratified**: 2025-01-11 | **Last Amended**: 2025-01-11
