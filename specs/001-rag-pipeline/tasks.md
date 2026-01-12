# Tasks: RAG Pipeline

**Input**: Design documents from `/specs/001-rag-pipeline/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/openapi.yaml ✓

**Tests**: ✅ REQUIRED - TDD approach with 85%+ coverage per plan.md constitutional requirements

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Project structure**: `app/` (not src/), `tests/` at repository root
- **Backend**: Python 3.11+ with FastAPI 0.115+, Pydantic 2.x, SQLAlchemy (async), ARQ 0.26+
- **Storage**: PostgreSQL 15+, Qdrant 1.9+, Redis 7+
- **External services**: TEI (embeddings), Crawl4AI (crawling)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project directory structure per plan.md (app/, tests/, alembic/, scripts/, docs/, .docs/)
- [ ] T002 Initialize Python 3.11+ project with pyproject.toml (uv package manager, FastAPI 0.115+, Pydantic 2.x dependencies)
- [ ] T003 [P] Configure linting and formatting tools (ruff check, ruff format, mypy --strict in pyproject.toml)
- [ ] T004 [P] Create .env.example with all required environment variables (DATABASE_URL, REDIS_URL, QDRANT_URL, TEI_URL, CRAWL4AI_URL, SECRET_KEY, etc.)
- [ ] T005 [P] Create docker-compose.yaml with all services (postgres:15, redis:7, qdrant:v1.9, tei, crawl4ai) using ports 52000+ range
- [ ] T006 [P] Create README.md with project overview, setup instructions, and quick start guide
- [ ] T007 [P] Configure pytest with pytest.ini and asyncio_mode=auto in pyproject.toml
- [ ] T008 [P] Create .gitignore with Python, Docker, IDE, and environment files

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Database & Migrations

- [ ] T009 Create Alembic configuration in alembic/ with env.py for async SQLAlchemy support
- [ ] T010 Setup initial database migration script (alembic/versions/001_initial_schema.py) with all tables from data-model.md
- [ ] T011 [P] Create database indexes per data-model.md (GIN indexes for FTS, B-tree for foreign keys, partial indexes for active records)

### Core Models & Abstractions

- [ ] T012 [P] Create abstract base classes in app/core/abstractions.py (VectorStore, DocumentStore, Cache, Embedder, Crawler interfaces)
- [ ] T013 Create Pydantic models in app/core/models.py (Document, Chunk, Collection, Tag, CrawlJob, SearchConfig, SearchFilters, SearchResult, etc. - all 16 entities from data-model.md)
- [ ] T014 [P] Create Pydantic settings configuration in app/core/config.py (BaseSettings for DATABASE_URL, REDIS_URL, QDRANT_URL, TEI_URL, etc.)
- [ ] T015 [P] Create FastAPI dependency injection functions in app/core/deps.py (get_db_session, get_redis_client, get_qdrant_client, etc.)

### Storage Layer (External Services)

- [ ] T016 [P] Implement PostgreSQL DocumentStore in app/storage/postgres.py (async SQLAlchemy, implements DocumentStore ABC, with FTS support)
- [ ] T017 [P] Implement Qdrant VectorStore in app/storage/qdrant.py (implements VectorStore ABC, with INT8 quantization config from research.md)
- [ ] T018 [P] Implement Redis Cache in app/storage/redis_cache.py (implements Cache ABC, connection pooling, TTL management)

### Authentication & Middleware

- [ ] T019 Create API key authentication middleware in app/api/middleware.py (bearer token validation, SHA256 key hash lookup, scope checking)
- [ ] T019a [P] Add API key expiration validation in app/api/middleware.py (check expires_at field, return 401 Unauthorized if key expired, include expiration info in error response)
- [ ] T020 [P] Create rate limiting middleware in app/api/middleware.py (token bucket algorithm with Redis Lua script from research.md)
- [ ] T021 [P] Create CORS middleware configuration in app/api/middleware.py (ALLOWED_ORIGINS from settings)
- [ ] T022 [P] Create structured logging setup in app/core/logging.py (structlog with JSON formatter, correlation ID injection)

### Error Handling & Health Checks

- [ ] T023 [P] Create global exception handlers in app/api/middleware.py (ValidationError → 422, HTTPException → status code, generic → 500)
- [ ] T024 [P] Create health check endpoints in app/api/v1/admin.py (GET /health for liveness, GET /health/ready for readiness with dependency checks)

### FastAPI Application Setup

- [ ] T025 Create main FastAPI application in app/main.py (app initialization, middleware registration, router inclusion, startup/shutdown events)
- [ ] T026 Create API router structure in app/api/v1/router.py (main v1 router, includes all sub-routers from endpoints)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Intelligent Content Search (Priority: P1) 🎯 MVP

**Goal**: Enable users to search through indexed content using natural language queries and keywords with hybrid vector + keyword search

**Independent Test**: Index sample documents (via manual insertion), execute searches, verify hybrid results combine semantic similarity and keyword matching with correct RRF fusion

### Tests for User Story 1 (TDD Required) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T027 [P] [US1] Contract test for POST /api/v1/search in tests/contract/test_search_api.py (validates OpenAPI schema compliance)
- [ ] T028 [P] [US1] Integration test for hybrid search flow in tests/integration/test_search_flow.py (end-to-end: embed query → vector search → keyword search → RRF fusion → return results)
- [ ] T029 [P] [US1] Unit test for RRF fusion algorithm in tests/unit/test_rrf_fusion.py (verify k=60 default, weight configuration, rank combination)
- [ ] T030 [P] [US1] Unit test for semantic chunker in tests/unit/test_chunker.py (boundary detection, token counting, section header extraction)
- [ ] T031 [P] [US1] Unit test for TEI embedder in tests/unit/test_embedder.py (batching, caching, circuit breaker integration)

### Implementation for User Story 1

#### Services Layer

- [ ] T032 [P] [US1] Implement semantic chunker in app/services/chunker.py (sliding window strategy, paragraph detection, token counting with tiktoken)
- [ ] T033 [P] [US1] Implement TEI embedder wrapper in app/services/embedder.py (implements Embedder ABC, async httpx client, batch support, SHA256 content caching)
- [ ] T034 [P] [US1] Implement circuit breaker in app/services/circuit_breaker.py (Redis-backed state, CLOSED/OPEN/HALF_OPEN transitions, configurable threshold and timeout)
- [ ] T035 [US1] Implement hybrid search service in app/services/search_service.py (RRF fusion algorithm, vector search via Qdrant, FTS via PostgreSQL, optional reranking, depends on T032, T033, T034)

#### Storage Layer Extensions

- [ ] T036 [US1] Add vector search methods to app/storage/qdrant.py (search by embedding, payload filtering, score normalization)
- [ ] T037 [US1] Add full-text search methods to app/storage/postgres.py (ts_rank_cd queries, tsvector matching, result pagination)

#### API Endpoints

- [ ] T038 [US1] Implement search endpoints in app/api/v1/search.py (POST /search for hybrid, POST /search/vector for vector-only, POST /search/keyword for FTS-only)
- [ ] T039 [US1] Create API request/response schemas in app/api/v1/schemas.py (SearchRequest, SearchFilters, SearchResponse, SearchResult per openapi.yaml)
- [ ] T040 [US1] Add search result caching in app/services/search_service.py (Redis cache with query hash key, 1h TTL, query embedding cache with 24h TTL)

#### Integration & Validation

- [ ] T041 [US1] Add input validation for search queries in app/api/v1/search.py (1-500 chars, min_score 0.0-1.0, limit 1-100, sanitize special characters)
- [ ] T042 [US1] Add performance logging for search operations in app/services/search_service.py (log latency_ms, cache hits, result counts with correlation IDs)
- [ ] T043 [US1] Run integration tests for User Story 1 to verify independent functionality (pytest tests/integration/test_search_flow.py)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently. Users can search indexed content with hybrid retrieval.

---

## Phase 4: User Story 2 - Document Management (Priority: P2)

**Goal**: Enable users to upload, tag, update, and delete documents through an API without manual database operations

**Independent Test**: Upload documents via API, verify storage in PostgreSQL + Qdrant, retrieve metadata, update tags, confirm deletion removes document + chunks + embeddings

### Tests for User Story 2 (TDD Required) ⚠️

- [ ] T044 [P] [US2] Contract test for POST /api/v1/documents in tests/contract/test_documents_api.py (validates OpenAPI schema for upload)
- [ ] T045 [P] [US2] Contract test for GET /api/v1/documents/{id} in tests/contract/test_documents_api.py (validates OpenAPI schema for retrieval)
- [ ] T046 [P] [US2] Integration test for document upload flow in tests/integration/test_document_flow.py (upload → chunk → embed → index → verify searchable)
- [ ] T047 [P] [US2] Integration test for document deletion in tests/integration/test_document_flow.py (delete document → verify chunks removed from PostgreSQL and Qdrant)
- [ ] T048 [P] [US2] Unit test for content hash deduplication in tests/unit/test_document_store.py (same content → same hash → deduplicated or merged metadata)

### Implementation for User Story 2

#### Storage Layer Extensions

- [ ] T049 [P] [US2] Add document CRUD methods to app/storage/postgres.py (create_document, get_document, update_document, soft_delete_document with deleted_at timestamp)
- [ ] T050 [P] [US2] Add chunk CRUD methods to app/storage/postgres.py (create_chunks batch insert, get_chunks_by_document, delete_chunks)
- [ ] T051 [P] [US2] Add collection CRUD methods to app/storage/postgres.py (create_collection, list_collections, update_collection, delete_collection)
- [ ] T052 [P] [US2] Add tag CRUD methods to app/storage/postgres.py (create_tag, list_tags, add_tags_to_document, remove_tag_from_document)
- [ ] T053 [US2] Add vector upsert/delete methods to app/storage/qdrant.py (upsert_vectors batch operation, delete_vectors_by_document_id)

#### API Endpoints

- [ ] T054 [US2] Implement document endpoints in app/api/v1/documents.py (POST /documents for upload, GET /documents for list, GET /documents/{id} for retrieve, DELETE /documents/{id} for soft delete)
- [ ] T055 [P] [US2] Implement collection endpoints in app/api/v1/collections.py (GET /collections, POST /collections, GET /collections/{id}, PUT /collections/{id}, DELETE /collections/{id})
- [ ] T056 [P] [US2] Implement tag endpoints in app/api/v1/tags.py (GET /tags, POST /tags, POST /documents/{id}/tags, DELETE /documents/{id}/tags/{tag_id})
- [ ] T057 [US2] Implement batch upload endpoint in app/api/v1/documents.py (POST /documents/batch, max 50 documents, per-item success/failure status)

#### Document Processing Pipeline

- [ ] T058 [US2] Create document processing orchestrator in app/services/document_processor.py (chunk content → generate embeddings → store in PostgreSQL and Qdrant → return document ID)
- [ ] T059 [US2] Add content hash deduplication logic in app/services/document_processor.py (SHA256 hash check, merge metadata if same URL or merge tags if different URL with same content hash)
- [ ] T060 [US2] Add document update handling in app/services/document_processor.py (update metadata, regenerate chunks if content changed, update embeddings)
- [ ] T060a [P] [US2] Implement audit log creation in app/services/document_processor.py (create audit_log entries on document created/updated/deleted events with user_id, action, timestamp, old_values, new_values)

#### Validation & Error Handling

- [ ] T061 [US2] Add document validation in app/api/v1/documents.py (URL format, content non-empty, max 50MB size, valid collection_id if provided)
- [ ] T062 [US2] Add tag validation in app/api/v1/tags.py (name 1-50 chars, lowercase, unique, alphanumeric + hyphens/underscores)
- [ ] T063 [US2] Run integration tests for User Story 2 to verify independent functionality (pytest tests/integration/test_document_flow.py)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently. Users can manage documents via API and search them.

---

## Phase 5: User Story 3 - Automated Web Crawling (Priority: P3)

**Goal**: Enable users to submit URLs for crawling so the system automatically extracts, processes, and indexes web content

**Independent Test**: Submit URL for crawling, verify Crawl4AI extraction, content conversion to markdown, chunking, embedding, indexing, and document appears in search results

### Tests for User Story 3 (TDD Required) ⚠️

- [ ] T064 [P] [US3] Contract test for POST /api/v1/crawl in tests/contract/test_crawl_api.py (validates OpenAPI schema for crawl submission)
- [ ] T065 [P] [US3] Contract test for GET /api/v1/crawl/{job_id} in tests/contract/test_crawl_api.py (validates job status response)
- [ ] T066 [P] [US3] Integration test for crawl job lifecycle in tests/integration/test_crawl_flow.py (submit → pending → running → completed → document indexed)
- [ ] T067 [P] [US3] Integration test for robots.txt compliance in tests/integration/test_crawl_flow.py (verify robots.txt checked, disallowed paths blocked)
- [ ] T068 [P] [US3] Unit test for URL validator (SSRF prevention) in tests/unit/test_url_validator.py (block private IPs, metadata endpoints, validate scheme)
- [ ] T069 [P] [US3] Unit test for rate limiter in tests/unit/test_rate_limiter.py (token bucket algorithm, per-domain limits, Redis Lua script execution)

### Implementation for User Story 3

#### Core Services

- [ ] T070 [P] [US3] Implement URL validator in app/services/url_validator.py (SSRF prevention: block 10.0.0.0/8, 192.168.0.0/16, 172.16.0.0/12, 169.254.0.0/16, metadata.google.internal)
- [ ] T071 [P] [US3] Implement rate limiter in app/services/rate_limiter.py (token bucket with Redis Lua script, per-domain 1 rps default, per-API-key 60 rpm)
- [ ] T072 [P] [US3] Implement Crawl4AI wrapper in app/services/crawler.py (implements Crawler ABC, async httpx client, markdown extraction, robots.txt checking with 7-day cache)
- [ ] T073 [US3] Add robots.txt checker to app/services/crawler.py (fetch robots.txt, parse with robots parser, cache in Redis for 7 days, respect Crawl-delay directive)

#### Background Workers (ARQ)

- [ ] T074 [P] [US3] Create ARQ worker settings in app/workers/__init__.py (WorkerSettings with Redis connection, queue names per priority, max_jobs=10, job_timeout=300s)
- [ ] T075 [US3] Implement crawl worker in app/workers/crawl_worker.py (crawl_job function: validate URL → check rate limit → crawl → process → save document → update job status)
- [ ] T076 [P] [US3] Implement embedding worker in app/workers/embedding_worker.py (embedding_job function: batch chunks → generate embeddings via TEI → upsert to Qdrant)

#### Storage Layer Extensions

- [ ] T077 [P] [US3] Add crawl job CRUD methods to app/storage/postgres.py (create_crawl_job, get_crawl_job, update_job_status, list_jobs_by_status)
- [ ] T078 [P] [US3] Add crawl config CRUD methods to app/storage/postgres.py (create_config, get_config, update_config, list_configs)
- [ ] T079 [P] [US3] Add domain settings CRUD methods to app/storage/postgres.py (get_domain_settings, update_domain_settings with rate limits and circuit breaker config)

#### API Endpoints

- [ ] T080 [US3] Implement crawl endpoints in app/api/v1/crawl.py (POST /crawl for submission, GET /crawl/{job_id} for status, POST /crawl/{job_id}/cancel)
- [ ] T081 [US3] Implement job enqueueing logic in app/api/v1/crawl.py (validate URLs → create jobs → enqueue to ARQ with priority → return job IDs)
- [ ] T082 [US3] Add batch crawl endpoint in app/api/v1/crawl.py (POST /crawl with multiple URLs, max 100, enqueue all jobs)

#### Resilience & Monitoring

- [ ] T083 [US3] Add retry logic to crawl worker in app/workers/crawl_worker.py (exponential backoff with jitter: 1s, 2s, 4s delays, max 3 retries)
- [ ] T084 [US3] Add circuit breaker integration to crawler service in app/services/crawler.py (wrap external calls, threshold=5 failures, timeout=300s)
- [ ] T085 [US3] Add crawl job monitoring in app/workers/crawl_worker.py (log job start/completion, duration, error details with structured logging)
- [ ] T086 [US3] Run integration tests for User Story 3 to verify independent functionality (pytest tests/integration/test_crawl_flow.py)

**Checkpoint**: All user stories 1-3 should now be independently functional. Users can search, manage documents, and crawl web content.

---

## Phase 6: User Story 4 - External System Integration (Priority: P4)

**Goal**: Enable users to receive webhooks when documents are created or crawl jobs complete for real-time external system integration

**Independent Test**: Configure webhook endpoint, trigger document creation and crawl completion events, verify webhook delivery with proper HMAC signature and retry logic

### Tests for User Story 4 (TDD Required) ⚠️

- [ ] T087 [P] [US4] Contract test for POST /api/v1/webhooks in tests/contract/test_webhooks_api.py (validates webhook configuration creation)
- [ ] T088 [P] [US4] Integration test for webhook delivery in tests/integration/test_webhook_flow.py (trigger event → webhook sent → verify payload and signature)
- [ ] T089 [P] [US4] Integration test for webhook retry logic in tests/integration/test_webhook_flow.py (simulate endpoint failure → verify exponential backoff retries)
- [ ] T090 [P] [US4] Unit test for HMAC signature generation in tests/unit/test_webhook_sender.py (verify SHA256 signature matches expected)
- [ ] T091 [P] [US4] Unit test for webhook retry backoff in tests/unit/test_webhook_sender.py (verify 1s, 2s, 4s, 8s, 16s delays)

### Implementation for User Story 4

#### Services Layer

- [ ] T092 [P] [US4] Implement webhook sender in app/services/webhook_sender.py (HMAC-SHA256 signing, async httpx POST, signature in X-Webhook-Signature header)
- [ ] T093 [US4] Add webhook retry logic in app/services/webhook_sender.py (exponential backoff: 1s, 2s, 4s, 8s, 16s intervals, max 5 attempts)

#### Background Workers

- [ ] T094 [US4] Implement webhook worker in app/workers/webhook_worker.py (webhook_delivery_job function: send webhook → update delivery status → retry on failure)
- [ ] T095 [US4] Add webhook event triggers in app/workers/crawl_worker.py (enqueue webhook_delivery_job on crawl completion/failure)
- [ ] T096 [US4] Add webhook event triggers in app/services/document_processor.py (enqueue webhook_delivery_job on document creation)

#### Storage Layer Extensions

- [ ] T097 [P] [US4] Add webhook CRUD methods to app/storage/postgres.py (create_webhook, list_webhooks, update_webhook, delete_webhook)
- [ ] T098 [P] [US4] Add webhook delivery tracking methods to app/storage/postgres.py (create_delivery, update_delivery_status, get_pending_deliveries)

#### API Endpoints

- [ ] T099 [US4] Implement webhook endpoints in app/api/v1/webhooks.py (POST /webhooks, GET /webhooks, GET /webhooks/{id}, DELETE /webhooks/{id})
- [ ] T100 [US4] Implement webhook delivery history endpoint in app/api/v1/webhooks.py (GET /webhooks/{id}/deliveries with pagination)

#### Validation & Security

- [ ] T101 [US4] Add webhook URL validation in app/api/v1/webhooks.py (HTTPS required in production, URL format validation, no private IPs)
- [ ] T102 [US4] Add webhook secret validation in app/api/v1/webhooks.py (min 16 chars, stored securely, used for HMAC signing)
- [ ] T103 [US4] Run integration tests for User Story 4 to verify independent functionality (pytest tests/integration/test_webhook_flow.py)

**Checkpoint**: All user stories 1-4 should now be independently functional. Users can integrate external systems via webhooks.

---

## Phase 7: User Story 5 - High-Volume Operations (Priority: P5)

**Goal**: Enable users to perform batch operations on hundreds of documents efficiently for large-scale knowledge base management

**Independent Test**: Submit batch uploads (500 documents), batch deletions (100 documents), batch crawls (50 URLs), verify all items processed with per-item success/failure status

### Tests for User Story 5 (TDD Required) ⚠️

- [ ] T104 [P] [US5] Contract test for POST /api/v1/documents/batch in tests/contract/test_batch_api.py (validates batch upload schema, max 50 items)
- [ ] T105 [P] [US5] Integration test for batch upload in tests/integration/test_batch_flow.py (submit 50 documents → verify all processed → check per-item status)
- [ ] T106 [P] [US5] Integration test for batch crawl in tests/integration/test_batch_flow.py (submit 50 URLs → verify all jobs created → check job statuses)
- [ ] T107 [P] [US5] Unit test for batch processing logic in tests/unit/test_batch_processor.py (verify error handling, partial failures, rollback behavior)

### Implementation for User Story 5

#### Services Layer

- [ ] T108 [US5] Extend document processor for batch operations in app/services/document_processor.py (batch_upload: process up to 50 documents, best-effort, return per-item status)
- [ ] T109 [US5] Add batch deletion service in app/services/document_processor.py (batch_delete: soft delete multiple documents, remove chunks and vectors, return per-item status)

#### Storage Layer Extensions

- [ ] T110 [US5] Add batch insert methods to app/storage/postgres.py (batch_create_documents, batch_create_chunks using SQLAlchemy bulk operations)
- [ ] T111 [US5] Add batch delete methods to app/storage/postgres.py (batch_soft_delete_documents, batch_delete_chunks)
- [ ] T112 [US5] Add batch vector operations to app/storage/qdrant.py (batch_upsert_vectors max 500 points, batch_delete_vectors)

#### API Endpoints

- [ ] T113 [US5] Extend batch upload endpoint in app/api/v1/documents.py (handle up to 50 documents, return BatchResponse with total_submitted, successful_count, failed_count, per-item results)
- [ ] T114 [US5] Add batch delete endpoint in app/api/v1/documents.py (POST /documents/batch/delete with document IDs array, return per-item status)
- [ ] T115 [US5] Extend batch crawl in app/api/v1/crawl.py (handle up to 100 URLs, create jobs, return BatchJobResponse)

#### Performance Optimization

- [ ] T116 [US5] Add database connection pooling optimization in app/core/config.py (pool_size=20, max_overflow=10 for SQLAlchemy engine)
- [ ] T117 [US5] Add batch embedding generation in app/services/embedder.py (batch size 128, parallel processing for multiple batches)
- [ ] T118 [US5] Run integration tests for User Story 5 to verify independent functionality (pytest tests/integration/test_batch_flow.py)

**Checkpoint**: All user stories 1-5 should now be independently functional. Users can perform high-volume operations efficiently.

---

## Phase 8: Deep Crawl & Discovery (Additional Features)

**Goal**: Advanced crawling features (deep crawl with BFS/DFS/BestFirst, URL discovery from sitemaps/Common Crawl)

### Tests for Deep Crawl ⚠️

- [ ] T119 [P] Contract test for POST /api/v1/crawl/deep in tests/contract/test_deep_crawl_api.py
- [ ] T120 [P] Integration test for BFS deep crawl in tests/integration/test_deep_crawl_flow.py
- [ ] T121 [P] Unit test for frontier management in tests/unit/test_deep_crawl_frontier.py

### Implementation for Deep Crawl

- [ ] T122 [P] Implement deep crawl worker in app/workers/deep_crawl_worker.py (BFS/DFS/BestFirst strategies, frontier management, max_depth and max_pages limits)
- [ ] T123 [P] Add deep crawl frontier management to app/storage/postgres.py (create_frontier_item, get_next_pending_items with strategy-specific ordering)
- [ ] T124 Implement deep crawl endpoints in app/api/v1/crawl.py (POST /crawl/deep, GET /crawl/deep/{job_id})
- [ ] T125 Add link extraction to crawler service in app/services/crawler.py (extract href links, normalize URLs, filter by domain)
- [ ] T126 Add URL scoring for best-first strategy in app/services/crawler.py (keyword matching, relevance scoring 0.0-1.0)

### Tests for Discovery ⚠️

- [ ] T127 [P] Contract test for POST /api/v1/discovery in tests/contract/test_discovery_api.py
- [ ] T128 [P] Integration test for sitemap discovery in tests/integration/test_discovery_flow.py

### Implementation for Discovery

- [ ] T129 [P] Implement sitemap parser in app/services/discovery.py (fetch sitemap.xml, parse URLs, apply pattern filters)
- [ ] T130 [P] Implement Common Crawl discovery in app/services/discovery.py (query Common Crawl index API, filter by domain and pattern)
- [ ] T131 Implement discovery endpoints in app/api/v1/discovery.py (POST /discovery, GET /discovery/{job_id})
- [ ] T132 Add discovery worker in app/workers/discovery_worker.py (discovery_job function: fetch URLs from sources, score by relevance, return top N)

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and production readiness

### Documentation

- [ ] T133 [P] Update README.md with comprehensive setup instructions, API usage examples, and architecture overview
- [ ] T134 [P] Create API documentation from OpenAPI spec in docs/api.md (auto-generated from contracts/openapi.yaml)
- [ ] T135 [P] Create deployment guide in docs/deployment.md (Docker Compose, environment variables, scaling strategies)
- [ ] T136 [P] Create troubleshooting guide in docs/troubleshooting.md (common issues, debugging tips, FAQ)

### Performance Optimization

- [ ] T137 Add database query optimization in app/storage/postgres.py (analyze slow queries with EXPLAIN, add missing indexes, optimize N+1 patterns)
- [ ] T138 Add Qdrant performance tuning in app/storage/qdrant.py (adjust HNSW parameters: m=16, ef_construct=100, ef=128 per research.md)
- [ ] T139 Add Redis memory optimization in docker-compose.yaml (maxmemory 2gb, maxmemory-policy allkeys-lru)
- [ ] T140 Add response compression in app/main.py (GZipMiddleware for responses > 1KB)

### Security Hardening

- [ ] T141 Add input sanitization across all endpoints in app/api/v1/ (XSS prevention, SQL injection prevention via parameterized queries)
- [ ] T142 Add API key rotation support in app/api/v1/admin.py (POST /admin/api-keys/rotate endpoint)
- [ ] T143 Add HTTPS enforcement in production in app/main.py (redirect HTTP → HTTPS, HSTS headers)
- [ ] T144 Add secrets management documentation in docs/security.md (environment variables, key rotation, webhook secrets)

### Monitoring & Observability

- [ ] T145 [P] Add metrics collection in app/core/metrics.py (Prometheus-compatible metrics: search_latency_ms, crawl_success_rate, queue_depth, cache_hit_rate)
- [ ] T146 [P] Add metrics endpoint in app/api/v1/admin.py (GET /metrics for Prometheus scraping)
- [ ] T147 Add structured logging for all services in app/services/ (correlation IDs, latency tracking, error context)
- [ ] T148 Add admin dashboard endpoint in app/api/v1/admin.py (GET /stats with documents_count, chunks_count, pending_jobs, cache_hit_rate)

### Testing & Quality Assurance

- [ ] T149 [P] Add unit tests for all utility functions in tests/unit/ (achieve 85%+ coverage per plan.md requirement)
- [ ] T150 [P] Add integration tests for error scenarios in tests/integration/ (network failures, service unavailable, timeout handling)
- [ ] T151 Run full test suite with coverage report (pytest --cov=app --cov-report=html --cov-report=term-missing)
- [ ] T152 Run type checking with mypy in strict mode (mypy app --strict, zero errors required)
- [ ] T153 Run linting with ruff (ruff check app tests, ruff format app tests, zero issues required)

### Deployment Validation

- [ ] T154 Run quickstart.md validation (follow quickstart.md from scratch, verify all services start, API accessible, tests pass)
- [ ] T155 Perform load testing (siege or locust, verify p95 latency < 200ms for search, < 100ms for CRUD per plan.md performance goals)
- [ ] T156 Verify all constitutional requirements (TDD ✓, API-first ✓, self-hosted ✓, 85%+ coverage ✓, type checking ✓)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion
- **User Story 2 (Phase 4)**: Depends on Foundational completion
- **User Story 3 (Phase 5)**: Depends on Foundational completion
- **User Story 4 (Phase 6)**: Depends on Foundational completion
- **User Story 5 (Phase 7)**: Depends on Foundational completion + User Stories 1-3 (extends existing features)
- **Deep Crawl & Discovery (Phase 8)**: Depends on User Story 3 completion (extends crawling)
- **Polish (Phase 9)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (Search - P1)**: No dependencies on other stories - can start after Foundational
- **User Story 2 (Document Management - P2)**: No dependencies on other stories - can start after Foundational (provides test data for US1)
- **User Story 3 (Crawling - P3)**: No dependencies on other stories - can start after Foundational (uses US2 document processing)
- **User Story 4 (Webhooks - P4)**: No dependencies on other stories - can start after Foundational (integrates with US2 and US3 events)
- **User Story 5 (Batch Operations - P5)**: Depends on US1, US2, US3 (extends their APIs with batch endpoints)

### Within Each User Story

**TDD Cycle (MANDATORY)**:
1. **RED**: Write tests FIRST, ensure they FAIL
2. **GREEN**: Implement minimal code to pass tests
3. **REFACTOR**: Improve code while keeping tests green

**Execution Order**:
- Tests before implementation (TDD required)
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

**Within Phases**:
- All Setup tasks marked [P] can run in parallel (T003, T004, T005, T006, T007, T008)
- All Foundational tasks marked [P] can run in parallel within their categories (T011, T012, T014, T015, T016, T017, T018, T020, T021, T022, T023, T024)
- All tests for a user story marked [P] can run in parallel (e.g., T027-T031 for US1)
- Models within a story marked [P] can run in parallel
- Services marked [P] can run in parallel if they have no interdependencies

**Across User Stories**:
- Once Foundational phase completes, **User Stories 1-4 can all start in parallel** (if team capacity allows)
- User Story 5 must wait for US1-3 completion (it extends their APIs)
- Deep Crawl & Discovery must wait for US3 completion

---

## Parallel Example: User Story 1

```bash
# Step 1: Launch all tests for User Story 1 together (TDD - write tests first):
Task T027: "Contract test for POST /api/v1/search in tests/contract/test_search_api.py"
Task T028: "Integration test for hybrid search flow in tests/integration/test_search_flow.py"
Task T029: "Unit test for RRF fusion algorithm in tests/unit/test_rrf_fusion.py"
Task T030: "Unit test for semantic chunker in tests/unit/test_chunker.py"
Task T031: "Unit test for TEI embedder in tests/unit/test_embedder.py"

# Step 2: Verify all tests FAIL (RED phase)

# Step 3: Launch all parallelizable services together (GREEN phase):
Task T032: "Implement semantic chunker in app/services/chunker.py"
Task T033: "Implement TEI embedder wrapper in app/services/embedder.py"
Task T034: "Implement circuit breaker in app/services/circuit_breaker.py"

# Step 4: Implement sequential dependencies:
Task T035: "Implement hybrid search service in app/services/search_service.py" (depends on T032-T034)
Task T036: "Add vector search methods to app/storage/qdrant.py"
Task T037: "Add full-text search methods to app/storage/postgres.py"

# Step 5: Implement API layer:
Task T038: "Implement search endpoints in app/api/v1/search.py"
Task T039: "Create API request/response schemas in app/api/v1/schemas.py"
Task T040: "Add search result caching in app/services/search_service.py"

# Step 6: Verify all tests PASS (GREEN phase complete)

# Step 7: REFACTOR while keeping tests green
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T008)
2. Complete Phase 2: Foundational (T009-T026) - **CRITICAL - blocks all stories**
3. Complete Phase 3: User Story 1 (T027-T043)
4. **STOP and VALIDATE**: Run `pytest tests/integration/test_search_flow.py` to verify US1 works independently
5. Deploy/demo search functionality if ready

**Estimated Effort**: ~40% of total project (foundation is heavy, enables all other stories)

### Incremental Delivery (Recommended)

1. **Sprint 1**: Setup + Foundational → Foundation ready (T001-T026)
2. **Sprint 2**: User Story 1 → Test independently → **Deploy/Demo (MVP!)** (T027-T043)
3. **Sprint 3**: User Story 2 → Test independently → Deploy/Demo (document management) (T044-T063)
4. **Sprint 4**: User Story 3 → Test independently → Deploy/Demo (web crawling) (T064-T086)
5. **Sprint 5**: User Story 4 → Test independently → Deploy/Demo (webhooks) (T087-T103)
6. **Sprint 6**: User Story 5 → Test independently → Deploy/Demo (batch operations) (T104-T118)
7. **Sprint 7**: Deep Crawl & Discovery (optional) → Deploy/Demo (T119-T132)
8. **Sprint 8**: Polish & Production Readiness (T133-T156)

Each sprint delivers working, testable, deployable functionality.

### Parallel Team Strategy

With 3 developers after Foundational phase completes:

- **Developer A**: User Story 1 (Search) - T027-T043
- **Developer B**: User Story 2 (Document Management) - T044-T063
- **Developer C**: User Story 3 (Crawling) - T064-T086

Stories complete independently, then integrate for full system testing.

---

## Task Summary

**Total Tasks**: 158

**By Phase**:
- Phase 1 (Setup): 8 tasks
- Phase 2 (Foundational): 19 tasks (27 total to this point) - includes T019a for API key expiration validation
- Phase 3 (US1 - Search): 17 tasks (44 total)
- Phase 4 (US2 - Document Management): 21 tasks (65 total) - includes T060a for audit log creation
- Phase 5 (US3 - Crawling): 23 tasks (88 total)
- Phase 6 (US4 - Webhooks): 17 tasks (105 total)
- Phase 7 (US5 - Batch Operations): 15 tasks (120 total)
- Phase 8 (Deep Crawl & Discovery): 14 tasks (134 total)
- Phase 9 (Polish): 24 tasks (158 total)

**By User Story**:
- US1 (Search - P1): 17 tasks 🎯 MVP
- US2 (Document Management - P2): 21 tasks - includes audit logging
- US3 (Crawling - P3): 23 tasks
- US4 (Webhooks - P4): 17 tasks
- US5 (Batch Operations - P5): 15 tasks

**Parallelizable Tasks**: 69 tasks marked [P] (44% can run in parallel within phases)

**Test Tasks**: 44 tasks (28% are tests - exceeds TDD minimum requirement)

**Independent Test Criteria Met**: ✅ Each user story has clear acceptance criteria and can be validated independently

**MVP Scope** (User Story 1 only): 44 tasks (Setup + Foundational + US1) - includes API key expiration validation

---

## Notes

- **[P] tasks**: Different files, no dependencies, can run in parallel
- **[Story] label**: Maps task to specific user story for traceability (US1, US2, US3, US4, US5)
- **TDD Required**: Write tests FIRST (RED), implement (GREEN), refactor (REFACTOR)
- **Each user story**: Independently completable and testable
- **85%+ coverage**: Mandatory per plan.md constitutional requirements
- **Strict checklist format**: All tasks follow `- [ ] [TaskID] [P?] [Story?] Description with file path` format
- **File paths**: All tasks include exact file paths for implementation
- **Verify tests fail**: Before implementing, run tests to ensure they fail (RED phase)
- **Commit frequently**: After each task or logical group of related tasks
- **Stop at checkpoints**: Validate story independently before moving to next priority
- **Constitutional compliance**: TDD ✓, API-first ✓, self-hosted ✓, 85%+ coverage ✓, type checking ✓

---

**Status**: ✅ Tasks generated and validated
**Last Updated**: 2025-01-11 (analysis remediation completed)
**Feature**: 001-rag-pipeline
**Ready for Implementation**: Yes - all prerequisites met, tasks organized by user story, all analysis issues addressed
