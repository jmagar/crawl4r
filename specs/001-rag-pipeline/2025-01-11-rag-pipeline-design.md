# RAG Pipeline Design Document

**Project:** crawl4r
**Date:** 2025-01-11
**Status:** Draft

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Data Models](#3-data-models)
4. [Database Schema](#4-database-schema)
5. [Redis Structure](#5-redis-structure)
6. [Qdrant Configuration](#6-qdrant-configuration)
7. [API Endpoints](#7-api-endpoints)
8. [Service Components](#8-service-components)
9. [Pipeline Flows](#9-pipeline-flows)
10. [Worker Tasks](#10-worker-tasks)
11. [Configuration](#11-configuration)
12. [Project Structure](#12-project-structure)
13. [Security Considerations](#13-security-considerations)
14. [Observability](#14-observability)
15. [Dependencies](#15-dependencies)
16. [Deployment](#16-deployment)

---

## 1. Overview

### 1.1 Purpose

A modular, pluggable RAG (Retrieval-Augmented Generation) pipeline that crawls web content, generates embeddings, and provides hybrid semantic + keyword search with RRF fusion and optional reranking.

### 1.2 Use Cases

- **Knowledge base search** — Index documentation, articles, reference material for semantic Q&A
- **Web research assistant** — Crawl and index arbitrary URLs on-demand for research queries
- **Content aggregation** — Continuously monitor and index content from specific sites/feeds

### 1.3 Key Features

- Hybrid retrieval (vector + keyword) with RRF fusion
- Deep crawling with BFS/DFS/BestFirst strategies
- URL discovery from sitemaps and Common Crawl
- Scheduled crawling for content monitoring
- Full caching layer (crawl results, embeddings, query results)
- Circuit breaker and rate limiting per domain
- Webhook notifications on job completion
- Pluggable backends via abstract interfaces

### 1.4 External Services

| Service | Local Endpoint | Docker Endpoint | Purpose |
|---------|----------------|-----------------|---------|
| Crawl4AI | `localhost:52001` | `crawl4ai:11235` | Web crawling and chunking |
| Qdrant | `100.74.16.82:52002` | `100.74.16.82:52002` | Vector storage (1024 dims) |
| TEI | `100.74.16.82:52000` | `100.74.16.82:52000` | Embeddings (Qwen3 0.6B) |
| Postgres | `localhost:53432` | `postgres:5432` | Document storage + FTS |
| Redis | `localhost:53379` | `redis:6379` | Cache, queues, rate limiting |

> **Note:** Qdrant uses port 52002 to avoid conflict with Crawl4AI's external port 52001.

### 1.5 Constraints

- English content only (filter non-English)
- 1024 vector dimensions
- Self-hosted infrastructure only

---

## 2. Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         API Service                                  │
│  FastAPI + WebSocket + Reranker (gte-reranker-modernbert-base)     │
└──────────────┬───────────────────────────────────┬──────────────────┘
               │                                   │
               ▼                                   ▼
┌──────────────────────────┐         ┌────────────────────────────────┐
│     Worker Service       │         │           Redis                 │
│  ARQ (2-3 processes)     │◄───────►│  Cache + Queues + Rate Limit   │
│  Single container        │         │  Circuit Breaker + Pub/Sub     │
└──────────────┬───────────┘         └────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        External Services                              │
├─────────────────┬─────────────────┬─────────────────┬────────────────┤
│   Crawl4AI      │    Postgres     │     Qdrant      │      TEI       │
│   :52001        │   (internal)    │    :52001       │    :52000      │
│   Crawl+Chunk   │   Docs + FTS    │    Vectors      │   Embeddings   │
└─────────────────┴─────────────────┴─────────────────┴────────────────┘
```

### 2.2 Design Principles

- **Protocol classes (ABCs)** for pluggable backends
- **Pydantic Settings** for configuration with validation
- **ARQ** for async background jobs (Redis-backed)
- **Hybrid container** approach for workers (2-3 processes per container)

---

## 3. Data Models

### 3.1 Core Models

```python
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID
from pydantic import BaseModel, Field


# === Enums ===

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class DocSource(str, Enum):
    CRAWL = "crawl"
    UPLOAD = "upload"
    API = "api"


class DeepCrawlStrategy(str, Enum):
    BFS = "bfs"
    DFS = "dfs"
    BEST_FIRST = "best_first"


class ChunkContentType(str, Enum):
    PROSE = "prose"
    CODE = "code"
    TABLE = "table"
    LIST = "list"


# === Auth ===

class ApiKey(BaseModel):
    id: UUID
    key_hash: str                           # SHA256 of actual key
    name: str                               # Human-readable label
    scopes: list[str] = ["read"]            # read, write, admin
    rate_limit_rpm: int = 60                # Requests per minute
    created_at: datetime
    expires_at: datetime | None = None
    is_active: bool = True


# === Organization ===

class Collection(BaseModel):
    id: UUID
    name: str
    description: str | None = None
    created_at: datetime


class Tag(BaseModel):
    id: UUID
    name: str                               # Unique, lowercase


# === Crawl Configuration ===

class CrawlConfig(BaseModel):
    id: UUID
    name: str
    max_depth: int = 0                      # 0 = single page
    rate_limit_rps: float = 1.0             # Requests per second per domain
    respect_robots_txt: bool = True
    headers: dict[str, str] = {}            # Custom headers
    extraction_strategy: str = "markdown"   # markdown, structured, raw
    chunking_strategy: str = "sliding_window"
    chunk_size: int = 512                   # Tokens per chunk
    chunk_overlap: int = 50                 # Overlap tokens
    min_chunk_size: int = 50                # Minimum tokens per chunk
    page_timeout_ms: int = 30000            # Page load timeout
    max_page_size_mb: int = 50              # Max response size
    max_redirects: int = 10


class CrawlSource(BaseModel):
    """Monitored sources for content aggregation."""
    id: UUID
    name: str
    url_pattern: str                        # Base URL or pattern
    crawl_config_id: UUID
    collection_id: UUID | None = None
    schedule_cron: str | None = None        # "0 */6 * * *" = every 6h
    is_active: bool = True
    last_crawled_at: datetime | None = None
    created_at: datetime


class DomainSettings(BaseModel):
    domain: str                             # Primary key
    rate_limit_rps: float = 1.0             # Override default
    circuit_breaker_threshold: int = 5      # Failures before open
    circuit_breaker_timeout_s: int = 300    # Seconds before retry
    is_blocked: bool = False                # Manual block
    updated_at: datetime


class ProxyConfig(BaseModel):
    id: UUID
    name: str
    servers: list[dict[str, str]]           # [{url, username?, password?}]
    rotation_strategy: str = "round_robin"  # round_robin, least_used, random
    is_active: bool = True
    created_at: datetime


# === Documents ===

class DocumentMetadata(BaseModel):
    """Extracted metadata from crawled content."""
    author: str | None = None
    publish_date: datetime | None = None
    description: str | None = None
    keywords: list[str] = []
    og_image: str | None = None
    canonical_url: str | None = None


class Document(BaseModel):
    id: UUID
    url: str
    domain: str                             # Extracted from URL
    parent_url: str | None = None           # If discovered via link
    title: str | None = None
    content: str                            # Full extracted text
    content_hash: str                       # SHA256 for dedup/cache
    language: str = "en"
    source: DocSource
    collection_id: UUID | None = None
    metadata: DocumentMetadata = DocumentMetadata()
    crawled_at: datetime
    deleted_at: datetime | None = None


class Chunk(BaseModel):
    id: UUID
    document_id: UUID
    content: str
    chunk_index: int                        # Order within document
    start_char: int                         # Offset in original doc
    end_char: int                           # For highlighting
    token_count: int
    embedding_model: str                    # "qwen3-0.6b-embedding"
    section_header: str | None = None       # Nearest H1/H2/H3 above
    content_type: ChunkContentType = ChunkContentType.PROSE


# === Jobs ===

class CrawlJob(BaseModel):
    id: UUID
    url: str
    domain: str
    crawl_config_id: UUID | None = None
    crawl_source_id: UUID | None = None     # If from scheduled source
    document_id: UUID | None = None         # Result after success
    collection_id: UUID | None = None
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    result: dict[str, Any] | None = None    # Stored result
    result_expires_at: datetime | None = None
    webhook_url: str | None = None
    webhook_headers: dict[str, str] = {}
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class DeepCrawlJob(BaseModel):
    id: UUID
    root_url: str
    strategy: DeepCrawlStrategy
    max_depth: int
    max_pages: int
    score_threshold: float = 0.0
    keywords: list[str] = []                # For best-first scoring
    crawl_config_id: UUID
    collection_id: UUID | None = None
    webhook_url: str | None = None
    webhook_headers: dict[str, str] = {}

    # State
    status: JobStatus = JobStatus.PENDING
    pages_discovered: int = 0
    pages_crawled: int = 0
    pages_failed: int = 0
    current_depth: int = 0

    # Timestamps
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class DeepCrawlFrontierItem(BaseModel):
    id: UUID
    deep_crawl_job_id: UUID
    url: str
    depth: int
    score: float | None = None              # For best-first
    status: JobStatus = JobStatus.PENDING
    discovered_at: datetime


class DiscoveryJob(BaseModel):
    id: UUID
    domain: str
    sources: list[str]                      # ["sitemap", "common_crawl"]
    pattern: str | None = None              # URL pattern filter
    max_urls: int = 500
    score_query: str | None = None
    score_threshold: float = 0.0
    status: JobStatus = JobStatus.PENDING
    urls_found: int = 0
    result: list[str] | None = None
    created_at: datetime
    completed_at: datetime | None = None


# === Webhooks ===

class Webhook(BaseModel):
    id: UUID
    source_id: UUID | None = None           # Link to crawl_source
    url: str
    headers: dict[str, str] = {}
    secret: str | None = None               # For HMAC signing
    events: list[str] = ["completed", "failed"]
    is_active: bool = True
    created_at: datetime


class WebhookDelivery(BaseModel):
    id: UUID
    webhook_id: UUID
    job_id: UUID | None = None
    event: str
    payload: dict[str, Any]
    status: Literal["pending", "delivered", "failed"] = "pending"
    attempts: int = 0
    max_attempts: int = 5
    last_attempt_at: datetime | None = None
    next_attempt_at: datetime | None = None
    response_status: int | None = None
    response_body: str | None = None
    created_at: datetime


# === Search ===

class SearchFilters(BaseModel):
    collection_ids: list[UUID] | None = None
    tag_ids: list[UUID] | None = None
    domains: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    source_types: list[DocSource] | None = None


class SearchConfig(BaseModel):
    query: str
    filters: SearchFilters = SearchFilters()
    min_score: float = 0.0
    use_reranker: bool = False
    rerank_top_n: int = 20
    expand_chunks: bool = False             # Include surrounding chunks
    rrf_k: int = 60                         # RRF constant
    vector_weight: float = 1.0              # Weight for vector results
    keyword_weight: float = 1.0             # Weight for FTS results
    limit: int = 10
    cursor: str | None = None               # For pagination


class SearchResult(BaseModel):
    document_id: UUID
    chunk_id: UUID
    url: str
    title: str | None
    content: str
    score: float                            # RRF fused score
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    source: Literal["vector", "keyword", "fused"]
    highlights: list[tuple[int, int]] | None = None  # Character ranges
    section_header: str | None = None
    expanded_chunks: list[str] | None = None  # Surrounding context


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_count: int | None = None          # Estimated total
    next_cursor: str | None = None
    query_embedding_cached: bool = False
    result_cached: bool = False
    latency_ms: float


# === Validated URL ===

class ValidatedURL(BaseModel):
    original: str
    normalized: str                         # Lowercase host, sorted params
    scheme: str
    host: str
    domain: str                             # Extracted domain
    path: str
    is_valid: bool = True
    rejection_reason: str | None = None


# === Crawl Results ===

class CrawlResult(BaseModel):
    """Result from crawling a single URL."""
    url: str
    normalized_url: str
    success: bool
    status_code: int | None = None
    content_type: str | None = None
    html: str | None = None                 # Raw HTML
    markdown: str | None = None             # Extracted markdown
    text: str | None = None                 # Plain text
    title: str | None = None
    metadata: DocumentMetadata = DocumentMetadata()
    links: list[str] = []                   # Discovered links
    chunks: list[dict[str, Any]] = []       # Pre-chunked content from Crawl4AI
    error: str | None = None
    error_type: str | None = None           # timeout, connection, http_error, etc.
    crawled_at: datetime
    duration_ms: int
```

### 3.2 Request/Response Models

```python
# === Crawl Requests ===

class CrawlRequest(BaseModel):
    urls: list[str]
    config_id: UUID | None = None
    collection_id: UUID | None = None
    tags: list[str] | None = None
    priority: JobPriority = JobPriority.NORMAL
    webhook_url: str | None = None
    webhook_headers: dict[str, str] = {}


class DeepCrawlRequest(BaseModel):
    url: str
    strategy: DeepCrawlStrategy = DeepCrawlStrategy.BFS
    max_depth: int = 2
    max_pages: int = 100
    score_threshold: float = 0.4
    keywords: list[str] = []
    config_id: UUID | None = None
    collection_id: UUID | None = None
    webhook_url: str | None = None
    webhook_headers: dict[str, str] = {}


class DiscoveryRequest(BaseModel):
    domain: str
    sources: list[str] = ["sitemap"]        # sitemap, common_crawl
    pattern: str | None = None
    max_urls: int = 500
    score_query: str | None = None
    score_threshold: float = 0.3


# === Search Requests ===

class SearchRequest(BaseModel):
    query: str
    collection_ids: list[UUID] | None = None
    tag_ids: list[UUID] | None = None
    domains: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    min_score: float = 0.0
    use_reranker: bool = False
    expand_chunks: bool = False
    limit: int = 10
    cursor: str | None = None


# === Document Requests ===

class DocumentUploadRequest(BaseModel):
    url: str                                # Can be synthetic URL for uploads
    title: str | None = None
    content: str
    collection_id: UUID | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] = {}


# === Media Requests ===

class ScreenshotRequest(BaseModel):
    url: str
    wait_for: str | None = None             # CSS selector to wait for
    wait_seconds: float = 2.0
    full_page: bool = True


class PDFRequest(BaseModel):
    url: str
    wait_for: str | None = None
    wait_seconds: float = 2.0


class ExecuteJSRequest(BaseModel):
    url: str
    scripts: list[str]
    wait_for: str | None = None
```

---

## 4. Database Schema

### 4.1 Postgres Schema

```sql
-- ============================================
-- ENUMS
-- ============================================

CREATE TYPE job_status AS ENUM (
    'pending', 'running', 'completed', 'failed', 'cancelled'
);

CREATE TYPE job_priority AS ENUM ('high', 'normal', 'low');

CREATE TYPE doc_source AS ENUM ('crawl', 'upload', 'api');

CREATE TYPE deep_crawl_strategy AS ENUM ('bfs', 'dfs', 'best_first');

CREATE TYPE chunk_content_type AS ENUM ('prose', 'code', 'table', 'list');


-- ============================================
-- AUTH
-- ============================================

CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    scopes TEXT[] DEFAULT '{read}',
    rate_limit_rpm INT DEFAULT 60,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_active ON api_keys(is_active) WHERE is_active = TRUE;


-- ============================================
-- ORGANIZATION
-- ============================================

CREATE TABLE collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================
-- CRAWL CONFIGURATION
-- ============================================

CREATE TABLE crawl_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    max_depth INT DEFAULT 0,
    rate_limit_rps FLOAT DEFAULT 1.0,
    respect_robots_txt BOOLEAN DEFAULT TRUE,
    headers JSONB DEFAULT '{}',
    extraction_strategy TEXT DEFAULT 'markdown',
    chunking_strategy TEXT DEFAULT 'sliding_window',
    chunk_size INT DEFAULT 512,
    chunk_overlap INT DEFAULT 50,
    min_chunk_size INT DEFAULT 50,
    page_timeout_ms INT DEFAULT 30000,
    max_page_size_mb INT DEFAULT 50,
    max_redirects INT DEFAULT 10,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE crawl_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    url_pattern TEXT NOT NULL,
    crawl_config_id UUID REFERENCES crawl_configs(id),
    collection_id UUID REFERENCES collections(id),
    schedule_cron TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    last_crawled_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_crawl_sources_active ON crawl_sources(is_active) WHERE is_active = TRUE;

CREATE TABLE domain_settings (
    domain TEXT PRIMARY KEY,
    rate_limit_rps FLOAT DEFAULT 1.0,
    circuit_breaker_threshold INT DEFAULT 5,
    circuit_breaker_timeout_s INT DEFAULT 300,
    is_blocked BOOLEAN DEFAULT FALSE,
    blocked_reason TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_domain_settings_blocked ON domain_settings(is_blocked) WHERE is_blocked = TRUE;

CREATE TABLE proxy_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    servers JSONB NOT NULL,
    rotation_strategy TEXT DEFAULT 'round_robin',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================
-- DOCUMENTS
-- ============================================

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT UNIQUE NOT NULL,
    domain TEXT NOT NULL,
    parent_url TEXT,
    title TEXT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    language TEXT DEFAULT 'en',
    source doc_source NOT NULL DEFAULT 'crawl',
    collection_id UUID REFERENCES collections(id),
    metadata JSONB DEFAULT '{}',
    crawled_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,

    -- Full-text search vector (on title + content)
    fts_vector TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', content), 'B')
    ) STORED
);

-- Trigger to update updated_at on documents
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE INDEX idx_documents_fts ON documents USING GIN(fts_vector);
CREATE INDEX idx_documents_url ON documents(url);
CREATE INDEX idx_documents_domain ON documents(domain);
CREATE INDEX idx_documents_hash ON documents(content_hash);
CREATE INDEX idx_documents_collection ON documents(collection_id);
CREATE INDEX idx_documents_crawled_at ON documents(crawled_at);
CREATE INDEX idx_documents_not_deleted ON documents(id) WHERE deleted_at IS NULL;

CREATE TABLE document_tags (
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    tag_id UUID REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (document_id, tag_id)
);

CREATE INDEX idx_document_tags_tag ON document_tags(tag_id);


-- ============================================
-- CHUNKS
-- ============================================

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INT NOT NULL,
    start_char INT NOT NULL,
    end_char INT NOT NULL,
    token_count INT NOT NULL,
    embedding_model TEXT NOT NULL,
    section_header TEXT,
    content_type chunk_content_type DEFAULT 'prose',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Full-text search vector for chunk-level search
    fts_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED,

    UNIQUE(document_id, chunk_index)
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_fts ON chunks USING GIN(fts_vector);
CREATE INDEX idx_chunks_embedding_model ON chunks(embedding_model);

CREATE TRIGGER update_chunks_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- ============================================
-- JOBS
-- ============================================

CREATE TABLE crawl_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL,
    domain TEXT NOT NULL,
    crawl_config_id UUID REFERENCES crawl_configs(id),
    crawl_source_id UUID REFERENCES crawl_sources(id),
    document_id UUID REFERENCES documents(id),
    collection_id UUID REFERENCES collections(id),
    priority job_priority NOT NULL DEFAULT 'normal',
    status job_status NOT NULL DEFAULT 'pending',
    error TEXT,
    retry_count INT DEFAULT 0 CHECK (retry_count >= 0),
    max_retries INT DEFAULT 3,
    result JSONB,
    result_expires_at TIMESTAMPTZ,
    webhook_url TEXT,
    webhook_headers JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_crawl_jobs_status ON crawl_jobs(status);
CREATE INDEX idx_crawl_jobs_priority ON crawl_jobs(priority, created_at) WHERE status = 'pending';
CREATE INDEX idx_crawl_jobs_domain ON crawl_jobs(domain);
CREATE INDEX idx_crawl_jobs_created ON crawl_jobs(created_at);
CREATE INDEX idx_crawl_jobs_source ON crawl_jobs(crawl_source_id);

CREATE TRIGGER update_crawl_jobs_updated_at
    BEFORE UPDATE ON crawl_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TABLE deep_crawl_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    root_url TEXT NOT NULL,
    strategy deep_crawl_strategy NOT NULL,
    max_depth INT NOT NULL,
    max_pages INT NOT NULL,
    score_threshold FLOAT DEFAULT 0.0,
    keywords TEXT[] DEFAULT '{}',
    crawl_config_id UUID REFERENCES crawl_configs(id),
    collection_id UUID REFERENCES collections(id),
    webhook_url TEXT,
    webhook_headers JSONB DEFAULT '{}',
    status job_status NOT NULL DEFAULT 'pending',
    pages_discovered INT DEFAULT 0,
    pages_crawled INT DEFAULT 0,
    pages_failed INT DEFAULT 0,
    current_depth INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_deep_crawl_jobs_status ON deep_crawl_jobs(status);

CREATE TABLE deep_crawl_frontier (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deep_crawl_job_id UUID REFERENCES deep_crawl_jobs(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    depth INT NOT NULL,
    score FLOAT,
    status job_status DEFAULT 'pending',
    discovered_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(deep_crawl_job_id, url)
);

CREATE INDEX idx_frontier_job_status ON deep_crawl_frontier(deep_crawl_job_id, status);
CREATE INDEX idx_frontier_job_pending_score ON deep_crawl_frontier(deep_crawl_job_id, score DESC)
    WHERE status = 'pending';
CREATE INDEX idx_frontier_job_pending_depth ON deep_crawl_frontier(deep_crawl_job_id, depth, discovered_at)
    WHERE status = 'pending';

CREATE TABLE discovery_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain TEXT NOT NULL,
    sources TEXT[] NOT NULL,
    pattern TEXT,
    max_urls INT DEFAULT 500,
    score_query TEXT,
    score_threshold FLOAT DEFAULT 0.0,
    status job_status NOT NULL DEFAULT 'pending',
    urls_found INT DEFAULT 0,
    result JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);


-- ============================================
-- WEBHOOKS
-- ============================================

CREATE TABLE webhooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID REFERENCES crawl_sources(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    headers JSONB DEFAULT '{}',
    secret TEXT,
    events TEXT[] DEFAULT '{completed,failed}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_webhooks_source ON webhooks(source_id);
CREATE INDEX idx_webhooks_active ON webhooks(is_active) WHERE is_active = TRUE;

CREATE TABLE webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    webhook_id UUID REFERENCES webhooks(id) ON DELETE CASCADE,
    job_id UUID,
    job_type TEXT,                          -- 'crawl', 'deep_crawl', 'discovery'
    event TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT DEFAULT 'pending',
    attempts INT DEFAULT 0,
    max_attempts INT DEFAULT 5,
    last_attempt_at TIMESTAMPTZ,
    next_attempt_at TIMESTAMPTZ,
    response_status INT,
    response_body TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_webhook_deliveries_pending ON webhook_deliveries(next_attempt_at)
    WHERE status = 'pending';
CREATE INDEX idx_webhook_deliveries_webhook ON webhook_deliveries(webhook_id);


-- ============================================
-- ROBOTS.TXT CACHE
-- ============================================

CREATE TABLE robots_txt_cache (
    domain TEXT PRIMARY KEY,
    content TEXT,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '7 days'
);


-- ============================================
-- URL NORMALIZATION CACHE (for dedup)
-- ============================================

CREATE TABLE canonical_urls (
    url_hash TEXT PRIMARY KEY,              -- SHA256 of normalized URL
    canonical_url TEXT NOT NULL,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 4.2 Migrations

Use Alembic for schema migrations. Initial migration creates all tables above.

---

## 5. Redis Structure

### 5.1 Key Naming Convention

```
crawl4r:{category}:{subcategory}:{identifier}
```

**Hash Truncation:** All SHA256 hashes in cache keys are truncated to 32 characters (128 bits) to balance key length with collision resistance. This provides ~2^64 collision resistance which is sufficient for cache deduplication.

### 5.2 Cache Keys

```
# Crawl result cache (raw HTML + markdown)
crawl4r:cache:crawl:{url_sha256[:32]}
  Type: STRING (JSON, gzip compressed for values > 10KB)
  TTL: 24 hours
  Value: {html, markdown, metadata, chunked_at}
  Max size: 50MB (reject larger)

# Embedding cache (per content hash)
crawl4r:cache:embed:{content_sha256[:32]}
  Type: STRING (JSON)
  TTL: 7 days
  Value: {embedding: [...], model: "qwen3-0.6b", created_at}

# Query result cache
crawl4r:cache:query:{query_hash[:32]}
  Type: STRING (JSON)
  TTL: 1 hour
  Value: {results: [...], total_count, created_at}
  Note: query_hash = SHA256(normalized_query + "|" + sorted_filters_json)

# Query embedding cache
crawl4r:cache:query_embed:{query_sha256[:32]}
  Type: STRING (JSON)
  TTL: 24 hours
  Value: {embedding: [...], model: "qwen3-0.6b"}
```

### 5.3 Rate Limiting

```
# Token bucket per domain
crawl4r:rate:{domain}
  Type: STRING (counter)
  TTL: 1 second
  Value: remaining tokens

# Sliding window rate limit per API key
crawl4r:rate:api:{api_key_hash}:{window_minute}
  Type: STRING (counter)
  TTL: 2 minutes
  Value: request count
```

### 5.4 Circuit Breaker

```
# Circuit breaker state per domain
crawl4r:circuit:{domain}
  Type: HASH
  TTL: 1 hour (reset after timeout)
  Fields:
    - failures: int
    - last_failure: timestamp
    - state: closed|open|half_open
    - opened_at: timestamp (when circuit opened)
```

### 5.5 Job Queues (ARQ)

```
# Priority queues
crawl4r:queue:high
crawl4r:queue:normal
crawl4r:queue:low

# Dead letter queue
crawl4r:queue:dlq

# Retry queues (for failed storage operations)
crawl4r:queue:qdrant_retry
crawl4r:queue:webhook_retry
```

### 5.6 Distributed Locks

```
# URL crawl lock (prevent duplicate simultaneous crawls)
crawl4r:lock:crawl:{url_sha256}
  Type: STRING
  TTL: 5 minutes
  Value: worker_id

# Query cache lock (prevent stampede)
crawl4r:lock:query:{query_hash}
  Type: STRING
  TTL: 30 seconds
  Value: worker_id
```

### 5.7 Pub/Sub Channels

```
# Job progress updates (for WebSocket)
crawl4r:ws:job:{job_id}

# Deep crawl progress
crawl4r:ws:deep_crawl:{job_id}
```

### 5.8 Metrics

```
# Counter metrics
crawl4r:metrics:counter:{metric_name}:{date}
  Type: STRING (counter)
  TTL: 7 days

# Gauge metrics (current values)
crawl4r:metrics:gauge:{metric_name}
  Type: STRING
  No TTL

# Histogram buckets
crawl4r:metrics:histogram:{metric_name}:{date}
  Type: SORTED SET
  Score: timestamp
  Member: value
  TTL: 24 hours
```

---

## 6. Qdrant Configuration

### 6.1 Collection Setup

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    ScalarQuantizationConfig,
    ScalarType,
    PayloadSchemaType,
)

COLLECTION_NAME = "crawl4r"
VECTOR_SIZE = 1024

async def setup_collection(client: QdrantClient):
    """Create or recreate the Qdrant collection."""

    # Delete existing collection if present
    collections = await client.get_collections()
    if COLLECTION_NAME in [c.name for c in collections.collections]:
        await client.delete_collection(COLLECTION_NAME)

    # Create collection with optimized settings
    await client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
            on_disk=True,  # Store vectors on disk for large collections
        ),
        hnsw_config=HnswConfigDiff(
            m=16,                    # Connections per node (default 16)
            ef_construct=100,        # Construction-time accuracy (default 100)
            full_scan_threshold=10000,  # Switch to full scan below this
        ),
        quantization_config=ScalarQuantizationConfig(
            type=ScalarType.INT8,    # 4x memory reduction
            quantile=0.99,           # Clip outliers
            always_ram=True,         # Keep quantized vectors in RAM
        ),
    )

    # Create payload indexes for filtering
    await client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="document_id",
        field_schema=PayloadSchemaType.UUID,
    )
    await client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="collection_id",
        field_schema=PayloadSchemaType.UUID,
    )
    await client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="domain",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="tags",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="source",
        field_schema=PayloadSchemaType.KEYWORD,
    )
```

### 6.2 Point Structure

```python
from qdrant_client.models import PointStruct

point = PointStruct(
    id=str(chunk.id),                # UUID as string
    vector=embedding,                 # 1024-dim float list
    payload={
        "document_id": str(chunk.document_id),
        "chunk_id": str(chunk.id),
        "chunk_index": chunk.chunk_index,
        "collection_id": str(document.collection_id) if document.collection_id else None,
        "domain": document.domain,
        "url": document.url,
        "title": document.title,
        "source": document.source.value,          # crawl, upload, api - for filtering
        "tags": [str(t) for t in document_tags],  # Denormalized for filtering
        "section_header": chunk.section_header,
        "content_type": chunk.content_type.value,
        "crawled_at": document.crawled_at.isoformat(),
    },
)
```

### 6.3 Search Parameters

```python
from qdrant_client.models import Filter, FieldCondition, MatchAny, DatetimeRange

async def vector_search(
    client: QdrantClient,
    query_embedding: list[float],
    filters: SearchFilters,
    limit: int = 10,
    ef: int = 128,  # Search-time accuracy parameter
) -> list[ScoredPoint]:

    # Build filter conditions
    must_conditions = []

    if filters.collection_ids:
        must_conditions.append(
            FieldCondition(
                key="collection_id",
                match=MatchAny(any=[str(c) for c in filters.collection_ids]),
            )
        )

    if filters.domains:
        must_conditions.append(
            FieldCondition(
                key="domain",
                match=MatchAny(any=filters.domains),
            )
        )

    if filters.tag_ids:
        must_conditions.append(
            FieldCondition(
                key="tags",
                match=MatchAny(any=[str(t) for t in filters.tag_ids]),
            )
        )

    if filters.source_types:
        must_conditions.append(
            FieldCondition(
                key="source",
                match=MatchAny(any=[s.value for s in filters.source_types]),
            )
        )

    if filters.date_from or filters.date_to:
        must_conditions.append(
            FieldCondition(
                key="crawled_at",
                range=DatetimeRange(
                    gte=filters.date_from.isoformat() if filters.date_from else None,
                    lte=filters.date_to.isoformat() if filters.date_to else None,
                ),
            )
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    return await client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        query_filter=query_filter,
        limit=limit,
        search_params={"ef": ef},
        with_payload=True,
    )
```

---

## 7. API Endpoints

> **API Version Prefix:** All endpoints are prefixed with `/api/v1/`. Example: `POST /api/v1/crawl`

### 7.0 Pagination & Response Models

```python
# app/api/schemas.py

from typing import Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Standard paginated response wrapper."""
    items: list[T]
    total: int                              # Total items matching query
    limit: int                              # Items per page
    offset: int                             # Current offset
    has_more: bool                          # More items available


class CursorPaginatedResponse(BaseModel, Generic[T]):
    """Cursor-based pagination for efficient large dataset traversal."""
    items: list[T]
    next_cursor: str | None = None          # Opaque cursor for next page
    has_more: bool


class JobResponse(BaseModel):
    """Standard response for job creation."""
    job_id: str
    status: str
    message: str | None = None


class BatchJobResponse(BaseModel):
    """Response for batch job creation."""
    jobs: list[JobResponse]
    total_submitted: int
    failed_count: int = 0


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    code: str                               # Machine-readable error code
    details: dict[str, Any] | None = None
    request_id: str | None = None
```

### 7.1 Crawl Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/crawl` | Submit URL(s) for crawling. Returns job ID(s) |
| `POST` | `/api/v1/crawl/batch` | Batch submit URLs (up to 100) |
| `POST` | `/api/v1/crawl/deep` | Deep crawl with BFS/DFS/BestFirst strategy |
| `GET` | `/api/v1/crawl/{job_id}` | Get job status and result |
| `GET` | `/api/v1/crawl/{job_id}/stream` | SSE stream of crawl progress |
| `POST` | `/api/v1/crawl/{job_id}/cancel` | Cancel pending/running job |
| `WS` | `/api/v1/ws/crawl/{job_id}` | WebSocket for real-time progress |

### 7.2 URL Discovery

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/discover` | Discover URLs from sitemap/Common Crawl |
| `POST` | `/api/v1/discover/score` | Score URLs for relevance |
| `GET` | `/api/v1/discover/{job_id}` | Get discovery job results |

### 7.3 Search Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/search` | Hybrid search with RRF fusion |
| `POST` | `/api/v1/search/vector` | Vector-only search |
| `POST` | `/api/v1/search/keyword` | Keyword-only (FTS) search |

### 7.4 Document CRUD

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/documents` | List documents (paginated, filterable) |
| `GET` | `/api/v1/documents/{id}` | Get document with chunks |
| `POST` | `/api/v1/documents` | Manual document upload |
| `POST` | `/api/v1/documents/batch` | Batch upload documents (up to 50) |
| `DELETE` | `/api/v1/documents/{id}` | Soft delete document |
| `POST` | `/api/v1/documents/{id}/reindex` | Re-embed with current model |
| `GET` | `/api/v1/documents/{id}/chunks` | Get chunks for a document |

### 7.5 Media Capture

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/screenshot` | Capture screenshot of URL |
| `POST` | `/api/v1/pdf` | Generate PDF of URL |
| `POST` | `/api/v1/execute-js` | Execute JavaScript on URL |

### 7.6 Collections & Tags

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET/POST` | `/api/v1/collections` | List/create collections |
| `GET/PUT/DELETE` | `/api/v1/collections/{id}` | Collection CRUD |
| `GET/POST` | `/api/v1/tags` | List/create tags |
| `POST` | `/api/v1/documents/{id}/tags` | Add tags to document |
| `DELETE` | `/api/v1/documents/{id}/tags/{tag_id}` | Remove tag |

### 7.7 Crawl Sources (Monitoring)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET/POST` | `/api/v1/sources` | List/create monitored sources |
| `GET/PUT/DELETE` | `/api/v1/sources/{id}` | Source CRUD |
| `POST` | `/api/v1/sources/{id}/trigger` | Manually trigger |
| `GET/POST` | `/api/v1/sources/{id}/webhooks` | Manage webhooks |

### 7.8 Crawl Configs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET/POST` | `/api/v1/configs` | List/create crawl configurations |
| `GET/PUT/DELETE` | `/api/v1/configs/{id}` | Config CRUD |

### 7.9 Proxy Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET/POST` | `/api/v1/proxies` | List/create proxy configurations |
| `GET/PUT/DELETE` | `/api/v1/proxies/{id}` | Proxy CRUD |
| `POST` | `/api/v1/proxies/test` | Test proxy connectivity |

### 7.10 Domain Settings

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/domains` | List domains with settings |
| `GET/PUT` | `/api/v1/domains/{domain}` | Get/update domain settings |
| `POST` | `/api/v1/domains/{domain}/block` | Block domain |
| `POST` | `/api/v1/domains/{domain}/unblock` | Unblock domain |

### 7.11 Cache Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `DELETE` | `/api/v1/cache/crawl` | Clear crawl cache |
| `DELETE` | `/api/v1/cache/query` | Clear query cache |
| `DELETE` | `/api/v1/cache/embeddings` | Clear embedding cache |
| `DELETE` | `/api/v1/cache/domain/{domain}` | Clear all for domain |
| `GET` | `/api/v1/cache/stats` | Cache statistics |

### 7.12 Admin & Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/health/ready` | Readiness check (deps healthy) |
| `GET` | `/api/v1/stats` | System stats |
| `GET` | `/api/v1/stats/domains` | Per-domain statistics |
| `GET/POST` | `/api/v1/api-keys` | Manage API keys |
| `GET` | `/api/v1/jobs` | List all jobs |
| `GET` | `/api/v1/jobs/failed` | List failed jobs (DLQ) |
| `POST` | `/api/v1/jobs/{job_id}/retry` | Retry failed job |

---

## 8. Service Components

### 8.1 Abstract Interfaces

```python
# app/core/abstractions.py

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID


class VectorStore(ABC):
    """Abstract interface for vector storage backends."""

    @abstractmethod
    async def upsert(self, points: list[dict[str, Any]]) -> None:
        """Upsert vectors with payloads."""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID."""
        pass

    @abstractmethod
    async def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """Delete vectors matching filter, return count."""
        pass


class DocumentStore(ABC):
    """Abstract interface for document storage backends."""

    @abstractmethod
    async def save(self, document: "Document") -> "Document":
        """Save or update a document."""
        pass

    @abstractmethod
    async def get(self, doc_id: UUID) -> "Document | None":
        """Get document by ID."""
        pass

    @abstractmethod
    async def get_by_url(self, url: str) -> "Document | None":
        """Get document by URL."""
        pass

    @abstractmethod
    async def search_fts(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple["Chunk", float]]:
        """Full-text search, returns chunks with scores."""
        pass

    @abstractmethod
    async def delete(self, doc_id: UUID, hard: bool = False) -> None:
        """Soft or hard delete document."""
        pass


class Embedder(ABC):
    """Abstract interface for embedding generation."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class Crawler(ABC):
    """Abstract interface for web crawling."""

    @abstractmethod
    async def crawl_url(
        self,
        url: str,
        config: "CrawlConfig",
    ) -> "CrawlResult":
        """Crawl a single URL."""
        pass

    @abstractmethod
    async def crawl_batch(
        self,
        urls: list[str],
        config: "CrawlConfig",
    ) -> list["CrawlResult"]:
        """Crawl multiple URLs."""
        pass


class Cache(ABC):
    """Abstract interface for caching."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Set value with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete key."""
        pass

    @abstractmethod
    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern, return count."""
        pass
```

### 8.2 URL Validator

```python
# app/services/url_validator.py

import ipaddress
import re
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl

from app.core.models import ValidatedURL


class URLValidator:
    """Validates and normalizes URLs, preventing SSRF attacks."""

    # Blocked IP ranges (SSRF prevention)
    BLOCKED_RANGES = [
        ipaddress.ip_network("127.0.0.0/8"),       # Loopback
        ipaddress.ip_network("10.0.0.0/8"),        # Private
        ipaddress.ip_network("172.16.0.0/12"),     # Private
        ipaddress.ip_network("192.168.0.0/16"),    # Private
        ipaddress.ip_network("169.254.0.0/16"),    # Link-local
        ipaddress.ip_network("::1/128"),           # IPv6 loopback
        ipaddress.ip_network("fc00::/7"),          # IPv6 private
        ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
    ]

    # Cloud metadata endpoints
    BLOCKED_HOSTS = {
        "metadata.google.internal",
        "169.254.169.254",                         # AWS/GCP metadata
        "metadata.azure.com",
    }

    # Allowed schemes
    ALLOWED_SCHEMES = {"http", "https"}

    # Max URL length
    MAX_URL_LENGTH = 2048

    def validate(self, url: str) -> ValidatedURL:
        """Validate and normalize a URL."""

        # Length check
        if len(url) > self.MAX_URL_LENGTH:
            return ValidatedURL(
                original=url,
                normalized=url,
                scheme="",
                host="",
                domain="",
                path="",
                is_valid=False,
                rejection_reason=f"URL exceeds max length ({self.MAX_URL_LENGTH})",
            )

        try:
            parsed = urlparse(url)
        except Exception as e:
            return ValidatedURL(
                original=url,
                normalized=url,
                scheme="",
                host="",
                domain="",
                path="",
                is_valid=False,
                rejection_reason=f"Failed to parse URL: {e}",
            )

        # Scheme check
        if parsed.scheme.lower() not in self.ALLOWED_SCHEMES:
            return ValidatedURL(
                original=url,
                normalized=url,
                scheme=parsed.scheme,
                host=parsed.netloc,
                domain="",
                path=parsed.path,
                is_valid=False,
                rejection_reason=f"Scheme not allowed: {parsed.scheme}",
            )

        # Host check
        host = parsed.netloc.lower().split(":")[0]  # Remove port

        if not host:
            return ValidatedURL(
                original=url,
                normalized=url,
                scheme=parsed.scheme,
                host="",
                domain="",
                path=parsed.path,
                is_valid=False,
                rejection_reason="No host specified",
            )

        # Blocked hosts
        if host in self.BLOCKED_HOSTS:
            return ValidatedURL(
                original=url,
                normalized=url,
                scheme=parsed.scheme,
                host=host,
                domain="",
                path=parsed.path,
                is_valid=False,
                rejection_reason=f"Host is blocked: {host}",
            )

        # IP address check
        try:
            ip = ipaddress.ip_address(host)
            for blocked_range in self.BLOCKED_RANGES:
                if ip in blocked_range:
                    return ValidatedURL(
                        original=url,
                        normalized=url,
                        scheme=parsed.scheme,
                        host=host,
                        domain="",
                        path=parsed.path,
                        is_valid=False,
                        rejection_reason=f"IP address in blocked range: {host}",
                    )
        except ValueError:
            # Not an IP address, continue with domain validation
            pass

        # Extract domain
        domain = self._extract_domain(host)

        # Normalize URL
        normalized = self._normalize_url(parsed)

        return ValidatedURL(
            original=url,
            normalized=normalized,
            scheme=parsed.scheme.lower(),
            host=host,
            domain=domain,
            path=parsed.path,
            is_valid=True,
        )

    def _extract_domain(self, host: str) -> str:
        """Extract the registrable domain from a host using tldextract."""
        import tldextract
        extracted = tldextract.extract(host)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}"
        return host

    def _normalize_url(self, parsed) -> str:
        """Normalize URL for consistent caching/dedup."""
        # Lowercase scheme and host
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        # Normalize path (remove trailing slash except for root)
        path = parsed.path
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        if not path:
            path = "/"

        # Sort query parameters
        query_params = parse_qsl(parsed.query, keep_blank_values=True)
        sorted_query = urlencode(sorted(query_params))

        # Remove fragment
        return urlunparse((scheme, netloc, path, "", sorted_query, ""))
```

### 8.3 Content Processor

```python
# app/services/content_processor.py

import hashlib
import re
from dataclasses import dataclass

import charset_normalizer
from bs4 import BeautifulSoup

from app.core.models import DocumentMetadata


@dataclass
class ProcessedContent:
    content: str
    content_hash: str
    title: str | None
    language: str
    metadata: DocumentMetadata
    is_valid: bool = True
    rejection_reason: str | None = None


class ContentProcessor:
    """Processes and validates crawled content."""

    ALLOWED_CONTENT_TYPES = {
        "text/html",
        "text/plain",
        "application/xhtml+xml",
        "application/xml",
        "text/xml",
    }

    MAX_CONTENT_SIZE = 50 * 1024 * 1024  # 50MB

    def process(
        self,
        raw_html: str,
        content_type: str | None = None,
        detected_encoding: str | None = None,
    ) -> ProcessedContent:
        """Process raw crawl response into clean content."""

        # Content type check
        if content_type:
            base_type = content_type.split(";")[0].strip().lower()
            if base_type not in self.ALLOWED_CONTENT_TYPES:
                return ProcessedContent(
                    content="",
                    content_hash="",
                    title=None,
                    language="unknown",
                    metadata=DocumentMetadata(),
                    is_valid=False,
                    rejection_reason=f"Content type not allowed: {base_type}",
                )

        # Size check
        if len(raw_html) > self.MAX_CONTENT_SIZE:
            return ProcessedContent(
                content="",
                content_hash="",
                title=None,
                language="unknown",
                metadata=DocumentMetadata(),
                is_valid=False,
                rejection_reason=f"Content exceeds max size ({self.MAX_CONTENT_SIZE} bytes)",
            )

        # Encoding normalization
        if detected_encoding:
            try:
                raw_html = raw_html.encode(detected_encoding).decode("utf-8", errors="replace")
            except Exception:
                pass

        # Parse HTML
        soup = BeautifulSoup(raw_html, "html.parser")

        # Extract metadata
        metadata = self._extract_metadata(soup)

        # Extract title
        title = self._extract_title(soup)

        # Detect language
        language = self._detect_language(soup, raw_html)

        # Extract text content
        content = self._extract_text(soup)

        # Compute content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        return ProcessedContent(
            content=content,
            content_hash=content_hash,
            title=title,
            language=language,
            metadata=metadata,
        )

    def _extract_metadata(self, soup: BeautifulSoup) -> DocumentMetadata:
        """Extract metadata from HTML head."""
        metadata = DocumentMetadata()

        # Description
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and desc_tag.get("content"):
            metadata.description = desc_tag["content"]

        # Author
        author_tag = soup.find("meta", attrs={"name": "author"})
        if author_tag and author_tag.get("content"):
            metadata.author = author_tag["content"]

        # Keywords
        keywords_tag = soup.find("meta", attrs={"name": "keywords"})
        if keywords_tag and keywords_tag.get("content"):
            metadata.keywords = [k.strip() for k in keywords_tag["content"].split(",")]

        # Open Graph
        og_image = soup.find("meta", attrs={"property": "og:image"})
        if og_image and og_image.get("content"):
            metadata.og_image = og_image["content"]

        # Canonical URL
        canonical = soup.find("link", attrs={"rel": "canonical"})
        if canonical and canonical.get("href"):
            metadata.canonical_url = canonical["href"]

        # Publish date (various formats)
        for attr in ["article:published_time", "datePublished", "date"]:
            date_tag = soup.find("meta", attrs={"property": attr}) or \
                       soup.find("meta", attrs={"name": attr})
            if date_tag and date_tag.get("content"):
                try:
                    from dateutil import parser
                    metadata.publish_date = parser.parse(date_tag["content"])
                    break
                except Exception:
                    pass

        return metadata

    def _extract_title(self, soup: BeautifulSoup) -> str | None:
        """Extract page title."""
        # Try og:title first
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        # Fall back to title tag
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text().strip()

        # Fall back to first h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text().strip()

        return None

    def _detect_language(self, soup: BeautifulSoup, raw_html: str) -> str:
        """Detect content language."""
        # Check html lang attribute
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            lang = html_tag["lang"].split("-")[0].lower()
            if lang:
                return lang

        # Check Content-Language meta
        lang_meta = soup.find("meta", attrs={"http-equiv": "Content-Language"})
        if lang_meta and lang_meta.get("content"):
            return lang_meta["content"].split("-")[0].lower()

        # Fall back to charset_normalizer language detection
        try:
            result = charset_normalizer.from_bytes(raw_html.encode()[:10000])
            if result.best():
                return result.best().language or "unknown"
        except Exception:
            pass

        return "unknown"

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML."""
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Get text
        text = soup.get_text(separator="\n")

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text
```

### 8.4 Embedding Validator

```python
# app/services/embedding_validator.py

import math
from dataclasses import dataclass


@dataclass
class ValidationResult:
    is_valid: bool
    reason: str | None = None


class EmbeddingValidator:
    """Validates embedding quality before storage."""

    def __init__(self, expected_dimension: int = 1024):
        self.expected_dimension = expected_dimension

    def validate(self, embedding: list[float]) -> ValidationResult:
        """Validate an embedding vector."""

        # Dimension check
        if len(embedding) != self.expected_dimension:
            return ValidationResult(
                is_valid=False,
                reason=f"Wrong dimension: {len(embedding)}, expected {self.expected_dimension}",
            )

        # Check for NaN/Inf
        for i, val in enumerate(embedding):
            if math.isnan(val):
                return ValidationResult(
                    is_valid=False,
                    reason=f"NaN value at index {i}",
                )
            if math.isinf(val):
                return ValidationResult(
                    is_valid=False,
                    reason=f"Infinite value at index {i}",
                )

        # Check not all zeros
        if all(v == 0 for v in embedding):
            return ValidationResult(
                is_valid=False,
                reason="All zero values",
            )

        # Check L2 norm is reasonable (should be close to 1 for normalized embeddings)
        l2_norm = math.sqrt(sum(v * v for v in embedding))
        if l2_norm < 0.1:
            return ValidationResult(
                is_valid=False,
                reason=f"L2 norm too small: {l2_norm}",
            )

        return ValidationResult(is_valid=True)

    def validate_batch(self, embeddings: list[list[float]]) -> list[ValidationResult]:
        """Validate multiple embeddings."""
        return [self.validate(emb) for emb in embeddings]
```

### 8.5 Query Preprocessor

```python
# app/services/query_preprocessor.py

import hashlib
import re
from dataclasses import dataclass


@dataclass
class ProcessedQuery:
    original: str
    normalized: str
    cache_key: str
    tokens: list[str]
    is_valid: bool = True
    rejection_reason: str | None = None


class QueryPreprocessor:
    """Preprocesses search queries for consistency."""

    MAX_QUERY_LENGTH = 1000
    MAX_TOKENS = 512

    def process(self, query: str, filters_hash: str = "") -> ProcessedQuery:
        """Process a search query."""

        # Trim whitespace
        query = query.strip()

        # Length check
        if len(query) > self.MAX_QUERY_LENGTH:
            return ProcessedQuery(
                original=query,
                normalized=query[:self.MAX_QUERY_LENGTH],
                cache_key="",
                tokens=[],
                is_valid=False,
                rejection_reason=f"Query exceeds max length ({self.MAX_QUERY_LENGTH})",
            )

        if not query:
            return ProcessedQuery(
                original=query,
                normalized="",
                cache_key="",
                tokens=[],
                is_valid=False,
                rejection_reason="Empty query",
            )

        # Normalize
        normalized = self._normalize(query)

        # Tokenize (simple whitespace split)
        tokens = normalized.split()

        # Truncate tokens if needed
        if len(tokens) > self.MAX_TOKENS:
            tokens = tokens[:self.MAX_TOKENS]
            normalized = " ".join(tokens)

        # Generate cache key
        cache_input = f"{normalized}|{filters_hash}"
        cache_key = hashlib.sha256(cache_input.encode()).hexdigest()[:32]

        return ProcessedQuery(
            original=query,
            normalized=normalized,
            cache_key=cache_key,
            tokens=tokens,
        )

    def _normalize(self, query: str) -> str:
        """Normalize query for caching."""
        # Lowercase
        query = query.lower()

        # Collapse whitespace
        query = re.sub(r"\s+", " ", query)

        # Strip
        query = query.strip()

        return query
```

### 8.6 RRF Fusion

```python
# app/core/rrf.py

from dataclasses import dataclass
from uuid import UUID


@dataclass
class RankedItem:
    chunk_id: UUID
    document_id: UUID
    content: str
    vector_rank: int | None = None
    vector_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    rrf_score: float = 0.0


def reciprocal_rank_fusion(
    vector_results: list[dict],
    keyword_results: list[dict],
    k: int = 60,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> list[RankedItem]:
    """
    Combine vector and keyword search results using RRF.

    RRF formula: score = sum(weight / (k + rank))

    Args:
        vector_results: Results from vector search with 'chunk_id', 'score'
        keyword_results: Results from keyword search with 'chunk_id', 'score'
        k: RRF constant (default 60)
        vector_weight: Weight for vector results
        keyword_weight: Weight for keyword results

    Returns:
        Combined and re-ranked results
    """

    # Build lookup by chunk_id
    items: dict[UUID, RankedItem] = {}

    # Process vector results
    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result["chunk_id"]
        items[chunk_id] = RankedItem(
            chunk_id=chunk_id,
            document_id=result["document_id"],
            content=result["content"],
            vector_rank=rank,
            vector_score=result.get("score"),
        )

    # Process keyword results
    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = result["chunk_id"]
        if chunk_id in items:
            items[chunk_id].keyword_rank = rank
            items[chunk_id].keyword_score = result.get("score")
        else:
            items[chunk_id] = RankedItem(
                chunk_id=chunk_id,
                document_id=result["document_id"],
                content=result["content"],
                keyword_rank=rank,
                keyword_score=result.get("score"),
            )

    # Calculate RRF scores
    for item in items.values():
        score = 0.0
        if item.vector_rank is not None:
            score += vector_weight / (k + item.vector_rank)
        if item.keyword_rank is not None:
            score += keyword_weight / (k + item.keyword_rank)
        item.rrf_score = score

    # Sort by RRF score descending
    ranked = sorted(items.values(), key=lambda x: x.rrf_score, reverse=True)

    return ranked
```

### 8.7 Reranker

```python
# app/core/reranker.py

from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    """Cross-encoder reranker for improving search precision."""

    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls):
        """Singleton pattern for lazy loading."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self):
        """Lazy load the reranker model."""
        if self._model is None:
            model_name = "Alibaba-NLP/gte-reranker-modernbert-base"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: List of dicts with 'content' field
            top_n: Number of top results to return

        Returns:
            Reranked documents with 'rerank_score' added
        """
        self._load_model()

        if not documents:
            return []

        # Prepare pairs
        pairs = [(query, doc["content"]) for doc in documents]

        # Tokenize
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Score
        with torch.no_grad():
            scores = self._model(**inputs).logits.squeeze(-1)
            scores = torch.sigmoid(scores).cpu().numpy()

        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sort by rerank score and return top_n
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
```

### 8.8 Circuit Breaker

```python
# app/services/circuit_breaker.py

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import redis.asyncio as redis


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStatus:
    state: CircuitState
    failures: int
    last_failure: datetime | None
    opened_at: datetime | None


class CircuitBreaker:
    """Redis-backed circuit breaker for domain-level failure handling."""

    KEY_PREFIX = "crawl4r:circuit"

    def __init__(
        self,
        redis_client: redis.Redis,
        threshold: int = 5,
        timeout_seconds: int = 300,
    ):
        self.redis = redis_client
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds

    def _key(self, domain: str) -> str:
        return f"{self.KEY_PREFIX}:{domain}"

    async def get_status(self, domain: str) -> CircuitStatus:
        """Get current circuit breaker status for a domain."""
        data = await self.redis.hgetall(self._key(domain))
        if not data:
            return CircuitStatus(
                state=CircuitState.CLOSED,
                failures=0,
                last_failure=None,
                opened_at=None,
            )

        state = CircuitState(data.get(b"state", b"closed").decode())
        failures = int(data.get(b"failures", 0))
        last_failure = None
        opened_at = None

        if b"last_failure" in data:
            last_failure = datetime.fromisoformat(data[b"last_failure"].decode())
        if b"opened_at" in data:
            opened_at = datetime.fromisoformat(data[b"opened_at"].decode())

        # Check if circuit should transition to half-open
        if state == CircuitState.OPEN and opened_at:
            if datetime.utcnow() > opened_at + timedelta(seconds=self.timeout_seconds):
                await self._set_state(domain, CircuitState.HALF_OPEN)
                state = CircuitState.HALF_OPEN

        return CircuitStatus(
            state=state,
            failures=failures,
            last_failure=last_failure,
            opened_at=opened_at,
        )

    async def is_allowed(self, domain: str) -> bool:
        """Check if requests to domain are allowed."""
        status = await self.get_status(domain)
        return status.state != CircuitState.OPEN

    async def record_success(self, domain: str) -> None:
        """Record a successful request, potentially closing the circuit."""
        status = await self.get_status(domain)
        if status.state == CircuitState.HALF_OPEN:
            await self.redis.delete(self._key(domain))

    async def record_failure(self, domain: str) -> CircuitState:
        """Record a failed request, potentially opening the circuit."""
        key = self._key(domain)
        now = datetime.utcnow().isoformat()

        # Increment failures
        failures = await self.redis.hincrby(key, "failures", 1)
        await self.redis.hset(key, "last_failure", now)
        await self.redis.expire(key, 3600)  # 1 hour TTL

        # Check threshold
        if failures >= self.threshold:
            await self._set_state(domain, CircuitState.OPEN)
            await self.redis.hset(key, "opened_at", now)
            return CircuitState.OPEN

        return CircuitState.CLOSED

    async def _set_state(self, domain: str, state: CircuitState) -> None:
        """Update circuit state."""
        await self.redis.hset(self._key(domain), "state", state.value)

    async def reset(self, domain: str) -> None:
        """Manually reset circuit breaker for a domain."""
        await self.redis.delete(self._key(domain))
```

### 8.9 Rate Limiter

```python
# app/services/rate_limiter.py

import asyncio
import time
from dataclasses import dataclass

import redis.asyncio as redis


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after: float | None = None


class TokenBucketRateLimiter:
    """Redis-backed token bucket rate limiter for per-domain throttling."""

    KEY_PREFIX = "crawl4r:rate"

    def __init__(
        self,
        redis_client: redis.Redis,
        default_rate: float = 1.0,  # requests per second
        burst_size: int = 5,
    ):
        self.redis = redis_client
        self.default_rate = default_rate
        self.burst_size = burst_size

    def _key(self, domain: str) -> str:
        return f"{self.KEY_PREFIX}:{domain}"

    async def acquire(
        self,
        domain: str,
        rate: float | None = None,
        tokens: int = 1,
    ) -> RateLimitResult:
        """
        Attempt to acquire tokens for a domain.

        Args:
            domain: The domain to rate limit
            rate: Custom rate (requests/second), defaults to default_rate
            tokens: Number of tokens to acquire

        Returns:
            RateLimitResult with allowed status and remaining tokens
        """
        rate = rate or self.default_rate
        key = self._key(domain)
        now = time.time()

        # Lua script for atomic token bucket
        lua_script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local burst = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local tokens_needed = tonumber(ARGV[4])

        -- Get current bucket state
        local bucket = redis.call('HMGET', key, 'tokens', 'last_update')
        local current_tokens = tonumber(bucket[1]) or burst
        local last_update = tonumber(bucket[2]) or now

        -- Calculate tokens to add based on time passed
        local time_passed = now - last_update
        local tokens_to_add = time_passed * rate
        current_tokens = math.min(burst, current_tokens + tokens_to_add)

        -- Check if we can acquire tokens
        if current_tokens >= tokens_needed then
            current_tokens = current_tokens - tokens_needed
            redis.call('HMSET', key, 'tokens', current_tokens, 'last_update', now)
            redis.call('EXPIRE', key, 60)
            return {1, current_tokens, 0}
        else
            -- Calculate wait time
            local tokens_short = tokens_needed - current_tokens
            local wait_time = tokens_short / rate
            redis.call('HMSET', key, 'tokens', current_tokens, 'last_update', now)
            redis.call('EXPIRE', key, 60)
            return {0, current_tokens, wait_time}
        end
        """

        result = await self.redis.eval(
            lua_script,
            1,
            key,
            str(rate),
            str(self.burst_size),
            str(now),
            str(tokens),
        )

        allowed, remaining, retry_after = result
        return RateLimitResult(
            allowed=bool(allowed),
            remaining=int(remaining),
            retry_after=float(retry_after) if retry_after > 0 else None,
        )

    async def get_rate(self, domain: str) -> float:
        """Get configured rate for a domain from DB, with caching."""
        # In practice, this would check domain_settings table
        # Here we return the default rate
        return self.default_rate

    async def wait_and_acquire(
        self,
        domain: str,
        rate: float | None = None,
        timeout: float = 30.0,
    ) -> bool:
        """Wait until rate limit allows, or timeout."""
        start = time.time()
        while time.time() - start < timeout:
            result = await self.acquire(domain, rate)
            if result.allowed:
                return True
            if result.retry_after:
                await asyncio.sleep(min(result.retry_after, timeout - (time.time() - start)))
        return False
```

### 8.10 Robots.txt Checker

```python
# app/services/robots_checker.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from app.services.cache import RedisCache


@dataclass
class RobotsResult:
    allowed: bool
    crawl_delay: float | None = None
    reason: str | None = None


class RobotsTxtChecker:
    """Checks robots.txt compliance with caching."""

    USER_AGENT = "crawl4r/1.0"
    CACHE_TTL = 86400 * 7  # 7 days

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        cache: RedisCache,
    ):
        self.http = http_client
        self.cache = cache

    async def is_allowed(self, url: str, respect_robots: bool = True) -> RobotsResult:
        """
        Check if URL can be crawled according to robots.txt.

        Args:
            url: The URL to check
            respect_robots: If False, always return allowed

        Returns:
            RobotsResult with allowed status and crawl delay
        """
        if not respect_robots:
            return RobotsResult(allowed=True)

        parsed = urlparse(url)
        domain = parsed.netloc
        robots_url = f"{parsed.scheme}://{domain}/robots.txt"

        # Try to get cached robots.txt
        cache_key = f"crawl4r:robots:{domain}"
        cached = await self.cache.get(cache_key)

        if cached is None:
            # Fetch robots.txt
            try:
                response = await self.http.get(
                    robots_url,
                    timeout=10.0,
                    follow_redirects=True,
                )
                if response.status_code == 200:
                    robots_content = response.text
                else:
                    # No robots.txt or error = allow all
                    robots_content = ""
            except Exception:
                # Network error = allow (fail open)
                robots_content = ""

            # Cache the content
            await self.cache.set(cache_key, robots_content, ttl_seconds=self.CACHE_TTL)
            cached = robots_content

        # Parse robots.txt
        parser = RobotFileParser()
        parser.parse(cached.split("\n") if cached else [])

        # Check if allowed
        path = parsed.path or "/"
        allowed = parser.can_fetch(self.USER_AGENT, path)

        # Get crawl delay
        crawl_delay = parser.crawl_delay(self.USER_AGENT)

        return RobotsResult(
            allowed=allowed,
            crawl_delay=crawl_delay,
            reason=None if allowed else "Blocked by robots.txt",
        )
```

### 8.11 Proxy Selector

```python
# app/services/proxy_selector.py

import random
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis

from app.core.models import ProxyConfig


@dataclass
class SelectedProxy:
    url: str
    username: str | None = None
    password: str | None = None


class ProxySelector:
    """Selects proxies using various rotation strategies."""

    KEY_PREFIX = "crawl4r:proxy"

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._configs: dict[str, ProxyConfig] = {}

    def register_config(self, config: ProxyConfig) -> None:
        """Register a proxy configuration."""
        self._configs[str(config.id)] = config

    async def select(
        self,
        config_id: str | None = None,
        domain: str | None = None,
    ) -> SelectedProxy | None:
        """
        Select a proxy based on configuration strategy.

        Args:
            config_id: Specific proxy config to use
            domain: Domain being crawled (for domain-specific rules)

        Returns:
            SelectedProxy or None if no proxy configured
        """
        if not config_id:
            return None

        config = self._configs.get(config_id)
        if not config or not config.is_active or not config.servers:
            return None

        if config.rotation_strategy == "round_robin":
            proxy = await self._round_robin_select(config)
        elif config.rotation_strategy == "least_used":
            proxy = await self._least_used_select(config)
        elif config.rotation_strategy == "random":
            proxy = self._random_select(config)
        else:
            proxy = self._random_select(config)

        return SelectedProxy(
            url=proxy.get("url"),
            username=proxy.get("username"),
            password=proxy.get("password"),
        )

    async def _round_robin_select(self, config: ProxyConfig) -> dict[str, str]:
        """Round-robin proxy selection."""
        key = f"{self.KEY_PREFIX}:rr:{config.id}"
        idx = await self.redis.incr(key)
        await self.redis.expire(key, 3600)
        return config.servers[(idx - 1) % len(config.servers)]

    async def _least_used_select(self, config: ProxyConfig) -> dict[str, str]:
        """Select least-used proxy."""
        key = f"{self.KEY_PREFIX}:usage:{config.id}"

        # Get usage counts
        usage = await self.redis.hgetall(key)

        # Find least used
        min_count = float("inf")
        selected = config.servers[0]

        for server in config.servers:
            url = server["url"]
            count = int(usage.get(url.encode(), 0))
            if count < min_count:
                min_count = count
                selected = server

        # Increment usage
        await self.redis.hincrby(key, selected["url"], 1)
        await self.redis.expire(key, 3600)

        return selected

    def _random_select(self, config: ProxyConfig) -> dict[str, str]:
        """Random proxy selection."""
        return random.choice(config.servers)

    async def report_failure(self, config_id: str, proxy_url: str) -> None:
        """Report a proxy failure for health tracking."""
        key = f"{self.KEY_PREFIX}:failures:{config_id}"
        await self.redis.hincrby(key, proxy_url, 1)
        await self.redis.expire(key, 3600)

    async def report_success(self, config_id: str, proxy_url: str) -> None:
        """Report a proxy success, resetting failure count."""
        key = f"{self.KEY_PREFIX}:failures:{config_id}"
        await self.redis.hset(key, proxy_url, 0)
```

### 8.12 TEI Embedder Client

```python
# app/services/embedder.py

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.abstractions import Embedder
from app.services.embedding_validator import EmbeddingValidator


@dataclass
class EmbeddingResult:
    embedding: list[float]
    model: str
    token_count: int | None = None


class TEIEmbedder(Embedder):
    """Text Embeddings Inference client implementation."""

    def __init__(
        self,
        base_url: str,
        model_name: str = "qwen3-0.6b-embedding",
        dimension: int = 1024,
        batch_size: int = 32,
        timeout_seconds: int = 60,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._dimension = dimension
        self.batch_size = batch_size
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.validator = EmbeddingValidator(expected_dimension=dimension)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if TEI service is healthy."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts with batching and retry."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch with retry logic."""
        client = await self._get_client()
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    "/embed",
                    json={"inputs": texts},
                )
                response.raise_for_status()
                embeddings = response.json()

                # Validate embeddings
                for i, emb in enumerate(embeddings):
                    result = self.validator.validate(emb)
                    if not result.is_valid:
                        raise ValueError(f"Invalid embedding at index {i}: {result.reason}")

                return embeddings

            except httpx.TimeoutException as e:
                last_error = e
                await asyncio.sleep(2 ** attempt)
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_error = e
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
            except Exception as e:
                last_error = e
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"Failed to embed after {self.max_retries} attempts: {last_error}")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension
```

### 8.13 Webhook Sender with HMAC

```python
# app/services/webhook_sender.py

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx


@dataclass
class WebhookDeliveryResult:
    success: bool
    status_code: int | None = None
    response_body: str | None = None
    error: str | None = None


class WebhookSender:
    """Sends webhook notifications with HMAC signing."""

    def __init__(
        self,
        timeout_seconds: int = 30,
        max_retries: int = 5,
    ):
        self.timeout = timeout_seconds
        self.max_retries = max_retries

    def _sign_payload(self, payload: dict[str, Any], secret: str) -> str:
        """Generate HMAC-SHA256 signature for payload."""
        payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        signature = hmac.new(
            secret.encode(),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={signature}"

    async def send(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        secret: str | None = None,
    ) -> WebhookDeliveryResult:
        """
        Send a webhook notification.

        Args:
            url: Webhook endpoint URL
            payload: JSON payload to send
            headers: Additional headers
            secret: HMAC secret for signing

        Returns:
            WebhookDeliveryResult with delivery status
        """
        request_headers = {
            "Content-Type": "application/json",
            "User-Agent": "crawl4r/1.0",
            "X-Webhook-Timestamp": datetime.utcnow().isoformat(),
        }

        if headers:
            request_headers.update(headers)

        # Add HMAC signature if secret provided
        if secret:
            signature = self._sign_payload(payload, secret)
            request_headers["X-Webhook-Signature"] = signature

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers=request_headers,
                )

                return WebhookDeliveryResult(
                    success=200 <= response.status_code < 300,
                    status_code=response.status_code,
                    response_body=response.text[:1000] if response.text else None,
                )

        except httpx.TimeoutException:
            return WebhookDeliveryResult(
                success=False,
                error="Request timed out",
            )
        except httpx.RequestError as e:
            return WebhookDeliveryResult(
                success=False,
                error=str(e),
            )

    def verify_signature(
        self,
        payload: dict[str, Any],
        signature: str,
        secret: str,
    ) -> bool:
        """Verify an incoming webhook signature."""
        expected = self._sign_payload(payload, secret)
        return hmac.compare_digest(expected, signature)
```

---

## 9. Pipeline Flows

### 9.1 Ingest Pipeline

```
API Request
     │
     ▼
┌─────────────────┐
│ URL Validate    │  ← SSRF check, scheme check, normalize
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dedup Check     │  ← Content hash exists in Postgres?
│ (Postgres)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Circuit Breaker │  ← Domain blocked/open circuit?
│ Check (Redis)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Robots.txt      │  ← Cached check, respect robots.txt
│ Check           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Queue Job       │  ← Priority queue (high/normal/low)
│ (Redis/ARQ)     │
└────────┬────────┘
         │
     [Worker]
         │
         ▼
┌─────────────────┐
│ Acquire Lock    │  ← Distributed lock on URL
│ (Redis)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cache Check     │  ← Fresh crawl result exists?
│ (Redis)         │
└────────┬────────┘
         │ Miss
         ▼
┌─────────────────┐
│ Rate Limiter    │  ← Token bucket per domain
│ (Redis)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Select Proxy    │  ← Round-robin if configured
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Crawl4AI        │  ← Crawl + extract + chunk
│ (External)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Content Process │  ← Validate, sanitize, metadata
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Language Check  │  ← Reject non-English
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Batch Embed     │  ← All chunks in batches of 32
│ (TEI)           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Validate Embeds │  ← Check dimension, no NaN, etc.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Postgres Write  │  ← Document + chunks (transaction)
│ (Source of      │
│  Truth)         │
└────────┬────────┘
         │ Success
         ▼
┌─────────────────┐
│ Qdrant Upsert   │  ← With retry on failure
└────────┬────────┘
         │ Failure?
         ▼
┌─────────────────┐
│ Queue Retry     │  ← Qdrant retry queue
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cache Populate  │  ← Crawl result, embeddings
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Webhook Notify  │  ← If configured
└─────────────────┘
```

### 9.2 Retrieval Pipeline

```
Query Request
     │
     ▼
┌─────────────────┐
│ Auth Check      │  ← Validate API key, check rate limit
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Preprocess│  ← Normalize, generate cache key
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Result Cache    │  ← Full response cached?
│ Check           │─────► Hit ──► Return
└────────┬────────┘
         │ Miss
         ▼
┌─────────────────┐
│ Query Embed     │  ← Check embedding cache first
│ Cache Check     │─────► Hit ──► Use cached
└────────┬────────┘
         │ Miss
         ▼
┌───────────────────────────────────────────────┐
│              PARALLEL EXECUTION                │
│  ┌──────────────┐      ┌────────────────────┐ │
│  │ Embed Query  │      │ Postgres FTS       │ │
│  │ (TEI)        │      │ (with filters)     │ │
│  └──────┬───────┘      └─────────┬──────────┘ │
│         │                        │            │
│         ▼                        │            │
│  ┌──────────────┐                │            │
│  │ Qdrant Search│                │            │
│  │ (with filters)                │            │
│  └──────┬───────┘                │            │
│         │                        │            │
└─────────┴────────────────────────┴────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │ Deduplicate     │  ← By chunk_id
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ RRF Fusion      │  ← Combine with weights
          │ (k=60)          │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Score Filter    │  ← Remove below min_score
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Chunk Expansion │  ← Optional: fetch ±1 chunks
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Reranker        │  ← Optional: top N rerank
          │ (if requested)  │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Highlight Gen   │  ← Mark matching terms
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Cache Result    │  ← Cache with TTL
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Return Response │
          └─────────────────┘
```

### 9.3 Error Handling Flow

```
Crawl Attempt
     │
     ▼
┌─────────────────┐
│ Execute Crawl   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
 Success   Error
    │         │
    ▼         ▼
┌────────┐  ┌─────────────────┐
│Process │  │ Classify Error  │
│Document│  └────────┬────────┘
└────────┘           │
              ┌──────┴──────┐
              │             │
         Transient     Permanent
         (5xx,timeout) (4xx except 429)
              │             │
              ▼             ▼
    ┌─────────────────┐  ┌─────────────────┐
    │ Retry Count < 3?│  │ Mark Failed     │
    └────────┬────────┘  │ → DLQ           │
             │           └────────┬────────┘
        ┌────┴────┐               │
        │         │               ▼
       Yes       No         ┌─────────────────┐
        │         │         │ Send Webhook    │
        ▼         ▼         │ (if configured) │
┌────────────┐ ┌──────────┐ └─────────────────┘
│ Exp Backoff│ │ Check    │
│ Retry      │ │ Circuit  │
│ (1s,2s,4s) │ │ Breaker  │
└────────────┘ └────┬─────┘
                    │
              ┌─────┴─────┐
              │           │
         Threshold   Under
         Exceeded    Threshold
              │           │
              ▼           ▼
    ┌──────────────┐ ┌──────────────┐
    │ Open Circuit │ │ DLQ + Alert  │
    │ for Domain   │ └──────────────┘
    │ (5 min)      │
    └──────────────┘


Rate Limit (429) Handling:
     │
     ▼
┌─────────────────┐
│ Extract Retry-  │
│ After header    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Back off for    │
│ domain (Redis)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Requeue with    │
│ delay           │
└─────────────────┘
```

---

## 10. Worker Tasks

### 10.1 Task Definitions

| Task | Purpose | Queue | Priority |
|------|---------|-------|----------|
| `crawl_url` | Single URL crawl | normal | Configurable |
| `crawl_deep` | Orchestrate deep crawl | normal | normal |
| `crawl_frontier_batch` | Process deep crawl frontier batch | normal | normal |
| `discover_urls` | URL discovery | low | low |
| `process_document` | Chunk + embed + store | normal | normal |
| `reindex_document` | Re-embed existing doc | low | low |
| `deliver_webhook` | Send webhook notification | high | high |
| `retry_webhook` | Retry failed webhook | normal | normal |
| `sync_qdrant` | Retry failed Qdrant upserts | normal | normal |
| `scheduled_crawl` | Cron-triggered source crawl | low | low |
| `cleanup_expired` | Clean old data | low | low |
| `reconcile_storage` | Postgres/Qdrant sync check | low | low |
| `reset_circuit_breakers` | Check half-open circuits | low | low |

### 10.2 Scheduled Jobs

| Job | Schedule | Purpose |
|-----|----------|---------|
| Cleanup expired cache | Hourly | Remove expired Redis keys |
| Cleanup old jobs | Daily | Delete job records > 30 days |
| Cleanup soft deletes | Daily | Hard delete docs with deleted_at > 30 days |
| Reconcile storage | Daily | Check Postgres/Qdrant consistency |
| Reset circuit breakers | Every 5 min | Check and reset half-open circuits |
| Retry DLQ | Every 5 min | Auto-retry failed jobs with exponential backoff |
| Scheduled crawls | Per source cron | Trigger crawl sources |

### 10.3 Scheduler Implementation

```python
# app/worker/scheduler.py

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine

from arq import cron
from arq.connections import ArqRedis
from croniter import croniter

from app.config import settings


class Scheduler:
    """ARQ-based scheduler for cron jobs and scheduled crawls."""

    def __init__(self, redis: ArqRedis):
        self.redis = redis

    @staticmethod
    def get_cron_jobs() -> list[cron]:
        """Define all scheduled jobs for ARQ worker."""
        return [
            # Cleanup expired cache - hourly at minute 0
            cron(
                cleanup_expired_cache,
                hour={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
                minute=0,
                run_at_startup=False,
            ),
            # Cleanup old jobs - daily at 3 AM
            cron(
                cleanup_old_jobs,
                hour=3,
                minute=0,
                run_at_startup=False,
            ),
            # Cleanup soft deletes - daily at 4 AM
            cron(
                cleanup_soft_deletes,
                hour=4,
                minute=0,
                run_at_startup=False,
            ),
            # Reconcile storage - daily at 5 AM
            cron(
                reconcile_storage,
                hour=5,
                minute=0,
                run_at_startup=False,
            ),
            # Reset circuit breakers - every 5 minutes
            cron(
                reset_circuit_breakers,
                minute={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55},
                run_at_startup=True,
            ),
            # Retry DLQ - every 5 minutes
            cron(
                retry_dlq_jobs,
                minute={2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57},
                run_at_startup=False,
            ),
            # Check scheduled crawl sources - every minute
            cron(
                check_scheduled_sources,
                minute=set(range(60)),
                run_at_startup=True,
            ),
        ]


async def cleanup_expired_cache(ctx: dict[str, Any]) -> int:
    """Remove expired Redis cache keys."""
    redis: ArqRedis = ctx["redis"]
    # Redis handles TTL expiration automatically, but we can scan for orphaned keys
    deleted = 0
    async for key in redis.scan_iter("crawl4r:cache:*"):
        if await redis.ttl(key) == -1:  # No TTL set
            await redis.delete(key)
            deleted += 1
    return deleted


async def cleanup_old_jobs(ctx: dict[str, Any]) -> dict[str, int]:
    """Delete job records older than 30 days."""
    from app.db.session import get_async_session
    from sqlalchemy import delete, and_
    from app.db.models import CrawlJob, DeepCrawlJob, DiscoveryJob

    cutoff = datetime.utcnow() - timedelta(days=30)
    counts = {"crawl_jobs": 0, "deep_crawl_jobs": 0, "discovery_jobs": 0}

    async with get_async_session() as session:
        # Delete old crawl jobs
        result = await session.execute(
            delete(CrawlJob).where(
                and_(
                    CrawlJob.created_at < cutoff,
                    CrawlJob.status.in_(["completed", "failed", "cancelled"]),
                )
            )
        )
        counts["crawl_jobs"] = result.rowcount

        # Delete old deep crawl jobs
        result = await session.execute(
            delete(DeepCrawlJob).where(
                and_(
                    DeepCrawlJob.created_at < cutoff,
                    DeepCrawlJob.status.in_(["completed", "failed", "cancelled"]),
                )
            )
        )
        counts["deep_crawl_jobs"] = result.rowcount

        # Delete old discovery jobs
        result = await session.execute(
            delete(DiscoveryJob).where(
                and_(
                    DiscoveryJob.created_at < cutoff,
                    DiscoveryJob.status.in_(["completed", "failed"]),
                )
            )
        )
        counts["discovery_jobs"] = result.rowcount

        await session.commit()

    return counts


async def cleanup_soft_deletes(ctx: dict[str, Any]) -> int:
    """Hard delete documents that were soft deleted > 30 days ago."""
    from app.db.session import get_async_session
    from sqlalchemy import delete
    from app.db.models import Document

    cutoff = datetime.utcnow() - timedelta(days=30)

    async with get_async_session() as session:
        result = await session.execute(
            delete(Document).where(
                Document.deleted_at < cutoff
            )
        )
        await session.commit()
        return result.rowcount


async def reconcile_storage(ctx: dict[str, Any]) -> dict[str, Any]:
    """Check Postgres/Qdrant consistency and report discrepancies."""
    from app.db.session import get_async_session
    from app.stores.qdrant import QdrantStore
    from sqlalchemy import select, func
    from app.db.models import Chunk

    async with get_async_session() as session:
        # Get chunk count from Postgres
        result = await session.execute(select(func.count(Chunk.id)))
        pg_count = result.scalar()

    # Get point count from Qdrant
    qdrant = QdrantStore(ctx["settings"].qdrant)
    qdrant_count = await qdrant.get_collection_count()

    discrepancy = abs(pg_count - qdrant_count)

    return {
        "postgres_chunks": pg_count,
        "qdrant_points": qdrant_count,
        "discrepancy": discrepancy,
        "status": "ok" if discrepancy < 100 else "warning",
    }


async def reset_circuit_breakers(ctx: dict[str, Any]) -> int:
    """Check and reset half-open circuit breakers."""
    from app.services.circuit_breaker import CircuitBreaker, CircuitState

    redis: ArqRedis = ctx["redis"]
    cb = CircuitBreaker(redis)
    reset_count = 0

    async for key in redis.scan_iter("crawl4r:circuit:*"):
        domain = key.decode().split(":")[-1]
        status = await cb.get_status(domain)

        if status.state == CircuitState.HALF_OPEN:
            # Already transitioned by get_status
            reset_count += 1

    return reset_count


async def retry_dlq_jobs(ctx: dict[str, Any]) -> int:
    """Auto-retry failed jobs from DLQ with exponential backoff."""
    redis: ArqRedis = ctx["redis"]
    retried = 0

    # Get jobs from DLQ
    dlq_key = "crawl4r:queue:dlq"
    jobs = await redis.lrange(dlq_key, 0, 100)

    for job_data in jobs:
        import json
        job = json.loads(job_data)

        # Check if enough time has passed (exponential backoff)
        retry_count = job.get("retry_count", 0)
        last_attempt = datetime.fromisoformat(job.get("last_attempt", "1970-01-01"))
        backoff_seconds = min(300, 2 ** retry_count)  # Max 5 minutes

        if datetime.utcnow() > last_attempt + timedelta(seconds=backoff_seconds):
            # Re-queue the job
            job["retry_count"] = retry_count + 1
            job["last_attempt"] = datetime.utcnow().isoformat()

            await redis.rpush("crawl4r:queue:normal", json.dumps(job))
            await redis.lrem(dlq_key, 1, job_data)
            retried += 1

    return retried


async def check_scheduled_sources(ctx: dict[str, Any]) -> int:
    """Check and trigger scheduled crawl sources."""
    from app.db.session import get_async_session
    from sqlalchemy import select, update
    from app.db.models import CrawlSource

    triggered = 0
    now = datetime.utcnow()

    async with get_async_session() as session:
        # Get active sources with cron schedules
        result = await session.execute(
            select(CrawlSource).where(
                CrawlSource.is_active == True,
                CrawlSource.schedule_cron.isnot(None),
            )
        )
        sources = result.scalars().all()

        for source in sources:
            # Check if it's time to run
            cron = croniter(source.schedule_cron, source.last_crawled_at or now - timedelta(days=1))
            next_run = cron.get_next(datetime)

            if next_run <= now:
                # Enqueue crawl job
                await ctx["redis"].rpush(
                    "crawl4r:queue:low",
                    json.dumps({
                        "task": "scheduled_crawl",
                        "source_id": str(source.id),
                        "url_pattern": source.url_pattern,
                        "config_id": str(source.crawl_config_id),
                        "collection_id": str(source.collection_id) if source.collection_id else None,
                    }),
                )

                # Update last_crawled_at
                await session.execute(
                    update(CrawlSource)
                    .where(CrawlSource.id == source.id)
                    .values(last_crawled_at=now)
                )
                triggered += 1

        await session.commit()

    return triggered
```

---

## 11. Configuration

### 11.1 Settings

```python
# app/config.py

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration. Env vars: DATABASE__*"""
    model_config = SettingsConfigDict(env_prefix="DATABASE__")

    postgres_url: str = Field(default="postgresql+asyncpg://user:pass@localhost:5432/crawl4r")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)


class RedisSettings(BaseSettings):
    """Redis configuration. Env vars: REDIS__*"""
    model_config = SettingsConfigDict(env_prefix="REDIS__")

    url: str = Field(default="redis://localhost:6379/0")
    pool_size: int = Field(default=10)


class QdrantSettings(BaseSettings):
    """Qdrant configuration. Env vars: QDRANT__*"""
    model_config = SettingsConfigDict(env_prefix="QDRANT__")

    url: str = Field(default="http://100.74.16.82:52002")
    collection: str = Field(default="crawl4r")
    api_key: str | None = Field(default=None)
    batch_size: int = Field(default=100)              # Upsert batch size


class TEISettings(BaseSettings):
    """TEI embedding service configuration. Env vars: TEI__*"""
    model_config = SettingsConfigDict(env_prefix="TEI__")

    url: str = Field(default="http://100.74.16.82:52000")
    batch_size: int = Field(default=32)
    timeout_s: int = Field(default=60)
    embedding_dimension: int = Field(default=1024)
    embedding_model: str = Field(default="qwen3-0.6b-embedding")


class Crawl4AISettings(BaseSettings):
    """Crawl4AI service configuration. Env vars: CRAWL4AI__*"""
    model_config = SettingsConfigDict(env_prefix="CRAWL4AI__")

    url: str = Field(default="http://localhost:52001")
    timeout_s: int = Field(default=60)


class CacheSettings(BaseSettings):
    """Cache TTL configuration. Env vars: CACHE__*"""
    model_config = SettingsConfigDict(env_prefix="CACHE__")

    crawl_ttl_s: int = Field(default=86400)           # 24 hours
    embed_ttl_s: int = Field(default=604800)          # 7 days
    query_ttl_s: int = Field(default=3600)            # 1 hour
    query_embed_ttl_s: int = Field(default=86400)     # 24 hours


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration. Env vars: RATE_LIMIT__*"""
    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT__")

    default_rps: float = Field(default=1.0)
    api_rpm: int = Field(default=60)


class CircuitBreakerSettings(BaseSettings):
    """Circuit breaker configuration. Env vars: CIRCUIT_BREAKER__*"""
    model_config = SettingsConfigDict(env_prefix="CIRCUIT_BREAKER__")

    threshold: int = Field(default=5)
    timeout_s: int = Field(default=300)


class JobSettings(BaseSettings):
    """Job processing configuration. Env vars: JOB__*"""
    model_config = SettingsConfigDict(env_prefix="JOB__")

    max_retries: int = Field(default=3)
    retry_delays_s: list[int] = Field(default=[1, 2, 4])
    timeout_s: int = Field(default=300)
    result_ttl_s: int = Field(default=86400)


class FeatureFlags(BaseSettings):
    """Feature flags. Env vars: FEATURES__*"""
    model_config = SettingsConfigDict(env_prefix="FEATURES__")

    enable_reranker: bool = Field(default=True)
    enable_deep_crawl: bool = Field(default=True)
    enable_webhooks: bool = Field(default=True)
    enable_proxy_rotation: bool = Field(default=False)
    enable_url_discovery: bool = Field(default=True)
    max_concurrent_deep_crawls: int = Field(default=5)
    max_pages_per_deep_crawl: int = Field(default=1000)


class Settings(BaseSettings):
    # App
    app_name: str = Field(default="crawl4r")
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=53001)

    # Components
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    tei: TEISettings = Field(default_factory=TEISettings)
    crawl4ai: Crawl4AISettings = Field(default_factory=Crawl4AISettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    circuit_breaker: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)
    jobs: JobSettings = Field(default_factory=JobSettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


settings = Settings()
```

### 11.2 Environment Variables

```bash
# .env.example

# === App ===
APP_ENV=development
DEBUG=false

# === Server ===
HOST=0.0.0.0
PORT=53001

# === Database ===
DATABASE__POSTGRES_URL=postgresql+asyncpg://crawl4r:password@postgres:5432/crawl4r
DATABASE__POOL_SIZE=10
DATABASE__MAX_OVERFLOW=20

# === Redis ===
REDIS__URL=redis://redis:6379/0
REDIS__POOL_SIZE=10

# === Qdrant ===
QDRANT__URL=http://100.74.16.82:52002
QDRANT__COLLECTION=crawl4r
QDRANT__BATCH_SIZE=100

# === TEI ===
TEI__URL=http://100.74.16.82:52000
TEI__BATCH_SIZE=32
TEI__TIMEOUT_S=60
TEI__EMBEDDING_DIMENSION=1024
TEI__EMBEDDING_MODEL=qwen3-0.6b-embedding

# === Crawl4AI ===
CRAWL4AI__URL=http://crawl4ai:11235
CRAWL4AI__TIMEOUT_S=60

# === Cache TTLs ===
CACHE__CRAWL_TTL_S=86400
CACHE__EMBED_TTL_S=604800
CACHE__QUERY_TTL_S=3600
CACHE__QUERY_EMBED_TTL_S=86400

# === Rate Limiting ===
RATE_LIMIT__DEFAULT_RPS=1.0
RATE_LIMIT__API_RPM=60

# === Circuit Breaker ===
CIRCUIT_BREAKER__THRESHOLD=5
CIRCUIT_BREAKER__TIMEOUT_S=300

# === Job Processing ===
JOB__MAX_RETRIES=3
JOB__TIMEOUT_S=300
JOB__RESULT_TTL_S=86400

# === Feature Flags ===
FEATURES__ENABLE_RERANKER=true
FEATURES__ENABLE_DEEP_CRAWL=true
FEATURES__ENABLE_WEBHOOKS=true
FEATURES__ENABLE_PROXY_ROTATION=false
FEATURES__ENABLE_URL_DISCOVERY=true
FEATURES__MAX_CONCURRENT_DEEP_CRAWLS=5
FEATURES__MAX_PAGES_PER_DEEP_CRAWL=1000
```

---

## 12. Project Structure

```
crawl4r/
├── docker-compose.yaml
├── Dockerfile
├── pyproject.toml
├── .env.example
├── .env                           # Git-ignored
├── alembic.ini
├── alembic/
│   └── versions/
│
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app factory
│   ├── config.py                  # Pydantic Settings
│   ├── dependencies.py            # DI container
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── middleware.py          # Auth, logging, error handling
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── crawl.py
│   │   │   ├── deep_crawl.py
│   │   │   ├── discover.py
│   │   │   ├── search.py
│   │   │   ├── documents.py
│   │   │   ├── collections.py
│   │   │   ├── tags.py
│   │   │   ├── sources.py
│   │   │   ├── configs.py
│   │   │   ├── proxies.py
│   │   │   ├── domains.py
│   │   │   ├── cache.py
│   │   │   ├── webhooks.py
│   │   │   ├── api_keys.py
│   │   │   ├── jobs.py
│   │   │   ├── media.py           # screenshot, pdf, execute-js
│   │   │   └── health.py
│   │   └── websocket.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── abstractions.py        # ABCs
│   │   ├── models.py              # Pydantic models
│   │   ├── rrf.py                 # RRF fusion
│   │   ├── reranker.py            # Cross-encoder reranker
│   │   └── exceptions.py          # Custom exceptions
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── session.py             # SQLAlchemy async session
│   │   ├── models.py              # SQLAlchemy ORM models
│   │   └── repositories/
│   │       ├── __init__.py
│   │       ├── documents.py
│   │       ├── chunks.py
│   │       ├── jobs.py
│   │       ├── collections.py
│   │       └── ...
│   │
│   ├── stores/
│   │   ├── __init__.py
│   │   ├── qdrant.py              # QdrantStore(VectorStore)
│   │   └── postgres.py            # PostgresStore(DocumentStore)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── crawler.py             # Crawl4AIClient(Crawler)
│   │   ├── embedder.py            # TEIEmbedder(Embedder)
│   │   ├── cache.py               # RedisCache(Cache)
│   │   ├── url_validator.py
│   │   ├── content_processor.py
│   │   ├── embedding_validator.py
│   │   ├── query_preprocessor.py
│   │   ├── circuit_breaker.py
│   │   ├── rate_limiter.py
│   │   ├── webhook_sender.py
│   │   └── search.py              # Search orchestration
│   │
│   └── worker/
│       ├── __init__.py
│       ├── main.py                # ARQ worker entrypoint
│       ├── tasks/
│       │   ├── __init__.py
│       │   ├── crawl.py
│       │   ├── deep_crawl.py
│       │   ├── discover.py
│       │   ├── process.py
│       │   ├── webhooks.py
│       │   ├── sync.py
│       │   └── cleanup.py
│       └── scheduler.py           # Scheduled jobs
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_url_validator.py
│   │   ├── test_content_processor.py
│   │   ├── test_rrf.py
│   │   └── ...
│   ├── integration/
│   │   ├── test_crawl_pipeline.py
│   │   ├── test_search_pipeline.py
│   │   └── ...
│   └── e2e/
│       └── test_api.py
│
├── docs/
│   └── plans/
│       └── 2025-01-11-rag-pipeline-design.md
│
└── .docs/
    ├── sessions/
    ├── deployment-log.md
    └── services-ports.md
```

---

## 13. Security Considerations

### 13.1 SSRF Prevention

- Block private IP ranges (127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
- Block cloud metadata endpoints (169.254.169.254, metadata.google.internal)
- Allow only http/https schemes
- Validate URLs before queuing

### 13.2 Input Validation

- Max URL length: 2048 characters
- Max request body: 10MB
- Max query length: 1000 characters
- Max chunk content: 10KB
- Sanitize HTML before storage

### 13.3 Authentication

- API key authentication (SHA256 hashed storage)
- Scopes: read, write, admin
- Rate limiting per API key
- Key expiration support

### 13.4 Secrets Management

- All secrets via environment variables
- Never log secrets
- Redact sensitive fields in logs
- Rotate API keys periodically

---

## 14. Observability

### 14.1 Structured Logging

```python
import structlog

logger = structlog.get_logger()

# Request logging
logger.info(
    "request_received",
    request_id=request_id,
    method=request.method,
    path=request.url.path,
    client_ip=request.client.host,
)

# Error logging
logger.error(
    "crawl_failed",
    request_id=request_id,
    url=url,
    error=str(e),
    error_type=type(e).__name__,
)
```

### 14.2 Metrics (Internal)

Track in Redis, expose via `/stats`:

**Counters:**
- `requests_total`
- `crawls_total`, `crawls_succeeded`, `crawls_failed`
- `searches_total`
- `cache_hits`, `cache_misses`
- `webhooks_delivered`, `webhooks_failed`

**Gauges:**
- `queue_depth_high`, `queue_depth_normal`, `queue_depth_low`, `queue_depth_dlq`
- `active_deep_crawls`
- `documents_total`, `chunks_total`

**Histograms:**
- `crawl_duration_ms`
- `search_duration_ms`
- `embedding_duration_ms`

### 14.3 Health Checks

```python
# GET /health - Liveness
{"status": "ok"}

# GET /health/ready - Readiness
{
    "status": "ok",
    "checks": {
        "postgres": "ok",
        "redis": "ok",
        "qdrant": "ok",
        "tei": "ok",
        "crawl4ai": "ok"
    }
}
```

---

## 15. Dependencies

### 15.1 Python Dependencies

```toml
# pyproject.toml [project.dependencies]

[project]
name = "crawl4r"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # Web Framework
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "python-multipart>=0.0.6",

    # Database
    "sqlalchemy[asyncio]>=2.0.25",
    "asyncpg>=0.29.0",
    "alembic>=1.13.0",

    # Redis
    "redis>=5.0.0",
    "arq>=0.25.0",

    # Qdrant
    "qdrant-client>=1.7.0",

    # HTTP Client
    "httpx>=0.26.0",

    # Data Validation
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",

    # Content Processing
    "beautifulsoup4>=4.12.0",
    "charset-normalizer>=3.3.0",
    "python-dateutil>=2.8.0",

    # URL Handling
    "tldextract>=5.1.0",

    # Scheduling
    "croniter>=2.0.0",

    # Reranker (optional, for local inference)
    "transformers>=4.36.0",
    "torch>=2.1.0",

    # Observability
    "structlog>=24.1.0",

    # Security
    "cryptography>=42.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "mypy>=1.8.0",
    "ruff>=0.1.0",
]
```

---

## 16. Deployment

### 16.1 Docker Compose

```yaml
# docker-compose.yaml

services:
  api:
    build: .
    container_name: crawl4r-api
    ports:
      - "53001:53001"
    environment:
      - APP_ENV=production
    env_file:
      - .env
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:53001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  worker:
    build: .
    container_name: crawl4r-worker
    command: python -m app.worker.main
    environment:
      - WORKER_PROCESSES=3
    env_file:
      - .env
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    container_name: crawl4r-postgres
    environment:
      POSTGRES_USER: crawl4r
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: crawl4r
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U crawl4r"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: crawl4r-redis
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  crawl4ai:
    image: unclecode/crawl4ai:latest
    container_name: crawl4r-crawl4ai
    ports:
      - "52001:11235"
    shm_size: 1g
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 16.2 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .

# Install dependencies
RUN uv pip install --system -e .

# Copy application
COPY app/ app/
COPY alembic/ alembic/
COPY alembic.ini .

# Run migrations and start server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 53001"]
```

### 16.3 Graceful Shutdown

```python
# app/main.py

import asyncio
import signal
from contextlib import asynccontextmanager

from fastapi import FastAPI


shutdown_event = asyncio.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await setup_connections()

    yield

    # Shutdown
    logger.info("shutdown_initiated")
    shutdown_event.set()

    # Wait for in-flight requests (max 30s)
    await asyncio.sleep(5)

    # Close connections
    await close_connections()
    logger.info("shutdown_complete")


def handle_signal(signum, frame):
    shutdown_event.set()


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)
```

---

## Appendix A: Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `URL_INVALID` | 400 | URL validation failed |
| `URL_BLOCKED` | 403 | URL blocked (SSRF, robots.txt) |
| `DOMAIN_BLOCKED` | 403 | Domain is blocked |
| `CIRCUIT_OPEN` | 503 | Circuit breaker is open |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource already exists |
| `AUTH_REQUIRED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `INTERNAL_ERROR` | 500 | Internal server error |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **RRF** | Reciprocal Rank Fusion - algorithm to combine ranked lists |
| **FTS** | Full-Text Search - Postgres tsvector/tsquery |
| **TEI** | Text Embeddings Inference - HuggingFace embedding server |
| **DLQ** | Dead Letter Queue - queue for failed jobs |
| **Circuit Breaker** | Pattern to prevent cascading failures |
| **SSRF** | Server-Side Request Forgery - security vulnerability |

---

*Document generated: 2025-01-11*
