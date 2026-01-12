# Data Model: RAG Pipeline

**Feature**: 001-rag-pipeline
**Date**: 2025-01-11
**Status**: Design Complete

---

## Overview

This document defines the complete data model for the RAG pipeline, including entities, relationships, validation rules, and state transitions. All models use Pydantic for validation and are stored in PostgreSQL (relational data) and Qdrant (vector embeddings).

**Design Principles**:
- **Flat Organization**: Collections are non-hierarchical, tags are global
- **Content-Addressable**: Documents deduplicated by SHA256 content hash
- **Audit Trail**: All mutations tracked with timestamps
- **Soft Deletes**: Documents marked deleted, not removed (recovery possible)
- **Async-First**: All models support async I/O operations

---

## 1. Authentication & Authorization

### 1.1 API Key

**Purpose**: Authenticate and authorize API requests with scope-based access control.

**Fields**:
```python
class ApiKey(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    key_hash: str                           # SHA256 of actual key (never store plaintext)
    name: str                               # Human-readable label (e.g., "Production API", "Dev Testing")
    scopes: list[str] = ["read"]            # Access levels: read, write, admin
    rate_limit_rpm: int = 60                # Requests per minute (per-key limit)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None      # Optional expiration
    is_active: bool = True                  # Soft disable

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "key_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "name": "Production API",
                "scopes": ["read", "write"],
                "rate_limit_rpm": 120,
                "created_at": "2025-01-11T12:00:00Z",
                "expires_at": null,
                "is_active": true
            }
        }
```

**Validation Rules**:
- `key_hash`: SHA256 hex string (64 characters)
- `name`: 1-100 characters
- `scopes`: Non-empty, each scope in `{"read", "write", "admin"}`
- `rate_limit_rpm`: 1-10000 (prevent abuse)
- `expires_at`: Must be future date if set

**Indexes**:
- Primary: `id` (UUID)
- Unique: `key_hash` (fast auth lookup)
- Filter: `is_active` (active keys only)

**Relationships**:
- None (API keys are system-level, not scoped to collections)

---

## 2. Organization & Structure

### 2.1 Collection

**Purpose**: Group related documents for organizational purposes (flat hierarchy).

**Fields**:
```python
class Collection(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str                               # Unique, human-readable
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("name")
    def validate_name(cls, v):
        if not 1 <= len(v) <= 100:
            raise ValueError("Name must be 1-100 characters")
        if not re.match(r"^[a-zA-Z0-9-_ ]+$", v):
            raise ValueError("Name must be alphanumeric with hyphens, underscores, spaces")
        return v.strip()
```

**Validation Rules**:
- `name`: 1-100 characters, alphanumeric + hyphens/underscores/spaces, unique
- `description`: 0-500 characters

**Indexes**:
- Primary: `id`
- Unique: `name` (enforce uniqueness)

**Relationships**:
- One-to-many with `Document` (collection → documents)
- One-to-many with `CrawlSource` (collection → monitored sources)

### 2.2 Tag

**Purpose**: Categorize documents with user-defined labels (global namespace).

**Fields**:
```python
class Tag(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str                               # Unique, lowercase

    @validator("name")
    def validate_name(cls, v):
        if not 1 <= len(v) <= 50:
            raise ValueError("Tag name must be 1-50 characters")
        return v.lower().strip()
```

**Validation Rules**:
- `name`: 1-50 characters, lowercase, unique
- No special characters except hyphens/underscores

**Indexes**:
- Primary: `id`
- Unique: `name`

**Relationships**:
- Many-to-many with `Document` (via `document_tags` junction table)

---

## 3. Content Storage

### 3.1 Document

**Purpose**: Represents a piece of content (web page, upload, API submission) with metadata and chunks.

**Fields**:
```python
class DocSource(str, Enum):
    CRAWL = "crawl"
    UPLOAD = "upload"
    API = "api"

class DocumentMetadata(BaseModel):
    """Extracted metadata from crawled content."""
    author: str | None = None
    publish_date: datetime | None = None
    description: str | None = None
    keywords: list[str] = Field(default_factory=list)
    og_image: str | None = None             # Open Graph image URL
    canonical_url: str | None = None        # Canonical URL (if different from source)

class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    url: str                                # Source URL or synthetic URL (for uploads)
    domain: str                             # Extracted domain (e.g., "example.com")
    parent_url: str | None = None           # If discovered via link (deep crawl)
    title: str | None = None
    content: str                            # Full extracted text (markdown or plain text)
    content_hash: str                       # SHA256 of content (deduplication)
    language: str = "en"                    # ISO 639-1 language code
    source: DocSource                       # How document entered system
    collection_id: UUID | None = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    crawled_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: datetime | None = None      # Soft delete timestamp

    @validator("content_hash")
    def validate_content_hash(cls, v):
        if len(v) != 64:
            raise ValueError("Content hash must be 64-character SHA256 hex")
        return v.lower()

    @validator("domain")
    def extract_domain(cls, v, values):
        """Auto-extract domain from URL if not provided."""
        if not v and "url" in values:
            parsed = urlparse(values["url"])
            return parsed.netloc.lower()
        return v.lower()
```

**Validation Rules**:
- `url`: Valid URL (validated with URLValidator)
- `content`: Non-empty, max 50MB
- `content_hash`: SHA256 hex string (64 chars)
- `language`: ISO 639-1 code (e.g., "en", "es", "zh")
- `source`: One of {crawl, upload, api}

**Indexes**:
- Primary: `id`
- Unique: `url` (one document per URL)
- Index: `domain` (filter by domain)
- Index: `content_hash` (deduplication check)
- Index: `collection_id` (filter by collection)
- Index: `crawled_at` (date range queries)
- GIN: `fts_vector` (full-text search)
- Partial: `deleted_at IS NULL` (active documents)

**Relationships**:
- Many-to-one with `Collection` (document → collection)
- One-to-many with `Chunk` (document → chunks)
- Many-to-many with `Tag` (via `document_tags`)
- One-to-one with `CrawlJob` (document ← crawl job)

**Generated Fields** (PostgreSQL):
```sql
fts_vector TSVECTOR GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
    setweight(to_tsvector('english', content), 'B')
) STORED
```

### 3.2 Chunk

**Purpose**: Text segment from a document, created via semantic boundary detection, with vector embedding.

**Fields**:
```python
class ChunkContentType(str, Enum):
    PROSE = "prose"
    CODE = "code"
    TABLE = "table"
    LIST = "list"

class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID                       # Parent document
    content: str                            # Text content
    chunk_index: int                        # Order within document (0-indexed)
    start_char: int                         # Offset in original document content
    end_char: int                           # End offset (for highlighting)
    token_count: int                        # Number of tokens (for context window tracking)
    embedding_model: str                    # Model used for embedding (e.g., "qwen3-0.6b-embedding")
    section_header: str | None = None       # Nearest H1/H2/H3 above chunk (context)
    content_type: ChunkContentType = ChunkContentType.PROSE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("chunk_index")
    def validate_chunk_index(cls, v):
        if v < 0:
            raise ValueError("Chunk index must be non-negative")
        return v

    @validator("token_count")
    def validate_token_count(cls, v):
        if v < 1:
            raise ValueError("Token count must be positive")
        return v
```

**Validation Rules**:
- `content`: Non-empty, max 2000 characters (typical chunk size)
- `chunk_index`: Non-negative integer
- `start_char`, `end_char`: `start_char < end_char`
- `token_count`: Positive integer
- `embedding_model`: Non-empty string

**Indexes**:
- Primary: `id`
- Unique: `(document_id, chunk_index)` (enforce order)
- Index: `document_id` (fetch chunks for document)
- Index: `embedding_model` (filter by model version)
- GIN: `fts_vector` (chunk-level full-text search)

**Relationships**:
- Many-to-one with `Document` (chunk → document)
- One-to-one with vector in Qdrant (chunk ← vector)

**Generated Fields** (PostgreSQL):
```sql
fts_vector TSVECTOR GENERATED ALWAYS AS (
    to_tsvector('english', content)
) STORED
```

---

## 4. Vector Storage (Qdrant)

### 4.1 Vector Point

**Purpose**: Qdrant point representing a chunk's semantic embedding.

**Structure**:
```python
from qdrant_client.models import PointStruct

point = PointStruct(
    id=str(chunk.id),                       # UUID as string
    vector=embedding,                       # 1024-dim float list
    payload={
        "document_id": str(chunk.document_id),
        "chunk_id": str(chunk.id),
        "chunk_index": chunk.chunk_index,
        "collection_id": str(document.collection_id) if document.collection_id else None,
        "domain": document.domain,
        "url": document.url,
        "title": document.title,
        "source": document.source.value,     # For filtering (crawl/upload/api)
        "tags": [str(t) for t in tags],      # Denormalized for fast filtering
        "section_header": chunk.section_header,
        "content_type": chunk.content_type.value,
        "crawled_at": document.crawled_at.isoformat()
    }
)
```

**Payload Fields**:
- `document_id`: UUID (string) - link to PostgreSQL document
- `chunk_id`: UUID (string) - link to PostgreSQL chunk
- `chunk_index`: int - order within document
- `collection_id`: UUID (string, nullable) - for filtering by collection
- `domain`: string - for filtering by domain
- `url`: string - source URL
- `title`: string (nullable) - document title
- `source`: enum string - "crawl", "upload", or "api"
- `tags`: list[string] - denormalized tag names
- `section_header`: string (nullable) - context for chunk
- `content_type`: enum string - "prose", "code", "table", "list"
- `crawled_at`: ISO 8601 string - for date range filtering

**Indexes** (Qdrant Payload):
- `document_id` (UUID)
- `collection_id` (UUID)
- `domain` (keyword)
- `tags` (keyword array)
- `source` (keyword)

**Validation**:
- `vector`: Exactly 1024 dimensions, normalized to unit length
- `payload.chunk_id`: Must match point ID
- `payload.crawled_at`: Valid ISO 8601 timestamp

---

## 5. Background Jobs

### 5.1 CrawlJob

**Purpose**: Asynchronous task to crawl a single URL and extract content.

**Fields**:
```python
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

class CrawlJob(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    url: str                                # Target URL
    domain: str                             # Extracted domain (for rate limiting)
    crawl_config_id: UUID | None = None     # Configuration to use
    crawl_source_id: UUID | None = None     # If from scheduled source
    document_id: UUID | None = None         # Result document (after success)
    collection_id: UUID | None = None       # Target collection
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    error: str | None = None                # Error message if failed
    retry_count: int = 0                    # Number of retries attempted
    max_retries: int = 3                    # Maximum retry attempts
    result: dict[str, Any] | None = None    # Cached result (JSON)
    result_expires_at: datetime | None = None
    webhook_url: str | None = None          # Notification endpoint
    webhook_headers: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @validator("retry_count")
    def validate_retry_count(cls, v, values):
        if v < 0:
            raise ValueError("Retry count cannot be negative")
        if "max_retries" in values and v > values["max_retries"]:
            raise ValueError("Retry count cannot exceed max_retries")
        return v
```

**Validation Rules**:
- `url`: Valid URL
- `domain`: Non-empty string
- `retry_count`: 0 ≤ retry_count ≤ max_retries
- `max_retries`: 0 ≤ max_retries ≤ 10
- `status`: One of {pending, running, completed, failed, cancelled}
- `priority`: One of {high, normal, low}

**State Transitions**:
```
pending → running → completed
         ↓        ↘ failed → pending (retry)
         cancelled
```

**Indexes**:
- Primary: `id`
- Index: `status` (fetch jobs by status)
- Index: `(priority, created_at)` WHERE `status = 'pending'` (queue ordering)
- Index: `domain` (rate limiting per domain)
- Index: `created_at` (job history queries)
- Index: `crawl_source_id` (jobs from scheduled source)

**Relationships**:
- Many-to-one with `CrawlConfig` (job → config)
- Many-to-one with `CrawlSource` (job → source)
- One-to-one with `Document` (job → document)
- Many-to-one with `Collection` (job → collection)

### 5.2 DeepCrawlJob

**Purpose**: Multi-page crawl orchestration with BFS/DFS/BestFirst strategies.

**Fields**:
```python
class DeepCrawlStrategy(str, Enum):
    BFS = "bfs"                             # Breadth-first search
    DFS = "dfs"                             # Depth-first search
    BEST_FIRST = "best_first"               # Score-based priority

class DeepCrawlJob(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    root_url: str                           # Starting URL
    strategy: DeepCrawlStrategy
    max_depth: int                          # Maximum link depth (0 = root only)
    max_pages: int                          # Maximum pages to crawl
    score_threshold: float = 0.0            # Minimum score for best-first (0.0-1.0)
    keywords: list[str] = Field(default_factory=list)  # For best-first scoring
    crawl_config_id: UUID
    collection_id: UUID | None = None
    webhook_url: str | None = None
    webhook_headers: dict[str, str] = Field(default_factory=dict)

    # State
    status: JobStatus = JobStatus.PENDING
    pages_discovered: int = 0
    pages_crawled: int = 0
    pages_failed: int = 0
    current_depth: int = 0

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @validator("max_depth")
    def validate_max_depth(cls, v):
        if v < 0:
            raise ValueError("Max depth cannot be negative")
        return v

    @validator("max_pages")
    def validate_max_pages(cls, v):
        if v < 1:
            raise ValueError("Max pages must be at least 1")
        return v

    @validator("score_threshold")
    def validate_score_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Score threshold must be between 0.0 and 1.0")
        return v
```

**Validation Rules**:
- `max_depth`: Non-negative integer
- `max_pages`: Positive integer, max 1000 (prevent runaway crawls)
- `score_threshold`: 0.0-1.0 (relevance threshold)
- `strategy`: One of {bfs, dfs, best_first}

**Indexes**:
- Primary: `id`
- Index: `status`

**Relationships**:
- One-to-many with `DeepCrawlFrontierItem` (job → frontier items)
- Many-to-one with `CrawlConfig` (job → config)
- Many-to-one with `Collection` (job → collection)

### 5.3 DeepCrawlFrontierItem

**Purpose**: URL in the crawl frontier (to be visited or already visited).

**Fields**:
```python
class DeepCrawlFrontierItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    deep_crawl_job_id: UUID
    url: str                                # URL to crawl
    depth: int                              # Link depth from root
    score: float | None = None              # Relevance score (for best-first)
    status: JobStatus = JobStatus.PENDING
    discovered_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("depth")
    def validate_depth(cls, v):
        if v < 0:
            raise ValueError("Depth cannot be negative")
        return v
```

**Validation Rules**:
- `url`: Valid URL
- `depth`: Non-negative integer
- `score`: 0.0-1.0 if set (for best-first strategy)
- `status`: One of {pending, running, completed, failed}

**Indexes**:
- Primary: `id`
- Unique: `(deep_crawl_job_id, url)` (prevent duplicate crawls)
- Index: `(deep_crawl_job_id, status)` (fetch pending items)
- Index: `(deep_crawl_job_id, score DESC)` WHERE `status = 'pending'` (best-first ordering)
- Index: `(deep_crawl_job_id, depth, discovered_at)` WHERE `status = 'pending'` (BFS/DFS ordering)

**Relationships**:
- Many-to-one with `DeepCrawlJob` (item → job)

### 5.4 DiscoveryJob

**Purpose**: Discover URLs from sitemaps and Common Crawl, optionally score by relevance.

**Fields**:
```python
class DiscoveryJob(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    domain: str                             # Target domain
    sources: list[str]                      # ["sitemap", "common_crawl"]
    pattern: str | None = None              # URL pattern filter (regex)
    max_urls: int = 500                     # Maximum URLs to return
    score_query: str | None = None          # Query for relevance scoring
    score_threshold: float = 0.0            # Minimum score (0.0-1.0)
    status: JobStatus = JobStatus.PENDING
    urls_found: int = 0
    result: list[str] | None = None         # Discovered URLs
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    @validator("sources")
    def validate_sources(cls, v):
        valid_sources = {"sitemap", "common_crawl"}
        if not v:
            raise ValueError("At least one source required")
        if not set(v).issubset(valid_sources):
            raise ValueError(f"Invalid sources. Must be subset of {valid_sources}")
        return v

    @validator("max_urls")
    def validate_max_urls(cls, v):
        if not 1 <= v <= 10000:
            raise ValueError("Max URLs must be between 1 and 10,000")
        return v
```

**Validation Rules**:
- `domain`: Valid domain name
- `sources`: Non-empty, subset of {"sitemap", "common_crawl"}
- `max_urls`: 1-10,000
- `score_threshold`: 0.0-1.0

**Indexes**:
- Primary: `id`
- Index: `domain`
- Index: `status`

**Relationships**: None (standalone job)

---

## 6. Configuration

### 6.1 CrawlConfig

**Purpose**: Configurable settings for crawl operations.

**Fields**:
```python
class CrawlConfig(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str                               # Unique, human-readable
    max_depth: int = 0                      # 0 = single page
    rate_limit_rps: float = 1.0             # Requests per second per domain
    respect_robots_txt: bool = True
    headers: dict[str, str] = Field(default_factory=dict)  # Custom HTTP headers
    extraction_strategy: str = "markdown"   # markdown, structured, raw
    chunking_strategy: str = "sliding_window"
    chunk_size: int = 512                   # Tokens per chunk
    chunk_overlap: int = 50                 # Overlap tokens
    min_chunk_size: int = 50                # Minimum tokens per chunk
    page_timeout_ms: int = 30000            # Page load timeout (milliseconds)
    max_page_size_mb: int = 50              # Max response size (megabytes)
    max_redirects: int = 10
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("rate_limit_rps")
    def validate_rate_limit(cls, v):
        if not 0.1 <= v <= 10.0:
            raise ValueError("Rate limit must be between 0.1 and 10.0 requests/second")
        return v

    @validator("chunk_size")
    def validate_chunk_size(cls, v):
        if not 100 <= v <= 2000:
            raise ValueError("Chunk size must be between 100 and 2000 tokens")
        return v

    @validator("page_timeout_ms")
    def validate_page_timeout(cls, v):
        if not 5000 <= v <= 120000:
            raise ValueError("Page timeout must be between 5s and 120s")
        return v
```

**Validation Rules**:
- `name`: 1-100 characters, unique
- `rate_limit_rps`: 0.1-10.0
- `chunk_size`: 100-2000 tokens
- `chunk_overlap`: 0-500 tokens
- `min_chunk_size`: 10-500 tokens
- `page_timeout_ms`: 5000-120000 (5s-120s)
- `max_page_size_mb`: 1-100 MB
- `max_redirects`: 0-20

**Indexes**:
- Primary: `id`
- Unique: `name`

**Relationships**:
- One-to-many with `CrawlJob` (config → jobs)
- One-to-many with `CrawlSource` (config → sources)

### 6.2 CrawlSource

**Purpose**: Monitored URL pattern for scheduled/continuous crawling.

**Fields**:
```python
class CrawlSource(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str                               # Unique, human-readable
    url_pattern: str                        # Base URL or pattern
    crawl_config_id: UUID
    collection_id: UUID | None = None
    schedule_cron: str | None = None        # Cron expression (e.g., "0 */6 * * *")
    is_active: bool = True
    last_crawled_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("schedule_cron")
    def validate_cron(cls, v):
        if v is not None:
            # Basic cron validation (5 fields)
            parts = v.split()
            if len(parts) != 5:
                raise ValueError("Cron expression must have 5 fields")
        return v
```

**Validation Rules**:
- `name`: 1-100 characters, unique
- `url_pattern`: Valid URL or regex pattern
- `schedule_cron`: Valid cron expression (5 fields)

**Indexes**:
- Primary: `id`
- Index: `is_active` WHERE `is_active = true`
- Index: `crawl_config_id`

**Relationships**:
- Many-to-one with `CrawlConfig` (source → config)
- Many-to-one with `Collection` (source → collection)
- One-to-many with `CrawlJob` (source → jobs)
- One-to-many with `Webhook` (source → webhooks)

### 6.3 DomainSettings

**Purpose**: Per-domain overrides for rate limiting and circuit breaker settings.

**Fields**:
```python
class DomainSettings(BaseModel):
    domain: str                             # Primary key (e.g., "example.com")
    rate_limit_rps: float = 1.0             # Override default rate limit
    circuit_breaker_threshold: int = 5      # Failures before circuit opens
    circuit_breaker_timeout_s: int = 300    # Seconds before retry (half-open state)
    is_blocked: bool = False                # Manual block
    blocked_reason: str | None = None       # Reason for block (if is_blocked=true)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("domain")
    def validate_domain(cls, v):
        # Basic domain validation
        if not re.match(r"^[a-z0-9.-]+\.[a-z]{2,}$", v.lower()):
            raise ValueError("Invalid domain format")
        return v.lower()
```

**Validation Rules**:
- `domain`: Valid domain name (lowercase)
- `rate_limit_rps`: 0.1-10.0
- `circuit_breaker_threshold`: 1-20
- `circuit_breaker_timeout_s`: 60-3600 (1min-1hour)

**Indexes**:
- Primary: `domain`
- Index: `is_blocked` WHERE `is_blocked = true`

**Relationships**: None (domain-level settings)

### 6.4 ProxyConfig

**Purpose**: Proxy server configuration for rotation strategies.

**Fields**:
```python
class ProxyConfig(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str                               # Unique, human-readable
    servers: list[dict[str, str]]           # [{"url": "...", "username": "...", "password": "..."}]
    rotation_strategy: str = "round_robin"  # round_robin, least_used, random
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("servers")
    def validate_servers(cls, v):
        if not v:
            raise ValueError("At least one proxy server required")
        for server in v:
            if "url" not in server:
                raise ValueError("Each server must have 'url' field")
        return v

    @validator("rotation_strategy")
    def validate_rotation_strategy(cls, v):
        valid_strategies = {"round_robin", "least_used", "random"}
        if v not in valid_strategies:
            raise ValueError(f"Invalid rotation strategy. Must be one of {valid_strategies}")
        return v
```

**Validation Rules**:
- `name`: 1-100 characters, unique
- `servers`: Non-empty list, each with `url` field
- `rotation_strategy`: One of {round_robin, least_used, random}

**Indexes**:
- Primary: `id`
- Unique: `name`
- Index: `is_active`

**Relationships**: None (global proxy config)

---

## 7. Webhooks & Integration

### 7.1 Webhook

**Purpose**: Configuration for external system notifications.

**Fields**:
```python
class Webhook(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source_id: UUID | None = None           # Optional link to CrawlSource
    url: str                                # Notification endpoint
    headers: dict[str, str] = Field(default_factory=dict)  # Custom headers
    secret: str | None = None               # HMAC secret (for signature)
    events: list[str] = ["completed", "failed"]  # Event types to notify
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("url")
    def validate_webhook_url(cls, v):
        # Validate webhook URL (must be HTTPS in production)
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Webhook URL must use HTTP or HTTPS")
        return v

    @validator("events")
    def validate_events(cls, v):
        valid_events = {"completed", "failed", "document_created", "document_updated", "document_deleted"}
        if not v:
            raise ValueError("At least one event type required")
        if not set(v).issubset(valid_events):
            raise ValueError(f"Invalid event types. Must be subset of {valid_events}")
        return v
```

**Validation Rules**:
- `url`: Valid HTTP/HTTPS URL
- `events`: Non-empty, subset of {"completed", "failed", "document_created", "document_updated", "document_deleted"}
- `secret`: Optional, used for HMAC-SHA256 signing

**Indexes**:
- Primary: `id`
- Index: `source_id`
- Index: `is_active` WHERE `is_active = true`

**Relationships**:
- Many-to-one with `CrawlSource` (webhook → source)
- One-to-many with `WebhookDelivery` (webhook → deliveries)

### 7.2 WebhookDelivery

**Purpose**: Track webhook delivery attempts and retry status.

**Fields**:
```python
class WebhookDelivery(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    webhook_id: UUID
    job_id: UUID | None = None              # Optional link to job
    job_type: str | None = None             # "crawl", "deep_crawl", "discovery"
    event: str                              # Event type (e.g., "completed", "failed")
    payload: dict[str, Any]                 # JSON payload
    status: Literal["pending", "delivered", "failed"] = "pending"
    attempts: int = 0
    max_attempts: int = 5
    last_attempt_at: datetime | None = None
    next_attempt_at: datetime | None = None  # For retry scheduling
    response_status: int | None = None      # HTTP status code
    response_body: str | None = None        # Response body (truncated to 1KB)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("attempts")
    def validate_attempts(cls, v, values):
        if v < 0:
            raise ValueError("Attempts cannot be negative")
        if "max_attempts" in values and v > values["max_attempts"]:
            raise ValueError("Attempts cannot exceed max_attempts")
        return v
```

**Validation Rules**:
- `event`: Non-empty string
- `payload`: Valid JSON object
- `attempts`: 0 ≤ attempts ≤ max_attempts
- `max_attempts`: 1-10
- `status`: One of {"pending", "delivered", "failed"}

**State Transitions**:
```
pending → delivered (200-299 response)
pending → pending (retry on 4xx/5xx/timeout)
pending → failed (after max_attempts)
```

**Indexes**:
- Primary: `id`
- Index: `webhook_id`
- Index: `next_attempt_at` WHERE `status = 'pending'` (retry queue)

**Relationships**:
- Many-to-one with `Webhook` (delivery → webhook)

---

## 8. Search Models

### 8.1 SearchFilters

**Purpose**: Filter criteria for search queries.

**Fields**:
```python
class SearchFilters(BaseModel):
    collection_ids: list[UUID] | None = None
    tag_ids: list[UUID] | None = None
    domains: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    source_types: list[DocSource] | None = None  # crawl, upload, api

    @validator("date_to")
    def validate_date_range(cls, v, values):
        if v and "date_from" in values and values["date_from"]:
            if v < values["date_from"]:
                raise ValueError("date_to must be after date_from")
        return v
```

**Validation Rules**:
- `collection_ids`: Valid UUIDs
- `tag_ids`: Valid UUIDs
- `domains`: Valid domain names
- `date_from`, `date_to`: Valid datetimes, date_to > date_from
- `source_types`: Subset of {crawl, upload, api}

### 8.2 SearchConfig

**Purpose**: Search query configuration with hybrid search parameters.

**Fields**:
```python
class SearchConfig(BaseModel):
    query: str                              # Search query text
    filters: SearchFilters = Field(default_factory=SearchFilters)
    min_score: float = 0.0                  # Minimum RRF score (0.0-1.0)
    use_reranker: bool = False              # Enable reranking (gte-reranker-modernbert-base)
    rerank_top_n: int = 20                  # Number of results to rerank
    expand_chunks: bool = False             # Include surrounding chunks
    rrf_k: int = 60                         # RRF constant
    vector_weight: float = 1.0              # Weight for vector results
    keyword_weight: float = 1.0             # Weight for FTS results
    limit: int = 10                         # Results per page
    cursor: str | None = None               # For pagination

    @validator("query")
    def validate_query(cls, v):
        if not 1 <= len(v.strip()) <= 500:
            raise ValueError("Query must be 1-500 characters")
        return v.strip()

    @validator("min_score")
    def validate_min_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Min score must be between 0.0 and 1.0")
        return v

    @validator("limit")
    def validate_limit(cls, v):
        if not 1 <= v <= 100:
            raise ValueError("Limit must be between 1 and 100")
        return v

    @validator("rrf_k")
    def validate_rrf_k(cls, v):
        if not 10 <= v <= 200:
            raise ValueError("RRF k must be between 10 and 200")
        return v
```

**Validation Rules**:
- `query`: 1-500 characters
- `min_score`: 0.0-1.0
- `rerank_top_n`: 1-50
- `rrf_k`: 10-200
- `vector_weight`, `keyword_weight`: 0.0-10.0
- `limit`: 1-100

### 8.3 SearchResult

**Purpose**: Single search result with scores and metadata.

**Fields**:
```python
class SearchResult(BaseModel):
    document_id: UUID
    chunk_id: UUID
    url: str
    title: str | None
    content: str                            # Chunk content
    score: float                            # RRF fused score
    vector_score: float | None = None       # Cosine similarity score
    keyword_score: float | None = None      # FTS rank score
    rerank_score: float | None = None       # Reranker score (if enabled)
    source: Literal["vector", "keyword", "fused"]
    highlights: list[tuple[int, int]] | None = None  # Character ranges for highlighting
    section_header: str | None = None       # Context from chunk
    expanded_chunks: list[str] | None = None  # Surrounding chunks (if expand_chunks=true)
```

**Validation Rules**:
- `score`: 0.0-1.0 (normalized)
- `source`: One of {"vector", "keyword", "fused"}
- `highlights`: List of (start, end) character ranges

### 8.4 SearchResponse

**Purpose**: Complete search response with results and metadata.

**Fields**:
```python
class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_count: int | None = None          # Estimated total (may be null for cursor pagination)
    next_cursor: str | None = None          # Opaque cursor for next page
    query_embedding_cached: bool = False    # Was query embedding cached?
    result_cached: bool = False             # Was entire result cached?
    latency_ms: float                       # Total query latency

    @validator("latency_ms")
    def validate_latency(cls, v):
        if v < 0:
            raise ValueError("Latency cannot be negative")
        return v
```

---

## 9. Utility Models

### 9.1 ValidatedURL

**Purpose**: URL validation and normalization result.

**Fields**:
```python
class ValidatedURL(BaseModel):
    original: str                           # Original URL
    normalized: str                         # Lowercase host, sorted params
    scheme: str                             # http or https
    host: str                               # Hostname
    domain: str                             # Extracted domain
    path: str                               # URL path
    is_valid: bool = True
    rejection_reason: str | None = None     # Reason if invalid
```

**Validation**:
- SSRF prevention: Block private IP ranges (10.0.0.0/8, 192.168.0.0/16, etc.)
- Metadata endpoints: Block 169.254.169.254, metadata.google.internal
- Scheme validation: Only http/https allowed
- Length limit: Max 2048 characters

### 9.2 CrawlResult

**Purpose**: Result from crawling a single URL (transient, not persisted).

**Fields**:
```python
class CrawlResult(BaseModel):
    url: str
    normalized_url: str
    success: bool
    status_code: int | None = None
    content_type: str | None = None
    html: str | None = None                 # Raw HTML
    markdown: str | None = None             # Extracted markdown
    text: str | None = None                 # Plain text
    title: str | None = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    links: list[str] = Field(default_factory=list)  # Discovered links
    chunks: list[dict[str, Any]] = Field(default_factory=list)  # Pre-chunked content
    error: str | None = None
    error_type: str | None = None           # timeout, connection, http_error, etc.
    crawled_at: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: int                        # Crawl duration

    def to_document(self) -> Document:
        """Convert crawl result to Document model."""
        content = self.markdown or self.text or ""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        return Document(
            url=self.normalized_url,
            domain=urlparse(self.normalized_url).netloc.lower(),
            title=self.title,
            content=content,
            content_hash=content_hash,
            source=DocSource.CRAWL,
            metadata=self.metadata
        )
```

---

## 10. Relationships Summary

**Entity Relationship Diagram (ERD)**:

```
ApiKey (authentication)

Collection
  ├── 1:N → Document
  ├── 1:N → CrawlSource
  └── 1:N → CrawlJob

Tag
  └── N:M → Document (via document_tags)

Document
  ├── 1:N → Chunk
  ├── 1:1 ← CrawlJob
  └── N:M ← Tag

Chunk
  └── 1:1 → Qdrant Vector Point

CrawlConfig
  ├── 1:N → CrawlJob
  ├── 1:N → CrawlSource
  └── 1:N → DeepCrawlJob

CrawlSource
  ├── 1:N → CrawlJob
  └── 1:N → Webhook

CrawlJob
  └── 1:1 → Document (after completion)

DeepCrawlJob
  └── 1:N → DeepCrawlFrontierItem

Webhook
  └── 1:N → WebhookDelivery

DomainSettings (standalone)
ProxyConfig (standalone)
DiscoveryJob (standalone)
```

---

## 11. State Machine Diagrams

### 11.1 CrawlJob State Machine

```
   ┌───────┐
   │PENDING│
   └───┬───┘
       │
       ├─[start]→ ┌───────┐
       │          │RUNNING│
       │          └───┬───┘
       │              │
       │              ├─[success]→ ┌─────────┐
       │              │             │COMPLETED│
       │              │             └─────────┘
       │              │
       │              ├─[failure, retry_count < max_retries]→ PENDING
       │              │
       │              └─[failure, retry_count ≥ max_retries]→ ┌──────┐
       │                                                       │FAILED│
       │                                                       └──────┘
       │
       └─[cancel]→ ┌─────────┐
                   │CANCELLED│
                   └─────────┘
```

### 11.2 Webhook Delivery State Machine

```
   ┌───────┐
   │PENDING│
   └───┬───┘
       │
       ├─[200-299 response]→ ┌─────────┐
       │                     │DELIVERED│
       │                     └─────────┘
       │
       ├─[4xx/5xx/timeout, attempts < max_attempts]→ PENDING (retry with backoff)
       │
       └─[attempts ≥ max_attempts]→ ┌──────┐
                                     │FAILED│
                                     └──────┘
```

---

## 12. Cache Key Design (Redis)

**Pattern**: `crawl4r:{category}:{subcategory}:{identifier}`

| Cache Type | Key Pattern | TTL | Value Type |
|------------|-------------|-----|------------|
| Crawl result | `crawl4r:cache:crawl:{url_sha256[:32]}` | 24h | JSON (gzipped if > 10KB) |
| Embedding | `crawl4r:cache:embed:{content_sha256[:32]}` | 7d | JSON array |
| Query result | `crawl4r:cache:query:{query_hash[:32]}` | 1h | JSON |
| Query embedding | `crawl4r:cache:query_embed:{query_sha256[:32]}` | 24h | JSON array |
| Rate limiter | `crawl4r:rate:{domain}` | 1s | Counter |
| API rate limit | `crawl4r:rate:api:{api_key_hash}:{minute}` | 2m | Counter |
| Circuit breaker | `crawl4r:circuit:{domain}` | 1h | Hash (failures, state, opened_at) |
| Crawl lock | `crawl4r:lock:crawl:{url_sha256}` | 5m | Worker ID |
| Query lock | `crawl4r:lock:query:{query_hash}` | 30s | Worker ID |

**Hash Truncation**: All SHA256 hashes truncated to 32 characters (128 bits) for balance between key length and collision resistance.

---

## 13. Database Indexes Summary

| Table | Index Type | Columns | Purpose |
|-------|-----------|---------|---------|
| api_keys | UNIQUE | key_hash | Fast auth lookup |
| api_keys | PARTIAL | is_active WHERE is_active = true | Active keys |
| collections | UNIQUE | name | Uniqueness constraint |
| tags | UNIQUE | name | Uniqueness constraint |
| documents | UNIQUE | url | One document per URL |
| documents | BTREE | domain, collection_id, crawled_at | Filtering |
| documents | GIN | fts_vector | Full-text search |
| documents | PARTIAL | id WHERE deleted_at IS NULL | Active documents |
| chunks | UNIQUE | (document_id, chunk_index) | Order enforcement |
| chunks | BTREE | document_id | Fetch chunks for document |
| chunks | GIN | fts_vector | Chunk-level FTS |
| crawl_jobs | BTREE | status, domain, created_at | Job queries |
| crawl_jobs | PARTIAL | (priority, created_at) WHERE status = 'pending' | Queue ordering |
| deep_crawl_frontier | UNIQUE | (deep_crawl_job_id, url) | Deduplication |
| deep_crawl_frontier | PARTIAL | (deep_crawl_job_id, score DESC) WHERE status = 'pending' | Best-first |
| deep_crawl_frontier | PARTIAL | (deep_crawl_job_id, depth, discovered_at) WHERE status = 'pending' | BFS/DFS |

---

## 14. Validation Summary

All models use Pydantic validators for:
- **Type safety**: Automatic type coercion and validation
- **Range checks**: Numeric bounds (e.g., 1 ≤ limit ≤ 100)
- **Format validation**: URLs, emails, ISO dates, cron expressions
- **Business rules**: Cross-field validation (e.g., date_to > date_from)
- **Sanitization**: Trimming whitespace, lowercasing domains
- **Custom validators**: SSRF prevention, content hash verification

**Philosophy**: Fail fast at API boundaries, prevent invalid data from entering the system.

---

## 15. Next Steps

**Phase 1 Remaining Deliverables**:
1. **API Contracts** (`contracts/`): OpenAPI 3.1 specifications for all endpoints
2. **Quickstart Guide** (`quickstart.md`): Local development setup
3. **Agent Context Update**: Add technology choices to `.claude/CLAUDE.md`

**Data Model Status**: ✅ Complete - Ready for implementation
