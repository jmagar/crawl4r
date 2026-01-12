# Feature Specification: RAG Pipeline (crawl4r)

**Feature Branch**: `001-rag-pipeline`
**Created**: 2025-01-11
**Status**: Draft
**Input**: User description: "Build crawl4r, a self-hosted RAG pipeline for intelligent web content retrieval. The system crawls web pages using headless browsers, extracts clean markdown content, chunks documents with semantic boundary detection, and generates embeddings for vector search. Documents are stored in PostgreSQL with full-text search indexes, while vectors live in Qdrant for similarity matching. Hybrid search combines vector and keyword results using Reciprocal Rank Fusion with optional reranking for relevance optimization. Users can filter by source type (crawl, upload, api), tags, and date ranges. The REST API supports document CRUD, batch operations, search queries, and crawl job submission. Background workers process crawl jobs asynchronously with rate limiting per domain and robots.txt compliance. Webhooks notify external systems on crawl completion and document events. All external service calls use circuit breakers and retry logic for resilience."

## User Scenarios & Testing

### User Story 1 - Intelligent Content Search (Priority: P1)

As a knowledge worker, I want to search through my indexed content using natural language queries and keywords, so I can quickly find relevant information even when I don't remember exact phrases.

**Why this priority**: Search is the core value proposition of a RAG pipeline. Without effective search, all other features are useless. This is the minimum viable product that demonstrates value.

**Independent Test**: Can be fully tested by indexing a small corpus of documents (via manual insertion or upload) and verifying that searches return relevant results ranked by both semantic similarity and keyword matching. Delivers immediate value by enabling information retrieval.

**Acceptance Scenarios**:

1. **Given** a collection of indexed documents about machine learning, **When** I search for "neural networks training techniques", **Then** I receive results ranked by relevance combining both semantic meaning and keyword matches
2. **Given** indexed documents with different tags ("tutorial", "research", "blog"), **When** I search with a tag filter for "tutorial", **Then** only documents tagged as "tutorial" appear in results
3. **Given** documents from multiple sources (crawled, uploaded, API), **When** I filter search results by source type "upload", **Then** only manually uploaded documents appear
4. **Given** documents indexed over the past 6 months, **When** I search with a date range filter for "last 30 days", **Then** only documents from the past month appear in results
5. **Given** a search query that matches many documents, **When** I request the first page of 10 results, **Then** I receive exactly 10 results with metadata indicating more results are available

---

### User Story 2 - Document Management (Priority: P2)

As a content curator, I want to upload, tag, update, and delete documents through an API, so I can build and maintain my searchable knowledge base without manual database operations.

**Why this priority**: This enables users to populate the system with content for testing search functionality. It's required before crawling exists, making it the second priority for getting a working MVP.

**Independent Test**: Can be tested by uploading documents via API, verifying storage, retrieving document metadata, updating tags, and confirming deletion. Delivers value by enabling content management without requiring crawling infrastructure.

**Acceptance Scenarios**:

1. **Given** I have a markdown file with metadata, **When** I upload it via the API with tags ["documentation", "api"], **Then** the document is stored, indexed, chunked, and searchable with those tags
2. **Given** an existing document in the system, **When** I retrieve its metadata by ID, **Then** I receive complete information including title, source type, tags, creation date, and chunk count
3. **Given** a document I uploaded yesterday, **When** I update its tags to add "archived", **Then** future searches can filter using the "archived" tag
4. **Given** a document that is no longer needed, **When** I delete it via API, **Then** the document, its chunks, and embeddings are removed and it no longer appears in search results
5. **Given** I need to update a large number of documents, **When** I use the batch upload endpoint with 100 documents, **Then** all documents are processed efficiently in a single request

---

### User Story 3 - Automated Web Crawling (Priority: P3)

As a researcher, I want to submit URLs for crawling so the system automatically extracts, processes, and indexes web content without manual copy-paste, keeping my knowledge base current.

**Why this priority**: Crawling automates content ingestion, but the system provides value even without it (via manual upload). This is an enhancement that improves efficiency but isn't required for core functionality.

**Independent Test**: Can be tested by submitting a URL, verifying the crawler extracts clean content, and confirming the document appears in search results. Delivers value by automating content acquisition from the web.

**Acceptance Scenarios**:

1. **Given** a URL to a blog post, **When** I submit a crawl job for that URL, **Then** the system extracts the content, converts it to clean markdown, and indexes it for search
2. **Given** a crawl job for a documentation site, **When** the crawler encounters a robots.txt file, **Then** the system respects the crawling rules and only accesses allowed paths
3. **Given** I'm crawling multiple pages from the same domain, **When** the crawler processes these URLs, **Then** rate limiting ensures requests are spaced appropriately to avoid overwhelming the server
4. **Given** a submitted crawl job, **When** the job is processing, **Then** I can check its status (pending, running, completed, failed) via the API
5. **Given** a crawl job has completed, **When** I retrieve the result, **Then** I receive the extracted content, metadata (title, URL, crawl timestamp), and any errors encountered

---

### User Story 4 - External System Integration (Priority: P4)

As a system integrator, I want to receive webhooks when documents are created or crawl jobs complete, so my external systems can react to content changes in real-time.

**Why this priority**: Webhooks enable integration with other systems but aren't required for core RAG functionality. This is valuable for production deployments but not essential for MVP validation.

**Independent Test**: Can be tested by configuring a webhook endpoint, triggering document events (upload, crawl completion), and verifying webhook delivery with proper payloads and signatures. Delivers value by enabling event-driven architectures.

**Acceptance Scenarios**:

1. **Given** I've configured a webhook URL for document creation events, **When** a new document is uploaded or crawled, **Then** my endpoint receives a POST request with the document metadata
2. **Given** a webhook is sent to my endpoint, **When** I validate the signature header, **Then** I can confirm the webhook came from the crawl4r system and wasn't tampered with
3. **Given** my webhook endpoint is temporarily down, **When** crawl4r attempts to deliver a webhook, **Then** the system retries with exponential backoff before marking it as failed
4. **Given** a crawl job has completed (success or failure), **When** the job finishes, **Then** a webhook is sent with the job status, URL crawled, and any error messages

---

### User Story 5 - High-Volume Operations (Priority: P5)

As a data engineer, I want to perform batch operations on hundreds of documents efficiently, so I can populate or update my knowledge base at scale without overwhelming the API.

**Why this priority**: Batch operations optimize for scale but single-document operations suffice for MVP. This is a performance enhancement valuable for production use cases with large datasets.

**Independent Test**: Can be tested by submitting batch uploads, batch deletions, and batch crawl jobs, verifying all items are processed efficiently and errors are reported per-item. Delivers value by improving throughput for large-scale operations.

**Acceptance Scenarios**:

1. **Given** I have 500 documents to upload, **When** I use the batch upload endpoint, **Then** all documents are processed in a single request with per-document success/failure status
2. **Given** I need to delete 100 obsolete documents, **When** I submit a batch delete request with document IDs, **Then** all valid deletions complete and any errors are reported with specific IDs
3. **Given** I have a list of 50 URLs to crawl, **When** I submit a batch crawl job, **Then** the system queues all jobs and processes them asynchronously with status tracking for each URL

---

### Edge Cases

- **What happens when a crawl job encounters a page that requires JavaScript rendering?** The headless browser must execute JavaScript to render dynamic content before extraction.
- **How does the system handle documents that exceed size limits?** Documents are rejected with clear error messages indicating the size limit and actual size.
- **What happens when embedding generation fails for a chunk?** The chunk is marked with an error state, the document is still searchable via keyword search, and retry logic attempts regeneration.
- **How does search behave when no results match the query?** Return an empty result set with clear messaging indicating no matches were found and suggestions for broadening the search.
- **What happens when two users upload the same document?** Documents are deduplicated based on content hash. If the same URL is uploaded again, metadata is merged (timestamps updated, tags merged). If a different URL has the same content, a separate entry is created since they are considered distinct documents from different sources.
- **How are circular links handled during crawling?** URL normalization and visited-URL tracking prevent infinite crawl loops.
- **What happens when a webhook endpoint returns an error?** Retry with exponential backoff (e.g., 1s, 2s, 4s, 8s, 16s) up to a maximum number of attempts, then mark as failed and log.
- **How does the system handle malformed HTML during crawling?** Robust parsing libraries extract what's possible, log warnings, and continue processing rather than failing the entire crawl job.
- **What happens when searches return thousands of results?** Pagination limits results per page, with metadata indicating total count and hasMore flag for client-side handling.
- **How are documents with mixed languages handled?** Multilingual embedding models support multiple languages; language detection may group chunks appropriately.

## Requirements

### Functional Requirements

#### Authentication & Authorization

- **FR-001**: System MUST authenticate API requests using bearer tokens with configurable scopes (read, write, admin)
- **FR-002**: System MUST enforce per-API-key rate limiting with configurable requests-per-minute limits
- **FR-003**: System MUST support API key expiration with optional expiration dates and MUST reject requests from expired keys with 401 Unauthorized (see FR-004 for scope definitions)
- **FR-004**: Users with read scope MUST be able to search and view documents; write scope required for upload/delete; admin scope required for system management

#### Organization & Structure

- **FR-005**: System MUST support collections for grouping related documents with flat hierarchy (no nested collections)
- **FR-006**: System MUST maintain global tag namespace accessible across all collections
- **FR-007**: Users MUST be able to assign documents to collections and apply multiple tags per document

#### Document Management

- **FR-008**: System MUST accept document uploads via REST API with support for markdown, plain text, and HTML formats
- **FR-009**: System MUST chunk documents using semantic boundary detection (paragraphs, sections, natural breaks)
- **FR-010**: System MUST generate vector embeddings for each document chunk to enable semantic similarity search
- **FR-011**: System MUST store documents with full-text search indexes for keyword matching
- **FR-012**: System MUST deduplicate documents based on content hash (SHA256) with the following rules: (a) same URL and content hash → merge metadata (update timestamps, merge tags), (b) different URL with same content hash → create separate entry (same content from different sources are distinct documents), (c) different content hash → always create new entry
- **FR-013**: System MUST track document metadata including title, source type, URL (if crawled), creation timestamp, update timestamp, and tags
- **FR-014**: System MUST maintain audit logs for document lifecycle events (created, updated, deleted)
- **FR-015**: System MUST provide REST API endpoints for document CRUD operations (create, read, update, delete)

#### Search Operations

- **FR-016**: System MUST provide hybrid search combining vector similarity and keyword matching using Reciprocal Rank Fusion
- **FR-017**: Users MUST be able to filter search results by source type (crawl, upload, api), tags, and date ranges
- **FR-018**: System MUST support optional reranking of search results to optimize relevance
- **FR-019**: Users MUST be able to customize hybrid search with configurable vector weight, keyword weight, and RRF k-parameter
- **FR-020**: System MUST support pagination for search results and document listings

#### Batch Operations

- **FR-021**: System MUST support batch operations for uploading, deleting, and crawling multiple items in a single request
- **FR-022**: Batch operations MUST use best-effort processing and return per-item success/failure status with total counts and error messages

#### Web Crawling

- **FR-023**: System MUST accept crawl job submissions via API and process them asynchronously in background workers
- **FR-024**: System MUST support three-tier priority queues (high, normal, low) for crawl job processing
- **FR-025**: System MUST respect robots.txt files when crawling websites
- **FR-026**: System MUST implement rate limiting per domain to avoid overwhelming target servers during crawling
- **FR-027**: System MUST extract clean markdown content from web pages using headless browser rendering
- **FR-028**: System MUST provide crawl job status tracking (pending, running, completed, failed)

#### Webhook Integration

- **FR-029**: System MUST send webhooks to configured endpoints on document creation and crawl completion events
- **FR-030**: System MUST sign webhook payloads with cryptographic signatures for authenticity verification
- **FR-031**: System MUST retry failed webhook deliveries with maximum 5 attempts using exponential backoff (1s, 2s, 4s, 8s, 16s intervals)
- **FR-032**: System MUST log webhook delivery failures after exhausting retry attempts

#### Resilience & Reliability

- **FR-033**: System MUST implement circuit breakers on all external service calls to prevent cascade failures
- **FR-034**: System MUST implement retry logic with exponential backoff for transient failures
- **FR-035**: System MUST provide health check endpoints reporting status of dependencies (database, vector store, embedding service, crawler)

### Key Entities

- **API Key**: Authentication credential with unique hash, configurable scopes (read/write/admin), per-key rate limiting, and optional expiration date
- **Collection**: Organizational grouping for related documents with flat hierarchy (no nesting), used for filtering and organization
- **Tag**: User-defined label in global namespace, attachable to documents for categorization and filtering across collections
- **Document**: Represents a piece of content (web page, uploaded file, API submission) with metadata including title, source type, source URL, creation/update timestamps, content hash for deduplication, tags, collection assignment, and associated chunks
- **Chunk**: A segment of a document created through semantic boundary detection, containing text content, position within parent document, token count, section header context, and vector embedding
- **Embedding**: A numerical vector representation of a chunk's semantic meaning, generated by specified model, used for similarity search
- **Crawl Job**: An asynchronous task to extract content from a URL, with priority level (high/normal/low), status tracking (pending, running, completed, failed), retry count, timestamps, and error information
- **Webhook Configuration**: Settings for external system integration including target URL, event types (document creation, crawl completion), secret for HMAC signing, and retry policy (max 5 attempts with exponential backoff)
- **Search Query**: User input combining text query, filters (source type, tags, date range, collections), hybrid search weights (vector/keyword), RRF k-parameter, reranking options, and pagination parameters
- **Search Result**: A ranked list of document chunks with fused relevance scores (vector, keyword, optional rerank), highlighting, section context, and metadata

## Success Criteria

### Measurable Outcomes

- **SC-001**: Users can find relevant documents in under 1 second for 95% of search queries
- **SC-002**: System maintains 99% uptime for search and document management operations
- **SC-003**: Crawl jobs extract clean, readable content from at least 90% of modern web pages, where "clean" means removal of boilerplate (navigation, ads, footers), and "readable" means valid markdown syntax with no garbled encoding, properly formatted headings, and preserved semantic structure
- **SC-004**: Hybrid search returns more relevant results than keyword-only or vector-only search in blind A/B testing
- **SC-005**: Users successfully complete document upload and search workflow on first attempt 95% of the time
- **SC-006**: System processes 10 concurrent crawl jobs without performance degradation
- **SC-007**: Batch operations reduce API call overhead by at least 80% compared to individual operations for bulk tasks
- **SC-008**: Webhook delivery success rate exceeds 99% under normal operating conditions
- **SC-009**: System remains operational when dependent services become temporarily unavailable
- **SC-010**: Search result relevance improves by at least 20% when reranking is enabled versus base hybrid search
- **SC-011**: System correctly honors website crawling policies for 100% of crawled domains
- **SC-012**: Crawling respects domain rate limits preventing more than one request per second to any single host

## Assumptions & Dependencies

### Technical Approach

The feature description specifies particular technical approaches that inform the requirements:

- **Storage Strategy**: Document storage with full-text indexing plus separate vector storage for embeddings enables hybrid search combining keyword matching and semantic similarity
- **Crawling Method**: Headless browser rendering required for JavaScript-heavy modern web applications to extract complete content
- **Search Fusion**: Reciprocal Rank Fusion (RRF) algorithm selected for combining vector and keyword search results based on proven effectiveness in information retrieval research
- **Resilience Patterns**: Circuit breakers and retry logic specified to handle transient failures in distributed system architecture
- **Integration Method**: Webhook-based event notifications chosen for real-time external system integration
- **Web Etiquette**: robots.txt compliance and domain-level rate limiting required for responsible web crawling

### Dependencies

- **External Services**: System depends on embedding generation service for vector creation, web crawler service for content extraction, and storage services for persistence
- **Network Connectivity**: Crawling functionality requires outbound internet access; webhook delivery requires ability to reach configured endpoints
- **Content Assumptions**: Source websites are assumed to be publicly accessible and contain textual content (not purely image or video based)

### Constraints

- **Self-Hosted Requirement**: All services must be deployable in self-hosted environment (no cloud-managed services)
- **Data Volume**: System designed for knowledge base scale (thousands to hundreds of thousands of documents), not web-scale (billions)
- **Language Support**: Multilingual embedding models assumed available for international content

## Clarifications

### Session 2025-01-11

The following clarifications were resolved by referencing the technical design document:

- **Q: How should users authenticate to the API?** → A: API key authentication with bearer tokens. Each API key has configurable scopes (read, write, admin), per-key rate limiting (requests per minute), and optional expiration dates.

- **Q: How is multi-tenancy organized?** → A: Collection-based organization with flat hierarchy. Collections group related documents for organizational purposes. Tags operate in global namespace. API keys are system-wide, not scoped to collections.

- **Q: How should duplicate documents be handled?** → A: Content hash-based deduplication using SHA256 of document content with explicit rules: (1) Same URL + same content hash = merge metadata (update timestamps, merge tags), (2) Different URL + same content hash = create separate entry (same content from different sources are distinct documents), (3) Different content hash = always create new entry. This ensures documents from different sources are tracked independently even if content is identical.

- **Q: What is the webhook retry policy?** → A: Maximum 5 delivery attempts with exponential backoff timing: 1s, 2s, 4s, 8s, 16s between attempts. After exhausting retries, webhook delivery marked as failed and logged.

- **Q: What access control model should be used?** → A: Scope-based authorization at API key level. Three scopes: read (search, view documents), write (upload, delete, manage documents), admin (manage API keys, system settings). Collections are organizational boundaries, not security boundaries.

- **Q: Can users customize hybrid search behavior?** → A: Yes. Users can configure vector weight, keyword weight, and RRF k-parameter per search request. Optional reranking can be enabled with configurable top-N candidates.

- **Q: How should batch operations report failures?** → A: Best-effort processing with per-item success/failure status. Batch response includes total submitted count, failed count, and individual job responses with status and error messages for failed items.

- **Q: How are crawl jobs prioritized?** → A: Three-tier priority system (high, normal, low) with separate Redis queues. Jobs processed in priority order within each queue, FIFO within same priority level.
