---
spec: rag-ingestion
phase: requirements
created: 2026-01-14
---

# Requirements: RAG Ingestion Pipeline

## Goal

Build an automated RAG ingestion pipeline that monitors a folder for markdown files, generates 1024-dimensional embeddings using HuggingFace TEI with Qwen3-Embedding-0.6B, and stores vectors in Qdrant with comprehensive metadata. The system must handle thousands of files with non-blocking async processing, support batch ingestion on startup, and automatically manage vector lifecycle for file modifications and deletions.

## User Stories

### US-1: Startup Batch Processing
**As a** developer initializing the RAG system
**I want to** automatically process all existing markdown files in the watched folder on startup
**So that** I can build the vector index from historical documents without manual intervention

**Acceptance Criteria:**
- [ ] AC-1.1: System detects all existing `.md` files recursively in the configured watch folder on startup
- [ ] AC-1.2: Files are processed in batches of 10-50 documents to optimize TEI API calls
- [ ] AC-1.3: Batch processing completes with progress logging (e.g., "Processed 50/250 files")
- [ ] AC-1.4: Failed files are logged with error details and do not block processing of other files
- [ ] AC-1.5: System enters watch mode only after initial batch processing completes
- [ ] AC-1.6: Processing status is visible through logs showing documents/minute throughput

### US-2: Real-Time File Monitoring
**As a** content creator adding new documentation
**I want** the system to automatically detect and ingest new markdown files
**So that** new content becomes searchable within minutes without manual triggering

**Acceptance Criteria:**
- [ ] AC-2.1: System detects new `.md` files within 1 second of creation in watched folder
- [ ] AC-2.2: Recursive monitoring captures files created in subdirectories
- [ ] AC-2.3: File events are debounced with 1-second threshold to prevent duplicate processing
- [ ] AC-2.4: New files are queued and processed asynchronously without blocking the watcher
- [ ] AC-2.5: Processing status is logged with filename and completion time
- [ ] AC-2.6: Non-markdown files and excluded directories (.git, .*, __pycache__, node_modules, venv, dist, build) are ignored, symlinks not followed

### US-3: File Modification Handling
**As a** content editor updating existing documentation
**I want** modified markdown files to trigger re-ingestion with old vectors removed
**So that** the vector index reflects current content without duplicates

**Acceptance Criteria:**
- [ ] AC-3.1: System detects markdown file modifications within 1 second
- [ ] AC-3.2: Modification events are debounced to handle editors that save multiple times
- [ ] AC-3.3: All existing vectors associated with the file are deleted from Qdrant before re-ingestion
- [ ] AC-3.4: Vector deletion uses file_path_relative as the filter criterion in Qdrant payload
- [ ] AC-3.5: Re-ingestion generates fresh embeddings for all chunks with updated metadata (new modification timestamp)
- [ ] AC-3.6: Failed re-ingestion preserves old vectors and logs error for manual review

### US-4: File Deletion Cleanup
**As a** content manager removing outdated documentation
**I want** deleted markdown files to automatically remove their vectors from Qdrant
**So that** search results only include currently existing documents

**Acceptance Criteria:**
- [ ] AC-4.1: System detects markdown file deletions within 1 second
- [ ] AC-4.2: All vectors associated with deleted file are removed from Qdrant collection
- [ ] AC-4.3: Vector deletion uses file_path_relative filter to identify all chunks from that document
- [ ] AC-4.4: Deletion is logged with filename and count of vectors removed
- [ ] AC-4.5: Failed deletions are logged but do not crash the watcher
- [ ] AC-4.6: Confirmation of successful cleanup is logged (e.g., "Deleted 15 vectors for doc.md")

### US-5: Markdown-Aware Chunking
**As a** system designer optimizing retrieval quality
**I want** documents split by semantic sections using markdown structure
**So that** each vector chunk represents coherent technical content

**Acceptance Criteria:**
- [ ] AC-5.1: Chunking respects markdown heading hierarchy (#, ##, ###) as primary split points
- [ ] AC-5.2: Chunk size targets 512 tokens with 15% overlap (77 tokens) between consecutive chunks
- [ ] AC-5.3: Each chunk preserves parent heading context in metadata (section path: "Guide > Installation > Requirements")
- [ ] AC-5.4: Chunks maintain markdown formatting (code blocks, lists) for full document reconstruction
- [ ] AC-5.5: Edge case handling: files without headings use paragraph-level splitting, section_path set to filename
- [ ] AC-5.6: Chunk metadata includes chunk index, start position, and heading level

### US-6: TEI Embedding Generation
**As a** system integrator connecting to external embedding service
**I want** reliable 1024-dimensional embeddings from TEI with Qwen3-Embedding-0.6B
**So that** vector representations are consistent and high-quality

**Acceptance Criteria:**
- [ ] AC-6.1: System connects to configurable TEI /embed endpoint (default: http://crawl4r-embeddings:80)
- [ ] AC-6.2: All embedding requests specify 1024 dimensions explicitly in API payload
- [ ] AC-6.3: Embedding responses are validated to confirm exactly 1024 dimensions
- [ ] AC-6.4: Batch embedding requests group multiple chunks to minimize API calls (max 32 texts per request)
- [ ] AC-6.5: TEI connection failures trigger exponential backoff retry (3 attempts: 1s, 2s, 4s delays)
- [ ] AC-6.6: Sustained TEI unavailability queues documents for later processing with circuit breaker pattern

### US-7: Qdrant Vector Storage
**As a** system architect managing vector persistence
**I want** reliable storage of embeddings with rich metadata in Qdrant
**So that** vectors support filtering, full document retrieval, and lifecycle management

**Acceptance Criteria:**
- [ ] AC-7.1: Vectors stored with cosine similarity distance metric in collection "crawl4r"
- [ ] AC-7.2: Each vector point includes payload: `{file_path_relative, file_path_absolute, filename, modification_date, chunk_index, section_path, heading_level, chunk_text, tags}` (tags optional from frontmatter)
- [ ] AC-7.3: Collection auto-created on first run with 1024-dimensional vector configuration
- [ ] AC-7.4: Upsert operations use deterministic IDs (SHA256 hash of file_path_relative:chunk_index converted to UUID) for idempotency
- [ ] AC-7.5: Failed upserts retry with exponential backoff (3 attempts) before logging error
- [ ] AC-7.6: Bulk upsert operations batch 100 points per request for performance

### US-8: Metadata Filtering and Search
**As a** query system developer
**I want** comprehensive metadata stored with vectors
**So that** I can filter by filename, date, tags, and retrieve full documents

**Acceptance Criteria:**
- [ ] AC-8.1: Metadata includes all required fields: `file_path_relative` (relative to WATCH_FOLDER), `file_path_absolute` (full system path), `filename`, `modification_date` (ISO 8601 timestamp)
- [ ] AC-8.2: Metadata supports optional `tags` field (array of strings) extracted from frontmatter if present
- [ ] AC-8.3: `chunk_text` field stores full chunk content enabling document reconstruction
- [ ] AC-8.4: Qdrant payload filters can match exact filename or file_path prefix for folder-level queries
- [ ] AC-8.5: Date range filtering supported through modification_date timestamps
- [ ] AC-8.6: Specific metadata fields indexed in Qdrant: file_path_relative (keyword), filename (keyword), modification_date (datetime), tags (keyword array)

### US-9: Quality Verification
**As a** system operator ensuring correctness
**I want** automated quality checks on embeddings and storage
**So that** I can detect configuration issues or data corruption early

**Acceptance Criteria:**
- [ ] AC-9.1: On startup, system validates TEI connection and requests test embedding to verify 1024 dimensions (retry 3 times with 5s, 10s, 20s delays, exit on failure)
- [ ] AC-9.2: On startup, system validates Qdrant connection and collection configuration matches expected 1024-dim setup (retry 3 times with 5s, 10s, 20s delays, exit on failure)
- [ ] AC-9.3: During processing, random 5% sample of embeddings are checked for L2 normalization (L2 norm = 1.0 ± 0.01 tolerance)
- [ ] AC-9.4: Dimension validation runs on every embedding response before storage
- [ ] AC-9.5: Quality check failures are logged with severity levels (WARNING for normalization, ERROR for dimension mismatch)
- [ ] AC-9.6: Post-ingestion verification counts total vectors in Qdrant and compares against expected chunk count

### US-10: Async Non-Blocking Processing
**As a** system architect managing scalability
**I want** async processing with queuing to prevent resource exhaustion
**So that** thousands of files can be processed without blocking or crashes

**Acceptance Criteria:**
- [ ] AC-10.1: File watcher runs in dedicated thread/process separate from document processing
- [ ] AC-10.2: Document processing uses asyncio for concurrent chunk embedding and storage
- [ ] AC-10.3: Processing queue has configurable max size (default: 1000) with backpressure handling
- [ ] AC-10.4: Queue overflow (>1000 items) triggers backpressure - pause watcher until queue drains below 800 items (80%)
- [ ] AC-10.5: Parallel processing limited to configurable max concurrency (default: 10 documents)
- [ ] AC-10.6: System remains responsive during heavy processing with queue depth logged periodically

### US-11: Error Handling and Recovery
**As a** DevOps engineer maintaining the pipeline
**I want** comprehensive error handling with retries and logging
**So that** transient failures self-recover and persistent issues are visible

**Acceptance Criteria:**
- [ ] AC-11.1: All errors are logged with human-readable format including: timestamp, level, component, message, traceback
- [ ] AC-11.2: Transient errors (network, timeout) trigger automatic retry with exponential backoff
- [ ] AC-11.3: Persistent errors (file corruption, API quota) are logged to failed_documents.jsonl and skipped after 3 retries (1s, 2s, 4s delays)
- [ ] AC-11.4: Failed documents are recorded in error log file with details for manual review
- [ ] AC-11.5: TEI/Qdrant service outages trigger circuit breaker to prevent cascading failures
- [ ] AC-11.6: System restarts resume by querying Qdrant for existing file_paths with modification_date, skip files where Qdrant mod_date >= filesystem mod_date

### US-12: Configuration Management
**As a** system administrator deploying to different environments
**I want** externalized configuration via environment variables and files
**So that** I can customize behavior without code changes

**Acceptance Criteria:**
- [ ] AC-12.1: All critical settings configurable via environment variables: `WATCH_FOLDER` (required, no default), `TEI_ENDPOINT` (default: http://crawl4r-embeddings:80), `QDRANT_URL` (default: http://crawl4r-vectors:6333), `COLLECTION_NAME` (default: crawl4r)
- [ ] AC-12.2: Chunking parameters configurable: `CHUNK_SIZE_TOKENS` (default: 512), `CHUNK_OVERLAP_PERCENT` (default: 15)
- [ ] AC-12.3: Performance parameters configurable: `MAX_CONCURRENT_DOCS` (default: 10), `QUEUE_MAX_SIZE` (default: 1000), `BATCH_SIZE` (default: 32)
- [ ] AC-12.4: Configuration validation on startup with clear error messages for invalid values
- [ ] AC-12.5: Support for `.env` file loading for local development
- [ ] AC-12.6: Configuration values logged on startup (with sensitive values redacted)

## Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | File system monitoring for markdown files in specified folder with recursive subdirectory support | P0 | Watchdog detects .md file events (create/modify/delete) within 1 second, recursive monitoring enabled |
| FR-2 | Debouncing mechanism to prevent duplicate processing of rapid file events | P0 | Events for same file within 1-second window are deduplicated, only final event processed |
| FR-3 | Batch processing of existing files on startup before entering watch mode | P0 | All existing .md files processed on startup with progress logging, system waits for completion before watching |
| FR-4 | Markdown-aware document chunking with configurable size and overlap | P0 | Documents split by headings with 512-token chunks (configurable), 15% overlap (77 tokens), metadata includes section context |
| FR-5 | TEI integration for embedding generation with 1024-dimensional output | P0 | HTTP client connects to TEI endpoint, requests specify 1024 dimensions, responses validated |
| FR-6 | Qdrant vector storage with metadata payload for filtering and retrieval | P0 | Vectors stored with file_path, filename, modification_date, chunk_index, section_path, chunk_text in payload |
| FR-7 | Automatic vector deletion for modified/deleted files before re-ingestion | P0 | Qdrant delete operations filter by file_path, all associated vectors removed before new ingestion |
| FR-8 | Async processing pipeline with queue management and backpressure handling | P0 | File events queued, processed asynchronously with configurable concurrency, queue overflow handled gracefully |
| FR-9 | Retry logic with exponential backoff for TEI and Qdrant transient failures | P0 | Failed operations retry up to 3 times with increasing delays (1s, 2s, 4s), permanent failures logged |
| FR-10 | Structured logging with levels (DEBUG/INFO/WARNING/ERROR) to stdout/file | P0 | All operations logged with human-readable format for development, rotating file logs (100MB, 5 backups), timestamps in ISO 8601 |
| FR-11 | Dimension validation for embeddings before storage to detect configuration drift | P1 | Every embedding response checked for exactly 1024 dimensions, mismatches logged as errors and skipped |
| FR-12 | Circuit breaker pattern for TEI/Qdrant service outages to prevent cascading failures | P1 | After 5 consecutive failures, circuit opens for 60 seconds, events queued for retry when circuit closes, status logged |
| FR-13 | Frontmatter parsing to extract optional tags from markdown files, skip invalid YAML gracefully | P1 | YAML frontmatter at document start parsed for tags field (array), invalid YAML skipped with tags=null, stored in Qdrant payload |
| FR-14 | Deterministic vector point IDs based on file path and chunk index for idempotency | P1 | Point ID = SHA256(file_path_relative:chunk_index) converted to UUID, enables safe re-ingestion without duplicates |
| ~~FR-15~~ | ~~Health check HTTP endpoint~~ | ~~P2~~ | **OUT OF SCOPE** - Deferred to future iteration |
| FR-16 | Sampling-based quality verification (5% of embeddings checked for normalization) | P2 | Random sample checked for L2 norm ≈ 1.0, warnings logged for anomalies, does not block processing |
| FR-17 | Post-ingestion vector count verification against expected chunk count | P2 | After batch processing, query Qdrant for total vectors, compare to chunking output, log discrepancies |
| FR-18 | Configuration file support (YAML/JSON) in addition to environment variables | P2 | Optional config file path via CONFIG_FILE env var, merged with env vars (env vars take precedence) |
| FR-19 | Intelligent restart resumption to skip already-processed files | P0 | Query Qdrant on startup for all unique file_paths with latest modification_date, compare with filesystem, skip files where Qdrant mod_date >= file system mod_date, process only new or modified files, log skipped files |
| FR-20 | Failed document logging with structured error details | P1 | Write failed documents to failed_documents.jsonl (one JSON object per line) with schema: {file_path, timestamp, error_type, error_message, traceback, retry_count}, log file location configurable via FAILED_DOCS_LOG env var (default: ./failed_documents.jsonl) |

## Non-Functional Requirements

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-1 | Processing throughput | Documents/minute | 50-100 docs/min on RTX 3050 GPU (dev), 100-200 docs/min on RTX 4070 (prod) |
| NFR-2 | Ingestion latency | Time from file creation to vector storage | < 5 seconds for single-page documents (< 2000 tokens) |
| NFR-3 | Scalability | Maximum supported files in watch folder | 10,000+ files without performance degradation |
| NFR-4 | Vector storage capacity | Total vectors in Qdrant | Up to 1,000,000 vectors (consistent with user deployment spec) |
| NFR-5 | Memory footprint | RAM usage during steady state | < 4 GB for watcher + processing queue, excludes TEI/Qdrant services |
| NFR-6 | Error recovery time | Time to resume after TEI/Qdrant service restart | < 10 seconds to detect and reconnect, automatic retry of queued items |
| NFR-7 | Startup time | Time from launch to ready state | < 60 seconds for initial validation and setup (worst case with retries, excludes batch processing) |
| NFR-8 | Batch processing efficiency | Files processed per API call | 10-50 documents per TEI batch request where possible (respecting API limits) |
| NFR-9 | Log file size management | Maximum log file size | 100 MB per log file with rotation to prevent disk exhaustion |
| NFR-10 | Reliability | Uptime during normal operation | 99.9% uptime (excluding external service failures) |
| NFR-11 | Resource utilization | CPU usage during processing | < 80% sustained CPU to maintain system responsiveness |
| NFR-12 | Data consistency | Vector-document synchronization accuracy | 100% of stored vectors traceable to source file via metadata |

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Ingestion Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌─────────────────┐                 │
│  │ File Watcher │─────>│ Processing Queue │                 │
│  │  (Watchdog)  │      │   (asyncio)     │                 │
│  └──────────────┘      └────────┬────────┘                 │
│         │                        │                           │
│         │                        v                           │
│  ┌──────▼─────────────────────────────────┐                │
│  │     Document Processor                 │                │
│  │  (LlamaIndex + Custom Chunking)        │                │
│  └────────────────┬───────────────────────┘                │
│                   │                                          │
│                   v                                          │
│  ┌────────────────────────────────────────┐                │
│  │    Embedding Generator (TEI Client)    │                │
│  │    - Batch requests                    │                │
│  │    - Dimension validation              │                │
│  │    - Retry with backoff                │                │
│  └────────────────┬───────────────────────┘                │
│                   │                                          │
│                   v                                          │
│  ┌────────────────────────────────────────┐                │
│  │    Vector Store Manager (Qdrant)       │                │
│  │    - Bulk upsert                       │                │
│  │    - Metadata storage                  │                │
│  │    - Lifecycle management              │                │
│  └────────────────────────────────────────┘                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
         │                                      │
         v                                      v
┌──────────────────┐              ┌─────────────────────┐
│   TEI Service    │              │  Qdrant Database    │
│  (Docker)        │              │   (Docker)          │
│  Port: 8080      │              │   Port: 6333        │
└──────────────────┘              └─────────────────────┘
```

### Data Flow

1. **Startup**: Batch process existing files → Watch mode
2. **File Event**: Detect → Debounce → Queue → Process
3. **Processing**: Load → Chunk → Embed → Store
4. **Modification**: Detect → Delete old vectors → Re-process
5. **Deletion**: Detect → Remove vectors → Log cleanup

### Module Structure

```
src/rag_ingestion/
├── __init__.py
├── config.py              # Configuration management (env vars, validation)
├── watcher.py             # File system monitoring (Watchdog)
├── processor.py           # Document processing orchestrator
├── chunker.py             # Markdown-aware chunking logic
├── embeddings.py          # TEI client wrapper with retry logic
├── vector_store.py        # Qdrant client wrapper with lifecycle management
├── queue_manager.py       # Async queue with backpressure handling
├── quality.py             # Validation and quality checks
└── logger.py              # Structured logging setup
```

## Glossary

- **Chunk**: A semantically coherent segment of a document, typically 400-600 tokens, used as the unit for embedding generation
- **Debouncing**: Technique to delay processing until a quiet period (1 second) after the last event, preventing duplicate processing
- **Embedding**: A dense vector representation (1024 dimensions) of text that captures semantic meaning
- **MRL (Multi-Representation Learning)**: Qwen3 model capability to generate embeddings at flexible dimensions (32-1024) without retraining
- **Point**: Qdrant's term for a vector entry with associated metadata (payload)
- **TEI (Text Embeddings Inference)**: HuggingFace toolkit for high-performance embedding generation
- **Vector Store**: Database optimized for similarity search over high-dimensional vectors (Qdrant)
- **Upsert**: Combined update/insert operation - creates new entry if missing, updates if exists
- **Circuit Breaker**: Pattern that stops calling failing service temporarily to prevent cascading failures
- **Backpressure**: System's ability to slow input when processing capacity is exceeded

## Out of Scope

- Query/retrieval API - separate specification required
- User authentication or authorization for file access
- Distributed deployment across multiple machines (single-node deployment only)
- Real-time streaming of embeddings to consumers
- Version history tracking for documents (only current version stored)
- Advanced metadata extraction beyond frontmatter tags (no NLP entity extraction)
- Custom embedding models beyond Qwen3-Embedding-0.6B
- Non-markdown file formats (PDF, DOCX, HTML) - markdown only
- Multi-language support for chunking (English-optimized chunking logic)
- Web UI for monitoring or management - CLI/logs only
- Backup and disaster recovery mechanisms for Qdrant
- Integration with external document management systems
- Health check HTTP endpoint (FR-15) - deferred to future iteration
- JSON structured logging - using human-readable format for development (can add later)

## Dependencies

### External Services
- **TEI Service**: Must be running and accessible at configured endpoint before pipeline starts
- **Qdrant Database**: Must be running and accessible at configured endpoint before pipeline starts
- **GPU Availability**: RTX 3050 (dev) or RTX 4070 (prod) required for optimal TEI performance

### Python Libraries
- `llama-index-core >= 0.11.0` - Document processing and orchestration
- `llama-index-vector-stores-qdrant` - Qdrant integration for LlamaIndex
- `qdrant-client` - Direct Qdrant API client
- `watchdog >= 3.0.0` - File system monitoring
- `huggingface-hub` - InferenceClient for TEI API calls
- `pydantic >= 2.0.0` - Configuration validation
- `python-dotenv` - Environment variable loading from .env files

### System Requirements
- **Python Version**: 3.10+
- **Operating System**: Linux (tested), macOS/Windows (should work but not primary target)
- **Disk Space**: 10 GB for model weights (TEI container), 100+ GB for Qdrant data depending on scale
- **Network**: Low-latency connection between pipeline and TEI/Qdrant services (localhost ideal)

## Risks and Mitigation

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| **TEI service becomes unavailable during batch processing** | Medium | High | Implement circuit breaker pattern, queue documents for retry when service recovers, log failed items for manual review |
| **Qdrant storage exhaustion with 1M+ vectors** | Low | High | Monitor disk usage, implement collection cleanup policies, document storage capacity planning |
| **Chunking strategy produces poor retrieval quality** | Medium | Medium | Make chunking parameters configurable, implement A/B testing framework for comparing strategies, collect retrieval metrics |
| **File watcher misses events under high load** | Low | High | Use reliable watchdog library, implement reconciliation scan comparing filesystem vs. Qdrant, add queue depth monitoring |
| **Dimension mismatch between TEI configuration and expectations** | Low | Critical | Validate embeddings on startup, check every response, fail fast with clear error messages |
| **Memory exhaustion with large queue during bulk ingestion** | Medium | Medium | Implement queue size limits (1000 items), add backpressure to slow file scanning, monitor memory usage |
| **Editor auto-save triggers excessive re-processing** | High | Low | Debounce file events with 1-second threshold, track file modification timestamps to detect actual changes |
| **Inconsistent state after crash during re-ingestion** | Medium | Medium | Use deterministic point IDs for idempotent operations, implement transaction logging, add recovery verification on startup |

## Success Criteria

The RAG ingestion pipeline will be considered successful when:

1. **Functional Completeness**:
   - Processes 1000+ existing markdown files on startup without errors
   - Automatically ingests new files within 5 seconds of creation
   - Successfully updates vectors for modified files with old vector cleanup
   - Removes vectors when files are deleted

2. **Performance Targets**:
   - Achieves 50+ documents/minute throughput on RTX 3050 GPU
   - Maintains < 5 second latency for single-page document ingestion
   - Supports 10,000+ files in watch folder without degradation

3. **Quality Assurance**:
   - 100% of embeddings are exactly 1024 dimensions
   - All vectors in Qdrant have complete metadata (file_path, modification_date, chunk_text)
   - No duplicate vectors for the same document chunk

4. **Reliability**:
   - 99.9% uptime during 24-hour test run (excluding external service failures)
   - Automatic recovery from TEI/Qdrant service restarts within 10 seconds
   - Zero data loss during normal operation (all file events processed)

5. **Operational Readiness**:
   - Comprehensive structured logs enable debugging of any issue
   - Configuration via environment variables works across dev/prod environments
   - System can be started, stopped, and restarted without manual intervention

6. **Verification**:
   - Integration test suite validates end-to-end pipeline (file → vectors)
   - Quality checks run on every batch showing 100% dimension compliance
   - Post-ingestion verification confirms vector count matches expected chunks
