---
spec: rag-ingestion
type: decisions
created: 2026-01-14
status: approved
---

# Specification Decisions

This document records all decisions made during the specification review and clarification process.

## Critical Architecture Decisions

### File Path Storage (Issue #3)
**Decision**: Store BOTH relative and absolute paths in metadata

**Rationale**:
- Relative paths enable portability between dev/prod environments
- Absolute paths simplify file system operations
- Minimal storage overhead for maximum flexibility

**Implementation**:
```json
{
  "file_path_relative": "docs/guide.md",
  "file_path_absolute": "/data/watched_folder/docs/guide.md"
}
```

### Point ID Generation (Issue #6)
**Decision**: Use SHA256 hash converted to UUID format

**Implementation**:
```python
import hashlib
import uuid

point_id = str(uuid.UUID(
    hashlib.sha256(f"{file_path_relative}:{chunk_index}".encode())
    .hexdigest()[:32]
))
```

**Rationale**: Industry standard, good collision resistance, Qdrant UUID compatibility

### TEI API Integration (Issue #11)
**Decision**: Use `/embed` endpoint (not `/v1/embeddings`)

**Rationale**: User preference, native TEI endpoint

**Implementation**: Custom BaseEmbedding class with direct HTTP calls to `/embed`

### Batch Processing Strategy (Issue #1)
**Decision**: Batch documents, embed all chunks per document

**Implementation**:
- Process 10-50 documents at a time
- For each document, send all chunks to TEI in single request (up to 32 chunks)
- If document has >32 chunks, split into multiple TEI requests

**Rationale**: Balances API efficiency with tracking complexity

### LlamaIndex-TEI Integration (Issue #20)
**Decision**: Implement custom BaseEmbedding class

**Implementation**:
```python
from llama_index.core.embeddings import BaseEmbedding
import requests

class TEIEmbedding(BaseEmbedding):
    def __init__(self, endpoint_url: str, dimensions: int = 1024):
        self.endpoint_url = endpoint_url
        self.dimensions = dimensions
        super().__init__()

    def _get_embedding(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.endpoint_url}/embed",
            json={"inputs": text}
        )
        return response.json()
```

**Rationale**: Full control over TEI integration, explicit dimension handling

## State Management Decisions

### Resume on Restart (Issue #5)
**Decision**: Query Qdrant metadata on startup

**Implementation**:
1. On startup, query Qdrant for all unique file_paths with latest modification_date
2. Compare with filesystem: skip files where Qdrant mod_date >= file mod_date
3. Process only new or modified files

**Rationale**: No separate state file needed, single source of truth

### Retry Failure Handling (Issue #10)
**Decision**: Log to error file and skip after 3 failed attempts

**Implementation**:
- Retry with exponential backoff: 1s, 2s, 4s
- After 3 failures, write to `failed_documents.jsonl` with full error details
- Continue processing other documents
- Manual review/reprocessing later

**Rationale**: Resilient to transient failures, doesn't block pipeline, enables manual recovery

### Queue Overflow (Issue #17)
**Decision**: Implement backpressure - pause watcher when queue full

**Implementation**:
- When queue reaches 1000 items, pause watchdog observer
- Resume watching when queue drains below 800 items (80% threshold)
- Log warnings at 90% and 100% queue capacity

**Rationale**: Prevents memory exhaustion, graceful degradation

### Circuit Breaker Behavior (Issue #7)
**Decision**: Queue events for retry when circuit closes

**Implementation**:
- After 5 consecutive TEI/Qdrant failures, open circuit for 60 seconds
- Continue accepting file events, queue them in-memory
- When circuit closes, process queued events
- If queue fills during circuit open, apply backpressure (pause watcher)

**Rationale**: No data loss, automatic recovery from service outages

## Metadata Schema Decisions

### Complete Schema (Issue #4)
**Decision**: Comprehensive metadata with core + structure + optional fields

**Schema**:
```json
{
  "file_path_relative": "string (required)",
  "file_path_absolute": "string (required)",
  "filename": "string (required)",
  "modification_date": "ISO 8601 timestamp (required)",
  "chunk_index": "integer (required)",
  "chunk_text": "string (required)",
  "section_path": "string (required, nullable)",
  "heading_level": "integer 0-6 (required)",
  "tags": "array of strings (optional, from frontmatter)"
}
```

### Headingless Documents (Issue #14)
**Decision**: Use filename as section_path when no headings exist

**Examples**:
- Document with headings: `section_path = "Installation > Prerequisites"`
- Document without headings: `section_path = "README.md"`

**Rationale**: Provides context without null values, simplifies querying

### Frontmatter Parsing (Issue #13)
**Decision**: Skip invalid YAML gracefully, don't extract other fields

**Implementation**:
- Try to parse YAML frontmatter
- If parsing fails, set `tags = null` and continue
- Only extract `tags` field (array of strings)
- Ignore other frontmatter fields (title, author, date)
- Log warnings for invalid frontmatter

**Rationale**: Resilient to malformed files, focused scope

### Qdrant Payload Indexing (Issue #19)
**Decision**: Index file_path, filename, modification_date, tags

**Implementation**:
```python
# Create indexes after collection creation
client.create_payload_index(
    collection_name="crawl4r",
    field_name="file_path_relative",
    field_schema="keyword"
)
client.create_payload_index(
    collection_name="crawl4r",
    field_name="filename",
    field_schema="keyword"
)
client.create_payload_index(
    collection_name="crawl4r",
    field_name="modification_date",
    field_schema="datetime"
)
client.create_payload_index(
    collection_name="crawl4r",
    field_name="tags",
    field_schema="keyword"
)
```

**Rationale**: Enable fast filtering by common query patterns

## Configuration Decisions

### Default Values (Issue #12)
**Decisions**:

| Variable | Default | Rationale |
|----------|---------|-----------|
| `WATCH_FOLDER` | **Required (no default)** | Force explicit configuration to avoid accidents |
| `TEI_ENDPOINT` | `http://crawl4r-embeddings:80` | Docker Compose service name |
| `QDRANT_URL` | `http://crawl4r-vectors:6333` | Docker Compose service name |
| `COLLECTION_NAME` | `llama` | User preference, project name |
| `CHUNK_SIZE_TOKENS` | `512` | Research-backed optimal size |
| `CHUNK_OVERLAP_PERCENT` | `15` | User-specified preference |
| `MAX_CONCURRENT_DOCS` | `10` | Balanced for RTX 3050 GPU |
| `QUEUE_MAX_SIZE` | `1000` | Memory-conscious buffer size |

### Environment-Specific Configuration (Issue #21)
**Decision**: Provide example .env.dev and .env.prod templates

**Implementation**:
- Ship `.env.dev.example` with RTX 3050-optimized settings
- Ship `.env.prod.example` with RTX 4070-optimized settings
- User copies and customizes for their environment

**Example .env.dev**:
```bash
MAX_CONCURRENT_DOCS=8
BATCH_SIZE=20
```

**Example .env.prod**:
```bash
MAX_CONCURRENT_DOCS=15
BATCH_SIZE=40
```

### File Watching Exclusions (Issue #15)
**Decision**: Ignore hidden directories, build folders, don't follow symlinks

**Implementation**:
```python
ignore_patterns = [
    "*/.git/*",
    "*/.*",  # All hidden directories
    "*/__pycache__/*",
    "*/node_modules/*",
    "*/venv/*",
    "*/dist/*",
    "*/build/*"
]

# Don't follow symlinks in watchdog configuration
observer.schedule(handler, path=watch_folder, recursive=True)
# Watchdog doesn't follow symlinks by default
```

**Rationale**: Avoid system files, prevent infinite loops, common-sense exclusions

## Operational Decisions

### Startup Validation Failure (Issue #18)
**Decision**: Retry 3 times with backoff, then exit

**Implementation**:
- Validate TEI: GET /health or test embedding request
- Validate Qdrant: Check connection and collection config
- Retry schedule: 5s, 10s, 20s delays
- After 3 failures, exit with clear error message and troubleshooting hints

**Rationale**: Handle transient startup issues, fail fast on persistent problems

### Logging Configuration (Issue #16)
**Decision**: Human-readable format for development

**Implementation**:
```python
import logging

# Development format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),  # stdout
        logging.handlers.RotatingFileHandler(
            'rag_ingestion.log',
            maxBytes=100*1024*1024,  # 100MB
            backupCount=5
        )
    ]
)
```

**Future**: Add JSON structured logging for production with `LOG_FORMAT=json` env var

**Rationale**: Easier to read during development, file rotation prevents disk exhaustion

### Health Check Endpoint (Issue #8)
**Decision**: Out of scope for initial implementation

**Rationale**: FR-15 is P2 priority, focus on core ingestion pipeline first

## Chunking Decisions

### Overlap Calculation (Issue #22)
**Decision**: Token-based overlap, 15% = 77 tokens

**Implementation**:
```python
chunk_size = 512  # tokens
overlap_percent = 15
overlap_tokens = int(chunk_size * overlap_percent / 100)  # 77 tokens

# Chunk positions:
# Chunk 0: tokens 0-512
# Chunk 1: tokens 435-947 (starts at 512-77)
# Chunk 2: tokens 870-1382 (starts at 947-77)
```

**Rationale**: Precise control over token budgets, consistent across documents

### Fixed Overlap Value (Issue #2)
**Decision**: Fixed at 15% (not 10-20% range)

**Update Required**: Change requirements.md AC-5.2 from "10-20%" to "15%"

**Rationale**: User explicitly specified 15% during initial clarifications

## Quality & Performance Decisions

### Normalization Verification (Issue #9)
**Decision**: Research Qwen3 characteristics first, then implement if needed

**Action Item**: Check Qwen3-Embedding-0.6B model card to determine if embeddings are L2-normalized

**Conditional Implementation**:
- If normalized: Check L2 norm with ±0.1 tolerance
- If not normalized: Skip normalization checks

### Memory Budget (Issue #24)
**Decision**: Revise NFR-5 from <2GB to <4GB

**Rationale**:
- 1000-item queue with documents and embeddings
- LlamaIndex overhead
- 4GB is realistic for production use

### Chunk Text Storage (Issue #25)
**Decision**: Store full chunk_text in Qdrant payload

**Trade-offs Accepted**:
- ✅ Can reconstruct documents from vectors alone
- ✅ Faster query responses (no file I/O)
- ❌ 2-3GB duplication for 1M vectors (acceptable)

**Rationale**: Query convenience and self-contained vectors worth the storage cost

## Out of Scope

The following items were explicitly marked as out of scope:

1. **Health Check HTTP Endpoint** (FR-15, P2) - Defer to future iteration
2. **Query/Retrieval API** - Separate specification required
3. **Web UI** - CLI/logs only for initial implementation
4. **Multi-format Support** - Markdown only
5. **Distributed Deployment** - Single-node only

---

## Summary Statistics

- **Total Issues Resolved**: 30
- **Critical Decisions**: 7
- **Configuration Defaults**: 8
- **Metadata Fields**: 9
- **Out of Scope Items**: 5

## Next Steps

1. ✅ Update requirements.md with all resolved specifications
2. ✅ Update spec-review.md to mark issues as resolved
3. ✅ Update .progress.md with decision summary
4. ⏭️ Proceed to design phase with /ralph-specum:design

---

*Approved: 2026-01-14*
*Status: Complete - Ready for Design Phase*
