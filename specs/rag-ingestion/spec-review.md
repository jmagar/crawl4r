---
spec: rag-ingestion
type: spec-review
created: 2026-01-14
updated: 2026-01-14
status: resolved
---

# Spec Review: RAG Ingestion Pipeline

## Resolution Status

**✅ ALL ISSUES RESOLVED**

All 30 identified issues have been reviewed and decisions made. See `decisions.md` for complete resolution details.

- **Critical Inconsistencies**: 3/3 resolved
- **Significant Ambiguities**: 7/7 resolved
- **Important Gaps**: 16/16 resolved
- **Technical Concerns**: 4/4 resolved

**Next Step**: Update requirements.md with resolved specifications, then proceed to design phase.

---

## Overview

This document identifies inconsistencies, ambiguities, gaps, and technical concerns found during comprehensive review of all specification artifacts (research.md, requirements.md, .progress.md).

---

## 🔴 CRITICAL INCONSISTENCIES

### 1. **Batch Request Confusion - TEI API Limits**

**Issue**: Mixed concepts between chunks vs documents in batch size specifications

- **AC-6.4**: "max 32 texts per request"
- **NFR-8**: "10-50 documents per TEI batch request"
- **AC-1.2**: "batches of 10-50 documents"

**Problem**: These are mixing different concepts (chunks vs documents). A document with 10 chunks would be 10 texts. Which limit applies where?

**Resolution Needed**: Clarify whether batch sizes refer to:
- Documents (which may contain multiple chunks)
- Individual chunks/texts (API request units)
- How to batch documents with variable chunk counts

---

### 2. **Chunk Overlap: Range or Fixed Value?**

**Issue**: Inconsistent overlap specification

- **Research.md**: "10-20% overlap"
- **Requirements AC-5.2**: "10-20% overlap"
- **User clarification**: "15% overlap"
- **.progress.md**: "15% overlap"

**Problem**: Requirements still say "10-20%" but user explicitly specified 15%.

**Resolution**: Update requirements to specify exactly 15% overlap.

---

### 3. **File Path: Absolute vs Relative**

**Issue**: Path storage strategy impacts portability

- **AC-8.1**: "file_path (absolute)"

**Problem**: Absolute paths break when moving between dev (RTX 3050) and prod (RTX 4070) environments. Vectors become invalid if watch folder moves.

**Example**:
- Dev: `/home/user/dev/docs/file.md`
- Prod: `/data/production/docs/file.md`
- Same file, different absolute paths → vectors can't be matched

**Resolution Needed**: Decide on path strategy:
- Option A: Relative paths from WATCH_FOLDER (portable)
- Option B: Absolute paths (simpler but not portable)

---

## 🟡 SIGNIFICANT AMBIGUITIES

### 4. **Metadata Schema Not Definitive**

**Issue**: Different sections list different metadata fields

- **AC-7.2**: `{file_path, filename, modification_date, chunk_index, section_path, heading_level, chunk_text}`
- **AC-8.2**: Also mentions `tags` (optional)

**Missing**: Complete schema with:
- Field names (exact spelling/casing)
- Data types (string, int, datetime, array)
- Required vs optional flags
- Example payload

**Recommended Schema**:
```json
{
  "file_path": "string (required)",
  "filename": "string (required)",
  "modification_date": "ISO 8601 timestamp (required)",
  "chunk_index": "integer (required)",
  "section_path": "string (required, nullable)",
  "heading_level": "integer 0-6 (required, 0 if no heading)",
  "chunk_text": "string (required)",
  "tags": "array of strings (optional)"
}
```

---

### 5. **State Persistence for Resume on Restart**

**Issue**: Resume capability mentioned but mechanism unspecified

- **AC-11.6**: "resume processing from last successful state without re-ingesting"

**Gap**: HOW is this achieved?
- File tracking database?
- Checksums stored somewhere?
- Timestamp comparison?
- Qdrant metadata query?

**Resolution Needed**: Specify persistence mechanism for tracking processed files.

**Options**:
1. Query Qdrant on startup for existing file_paths with latest modification_date
2. Maintain SQLite database of processed files
3. Use file hash checksums stored in local state file
4. Compare filesystem modification times with Qdrant metadata

---

### 6. **Deterministic Point IDs - Hash Function Unspecified**

**Issue**: Hash function not specified

- **AC-7.4**: "hash(file_path + chunk_index)"

**Gaps**:
- Which hash algorithm? (MD5, SHA256, Blake2)
- Output format? Qdrant supports UUID or u64 integers
- Collision handling?

**Recommendation**:
```python
point_id = str(uuid.UUID(hashlib.sha256(f"{file_path}:{chunk_index}".encode()).hexdigest()[:32]))
```

---

### 7. **Circuit Breaker Behavior**

**Issue**: Circuit breaker configuration incomplete

- **FR-12**: "After 5 consecutive failures, circuit opens for 60 seconds"

**Gaps**:
- What happens to incoming file events during those 60 seconds?
  - Queued for later?
  - Dropped entirely?
  - Backpressure to watcher?
- When circuit closes, does it retry immediately or gradually?
- Per-service circuit breakers (TEI vs Qdrant separate) or shared?

---

### 8. **Health Check HTTP Server**

**Issue**: Health endpoint mentioned without implementation details

- **FR-15** (P2): "/health endpoint returns JSON"

**Problems**:
- No mention of HTTP server anywhere else in requirements
- Is this a standalone server?
- What port?
- Who starts/manages it?
- Or is this just a conceptual requirement for future implementation?

**Resolution Needed**: Either remove (out of scope) or specify full HTTP server requirements.

---

### 9. **Normalization Check Tolerance**

**Issue**: Fuzzy comparison without tolerance specification

- **AC-9.3**: "L2 norm ≈ 1.0 if normalized"

**Gaps**:
- What tolerance range? ±0.01? ±0.1?
- Also, ARE Qwen3 embeddings normalized or not?
- What action on failure? Warning only? Block storage?

**Resolution Needed**: Research Qwen3 output characteristics and specify exact tolerance.

---

### 10. **After Max Retries, Then What?**

**Issue**: Post-retry behavior not defined

- **FR-9**: "3 attempts with delays (1s, 2s, 4s)"

**Gap**: After 3 failures, the document is:
- Dropped permanently?
- Queued indefinitely?
- Logged to error file and skipped?
- Moved to dead-letter queue?

**Resolution Needed**: Specify final failure handling.

---

## 🟠 IMPORTANT GAPS

### 11. **TEI Endpoint Choice**

**Issue**: Multiple endpoints mentioned, none specified in requirements

- **Research**: Mentions `/embed` and `/v1/embeddings`
- **Gap**: Requirements don't specify which to use

**Recommendation**: Use `/v1/embeddings` because:
- OpenAI-compatible
- Supports `dimensions` parameter explicitly
- May work with existing LlamaIndex OpenAI embedding class

---

### 12. **Default Configuration Values**

**Issue**: Missing defaults for required environment variables

**AC-12.1** lists variables but not defaults:
- `WATCH_FOLDER` → Missing (research suggests `./data/watched_folder`)
- `TEI_ENDPOINT` → Missing (should be `http://localhost:8080`)
- `QDRANT_URL` → Missing (should be `http://localhost:6333`)
- `COLLECTION_NAME` → Missing (AC-7.1 uses "markdown_embeddings" but not in config)
- `CHUNK_SIZE_TOKENS` → Missing (should be 512)
- `CHUNK_OVERLAP_PERCENT` → Missing (should be 15)
- `MAX_CONCURRENT_DOCS` → Missing (mentioned in AC-10.5 as 10)
- `QUEUE_MAX_SIZE` → Missing (mentioned in AC-10.3 as 1000)
- `BATCH_SIZE` → Missing (need to clarify documents vs chunks)

**Resolution Needed**: Document all defaults in requirements.

---

### 13. **Frontmatter Parsing Edge Cases**

**Issue**: FR-13 specifies parsing YAML frontmatter but lacks detail

**Gaps**:
- What if frontmatter exists but has no `tags` field?
- What if frontmatter is invalid YAML?
- Should we store other frontmatter fields (title, author, date)?
- What defines "document start"? First N bytes?
- What if `tags` field is not an array?

**Example Edge Cases**:
```markdown
---
title: My Doc
tags: "single-string-not-array"
---
```

```markdown
---
invalid: yaml: syntax:
---
```

**Resolution Needed**: Define parsing behavior for edge cases.

---

### 14. **Section Path for Headingless Documents**

**Issue**: Section path generation not specified for edge cases

- **AC-5.3**: Section path like "Guide > Installation > Requirements"
- **AC-5.5**: "files without headings use paragraph-level splitting"

**Gaps**:
- What is `section_path` value if document has no headings?
  - Empty string `""`?
  - `null`?
  - `"Untitled"`?
  - Filename?
- How deep does hierarchy go? All levels or capped at 3-4?
- What separator? `>` or `/` or `.`?

---

### 15. **Recursive Monitoring Depth & Exclusions**

**Issue**: Recursive monitoring behavior not fully specified

- **AC-2.2**: "Recursive monitoring captures subdirectories"

**Gaps**:
- Max depth limit? Unlimited?
- Ignore hidden directories (`.git`, `.vscode`, `__pycache__`)?
- Follow symlinks or not?
- Ignore patterns (e.g., `node_modules/`, `venv/`)?

**Resolution Needed**: Specify file watching exclusions and depth.

---

### 16. **Log Configuration**

**Issue**: Logging configuration incomplete

- **NFR-9**: "100 MB per log file with rotation"
- **FR-10**: "to stdout/file"
- **AC-11.4**: "Failed documents are recorded in error log file"

**Gaps**:
- Log file path?
- How many rotated files to keep (e.g., app.log.1, app.log.2)?
- Separate error log file or unified?
- Naming scheme for rotated logs?
- stdout AND file simultaneously, or configurable?

---

### 17. **Queue Behavior on Overflow**

**Issue**: Queue overflow handling not specified

- **AC-10.4**: "Queue overflow triggers warning but does not crash"

**Gaps**:
- What DOES happen when queue is full?
  - Drop new events?
  - Pause watcher until space available (backpressure)?
  - Block file event handler?
- Queue ordering? FIFO? Priority-based (deletions first)?

**Recommendation**: Implement backpressure - pause watcher when queue full.

---

### 18. **Startup Validation Failure Handling**

**Issue**: Failure modes during startup checks not defined

- **AC-9.1**: "Validate TEI/Qdrant on startup"

**Gap**: If validation fails:
- Does system exit immediately?
- Retry with backoff?
- Wait indefinitely for services to come online?
- Startup timeout?

**Resolution Needed**: Define startup failure behavior.

---

### 19. **Qdrant Payload Indexing**

**Issue**: Payload indexing requirement without implementation detail

- **AC-8.6**: "All metadata fields are indexed in Qdrant"

**Problem**: Qdrant doesn't auto-index payload fields. Requires explicit configuration.

**Gap**: Which fields need indexing? How to specify?

**Example Required**:
```python
client.create_payload_index(
    collection_name="markdown_embeddings",
    field_name="file_path",
    field_schema="keyword"
)
```

**Resolution Needed**: List fields to index and index types.

---

### 20. **LlamaIndex TEI Integration Approach**

**Issue**: Two approaches mentioned in research, none chosen in requirements

**Research Options**:
1. Custom `BaseEmbedding` subclass with `InferenceClient`
2. Use OpenAI-compatible endpoint with existing LlamaIndex `OpenAIEmbedding` class

**Gap**: Requirements don't specify which approach to implement.

**Recommendation**: Option 2 (OpenAI-compatible) is simpler and leverages existing code.

---

### 21. **Environment Detection**

**Issue**: Dev vs Prod environment distinction mentioned but not implemented

- User specified RTX 3050 (dev) vs RTX 4070 (prod)

**Gap**: How does system know which environment it's in?
- Different config files?
- Environment variable `ENV=dev|prod`?
- Different batch sizes per environment?

**Resolution Needed**: Specify if environment-specific configuration is needed.

---

### 22. **Chunk Overlap Mechanics**

**Issue**: Overlap implementation details not specified

- **AC-5.2**: "10-20% overlap" (should be 15%)

**Gap**: Overlap of what unit?
- If chunk is 512 tokens:
  - 15% = 77 tokens
  - Next chunk starts at token 435?
- Or overlap in characters?
- How handled at document boundaries?

**Resolution Needed**: Specify exact overlap calculation.

---

## 🔵 TECHNICAL CONCERNS

### 23. **Performance Targets May Be Optimistic**

**Issue**: Throughput targets may not be achievable

- **NFR-1**: 50-100 docs/min on RTX 3050

**Concern**:
- 50 docs/min = 0.83 docs/sec = 1.2 sec/doc
- 100 docs/min = 1.67 docs/sec = 0.6 sec/doc
- With chunking, embedding, and storage, this is aggressive
- Need validation that TEI batch requests can achieve this

**Recommendation**: Add performance testing to design phase.

---

### 24. **Memory Target May Be Too Low**

**Issue**: Memory budget may be insufficient

- **NFR-5**: "< 2 GB for watcher + processing queue"

**Concern**:
- 1000-item queue
- Each item: document chunks + metadata + embeddings
- 1000 docs × 10 chunks × (512 tokens + 1024-dim float32 embedding)
- = ~1000 × 10 × (2KB + 4KB) = ~60MB just for embeddings
- But documents themselves could be large
- LlamaIndex overhead
- Python interpreter overhead

**Recommendation**: Revise to 4 GB or make configurable.

---

### 25. **Chunk Text Storage Size**

**Issue**: Storing full chunk text in every vector may be expensive

- Storing full `chunk_text` (400-600 tokens ≈ 2-3KB) in every vector payload
- With 1M vectors: 2-3 GB of text storage in Qdrant
- Duplicated storage (source files + vector payloads)

**Question**: Is this acceptable?

**Alternatives**:
- Store only chunk offset/length, reconstruct from source file
- Store chunks separately, reference by ID
- Accept duplication for query convenience

**Resolution Needed**: Confirm storage approach acceptable.

---

### 26. **Missing Error Scenarios**

**Issue**: Requirements don't cover several failure modes

**Missing Error Scenarios**:
- Qdrant collection exists with wrong dimensions (1536 instead of 1024)
- TEI model doesn't support 1024 dimensions
- Watch folder doesn't exist or isn't readable
- Disk space exhaustion during processing
- File permissions errors (can't read .md file)
- Malformed markdown causes chunking failure
- Extremely large files (100MB markdown)
- Binary files with .md extension

**Resolution Needed**: Add error handling requirements for these scenarios.

---

## 📋 RECOMMENDED ACTIONS

### Immediate Clarifications Needed

Priority order for resolution:

#### P0 - Blocking Issues (Must Resolve Before Design)

1. ✅ **Fix overlap to 15%** in requirements.md AC-5.2
2. ❓ **Decide**: Absolute or relative file paths? (Recommend: relative)
3. ❓ **Specify**: Complete metadata schema with types and required/optional flags
4. ❓ **Specify**: Hash function for point IDs (Recommend: SHA256 → UUID)
5. ❓ **Specify**: TEI endpoint to use (Recommend: `/v1/embeddings`)
6. ❓ **Clarify**: Batch sizes - documents vs chunks (Recommend: batch docs, TEI gets all chunks)
7. ❓ **Define**: All default configuration values

#### P1 - Important Issues (Should Resolve Before Implementation)

8. ❓ **Specify**: State persistence mechanism (Recommend: query Qdrant on startup)
9. ❓ **Clarify**: Queue overflow behavior (Recommend: backpressure to watcher)
10. ❓ **Clarify**: Max retry behavior (Recommend: log to error file and skip)
11. ❓ **Decide**: Custom embedding class vs OpenAI-compatible (Recommend: OpenAI-compatible)
12. ❓ **Define**: Frontmatter edge case handling
13. ❓ **Define**: Section path for headingless documents
14. ❓ **Specify**: Qdrant payload fields to index

#### P2 - Nice to Have (Can Defer)

15. ❓ **Specify**: Recursive monitoring exclusions (hidden dirs, symlinks)
16. ❓ **Define**: Log configuration details (paths, rotation count)
17. ❓ **Clarify**: Circuit breaker queuing behavior
18. ❓ **Decide**: Health check HTTP server scope (in/out of scope)
19. ❓ **Specify**: Startup validation failure handling
20. ❓ **Review**: Performance targets achievability
21. ❓ **Review**: Memory budget adequacy
22. ❓ **Confirm**: Chunk text storage approach acceptable

---

## Summary Statistics

- **Critical Inconsistencies**: 3
- **Significant Ambiguities**: 7
- **Important Gaps**: 16
- **Technical Concerns**: 4

**Total Issues Identified**: 30

**Estimated Resolution Time**: 2-3 hours of clarification discussions + updates to requirements.md

---

## Next Steps

1. Review this document with stakeholders
2. Make decisions on all P0 issues
3. Update requirements.md with resolved specifications
4. Re-run spec review to confirm all issues addressed
5. Proceed to design phase

---

*Generated: 2026-01-14*
*Reviewer: Claude Sonnet 4.5*
*Status: Awaiting resolution*
