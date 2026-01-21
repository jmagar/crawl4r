---
spec: language-filter
phase: requirements
created: 2026-01-20T17:45:00Z
---

# Requirements: Language Filtering for Web Crawling

## Goal
Filter web-crawled documents by language during ingestion, defaulting to English only with configurable multi-language support. Use fast-langdetect for 95% accuracy at minimal overhead (~1-2ms per document).

## User Decisions

Based on interview responses:
- **Primary users:** Both developers (Crawl4AIReader API) and end users (MCP server interface)
- **Priority tradeoffs:** Feature completeness + code quality/maintainability
- **Scope:** Comprehensive edge case handling, clean code structure, thorough tests, clear docs

## User Stories

### US-1: Default English-Only Filtering
**As a** developer integrating Crawl4AIReader
**I want** documents filtered to English only by default
**So that** I don't accidentally ingest unwanted languages

**Acceptance Criteria:**
- [ ] AC-1.1: `Crawl4AIReader()` without config filters to English only (`allowed_languages=["en"]`)
- [ ] AC-1.2: Documents with detected language != "en" are filtered out before return
- [ ] AC-1.3: Filtered documents logged with structured fields (URL, detected language, confidence)
- [ ] AC-1.4: English documents pass through with `language="en"` and `language_confidence` in metadata

### US-2: Multi-Language Configuration
**As a** developer building multi-language RAG
**I want to** configure allowed languages explicitly
**So that** I can accept Spanish, French, or other specific languages

**Acceptance Criteria:**
- [ ] AC-2.1: `allowed_languages` config accepts list of ISO 639-1 codes (e.g., `["en", "es", "fr"]`)
- [ ] AC-2.2: Documents with detected language in allowed list pass through
- [ ] AC-2.3: Documents with detected language not in allowed list are filtered out
- [ ] AC-2.4: Empty list `allowed_languages=[]` accepts all languages (disable filter)

### US-3: Confidence Threshold Tuning
**As a** developer
**I want** adjustable confidence threshold
**So that** I can filter low-confidence detections

**Acceptance Criteria:**
- [ ] AC-3.1: `min_language_confidence` config accepts float 0.0-1.0 (default: 0.80)
- [ ] AC-3.2: Documents with confidence < threshold filtered regardless of detected language
- [ ] AC-3.3: Documents with confidence >= threshold pass language check
- [ ] AC-3.4: Filtered low-confidence documents logged with URL, language, confidence, reason

### US-4: Language Metadata Enrichment
**As a** developer
**I want** language fields in document metadata
**So that** I can filter/search by language in Qdrant

**Acceptance Criteria:**
- [ ] AC-4.1: All returned documents have `metadata["language"]` (ISO 639-1 code string)
- [ ] AC-4.2: All returned documents have `metadata["language_confidence"]` (float 0.0-1.0)
- [ ] AC-4.3: Metadata fields stored in Qdrant payload for filtering/search
- [ ] AC-4.4: CrawlResult dataclass updated with `language` and `language_confidence` fields

### US-5: Edge Case Handling
**As a** developer
**I want** robust handling of edge cases
**So that** pipeline doesn't break on unusual content

**Acceptance Criteria:**
- [ ] AC-5.1: Empty/short text (< 50 chars) skips detection, accepts document, logs skip reason
- [ ] AC-5.2: Multi-language pages detect primary language (highest confidence)
- [ ] AC-5.3: Detection failure (exception) logs error, accepts document (fail-open), sets `language="unknown"`
- [ ] AC-5.4: Code-heavy pages use confidence threshold to filter false positives

### US-6: Opt-Out for Testing
**As a** developer
**I want** ability to disable language filtering
**So that** I can test without restrictions

**Acceptance Criteria:**
- [ ] AC-6.1: `enable_language_filter=False` config disables filtering entirely
- [ ] AC-6.2: Disabled filter still detects and enriches metadata (language/confidence fields)
- [ ] AC-6.3: Disabled filter logs "filtering disabled" message on first crawl

## Functional Requirements

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| FR-1 | Integrate fast-langdetect library (thread-safe, 95% accuracy) | P0 | Passes `ruff check`, `ty check`, no new dependencies conflicts |
| FR-2 | Add `allowed_languages`, `min_language_confidence`, `enable_language_filter` to Crawl4AIReaderConfig | P0 | Config validation passes, defaults match spec (en-only, 0.80, True) |
| FR-3 | Detect language after crawling, before return (post-filter strategy) | P0 | Detection runs in `_aload_batch()` between line 648 and return |
| FR-4 | Filter documents by allowed_languages and confidence threshold | P0 | Filtered docs excluded from return, logged with structured fields |
| FR-5 | Enrich all documents with language/language_confidence metadata | P0 | Fields present in Document.metadata for all returned docs |
| FR-6 | Update CrawlResult dataclass with language fields | P1 | Fields optional (None for old crawls), populated for new crawls |
| FR-7 | Store language metadata in Qdrant for search/filtering | P1 | Payload includes language fields, queryable via filter |
| FR-8 | Log filtered documents with URL, detected language, confidence, reason | P0 | Structured logging with extra={} fields for monitoring |
| FR-9 | Handle empty/short text by skipping detection | P1 | Text < 50 chars skips detection, accepts doc, logs skip |
| FR-10 | Handle detection failures by fail-open (accept document) | P0 | Exception caught, logged, document accepted with `language="unknown"` |

## Non-Functional Requirements

| ID | Requirement | Metric | Target |
|----|-------------|--------|--------|
| NFR-1 | Performance | Detection overhead | < 5ms per document (p95) |
| NFR-2 | Memory | Library footprint | < 100KB loaded model |
| NFR-3 | Accuracy | Detection correctness | 95%+ on web text |
| NFR-4 | Maintainability | Code complexity | < 10 cyclomatic complexity per function |
| NFR-5 | Test Coverage | Unit + integration tests | 85%+ coverage for new code |
| NFR-6 | Thread Safety | Async compatibility | No blocking in event loop (< 5ms sync work) |
| NFR-7 | Backward Compatibility | Existing tests | Zero breaking changes to current API |

## Glossary

- **ISO 639-1 codes**: Two-letter language codes (e.g., "en" for English, "es" for Spanish)
- **Post-filter**: Language detection after crawling, before chunking/ingestion
- **Pre-filter**: Language detection before crawling (not used in this spec)
- **Fail-open**: Accept document on detection failure (prevents pipeline breakage)
- **Confidence threshold**: Minimum probability (0.0-1.0) required to accept detection result
- **Primary language**: Language with highest confidence in multi-language content
- **fast-langdetect**: Python library providing fast, thread-safe language detection (80x faster than langdetect)

## Out of Scope

- Language detection for file-based ingestion (separate spec for file_watcher.py)
- Pre-filtering (detecting language before crawl via URL heuristics)
- Multi-language document handling (detecting/tagging all languages in page)
- Upstream language detection in Crawl4AI service (future feature request)
- Storing rejected documents for review (logs only, no persistence)
- Custom language models or training (use fast-langdetect defaults)
- Language translation or content modification

## Dependencies

**External:**
- fast-langdetect>=0.4.0 (PyPI package, adds ~50KB to dependencies)
- Python 3.9+ (fast-langdetect requirement)

**Internal:**
- crawl4r.readers.crawl4ai (integration point for filtering logic)
- crawl4r.readers.crawl.models (CrawlResult dataclass modification)
- crawl4r.storage.qdrant (metadata storage for language fields)
- crawl4r.core.logger (structured logging for filtered documents)

**Specification Dependencies:**
- llamaindex-crawl4ai-reader (COMPLETE) - language filter extends this implementation
- web-crawl-cli (TASKS DEFINED) - CLI may expose language config in future

## Success Criteria

- All 6 user stories pass acceptance criteria (24 ACs total)
- Zero breaking changes to existing 786 Crawl4AIReader tests
- New language filtering tests achieve 85%+ coverage
- `ruff check .` and `ty check crawl4r/` pass with zero errors
- Performance benchmark: < 5ms p95 detection overhead per document
- Integration test: Crawl 100 multi-language URLs, verify filtering accuracy 95%+
- Documentation: README updated with language filtering examples
- Monitoring: Structured logs enable language filtering analytics

## Unresolved Questions

1. **Should confidence threshold be URL-specific or global?**
   - Decision: Global threshold in config (simplicity), can be overridden per reader instance
   - Rationale: Rare need for per-URL tuning, adds complexity without clear benefit

2. **How to handle language=None for detection failures?**
   - Decision: Use `language="unknown"` string instead of None for consistency
   - Rationale: Simplifies Qdrant filtering (string field always present)

3. **Should we expose language stats in batch crawl logs?**
   - Decision: Yes, log language distribution in batch summary (en: 45, es: 5, filtered: 3)
   - Rationale: Helps diagnose filtering behavior and tune confidence threshold

## Next Steps

1. Update `pyproject.toml` to add `fast-langdetect>=0.4.0` dependency
2. Create `LanguageDetector` component in `crawl4r/processing/language.py`
3. Extend `Crawl4AIReaderConfig` with language filtering fields
4. Integrate filtering logic into `_aload_batch()` method
5. Update `CrawlResult` dataclass with optional language fields
6. Write unit tests for LanguageDetector component
7. Write integration tests for Crawl4AIReader language filtering
8. Update CLAUDE.md with language filtering examples
9. Set awaitingApproval state for user review
