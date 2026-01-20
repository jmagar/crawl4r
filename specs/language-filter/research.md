---
spec: language-filter
phase: research
created: 2026-01-20T17:30:00Z
---

# Research: Language Filtering for Crawl4r

## Executive Summary

Language filtering should use **post-filter** strategy with **fast-langdetect** library. Detect language after crawling, filter before chunking/ingestion. Minimal performance impact (~1-2ms per document), high accuracy (95%+), thread-safe for async pipeline.

## External Research

### Python Language Detection Libraries Comparison

| Library | Speed | Accuracy | Async-Safe | Size | Use Case |
|---------|-------|----------|------------|------|----------|
| **fast-langdetect** | **80x faster** | **95%** | ✅ Thread-safe | ~50KB | **Recommended: production** |
| lingua-py | 3-8s (3K texts) | 99.7% (sentences) | ✅ Thread-safe | ~1GB | High accuracy, short text |
| langdetect | 10+ min (3K texts) | Good | ❌ Not thread-safe | ~1MB | Legacy only |
| fasttext | 120K texts/s | High | ⚠️ Manual | ~126MB | DIY integration |
| pycld3 (Google) | Very fast | Lower | ✅ Thread-safe | ~1MB | Speed over accuracy |

**Sources:**
- [fast-langdetect GitHub](https://github.com/LlmKira/fast-langdetect) - 80x faster, 95% accuracy, Python 3.9-3.13
- [lingua-py GitHub](https://github.com/pemistahl/lingua-py) - Most accurate for short text
- [Language Detection Comparison](https://modelpredict.com/language-identification-survey) - Comprehensive benchmark

### Best Practices from Web Crawling

**Pre-filter vs Post-filter Strategy:**

| Strategy | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **Pre-filter** (detect before crawl) | Known language in URL metadata | Saves bandwidth, reduces storage | Misses content, complex |
| **Post-filter** (detect after crawl) | **RAG ingestion pipelines** | **Accurate, simple, flexible** | **Minimal overhead** |

**Post-filter recommended** for Crawl4r because:
1. Already crawled markdown is lightweight (~12KB with `f=fit`)
2. Detection adds only ~1-2ms per document
3. Enables confidence thresholds and fallback logic
4. Supports multi-language documents (detect primary language)

**Sources:**
- [Language Specific Web Crawling](https://www.researchgate.net/publication/267257810_Language_Specific_and_Topic_Focused_Web_Crawling) - Pre-filter reduces storage
- [GPT Crawler Guide](https://scrapfly.io/blog/posts/gpt-crawler-a-complete-guide-to-automated-web-data-collection-for-ai-training) - Quality filtering via language detection
- [Common Crawl Processing](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-6-sourcing-acquiring-massive-text-datasets/utilizing-common-crawl-data) - Deduplication then LID order

## Codebase Analysis

### Current Architecture (Crawl4AIReader)

**Integration Point:** Between crawling and chunking

```python
# crawl4r/readers/crawl4ai.py (line 659-676)
async def aload_data(self, urls: list[str]) -> list[Document]:
    # 1. Health check
    # 2. Deduplication (delete old versions)
    # 3. Crawl URLs → Document objects
    # 4. Return Documents  <-- INSERT LANGUAGE FILTER HERE
```

**Current Metadata Fields:** (`crawl4r/readers/crawl/models.py`)
- `url`, `markdown`, `success`, `title`, `description`
- `status_code`, `error`, `timestamp`
- `internal_links_count`, `external_links_count`
- **Missing:** `language`, `language_confidence`

**Async Patterns in Codebase:**
- `asyncio.gather()` for concurrent operations (line 621)
- `asyncio.Semaphore` for rate limiting (line 610)
- Circuit breaker wraps async calls (line 535)
- All services use `httpx.AsyncClient` (async-first)

**Key Files:**
- `crawl4r/readers/crawl4ai.py` (831 lines) - Main reader, aload_data method
- `crawl4r/readers/crawl/models.py` (39 lines) - CrawlResult dataclass
- `crawl4r/core/metadata.py` (73 lines) - Metadata constants

### Existing Patterns to Follow

**1. Configuration Pattern:**
```python
# crawl4r/readers/crawl4ai.py (line 70-140)
class Crawl4AIReaderConfig(BaseModel):
    base_url: str = Field(default="http://localhost:52004")
    timeout: int = Field(default=30, ge=10, le=300)
    # Add: allowed_languages, min_confidence, default_language
```

**2. Filter Pattern:**
```python
# crawl4r/readers/crawl4ai.py (line 589-600)
# Validate URLs for SSRF protection before crawling
validation_results = [self.validate_url(url) for url in urls]
if not all(validation_results):
    # Skip invalid URLs

# APPLY SAME PATTERN for language filtering:
# Filter documents after crawling, before returning
```

**3. Metadata Enrichment:**
```python
# crawl4r/readers/crawl/metadata_builder.py
class MetadataBuilder:
    def build(self, result: CrawlResult) -> dict[str, Any]:
        # Add language fields to metadata
```

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | **High** | fast-langdetect is thread-safe, pure Python, no numpy |
| Effort Estimate | **S (3-5 tasks)** | Add library, detect language, filter results |
| Risk Level | **Low** | Non-breaking change, post-filter is optional |
| Performance Impact | **Minimal** | ~1-2ms per document, async-compatible |
| Test Coverage | **High** | 786 existing tests, easy to add language tests |

## Recommended Approach

### Architecture: Post-Filter with Confidence Thresholds

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Crawl URLs │ --> │ Detect Lang  │ --> │ Filter Lang │ --> │ Chunk/Ingest │
│ (existing)  │     │ (new: 1-2ms) │     │ (new: 0ms)  │     │ (existing)   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                           ↓
                    Add metadata:
                    - language: "en"
                    - language_confidence: 0.98
```

**Why Post-Filter:**
1. Markdown already extracted and lightweight (~12KB with `f=fit`)
2. Detection faster than network round-trip (~1-2ms vs 50-200ms crawl time)
3. Enables confidence-based decisions (e.g., skip if confidence < 0.80)
4. Supports multi-language pages (detect primary + list secondary)

### Implementation Plan

**Phase 1: Library Integration**
- Add `fast-langdetect>=0.4.0` to pyproject.toml dependencies
- Initialize detector once in Crawl4AIReader.__init__()
- Thread-safe singleton pattern (library handles this internally)

**Phase 2: Language Detection**
- Create `LanguageDetector` component in `crawl4r/processing/`
- Method: `detect(text: str) -> tuple[str, float]` (language code, confidence)
- Use ISO 639-1 codes: "en", "es", "fr", "de", etc.

**Phase 3: Filtering Integration**
- Add fields to `Crawl4AIReaderConfig`:
  - `allowed_languages: list[str] = ["en"]` (default English only)
  - `min_language_confidence: float = 0.80` (confidence threshold)
  - `enable_language_filter: bool = True` (opt-out for testing)
- Filter in `aload_data()` after crawling, before returning
- Log filtered documents with structured logging

**Phase 4: Metadata Enrichment**
- Add `language` and `language_confidence` to `CrawlResult` dataclass
- Update `MetadataBuilder.build()` to include language fields
- Store in Qdrant metadata for filtering/search

### Configuration Design

```python
from crawl4r.readers.crawl4ai import Crawl4AIReader

# Default: English only
reader = Crawl4AIReader()

# Custom: Multiple languages
reader = Crawl4AIReader(
    allowed_languages=["en", "es", "fr"],
    min_language_confidence=0.85,
    enable_language_filter=True
)

# Disable filtering (accept all)
reader = Crawl4AIReader(
    enable_language_filter=False
)
```

## Edge Cases & Solutions

| Edge Case | Solution |
|-----------|----------|
| **Multi-language page** | Detect primary language (highest confidence) |
| **Low confidence (< threshold)** | Skip document, log warning with URL and detected language |
| **Empty/short text (< 50 chars)** | Skip detection, accept document (assume valid) |
| **Language not in allowed list** | Filter out, log with detected language for debugging |
| **Detection failure (exception)** | Log error, accept document (fail-open for robustness) |
| **Code-heavy pages** | Detection works on comments/strings, may detect wrong language → use confidence threshold |

## Performance Considerations

**Benchmark (fast-langdetect):**
- Single document detection: ~1-2ms (Python, no numpy)
- Batch 100 documents: ~100-200ms total (parallelizable)
- Memory overhead: ~50KB model in memory (negligible)

**Async Integration:**
- Detection is CPU-bound (not I/O), runs synchronously
- Wrap in `asyncio.to_thread()` for true non-blocking? **No** - 1-2ms is acceptable blocking time
- Current bottleneck is network (50-200ms per URL), not detection

**Optimization Strategy:**
1. Start: Synchronous detection (simple, 1-2ms acceptable)
2. Later: If profiling shows bottleneck, batch detect with `asyncio.gather()`
3. Future: Pre-compute language in Crawl4AI service (upstream optimization)

## Quality Commands

| Type | Command | Source |
|------|---------|--------|
| Lint | `ruff check .` | pyproject.toml [tool.ruff] |
| TypeCheck | `ty check crawl4r/` | pyproject.toml [tool.ty] |
| Test (all) | `pytest` | pyproject.toml [tool.pytest.ini_options] |
| Test (unit) | `pytest tests/unit/` | Standard pytest pattern |
| Test (integration) | `pytest tests/integration/ -m integration` | Test marker pattern |
| Coverage | `pytest --cov=crawl4r --cov-report=term` | pyproject.toml [tool.coverage] |

**Local CI:** `ruff check . && ty check crawl4r/ && pytest --cov=crawl4r`

## Open Questions

1. **Should we support language detection for file-based ingestion too?**
   - Answer: Yes, but separate feature (file_watcher.py uses different reader)
   - Scope: This spec focuses on web crawling only

2. **What about language detection in Crawl4AI service itself?**
   - Answer: Upstream feature request for v0.8+ (future work)
   - Current: Post-filter in Crawl4AIReader (Python side)

3. **Should we store rejected documents for review/tuning?**
   - Answer: Log only (structured logging with URL + detected language)
   - Reason: Avoids storage bloat, users can replay crawl if needed

## Related Specs

### High Priority (Direct Overlap)

**llamaindex-crawl4ai-reader** (Complete, Production)
- **Relationship:** Language filter integrates directly into Crawl4AIReader
- **Impact:** Non-breaking addition, new config fields only
- **Status:** Completed, 786 tests passing, 87%+ coverage
- **mayNeedUpdate:** No - additive feature, no breaking changes

**web-crawl-cli** (Tasks Complete, Awaiting Implementation)
- **Relationship:** CLI scrape/crawl commands use Crawl4AIReader
- **Impact:** Language filtering automatic via reader config
- **Status:** 38 tasks defined, not yet implemented
- **mayNeedUpdate:** Yes - ScraperService/IngestionService should expose language config

### Medium Priority (Shared Components)

**rag-ingestion** (Design Complete, Not Implemented)
- **Relationship:** File-based ingestion may want language detection too
- **Impact:** Separate implementation for file_watcher.py
- **Status:** Specifications complete, implementation not started
- **mayNeedUpdate:** Maybe - consider language detection for markdown files

## Recommendations for Requirements

1. **Use fast-langdetect library**
   - Fastest (80x), accurate (95%), thread-safe, minimal dependencies
   - Mature ecosystem, active maintenance, Python 3.9-3.13 support

2. **Implement post-filter strategy**
   - Detect after crawling, filter before chunking
   - Lower risk, better accuracy, simpler implementation
   - Enables confidence thresholds and multi-language handling

3. **Default to English only, opt-in for multi-language**
   - `allowed_languages=["en"]` by default
   - Users explicitly enable other languages (intentional choice)
   - Prevents accidental ingestion of unwanted languages

4. **Add language metadata to Qdrant**
   - Store `language` and `language_confidence` in payload
   - Enable language-specific search/filtering in retrieval
   - Useful for multi-language RAG systems

5. **Fail-open on detection errors**
   - If detection fails, log error but accept document
   - Prevents pipeline breakage from library issues
   - Users can monitor logs for detection failures

6. **Log filtered documents with structured fields**
   - URL, detected language, confidence, reason
   - Enables tuning of confidence thresholds
   - Helps diagnose false negatives

## Sources

**Language Detection Libraries:**
- [fast-langdetect GitHub](https://github.com/LlmKira/fast-langdetect)
- [fast-langdetect PyPI](https://pypi.org/project/fast-langdetect/)
- [lingua-py GitHub](https://github.com/pemistahl/lingua-py)
- [lingua-py PyPI](https://pypi.org/project/lingua-language-detector/)
- [Language Detection Comparison](https://modelpredict.com/language-identification-survey)
- [Mastering Multilingualism](https://plainenglish.io/blog/mastering-multilingualism-top-5-python-language-detection-techniques-explained)

**Web Crawling Best Practices:**
- [Language Specific Web Crawling](https://www.researchgate.net/publication/267257810_Language_Specific_and_Topic_Focused_Web_Crawling)
- [GPT Crawler Guide](https://scrapfly.io/blog/posts/gpt-crawler-a-complete-guide-to-automated-web-data-collection-for-ai-training)
- [Common Crawl Processing](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-6-sourcing-acquiring-massive-text-datasets/utilizing-common-crawl-data)
- [Web Crawler Guide](https://www.promptcloud.com/blog/web-crawler-guide/)

**Codebase:**
- `/home/jmagar/workspace/crawl4r/crawl4r/readers/crawl4ai.py` (831 lines)
- `/home/jmagar/workspace/crawl4r/crawl4r/readers/crawl/models.py` (39 lines)
- `/home/jmagar/workspace/crawl4r/pyproject.toml` (96 lines)
