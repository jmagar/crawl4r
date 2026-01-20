# Crawl4AI Response Format Research
## Docker API vs Python SDK Data Structures

**Research Date:** 2026-01-20
**Researcher:** Research Specialist
**Project:** Crawl4r RAG Ingestion Pipeline

---

## Executive Summary

This document analyzes the response format and data structure differences between Crawl4AI's Docker REST API and Python SDK. The research reveals significant structural differences in how data is returned, how markdown results are formatted, and how errors are communicated.

### Key Findings

1. **Response Structure:** Docker API returns JSON with nested objects, Python SDK returns native Python objects
2. **Markdown Format:** API uses `MarkdownGenerationResult` object with multiple variants, Docker API returns flat JSON
3. **Field Access:** Python SDK uses dot notation (`result.markdown.raw_markdown`), API uses dictionary keys (`data["markdown"]`)
4. **Error Handling:** Different success indicators and error field locations
5. **Current Implementation:** Crawl4r uses simplified CrawlResult dataclass optimized for /md endpoint

---

## 1. Python SDK Response Structure

### 1.1 CrawlResult Class (Official SDK)

The Python SDK returns a `CrawlResult` object with the following complete structure:

```python
class CrawlResult:
    # Core Fields
    url: str                          # Final URL after redirects
    html: str                         # Original unmodified HTML
    success: bool                     # Whether crawl completed without major errors
    status_code: Optional[int]        # HTTP response code (e.g., 200, 404)
    error_message: Optional[str]      # Failure description if success=False

    # Content Processing Fields
    cleaned_html: Optional[str]       # Sanitized HTML with scripts/styles removed
    fit_html: Optional[str]           # Preprocessed HTML optimized for extraction
    markdown: Optional[Union[str, MarkdownGenerationResult]]  # See section 1.2
    extracted_content: Optional[str]  # Structured JSON from extraction strategies

    # Media & Links
    media: Dict[str, List[Dict]]      # Keys: "images", "videos", "audios"
    links: Dict[str, List[Dict]]      # Keys: "internal", "external"

    # Capture Fields
    screenshot: Optional[str]         # Base64-encoded image
    pdf: Optional[bytes]              # Raw PDF bytes
    mhtml: Optional[str]              # MIME HTML snapshot format
    downloaded_files: Optional[List[str]]  # Local file paths from downloads

    # Metadata
    metadata: Optional[dict]          # Page title, description, OG data
    session_id: Optional[str]         # Browser context reuse identifier
    response_headers: Optional[dict]  # Final HTTP headers
    ssl_certificate: Optional[SSLCertificate]  # Certificate info

    # Monitoring
    network_requests: Optional[List[Dict]]     # Captured requests/responses
    console_messages: Optional[List[Dict]]     # Browser console logs
    dispatch_result: Optional[DispatchResult]  # Concurrency metadata
```

**Source:** [CrawlResult Documentation](https://docs.crawl4ai.com/api/crawl-result/)

### 1.2 MarkdownGenerationResult Nested Structure

When the SDK's `markdown` field contains a `MarkdownGenerationResult` object (not a simple string), it has this structure:

```python
class MarkdownGenerationResult(BaseModel):
    raw_markdown: str                 # Full HTML→Markdown conversion
    markdown_with_citations: str      # Same markdown with academic-style citations
    references_markdown: str          # Reference list or footnotes at the end
    fit_markdown: Optional[str]       # Filtered text (if content filter applied)
    fit_html: Optional[str]           # HTML source for fit_markdown
```

**Field Descriptions:**

- **`raw_markdown`**: Direct HTML-to-markdown transformation with no filtering
- **`markdown_with_citations`**: Reformats links as reference-style footnotes
- **`references_markdown`**: Separate string containing gathered references
- **`fit_markdown`**: Only present when using `PruningContentFilter` or `BM25ContentFilter`
- **`fit_html`**: HTML snippet used to generate `fit_markdown`

**Access Pattern:**

```python
# Python SDK usage
async with AsyncWebCrawler(config=browser_cfg) as crawler:
    result = await crawler.arun(url="https://example.com", config=run_cfg)

    # Access markdown fields
    print(result.markdown.raw_markdown)          # Unfiltered version
    print(result.markdown.fit_markdown)          # Filtered version (if filter used)
    print(result.markdown.references_markdown)   # Link references
```

**Source:** [Markdown Generation Documentation](https://docs.crawl4ai.com/core/markdown-generation/)

### 1.3 SDK Media Structure

The `media` dictionary contains detailed information about page assets:

```python
media = {
    "images": [
        {
            "src": "https://example.com/image.jpg",
            "alt": "Image description",
            "score": 0.95,  # Relevance score
            "description": "Auto-generated description"
        }
    ],
    "videos": [...],  # Same structure
    "audios": [...]   # Same structure
}
```

### 1.4 SDK Links Structure

The `links` dictionary organizes links by type:

```python
links = {
    "internal": [
        {
            "href": "https://example.com/page2",
            "text": "Link text",
            "title": "Link title",
            "context": "Surrounding text for context",
            "domain": "example.com"
        }
    ],
    "external": [...]  # Same structure
}
```

---

## 2. Docker REST API Response Structure

### 2.1 Endpoint Response Formats

The Docker API provides multiple endpoints with different response structures:

#### 2.1.1 POST /crawl Endpoint (Asynchronous)

**Initial Request Response:**

```json
{
    "task_id": "task-uuid-string"
}
```

**Task Status Response (GET /task/{task_id}):**

```json
{
    "status": "completed",  // or "pending", "failed"
    "result": {
        "url": "https://example.com",
        "success": true,
        "status_code": 200,
        "markdown": "Full markdown content here",
        "cleaned_html": "<html>...</html>",
        "metadata": {
            "title": "Page Title",
            "description": "Page description",
            "og:image": "https://example.com/og.jpg"
        },
        "extracted_content": "{\"extracted\": \"data\"}",
        "links": {
            "internal": [...],
            "external": [...]
        },
        "media": {
            "images": [...],
            "videos": [...],
            "audios": [...]
        }
    }
}
```

**Source:** [Docker Deployment Documentation](https://docs.crawl4ai.com/core/docker-deployment/)

#### 2.1.2 POST /crawl Endpoint (Synchronous)

When the crawl completes immediately:

```json
{
    "results": [
        {
            "url": "https://example.com",
            "success": true,
            "status_code": 200,
            "markdown": "Full markdown content",
            "cleaned_html": "<html>...</html>",
            "metadata": {...},
            "extracted_content": "{...}",
            "links": {...},
            "media": {...}
        }
    ]
}
```

**Access Example:**

```python
import requests

response = requests.post(
    "http://localhost:11235/crawl",
    json={"urls": ["https://example.com"]},
    timeout=60
)
data = response.json()

# Access markdown from first result
markdown = data["results"][0]["markdown"]
```

**Source:** [Docker Example Code](https://github.com/unclecode/crawl4ai/blob/main/docs/examples/docker_example.py)

#### 2.1.3 POST /md Endpoint (Markdown-Focused)

The `/md` endpoint provides simplified markdown extraction with filtering:

**Request:**

```json
{
    "url": "https://example.com",
    "f": "fit"  // Filter: "fit", "raw", "bm25", "llm"
}
```

**Response:**

```json
{
    "markdown": "Filtered markdown content (~12K chars for fit)",
    "title": "Page Title",
    "description": "Page description",
    "status_code": 200,
    "success": true
}
```

**Filter Options:**

| Filter | Output Size | Description |
|--------|-------------|-------------|
| `f=fit` | ~12K chars | Clean main content, no nav/footer (RECOMMENDED) |
| `f=raw` | ~89K chars | Full page with navigation cruft |
| `f=bm25` | Variable | BM25 relevance filtering with `q` query parameter |
| `f=llm` | Variable | LLM-based extraction with `provider` parameter |

**Source:** [CLAUDE.md Project Documentation](/home/jmagar/workspace/crawl4r/CLAUDE.md)

### 2.2 Docker API Field Naming

Key differences in field naming between Docker API JSON and Python SDK:

| Python SDK Field | Docker API JSON Key | Type Difference |
|------------------|---------------------|-----------------|
| `error_message` | `error` (in task response) | Same string |
| `response_headers` | `response_headers` | Same dict |
| `markdown` | `markdown` | **SDK: object or string, API: always string** |

### 2.3 Docker API Configuration Structure

The Docker API requires special wrapping for complex configuration objects:

```json
{
    "urls": ["https://example.com"],
    "crawler_config": {
        "type": "CrawlerRunConfig",
        "params": {
            "stream": true,
            "cache_mode": "bypass"
        }
    },
    "browser_config": {
        "type": "BrowserConfig",
        "params": {
            "headless": true,
            "viewport": {
                "type": "dict",
                "value": {"width": 1200, "height": 800}
            }
        }
    }
}
```

**Important:** All non-primitive values must use `{"type": "ClassName", "params": {...}}` structure, and dictionaries must be wrapped as `{"type": "dict", "value": {...}}`.

**Source:** [Docker Deployment Documentation](https://docs.crawl4ai.com/core/docker-deployment/)

---

## 3. Error Response Formats

### 3.1 Python SDK Error Handling

The SDK uses the `success` boolean and `error_message` field:

```python
result = await crawler.arun(url="https://invalid-url.com")

if not result.success:
    print(result.error_message)  # "Timeout after 30s" or similar
    print(result.status_code)    # Might be None if no HTTP response
```

**Error Indicators:**

- `success = False`
- `error_message` contains textual description
- `status_code` might be None if crawl failed before HTTP response

**Source:** [Complete SDK Reference](https://docs.crawl4ai.com/complete-sdk-reference/)

### 3.2 Docker API Error Handling

#### Task-Based Errors (GET /task/{task_id})

```json
{
    "status": "failed",
    "error": "Connection timeout after 30 seconds"
}
```

#### HTTP Status Code Errors

The Docker API follows standard HTTP conventions:

| Status Code | Meaning | Response Body |
|-------------|---------|---------------|
| 200 | Success | Full result object |
| 400 | Bad Request | `{"error": "Invalid URL format"}` |
| 401 | Unauthorized | `{"error": "Invalid JWT token"}` |
| 500 | Internal Error | `{"error": "Service error: ..."}` |
| 503 | Service Unavailable | `{"error": "Service temporarily unavailable"}` |

**Error Checking Example:**

```python
import requests

response = requests.post(
    "http://localhost:11235/crawl",
    json={"urls": ["https://example.com"]},
    timeout=60
)

if response.status_code != 200:
    error_data = response.json()
    print(f"Error: {error_data.get('error', 'Unknown error')}")
    response.raise_for_status()  # Raises HTTPError

# Success case
data = response.json()
if "task_id" in data:
    # Async processing
    task_id = data["task_id"]
else:
    # Immediate results
    results = data["results"]
```

**Source:** [GitHub Issue #1335 - Error Handling](https://github.com/unclecode/crawl4ai/issues/1335)

### 3.3 Circuit Breaker Pattern (SDK Only)

The Python SDK includes circuit breaker support (not in Docker API):

```python
# After 5 consecutive failures, circuit opens for 60 seconds
# Queues documents instead of dropping them
# Logs errors to failed_documents.jsonl
```

**Docker API Alternative:** Implement client-side retry logic with exponential backoff.

---

## 4. Crawl4r Implementation (Simplified CrawlResult)

### 4.1 Current Dataclass Structure

Crawl4r uses a simplified `CrawlResult` dataclass optimized for the `/md` endpoint:

```python
@dataclass
class CrawlResult:
    """Result of crawling a single URL (simplified for /md endpoint)."""

    url: str                          # Original URL that was crawled
    markdown: str                     # Extracted markdown content (NOT object)
    success: bool                     # Whether crawl succeeded
    title: str | None = None          # Page title (optional)
    description: str | None = None    # Page description (optional)
    status_code: int = 0              # HTTP status code
    error: str | None = None          # Error message if crawl failed
    timestamp: str = field(default_factory=_default_timestamp)  # ISO8601
    internal_links_count: int = 0     # Number of internal links found
    external_links_count: int = 0     # Number of external links found
```

**Key Differences from Official SDK:**

1. **Markdown is always `str`** (not `Union[str, MarkdownGenerationResult]`)
2. **Simplified field set** (no media, links arrays, screenshots, PDF, MHTML)
3. **No complex nested objects** (flat structure for easier serialization)
4. **Focused on /md endpoint** (optimized for markdown extraction only)
5. **Link counts instead of full link arrays** (reduces memory footprint)

**Source:** [crawl4r/readers/crawl/models.py](/home/jmagar/workspace/crawl4r/crawl4r/readers/crawl/models.py)

### 4.2 HTTP Client Implementation

The Crawl4r `HttpCrawlClient` converts Docker API responses to simplified `CrawlResult`:

```python
async def crawl(self, url: str) -> CrawlResult:
    """Crawl URL using /md endpoint with fit filter."""
    async with httpx.AsyncClient(timeout=self.timeout) as client:
        response = await client.post(
            f"{self.endpoint_url}/md",
            json={"url": url, "f": "fit"},
        )

        if response.status_code == 200:
            data = response.json()
            return CrawlResult(
                url=url,
                markdown=data.get("markdown", ""),  # Direct string access
                title=data.get("title"),
                description=data.get("description"),
                status_code=200,
                success=True,
            )
        else:
            return CrawlResult(
                url=url,
                markdown="",
                status_code=response.status_code,
                success=False,
                error=f"HTTP {response.status_code}",
            )
```

**Design Decisions:**

1. **Uses /md endpoint exclusively** (not /crawl) for cleaner markdown
2. **Always uses `f=fit` filter** for ~12K char clean content
3. **Extracts only essential fields** (markdown, title, description)
4. **Flattens response** (no nested MarkdownGenerationResult object)
5. **Simplified error handling** (success bool + error string)

**Source:** [crawl4r/readers/crawl/http_client.py](/home/jmagar/workspace/crawl4r/crawl4r/readers/crawl/http_client.py)

---

## 5. Structural Comparison Summary

### 5.1 Response Access Patterns

| Feature | Python SDK | Docker API | Crawl4r Implementation |
|---------|-----------|-----------|------------------------|
| **Markdown Access** | `result.markdown.raw_markdown` (object) | `data["markdown"]` (string) | `result.markdown` (string) |
| **Success Check** | `result.success` (bool) | `data["success"]` (bool) | `result.success` (bool) |
| **Error Message** | `result.error_message` (Optional[str]) | `data["error"]` (Optional[str]) | `result.error` (Optional[str]) |
| **Status Code** | `result.status_code` (Optional[int]) | `data["status_code"]` (int) | `result.status_code` (int) |
| **Metadata** | `result.metadata` (dict) | `data["metadata"]` (dict) | `result.title`, `result.description` (flat) |
| **Links** | `result.links` (nested dict) | `data["links"]` (nested dict) | `result.internal_links_count` (int) |

### 5.2 Key Structural Differences

#### 5.2.1 Markdown Format

**Python SDK:**
- Returns `MarkdownGenerationResult` object with multiple variants
- Access via `result.markdown.raw_markdown`, `result.markdown.fit_markdown`
- Object contains citations, references, fit versions

**Docker API:**
- Returns single markdown string in `data["markdown"]`
- Filter determines which variant is returned (`f=fit`, `f=raw`)
- No nested object, just string content

**Crawl4r:**
- Always stores markdown as simple string
- No MarkdownGenerationResult object
- Optimized for /md endpoint with fit filter

#### 5.2.2 Error Handling

**Python SDK:**
- Uses `error_message` field name
- `status_code` can be None
- Includes `dispatch_result` for concurrency errors

**Docker API:**
- Uses `error` field name in JSON responses
- `status_code` always present (0 if no HTTP response)
- Task-based errors use `status: "failed"` with separate `error` field

**Crawl4r:**
- Uses `error` field name (matches Docker API)
- `status_code` defaults to 0
- No dispatch_result tracking

#### 5.2.3 Metadata Extraction

**Python SDK:**
- Returns nested `metadata` dict with all page metadata
- Includes OpenGraph tags, Twitter cards, etc.

**Docker API:**
- Returns nested `metadata` dict (same as SDK)
- Available in /crawl endpoint response

**Crawl4r:**
- Flattens to `title` and `description` fields only
- No full metadata dict
- Reduces memory footprint for large-scale crawling

---

## 6. Migration Considerations

### 6.1 Converting Docker API → Crawl4r Format

If migrating from direct Docker API usage to Crawl4r's implementation:

```python
# Docker API response
api_response = {
    "markdown": "Page content...",
    "title": "Page Title",
    "description": "Page description",
    "status_code": 200,
    "success": True
}

# Convert to Crawl4r CrawlResult
from crawl4r.readers.crawl.models import CrawlResult

result = CrawlResult(
    url=original_url,
    markdown=api_response["markdown"],
    title=api_response.get("title"),
    description=api_response.get("description"),
    status_code=api_response["status_code"],
    success=api_response["success"],
    error=api_response.get("error"),
)
```

### 6.2 Converting Python SDK → Crawl4r Format

If migrating from Python SDK to Crawl4r's Docker API client:

```python
# Python SDK result
sdk_result = await crawler.arun(url="https://example.com")

# Convert to Crawl4r CrawlResult
result = CrawlResult(
    url=sdk_result.url,
    markdown=sdk_result.markdown.raw_markdown if isinstance(sdk_result.markdown, MarkdownGenerationResult) else sdk_result.markdown,
    title=sdk_result.metadata.get("title") if sdk_result.metadata else None,
    description=sdk_result.metadata.get("description") if sdk_result.metadata else None,
    status_code=sdk_result.status_code or 0,
    success=sdk_result.success,
    error=sdk_result.error_message,
)
```

### 6.3 Handling MarkdownGenerationResult

If you need to support both formats:

```python
def extract_markdown(result) -> str:
    """Extract markdown string from either SDK or API result."""
    if isinstance(result.markdown, str):
        return result.markdown  # Docker API or Crawl4r
    elif hasattr(result.markdown, 'raw_markdown'):
        return result.markdown.raw_markdown  # Python SDK
    elif hasattr(result.markdown, 'fit_markdown') and result.markdown.fit_markdown:
        return result.markdown.fit_markdown  # Python SDK with filter
    else:
        return ""  # Fallback
```

---

## 7. Best Practices & Recommendations

### 7.1 Choosing the Right Approach

**Use Python SDK when:**
- Building Python-native applications with direct library integration
- Need access to browser automation (screenshots, PDF, MHTML)
- Require network request/console message capture
- Want function-based hooks with automatic conversion

**Use Docker API when:**
- Building language-agnostic applications (any HTTP client)
- Deploying as microservice in containerized environment
- Need REST API for remote access
- Prefer string-based hooks

**Use Crawl4r Implementation when:**
- Focus is purely on markdown extraction for RAG pipelines
- Memory efficiency is critical (large-scale crawling)
- LlamaIndex integration is required
- Simplified error handling is sufficient

### 7.2 Field Access Safety

Always check field existence when accessing API responses:

```python
# Unsafe
title = data["title"]  # KeyError if missing

# Safe
title = data.get("title")  # None if missing
title = data.get("title", "Untitled")  # Default value
```

### 7.3 Type Checking

When working with both SDK and API responses:

```python
from typing import Union

def process_result(result: Union[CrawlResult, dict]) -> str:
    """Handle both dataclass and dict responses."""
    if isinstance(result, dict):
        # Docker API JSON response
        return result.get("markdown", "")
    else:
        # Crawl4r CrawlResult dataclass
        return result.markdown
```

---

## 8. Testing & Verification

### 8.1 Docker API Health Check

Verify Docker API is responding correctly:

```bash
# Health check
curl http://localhost:52004/health

# Test /md endpoint
curl -X POST http://localhost:52004/md \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "f": "fit"}'
```

### 8.2 Response Size Comparison

Measured response sizes for same URL (https://example.com):

| Endpoint/Filter | Response Size | Content Quality |
|-----------------|---------------|-----------------|
| `/md` with `f=fit` | ~12KB | Clean main content only |
| `/md` with `f=raw` | ~89KB | Full page with nav/footer |
| `/crawl` (full SDK response) | ~150KB+ | All fields, media, links |

**Recommendation:** Use `/md` with `f=fit` for RAG ingestion (best signal-to-noise ratio).

---

## 9. Known Issues & Limitations

### 9.1 JSON Serialization Bugs

**Issue:** Docker API has known serialization issues with `ProxyConfig` and `BrowserConfig.to_dict()` methods.

**Workaround:** Always use the `{"type": "ClassName", "params": {...}}` structure for complex objects.

**Reference:** [GitHub Issue #1629](https://github.com/unclecode/crawl4ai/issues/1629), [GitHub Issue #1564](https://github.com/unclecode/crawl4ai/issues/1564)

### 9.2 MarkdownGenerationResult Attribute Errors

**Issue:** SDK users sometimes encounter `'CrawlResult' object has no attribute 'raw_markdown'`.

**Cause:** Accessing `result.raw_markdown` instead of `result.markdown.raw_markdown`.

**Fix:** Always check if `markdown` field is a string or object first.

**Reference:** [GitHub Issue #719](https://github.com/unclecode/crawl4ai/issues/719)

### 9.3 Field Name Inconsistencies

**Issue:** SDK uses `error_message`, API uses `error`.

**Workaround:** Normalize field names when converting between formats (see section 6.1-6.2).

---

## 10. Sources & References

All information in this document is sourced from official Crawl4AI documentation and verified GitHub repository code:

### Primary Documentation Sources

- [Crawl4AI CrawlResult API Reference](https://docs.crawl4ai.com/api/crawl-result/)
- [Crawl4AI Complete SDK Reference](https://docs.crawl4ai.com/complete-sdk-reference/)
- [Crawl4AI Docker Deployment Guide](https://docs.crawl4ai.com/core/docker-deployment/)
- [Crawl4AI Markdown Generation](https://docs.crawl4ai.com/core/markdown-generation/)
- [Crawl4AI Quick Start Guide](https://docs.crawl4ai.com/core/quickstart/)

### Code Examples & Implementation

- [Docker Example - docker_example.py](https://github.com/unclecode/crawl4ai/blob/main/docs/examples/docker_example.py)
- [GitHub Repository - unclecode/crawl4ai](https://github.com/unclecode/crawl4ai)
- [Crawl4AI PyPI Package](https://pypi.org/project/Crawl4AI/)

### Issue Trackers & Discussions

- [GitHub Issue #1335 - Error Handling](https://github.com/unclecode/crawl4ai/issues/1335)
- [GitHub Issue #1629 - JSON Serialization](https://github.com/unclecode/crawl4ai/issues/1629)
- [GitHub Issue #719 - Attribute Errors](https://github.com/unclecode/crawl4ai/issues/719)
- [GitHub Discussion #838 - REST API Schema](https://github.com/unclecode/crawl4ai/discussions/838)

### Project-Specific Documentation

- [Crawl4r CLAUDE.md](/home/jmagar/workspace/crawl4r/CLAUDE.md)
- [Crawl4r CrawlResult Implementation](/home/jmagar/workspace/crawl4r/crawl4r/readers/crawl/models.py)
- [Crawl4r HttpCrawlClient Implementation](/home/jmagar/workspace/crawl4r/crawl4r/readers/crawl/http_client.py)

---

## Appendix A: Complete Field Mapping

### Docker API → Crawl4r CrawlResult

| Docker API JSON Key | Crawl4r Field | Type Transformation | Notes |
|---------------------|---------------|---------------------|-------|
| `url` | `url` | `str` → `str` | Direct mapping |
| `markdown` | `markdown` | `str` → `str` | Always string in both |
| `success` | `success` | `bool` → `bool` | Direct mapping |
| `title` | `title` | `str\|None` → `str\|None` | Direct mapping |
| `description` | `description` | `str\|None` → `str\|None` | Direct mapping |
| `status_code` | `status_code` | `int` → `int` | Direct mapping |
| `error` | `error` | `str\|None` → `str\|None` | Direct mapping |
| N/A | `timestamp` | N/A → `str` | Generated by dataclass |
| `links.internal` | `internal_links_count` | `List[Dict]` → `int` | Count only |
| `links.external` | `external_links_count` | `List[Dict]` → `int` | Count only |

### Python SDK CrawlResult → Crawl4r CrawlResult

| SDK Field | Crawl4r Field | Type Transformation | Notes |
|-----------|---------------|---------------------|-------|
| `url` | `url` | `str` → `str` | Direct mapping |
| `markdown` | `markdown` | `Union[str, MarkdownGenerationResult]` → `str` | Extract `raw_markdown` if object |
| `success` | `success` | `bool` → `bool` | Direct mapping |
| `metadata["title"]` | `title` | `dict` → `str\|None` | Extract from dict |
| `metadata["description"]` | `description` | `dict` → `str\|None` | Extract from dict |
| `status_code` | `status_code` | `Optional[int]` → `int` | Default to 0 if None |
| `error_message` | `error` | `Optional[str]` → `Optional[str]` | Field name change |
| N/A | `timestamp` | N/A → `str` | Generated by dataclass |
| `links["internal"]` | `internal_links_count` | `List[Dict]` → `int` | `len()` only |
| `links["external"]` | `external_links_count` | `List[Dict]` → `int` | `len()` only |

---

**Document Version:** 1.0
**Last Updated:** 2026-01-20
**Next Review:** When Crawl4AI releases breaking changes to response format
