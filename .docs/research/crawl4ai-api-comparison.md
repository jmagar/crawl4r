# Crawl4AI API Comparison: Docker REST API vs Python SDK

**Research Date:** 2026-01-20
**Crawl4AI Version:** 0.8.0 (latest as of January 2026)
**Documentation Sources:** [Official Docs](https://docs.crawl4ai.com/), [GitHub Repository](https://github.com/unclecode/crawl4ai)

---

## Executive Summary

Crawl4AI provides two primary interfaces for web crawling:
1. **Docker REST API** - HTTP endpoints for language-agnostic access (default port: 11235)
2. **Python SDK** - Native async Python library with `AsyncWebCrawler` class

Both interfaces share similar capabilities but differ in synchronous/asynchronous job handling, parameter naming conventions, and ease of integration.

---

## Complete Endpoint/Method Comparison

### Core Crawling Operations

| Functionality | Docker REST API | Python SDK | Notes |
|--------------|----------------|------------|-------|
| **Single URL crawl** | `POST /crawl` | `await crawler.arun(url, config)` | REST returns JSON; SDK returns `CrawlResult` object |
| **Multiple URL crawl** | `POST /crawl` (array) | `await crawler.arun_many(urls, config)` | SDK has built-in concurrency control |
| **Streaming crawl** | `POST /crawl/stream` | `await crawler.arun_many(urls, stream=True)` | REST uses NDJSON; SDK uses AsyncGenerator |
| **Async job submission** | `POST /crawl/job` | N/A | REST-only; returns task_id for polling |
| **Job status check** | `GET /job/{task_id}` | N/A | REST-only; retrieves job results |

### Specialized Content Extraction

| Functionality | Docker REST API | Python SDK | Notes |
|--------------|----------------|------------|-------|
| **Markdown generation** | `POST /md` | `result.markdown` (auto) | REST accepts filter param (`f=fit/raw/bm25/llm`) |
| **LLM extraction** | `POST /llm/job` | `LLMExtractionStrategy` in config | REST supports async jobs with webhooks |
| **HTML extraction** | `POST /html` | `result.cleaned_html` | REST optimized for schema extraction |
| **Screenshot capture** | `POST /screenshot` | `config.screenshot=True` | Both support full-page PNG |
| **PDF generation** | `POST /pdf` | `config.pdf=True` | Both generate PDF from web page |
| **JavaScript execution** | `POST /execute_js` | `config.js_code="..."` | SDK more flexible with hooks |

### Monitoring & Operations

| Functionality | Docker REST API | Python SDK | Notes |
|--------------|----------------|------------|-------|
| **Health check** | `GET /monitor/health` | N/A | REST-only; system snapshot |
| **Request tracking** | `GET /monitor/requests` | N/A | REST-only; active/completed requests |
| **Browser pool status** | `GET /monitor/browsers` | N/A | REST-only; memory, age, hit count |
| **Endpoint statistics** | `GET /monitor/endpoints/stats` | N/A | REST-only; latency, success rate |
| **Timeline metrics** | `GET /monitor/timeline` | N/A | REST-only; time-series data |
| **Error logs** | `GET /monitor/logs/errors` | N/A | REST-only; error history |
| **Janitor logs** | `GET /monitor/logs/janitor` | N/A | REST-only; cleanup events |
| **Force cleanup** | `POST /monitor/actions/cleanup` | N/A | REST-only; kills cold browsers |
| **Real-time monitoring** | `WebSocket /monitor/ws` | N/A | REST-only; 2-second updates |

### Lifecycle & Configuration

| Functionality | Docker REST API | Python SDK | Notes |
|--------------|----------------|------------|-------|
| **Initialize crawler** | Container startup | `AsyncWebCrawler(config)` | SDK requires explicit initialization |
| **Start resources** | Automatic | `await crawler.start()` | SDK allows manual control |
| **Close resources** | Automatic | `await crawler.close()` | SDK cleanup required |
| **Context manager** | N/A | `async with AsyncWebCrawler()` | SDK auto-manages lifecycle |
| **Cache management** | Via request params | `config.cache_mode` | SDK deprecated `aclear_cache()` |

### Developer Tools

| Functionality | Docker REST API | Python SDK | Notes |
|--------------|----------------|------------|-------|
| **Interactive testing** | `GET /playground` | N/A | REST-only; web UI at port 11235 |
| **Monitoring dashboard** | `GET /monitor` | N/A | REST-only; real-time visibility |
| **API documentation** | `GET /docs` | Official docs site | REST auto-generates FastAPI docs |
| **Schema introspection** | `GET /schema` | N/A | REST-only; config schemas |
| **Hook information** | `GET /hooks/info` | N/A | REST-only; lifecycle hooks |
| **MCP schema** | `GET /mcp/schema` | N/A | REST-only; AI model integration |
| **MCP SSE transport** | `POST /mcp/sse` | N/A | REST-only; Model Context Protocol |

---

## Docker REST API Endpoints Reference

### Synchronous Crawling

#### `POST /crawl`
Full-featured synchronous crawl endpoint.

**Request Body:**
```json
{
  "urls": ["https://example.com"],
  "browser_config": {
    "type": "BrowserConfig",
    "params": {
      "headless": true,
      "viewport_width": 1080,
      "viewport_height": 600,
      "user_agent": "Mozilla/5.0..."
    }
  },
  "crawler_config": {
    "type": "CrawlerRunConfig",
    "params": {
      "cache_mode": "bypass",
      "word_count_threshold": 200,
      "screenshot": false,
      "pdf": false,
      "js_code": "return document.title",
      "wait_for": ".content-loaded"
    }
  }
}
```

**Response:**
```json
{
  "url": "https://example.com",
  "html": "<html>...</html>",
  "cleaned_html": "<div>...</div>",
  "markdown": "# Page Title\n\nContent...",
  "extracted_content": {},
  "media": {"images": [], "videos": [], "audios": []},
  "links": {"internal": [], "external": []},
  "screenshot": null,
  "pdf": null,
  "status_code": 200,
  "success": true
}
```

#### `POST /crawl/stream`
Streaming variant for real-time results.

**Response Format:** NDJSON (newline-delimited JSON)
```json
{"url": "https://example1.com", "markdown": "...", "success": true}
{"url": "https://example2.com", "markdown": "...", "success": true}
{"status": "completed"}
```

---

### Markdown & Content Extraction

#### `POST /md`
Generates markdown with optional filtering and LLM processing.

**Query Parameters:**
- `url` (required): Target webpage URL
- `f` (optional): Filter type - `raw`, `fit`, `bm25`, `llm` (default: `fit`)
- `q` (optional): Query for BM25/LLM filtering
- `provider` (optional): LLM provider (e.g., `openai/gpt-4o-mini`)
- `temperature` (optional): LLM temperature (0.0-2.0)
- `base_url` (optional): Custom LLM API endpoint
- `cache` (optional): Cache control parameter

**Filter Types:**
- **`raw`**: Full page markdown (~89K chars) with navigation/footer
- **`fit`**: Clean main content (~12K chars) - **RECOMMENDED**
- **`bm25`**: BM25 relevance filtering with query `q`
- **`llm`**: AI-powered extraction with provider config

**Example Request:**
```bash
curl -X POST "http://localhost:11235/md" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "f": "fit"
  }'
```

**Example Response:**
```json
{
  "markdown": "# Main Article\n\nClean content without nav/footer...",
  "filter_applied": "fit",
  "char_count": 12453
}
```

#### `POST /html`
Extracts preprocessed HTML optimized for schema extraction.

**Request:**
```json
{"url": "https://example.com"}
```

---

### Media Capture

#### `POST /screenshot`
Captures full-page PNG screenshot.

**Request:**
```json
{
  "url": "https://example.com",
  "screenshot_wait_for": 2,
  "output_path": "/path/to/screenshot.png"
}
```

**Response:** Base64-encoded PNG or file path

#### `POST /pdf`
Generates PDF from web page.

**Request:**
```json
{
  "url": "https://example.com",
  "output_path": "/path/to/document.pdf"
}
```

**Response:** Binary PDF or file path

---

### JavaScript Execution

#### `POST /execute_js`
Runs custom JavaScript on page.

**Request:**
```json
{
  "url": "https://example.com",
  "scripts": [
    "return document.title",
    "return Array.from(document.querySelectorAll('a')).map(a => a.href)"
  ]
}
```

**Response:**
```json
{
  "results": [
    "Example Domain",
    ["https://example.com/page1", "https://example.com/page2"]
  ]
}
```

---

### Asynchronous Jobs

#### `POST /crawl/job`
Submits async crawl job with webhook support.

**Request:**
```json
{
  "urls": ["https://example.com"],
  "crawler_config": {...},
  "webhook_config": {
    "webhook_url": "https://your-app.com/webhook",
    "webhook_data_in_payload": true,
    "webhook_headers": {"X-Webhook-Secret": "token"}
  }
}
```

**Response:**
```json
{
  "task_id": "crawl_1698765432",
  "message": "Crawl job submitted"
}
```

**Webhook Payload:**
```json
{
  "task_id": "crawl_1698765432",
  "task_type": "crawl",
  "status": "completed",
  "timestamp": "2026-01-20T10:30:00Z",
  "urls": ["https://example.com"],
  "result": {
    "markdown": "...",
    "extracted_content": {}
  }
}
```

**Webhook Features:**
- Exponential backoff retry (5 attempts)
- Success codes: 200-299
- Custom headers for authentication
- Optional result data in payload

#### `POST /llm/job`
Submits async LLM extraction job.

**Request:**
```json
{
  "url": "https://example.com/article",
  "q": "Extract title, author, and main points",
  "provider": "openai/gpt-4o-mini",
  "temperature": 0.7,
  "webhook_config": {
    "webhook_url": "https://your-app.com/webhook",
    "webhook_data_in_payload": true
  }
}
```

**Response:**
```json
{
  "task_id": "llm_1698765432",
  "message": "LLM job submitted"
}
```

#### `GET /job/{task_id}`
Retrieves job status and results.

**Response (In Progress):**
```json
{
  "task_id": "crawl_1698765432",
  "status": "processing",
  "message": "Job is being processed"
}
```

**Response (Completed):**
```json
{
  "task_id": "crawl_1698765432",
  "status": "completed",
  "result": {
    "markdown": "# Page Title\n\nContent...",
    "extracted_content": {},
    "links": {}
  }
}
```

---

### Monitoring Endpoints

#### `GET /monitor/health`
System health snapshot.

**Response:**
```json
{
  "status": "healthy",
  "memory": {"used": 1024, "total": 8192},
  "cpu": {"percent": 25.5},
  "uptime": 86400,
  "browser_pool": {"active": 3, "idle": 2}
}
```

#### `GET /monitor/requests`
Active and completed request tracking.

**Query Parameters:**
- `status`: Filter by `all`, `active`, `completed`, `success`, `error`
- `limit`: Number of requests (1-1000)

**Response:**
```json
{
  "active": 5,
  "completed": 1234,
  "requests": [
    {
      "request_id": "req_123",
      "url": "https://example.com",
      "status": "completed",
      "duration_ms": 1234
    }
  ]
}
```

#### `GET /monitor/browsers`
Browser pool status.

**Response:**
```json
{
  "browsers": [
    {
      "browser_id": "chrome_1",
      "type": "chromium",
      "age_seconds": 120,
      "memory_mb": 256,
      "hit_count": 15
    }
  ]
}
```

#### `GET /monitor/endpoints/stats`
Endpoint performance metrics.

**Response:**
```json
{
  "endpoints": {
    "/crawl": {
      "latency_ms": {"p50": 500, "p95": 1200, "p99": 2000},
      "success_rate": 0.98,
      "pool_hit_rate": 0.85
    }
  }
}
```

#### `GET /monitor/timeline`
Time-series metrics for charts.

**Query Parameters:**
- `metric`: `memory`, `requests`, `browsers`
- `window`: `5m` (5-minute window, 5-second resolution)

**Response:**
```json
{
  "metric": "memory",
  "window": "5m",
  "data": [
    {"timestamp": "2026-01-20T10:00:00Z", "value": 1024},
    {"timestamp": "2026-01-20T10:00:05Z", "value": 1056}
  ]
}
```

#### `GET /monitor/logs/errors`
Error log history.

**Query Parameters:**
- `limit`: Number of errors (default: 100)

**Response:**
```json
{
  "errors": [
    {
      "timestamp": "2026-01-20T10:15:30Z",
      "endpoint": "/crawl",
      "url": "https://example.com",
      "error_message": "Timeout after 60s"
    }
  ]
}
```

#### `GET /monitor/logs/janitor`
Cleanup event history.

**Query Parameters:**
- `limit`: Number of events (default: 100)

**Response:**
```json
{
  "events": [
    {
      "timestamp": "2026-01-20T10:20:00Z",
      "action": "cleanup",
      "browsers_killed": 3,
      "reason": "cold_pool"
    }
  ]
}
```

#### `POST /monitor/actions/cleanup`
Force immediate browser cleanup.

**Response:**
```json
{
  "success": true,
  "killed_browsers": 3
}
```

#### `WebSocket /monitor/ws`
Real-time monitoring stream (updates every 2 seconds).

**Message Format:**
```json
{
  "health": {...},
  "requests": {...},
  "browsers": {...},
  "timeline": {...},
  "errors": [...]
}
```

---

### Developer Tools

#### `GET /playground`
Interactive web UI for API testing at `http://localhost:11235/playground`.

Features:
- Configure `CrawlerRunConfig` and `BrowserConfig` visually
- Test crawling operations in real-time
- Generate JSON payloads for REST API
- View results immediately

#### `GET /monitor`
Monitoring dashboard at `http://localhost:11235/monitor`.

Features:
- Real-time system health
- Active request tracking
- Browser pool visualization
- Performance metrics charts

#### `GET /docs`
Auto-generated FastAPI documentation at `http://localhost:11235/docs`.

Features:
- Interactive API explorer
- Request/response schemas
- Try-it-now functionality
- OpenAPI specification

#### `GET /schema`
Returns configuration schemas.

**Response:**
```json
{
  "BrowserConfig": {...},
  "CrawlerRunConfig": {...},
  "LLMConfig": {...}
}
```

#### `GET /hooks/info`
Available hook points and signatures.

**Response:**
```json
{
  "hooks": [
    {
      "name": "on_browser_created",
      "signature": "async def on_browser_created(browser: Browser) -> None"
    }
  ]
}
```

#### `GET /mcp/schema`
Model Context Protocol tool schemas.

**Response:**
```json
{
  "tools": [
    {
      "name": "crawl_web",
      "description": "Crawl web pages and extract content",
      "parameters": {...}
    }
  ]
}
```

#### `POST /mcp/sse`
Model Context Protocol Server-Sent Events transport.

---

## Python SDK Methods Reference

### AsyncWebCrawler Class

#### Constructor

```python
def __init__(
    crawler_strategy: Optional[AsyncCrawlerStrategy] = None,
    config: Optional[BrowserConfig] = None,
    always_bypass_cache: bool = False,  # Deprecated
    base_directory: str = ...,
    thread_safe: bool = False,
    **kwargs
) -> None
```

**Parameters:**
- `crawler_strategy`: Custom crawling strategy (default: PlaywrightCrawlerStrategy)
- `config`: Browser configuration (BrowserConfig object)
- `base_directory`: Cache and data storage location
- `thread_safe`: Enable thread-safe operation
- `**kwargs`: Additional browser configuration parameters

**Example:**
```python
from crawl4ai import AsyncWebCrawler, BrowserConfig

browser_config = BrowserConfig(
    browser_type="chromium",
    headless=True,
    viewport_width=1080,
    viewport_height=600
)

crawler = AsyncWebCrawler(config=browser_config)
```

---

### Lifecycle Methods

#### `start()`

```python
async def start() -> None
```

Manually initializes browser resources. Required if not using context manager.

**Example:**
```python
await crawler.start()
# Perform crawls...
await crawler.close()
```

#### `close()`

```python
async def close() -> None
```

Cleans up and closes browser resources.

#### Context Manager (Recommended)

```python
async with AsyncWebCrawler(config=browser_config) as crawler:
    result = await crawler.arun("https://example.com")
    # Resources automatically managed
```

---

### Primary Crawling Methods

#### `arun()`

```python
async def arun(
    url: str,
    config: Optional[CrawlerRunConfig] = None
) -> CrawlResult
```

Crawls a single URL asynchronously.

**Parameters:**
- `url` (str, required): Target webpage URL
- `config` (CrawlerRunConfig, optional): Crawl configuration

**Returns:** `CrawlResult` object

**Key CrawlerRunConfig Options:**
- `cache_mode`: `CacheMode.ENABLED | DISABLED | BYPASS | READ_ONLY | WRITE_ONLY`
- `word_count_threshold`: Minimum words per text block (default: 200)
- `css_selector`: Focus on specific page region
- `excluded_tags`: HTML tags to remove
- `js_code`: JavaScript to execute before extraction
- `wait_for`: CSS selector or JS condition to await
- `page_timeout`: Maximum wait time in milliseconds
- `extraction_strategy`: Data extraction handler
- `markdown_generator`: Custom markdown generator
- `screenshot`: Capture screenshot (bool)
- `pdf`: Generate PDF (bool)
- `session_id`: Reuse browser session

**Example:**
```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

config = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS,
    word_count_threshold=100,
    screenshot=True,
    wait_for=".content-loaded"
)

async with AsyncWebCrawler() as crawler:
    result = await crawler.arun("https://example.com", config=config)

    print(result.markdown)
    print(result.screenshot)  # Base64-encoded PNG
```

#### `arun_many()`

```python
async def arun_many(
    urls: Union[List[str], List[Any]],
    config: Optional[Union[CrawlerRunConfig, List[CrawlerRunConfig]]] = None,
    dispatcher: Optional[BaseDispatcher] = None,
    ...
) -> Union[List[CrawlResult], AsyncGenerator[CrawlResult, None]]
```

Crawls multiple URLs with intelligent concurrency control.

**Parameters:**
- `urls`: List of URLs or task objects
- `config`: Single config or list of configs with URL matchers
- `dispatcher`: Concurrency controller (MemoryAdaptiveDispatcher, SemaphoreDispatcher)

**Returns:** List of `CrawlResult` or AsyncGenerator (if `stream=True`)

**Dispatcher Types:**
- **MemoryAdaptiveDispatcher** (default): Adjusts concurrency based on memory usage
- **SemaphoreDispatcher**: Fixed concurrency limit

**Example (Basic):**
```python
urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3"
]

async with AsyncWebCrawler() as crawler:
    results = await crawler.arun_many(urls)

    for result in results:
        if result.success:
            print(f"{result.url}: {len(result.markdown)} chars")
```

**Example (Streaming):**
```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

config = CrawlerRunConfig(stream=True)

async with AsyncWebCrawler() as crawler:
    async for result in await crawler.arun_many(urls, config=config):
        print(f"Completed: {result.url}")
```

**Example (URL-Specific Configs):**
```python
configs = [
    CrawlerRunConfig(
        url_matcher=r"https://example\.com/blog/.*",
        extraction_strategy=BlogExtractionStrategy()
    ),
    CrawlerRunConfig(
        url_matcher=r"https://example\.com/docs/.*",
        extraction_strategy=DocsExtractionStrategy()
    )
]

results = await crawler.arun_many(urls, config=configs)
```

**Example (Custom Dispatcher):**
```python
from crawl4ai.dispatchers import MemoryAdaptiveDispatcher

dispatcher = MemoryAdaptiveDispatcher(
    memory_threshold_percent=70.0,
    max_session_permit=10,
    check_interval=1.0
)

results = await crawler.arun_many(urls, dispatcher=dispatcher)
```

---

### Configuration Classes

#### BrowserConfig

```python
class BrowserConfig:
    def __init__(
        browser_type: str = "chromium",  # chromium | firefox | webkit
        headless: bool = True,
        proxy_config: Optional[Dict] = None,
        viewport_width: int = 1080,
        viewport_height: int = 600,
        verbose: bool = True,
        use_persistent_context: bool = False,
        user_data_dir: Optional[str] = None,
        cookies: Optional[List[Dict]] = None,
        headers: Optional[Dict[str, str]] = None,
        user_agent: Optional[str] = None,
        user_agent_mode: str = "",  # "random" for rotation
        text_mode: bool = False,
        light_mode: bool = False,
        ignore_https_errors: bool = True,
        java_script_enabled: bool = True,
        enable_stealth: bool = False,
        ...
    )
```

**Key Parameters:**
| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `browser_type` | str | "chromium" | Engine: chromium, firefox, webkit |
| `headless` | bool | True | Run without visible UI |
| `viewport_width` | int | 1080 | Page width (pixels) |
| `viewport_height` | int | 600 | Page height (pixels) |
| `proxy_config` | dict | None | Proxy server configuration |
| `user_agent` | str | Chrome UA | Custom user-agent string |
| `user_agent_mode` | str | "" | "random" for rotation |
| `ignore_https_errors` | bool | True | Continue on invalid certs |
| `java_script_enabled` | bool | True | Enable/disable JS |
| `enable_stealth` | bool | False | Anti-bot detection mode |
| `use_persistent_context` | bool | False | Maintain session across runs |
| `user_data_dir` | str | None | Directory for persistent data |
| `cookies` | list | None | Pre-set cookies |
| `headers` | dict | None | Custom HTTP headers |

**Example:**
```python
from crawl4ai import BrowserConfig

config = BrowserConfig(
    browser_type="chromium",
    headless=True,
    enable_stealth=True,
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
    viewport_width=1920,
    viewport_height=1080,
    cookies=[
        {"name": "session", "value": "abc123", "domain": ".example.com"}
    ],
    headers={
        "Accept-Language": "en-US,en;q=0.9"
    }
)
```

#### CrawlerRunConfig

```python
class CrawlerRunConfig:
    def __init__(
        word_count_threshold: int = 200,
        extraction_strategy: Optional[ExtractionStrategy] = None,
        markdown_generator: Optional[MarkdownGenerator] = None,
        cache_mode: Optional[CacheMode] = None,
        js_code: Optional[Union[str, List[str]]] = None,
        wait_for: Optional[str] = None,
        page_timeout: int = 60000,
        screenshot: bool = False,
        pdf: bool = False,
        capture_mhtml: bool = False,
        css_selector: Optional[str] = None,
        excluded_tags: Optional[List[str]] = None,
        exclude_external_links: bool = False,
        exclude_domains: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        locale: Optional[str] = None,
        timezone_id: Optional[str] = None,
        geolocation: Optional[GeolocationConfig] = None,
        enable_rate_limiting: bool = False,
        memory_threshold_percent: float = 70.0,
        max_session_permit: int = 20,
        check_robots_txt: bool = False,
        verbose: bool = True,
        stream: bool = False,
        ...
    )
```

**Key Parameters:**
| Category | Parameter | Type | Default | Purpose |
|----------|-----------|------|---------|---------|
| **Content** | `word_count_threshold` | int | 200 | Min words per block |
| | `css_selector` | str | None | Focus region |
| | `excluded_tags` | list | None | Tags to remove |
| **Timing** | `wait_until` | str | "domcontentloaded" | Page load state |
| | `page_timeout` | int | 60000 | Max wait (ms) |
| | `wait_for` | str | None | CSS/JS condition |
| **Caching** | `cache_mode` | CacheMode | ENABLED | Cache behavior |
| | `session_id` | str | None | Session identifier |
| **Interaction** | `js_code` | str/list | None | Custom JavaScript |
| | `scan_full_page` | bool | False | Scroll entire page |
| **Media** | `screenshot` | bool | False | Capture PNG |
| | `pdf` | bool | False | Generate PDF |
| | `capture_mhtml` | bool | False | Save MHTML |
| **Links** | `exclude_external_links` | bool | False | Filter external |
| | `exclude_domains` | list | [] | Domain blocklist |
| **Extraction** | `extraction_strategy` | object | None | Data extractor |
| | `markdown_generator` | object | None | MD generator |
| **Location** | `locale` | str | None | Language/region |
| | `timezone_id` | str | None | Timezone |
| | `geolocation` | object | None | Lat/long coords |
| **Performance** | `memory_threshold_percent` | float | 70.0 | Memory limit |
| | `max_session_permit` | int | 20 | Max concurrent |
| **Compliance** | `check_robots_txt` | bool | False | Respect robots |
| **Output** | `verbose` | bool | True | Logging level |
| | `stream` | bool | False | Streaming mode |

**Helper Method:**
```python
def clone(**kwargs) -> CrawlerRunConfig
```

Creates modified copy preserving original settings.

**Example:**
```python
from crawl4ai import CrawlerRunConfig, CacheMode

config = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS,
    word_count_threshold=100,
    css_selector=".main-content",
    excluded_tags=["nav", "footer", "script"],
    js_code="""
        // Wait for content to load
        await new Promise(r => setTimeout(r, 2000));
        return document.title;
    """,
    wait_for=".content-loaded",
    screenshot=True,
    pdf=False,
    exclude_external_links=True,
    session_id="my-session-123"
)

# Clone with modifications
config2 = config.clone(screenshot=False, pdf=True)
```

#### LLMConfig

```python
class LLMConfig:
    def __init__(
        provider: str = "openai/gpt-4o-mini",
        api_token: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        backoff_max_attempts: int = 3,
        ...
    )
```

**Supported Providers:**
- OpenAI: `openai/gpt-4`, `openai/gpt-4o-mini`
- Anthropic: `anthropic/claude-3-opus`, `anthropic/claude-3-sonnet`
- Google: `gemini/gemini-pro`, `gemini/gemini-2.0-flash`
- Groq: `groq/llama-3-70b`, `groq/mixtral-8x7b`
- DeepSeek: `deepseek/deepseek-coder`
- Ollama (local): `ollama/llama3`, `ollama/mistral`

**Example:**
```python
from crawl4ai import LLMConfig

llm_config = LLMConfig(
    provider="openai/gpt-4o-mini",
    api_token="sk-...",
    temperature=0.3,
    max_tokens=2000
)
```

---

### Return Types

#### CrawlResult

```python
class CrawlResult(BaseModel):
    url: str
    html: str
    success: bool
    cleaned_html: Optional[str]
    markdown: Optional[Union[str, MarkdownGenerationResult]]
    extracted_content: Optional[str]
    media: Dict[str, List[Dict]]
    links: Dict[str, List[Dict]]
    screenshot: Optional[str]  # Base64-encoded PNG
    pdf: Optional[bytes]
    mhtml: Optional[str]
    status_code: Optional[int]
    error_message: Optional[str]
    session_id: Optional[str]
    dispatch_result: Optional[DispatchResult]
    network_requests: Optional[List[Dict]]
    console_messages: Optional[List[Dict]]
```

**Key Fields:**
- `url`: Final URL (after redirects)
- `html`: Original HTML source
- `cleaned_html`: Sanitized HTML
- `markdown`: Generated markdown (string or MarkdownGenerationResult)
- `extracted_content`: Structured data from extraction strategy
- `media`: Images, videos, audio (with URLs and metadata)
- `links`: Internal/external links with anchor text
- `screenshot`: Base64-encoded PNG image
- `pdf`: Binary PDF document
- `status_code`: HTTP response code
- `success`: Overall crawl success status
- `network_requests`: Captured HTTP requests
- `console_messages`: Browser console logs

**MarkdownGenerationResult Structure:**
```python
class MarkdownGenerationResult:
    raw_markdown: str      # Full page markdown (~89K chars)
    fit_markdown: str      # Filtered content (~12K chars)
    markdown_with_citations: str  # With source links
    references_markdown: str      # Citation list
```

**Example:**
```python
result = await crawler.arun("https://example.com")

if result.success:
    print(f"URL: {result.url}")
    print(f"Status: {result.status_code}")
    print(f"Content length: {len(result.markdown)} chars")

    # Access fit markdown
    if isinstance(result.markdown, MarkdownGenerationResult):
        print(result.markdown.fit_markdown)

    # Media
    for img in result.media["images"]:
        print(f"Image: {img['src']}")

    # Links
    for link in result.links["internal"]:
        print(f"Link: {link['href']}")
else:
    print(f"Error: {result.error_message}")
```

---

### Extraction Strategies

#### JsonCssExtractionStrategy

Schema-based CSS selector extraction without LLM.

```python
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

schema = {
    "name": "Products",
    "baseSelector": ".product-card",
    "fields": [
        {"name": "title", "selector": ".title", "type": "text"},
        {"name": "price", "selector": ".price", "type": "text"},
        {"name": "image", "selector": "img", "type": "attribute", "attribute": "src"}
    ]
}

strategy = JsonCssExtractionStrategy(schema=schema)
config = CrawlerRunConfig(extraction_strategy=strategy)

result = await crawler.arun("https://example.com/products", config=config)
products = json.loads(result.extracted_content)
```

#### LLMExtractionStrategy

LLM-powered structured data extraction with Pydantic schemas.

```python
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    author: str
    publish_date: str
    summary: str
    tags: List[str]

llm_config = LLMConfig(provider="openai/gpt-4o-mini")
strategy = LLMExtractionStrategy(
    llm_config=llm_config,
    schema=Article.model_json_schema(),
    extraction_type="schema",
    instruction="Extract article metadata and content"
)

config = CrawlerRunConfig(extraction_strategy=strategy)
result = await crawler.arun("https://example.com/article", config=config)

article = Article.model_validate_json(result.extracted_content)
print(f"Title: {article.title}")
print(f"Author: {article.author}")
```

#### RegexExtractionStrategy

Pattern-based extraction for fast parsing.

```python
from crawl4ai.extraction_strategy import RegexExtractionStrategy

patterns = {
    "emails": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone_numbers": r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
    "urls": r"https?://[^\s<>\"]+|www\.[^\s<>\"]+"
}

strategy = RegexExtractionStrategy(patterns=patterns)
config = CrawlerRunConfig(extraction_strategy=strategy)

result = await crawler.arun("https://example.com/contact", config=config)
extracted = json.loads(result.extracted_content)

print(f"Emails: {extracted['emails']}")
print(f"Phone numbers: {extracted['phone_numbers']}")
```

---

### Content Filtering

#### PruningContentFilter

Removes low-value content blocks based on word count and structure.

```python
from crawl4ai.content_filter import PruningContentFilter

filter = PruningContentFilter(
    threshold=200,  # Minimum words per block
    threshold_type="fixed",
    remove_forms=True,
    remove_nav=True
)

config = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=filter
    )
)
```

#### BM25ContentFilter

Relevance-based filtering using BM25 ranking algorithm.

```python
from crawl4ai.content_filter import BM25ContentFilter

filter = BM25ContentFilter(
    user_query="machine learning tutorial",
    bm25_threshold=1.0
)

config = CrawlerRunConfig(
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=filter
    )
)
```

---

### Markdown Generation

#### DefaultMarkdownGenerator

```python
from crawl4ai.markdown_generator import DefaultMarkdownGenerator

generator = DefaultMarkdownGenerator(
    content_filter=PruningContentFilter(threshold=200),
    options={
        "include_links": True,
        "include_images": True,
        "body_width": 0  # No wrapping
    }
)

config = CrawlerRunConfig(markdown_generator=generator)
```

---

## Configuration Parameter Mapping

### Browser Configuration

| Feature | REST API (`browser_config.params`) | Python SDK (`BrowserConfig`) |
|---------|-----------------------------------|------------------------------|
| Browser type | `browser_type` | `browser_type` |
| Headless mode | `headless` | `headless` |
| Viewport size | `viewport_width`, `viewport_height` | `viewport_width`, `viewport_height` |
| User agent | `user_agent` | `user_agent` |
| User agent rotation | `user_agent_mode` | `user_agent_mode` |
| Proxy | `proxy_config` | `proxy_config` |
| HTTPS errors | `ignore_https_errors` | `ignore_https_errors` |
| JavaScript | `java_script_enabled` | `java_script_enabled` |
| Stealth mode | `enable_stealth` | `enable_stealth` |
| Persistent context | `use_persistent_context` | `use_persistent_context` |
| User data dir | `user_data_dir` | `user_data_dir` |
| Cookies | `cookies` | `cookies` |
| Headers | `headers` | `headers` |

### Crawler Configuration

| Feature | REST API (`crawler_config.params`) | Python SDK (`CrawlerRunConfig`) |
|---------|------------------------------------|---------------------------------|
| Cache mode | `cache_mode` | `cache_mode` |
| Word threshold | `word_count_threshold` | `word_count_threshold` |
| CSS selector | `css_selector` | `css_selector` |
| Excluded tags | `excluded_tags` | `excluded_tags` |
| JavaScript code | `js_code` | `js_code` |
| Wait condition | `wait_for` | `wait_for` |
| Wait state | `wait_until` | `wait_until` |
| Page timeout | `page_timeout` | `page_timeout` |
| Screenshot | `screenshot` | `screenshot` |
| PDF | `pdf` | `pdf` |
| MHTML | `capture_mhtml` | `capture_mhtml` |
| Session ID | `session_id` | `session_id` |
| External links | `exclude_external_links` | `exclude_external_links` |
| Domain exclusion | `exclude_domains` | `exclude_domains` |
| Extraction | `extraction_config` | `extraction_strategy` ⚠️ |
| Markdown | N/A | `markdown_generator` |
| Locale | `locale` | `locale` |
| Timezone | `timezone_id` | `timezone_id` |
| Geolocation | `geolocation` | `geolocation` |
| Rate limiting | `enable_rate_limiting` | `enable_rate_limiting` |
| Memory threshold | `memory_threshold_percent` | `memory_threshold_percent` |
| Max sessions | `max_session_permit` | `max_session_permit` |
| Robots.txt | `check_robots_txt` | `check_robots_txt` |
| Streaming | `stream` | `stream` |
| Verbose | `verbose` | `verbose` |

⚠️ **Parameter Naming Discrepancy:** REST API uses `extraction_config` while Python SDK uses `extraction_strategy`.

---

## Key Differences

### 1. Synchronous vs Asynchronous

| Feature | Docker REST API | Python SDK |
|---------|----------------|------------|
| **Synchronous operations** | ✅ `POST /crawl` | ❌ (async only) |
| **Asynchronous operations** | ✅ `POST /crawl/job` | ✅ `await arun()` |
| **Job polling** | ✅ `GET /job/{task_id}` | ❌ (direct results) |
| **Webhooks** | ✅ Webhook config | ❌ (not needed) |
| **Streaming** | ✅ `POST /crawl/stream` (NDJSON) | ✅ `stream=True` (AsyncGenerator) |

### 2. Monitoring & Observability

| Feature | Docker REST API | Python SDK |
|---------|----------------|------------|
| **Health checks** | ✅ `GET /monitor/health` | ❌ |
| **Request tracking** | ✅ `GET /monitor/requests` | ❌ |
| **Browser pool status** | ✅ `GET /monitor/browsers` | ❌ |
| **Performance metrics** | ✅ `GET /monitor/endpoints/stats` | ❌ |
| **Real-time monitoring** | ✅ WebSocket `/monitor/ws` | ❌ |
| **Error logs** | ✅ `GET /monitor/logs/errors` | ❌ |

**Implication:** Docker service provides production-grade monitoring; Python SDK requires custom instrumentation.

### 3. Developer Experience

| Feature | Docker REST API | Python SDK |
|---------|----------------|------------|
| **Interactive testing** | ✅ `/playground` web UI | ❌ |
| **API documentation** | ✅ Auto-generated `/docs` | ✅ Official docs site |
| **Language support** | ✅ Any HTTP client | ❌ Python only |
| **Type safety** | ❌ JSON validation | ✅ Pydantic models |
| **IDE support** | ❌ Limited | ✅ Full autocomplete |

### 4. Content Extraction

| Feature | Docker REST API | Python SDK |
|---------|----------------|------------|
| **Markdown filtering** | ✅ `f=fit/raw/bm25/llm` param | ⚠️ Via `content_filter` in config |
| **Quick markdown** | ✅ `POST /md` (simple) | ❌ (requires full crawl) |
| **LLM extraction** | ✅ `POST /llm/job` | ✅ `LLMExtractionStrategy` |
| **Schema extraction** | ⚠️ Via `extraction_config` | ✅ `JsonCssExtractionStrategy` |
| **Regex extraction** | ❌ | ✅ `RegexExtractionStrategy` |

### 5. Lifecycle Management

| Feature | Docker REST API | Python SDK |
|---------|----------------|------------|
| **Resource initialization** | Automatic (container) | Manual (`await start()`) |
| **Resource cleanup** | Automatic | Manual (`await close()`) |
| **Context manager** | N/A | ✅ `async with` |
| **Browser reuse** | ✅ Pool management | ✅ Session ID |
| **Force cleanup** | ✅ `POST /monitor/actions/cleanup` | ❌ |

### 6. Parameter Naming

| Concept | REST API | Python SDK |
|---------|----------|------------|
| **Extraction** | `extraction_config` | `extraction_strategy` |
| **Cache control** | `cache_mode` (params) | `cache_mode` (enum) |
| **Browser config wrapper** | `browser_config.type` + `params` | Direct `BrowserConfig` object |
| **Crawler config wrapper** | `crawler_config.type` + `params` | Direct `CrawlerRunConfig` object |

### 7. Feature Availability

#### REST API Only
- `/md` endpoint with filter parameter (`f=fit/raw/bm25/llm`)
- `/html` preprocessed extraction
- `/screenshot` and `/pdf` standalone endpoints
- `/execute_js` JavaScript execution
- Job-based async with webhooks
- Comprehensive monitoring suite
- Interactive playground
- MCP (Model Context Protocol) integration

#### Python SDK Only
- `RegexExtractionStrategy`
- Fine-grained content filtering (PruningContentFilter, BM25ContentFilter)
- Custom markdown generators
- Direct access to browser context
- Type-safe Pydantic models
- Programmatic hook system
- AsyncGenerator streaming

#### Both Support
- Web crawling (single and batch)
- LLM extraction
- CSS-based extraction
- Screenshot/PDF generation
- Session management
- Cache control
- Rate limiting

---

## Use Case Recommendations

### Use Docker REST API when:

1. **Language Agnostic**: Non-Python projects (JavaScript, Java, Go, etc.)
2. **Microservices**: Separate crawling service from main application
3. **Async Jobs**: Long-running crawls with webhook notifications
4. **Production Monitoring**: Need health checks, metrics, and dashboards
5. **Quick Markdown**: Only need `/md` endpoint with filtering
6. **Centralized Crawling**: Shared service for multiple consumers
7. **Zero Dependencies**: Don't want Python environment
8. **Rate Limiting**: Built-in request tracking and throttling

### Use Python SDK when:

1. **Python Projects**: Native integration with Python codebases
2. **Type Safety**: Leverage Pydantic models and IDE autocomplete
3. **Custom Logic**: Need programmatic hooks and callbacks
4. **Performance**: Avoid HTTP overhead for local crawling
5. **Advanced Extraction**: Use multiple extraction strategies
6. **Fine Control**: Direct browser context manipulation
7. **Streaming**: Real-time processing with AsyncGenerator
8. **Embedded**: Self-contained application without external services

### Hybrid Approach:

Use REST API for:
- Production crawling infrastructure
- Monitoring and observability
- Multi-language support

Use Python SDK for:
- Development and testing
- Local prototyping
- Custom extraction logic
- Integration testing

---

## Performance Considerations

### Docker REST API

**Advantages:**
- Browser pool reuse across requests
- Automatic resource cleanup (janitor)
- Memory-adaptive concurrency
- Centralized caching

**Overhead:**
- HTTP request/response serialization
- Network latency (even localhost)
- JSON encoding/decoding

**Best For:** Shared infrastructure, multiple consumers

### Python SDK

**Advantages:**
- No HTTP overhead
- Direct memory access to results
- Native async/await patterns
- Fine-grained control

**Overhead:**
- Manual resource management
- Browser lifecycle costs
- No built-in monitoring

**Best For:** Embedded applications, high-throughput batch jobs

---

## Authentication & Security

### Docker REST API

**Optional JWT Authentication:**
```bash
curl -X POST http://localhost:11235/crawl \
  -H "Authorization: Bearer <jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com"]}'
```

**Webhook Security:**
```json
{
  "webhook_config": {
    "webhook_url": "https://your-app.com/webhook",
    "webhook_headers": {
      "X-Webhook-Secret": "your-secret-token"
    }
  }
}
```

### Python SDK

**No Built-in Auth:**
- Application-level authentication required
- Direct process access (inherently trusted)

---

## Migration Path

### REST API → Python SDK

```python
# REST API (curl)
curl -X POST http://localhost:11235/crawl \
  -d '{
    "urls": ["https://example.com"],
    "crawler_config": {
      "type": "CrawlerRunConfig",
      "params": {"screenshot": true}
    }
  }'

# Python SDK
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

config = CrawlerRunConfig(screenshot=True)
async with AsyncWebCrawler() as crawler:
    result = await crawler.arun("https://example.com", config=config)
```

### Python SDK → REST API

```python
# Python SDK
result = await crawler.arun("https://example.com", config=config)

# REST API (Python requests)
import requests

response = requests.post(
    "http://localhost:11235/crawl",
    json={
        "urls": ["https://example.com"],
        "crawler_config": {
            "type": "CrawlerRunConfig",
            "params": config.model_dump()  # Convert Pydantic to dict
        }
    }
)
result = response.json()
```

---

## Sources

### Official Documentation
- [Crawl4AI Documentation](https://docs.crawl4ai.com/) - Complete SDK reference and guides
- [Complete SDK Reference](https://docs.crawl4ai.com/complete-sdk-reference/) - API methods and classes
- [Docker Deployment](https://docs.crawl4ai.com/core/docker-deployment/) - REST API endpoints
- [Self-Hosting Guide](https://docs.crawl4ai.com/core/self-hosting/) - Infrastructure setup
- [Fit Markdown](https://docs.crawl4ai.com/core/fit-markdown/) - Content filtering
- [AsyncWebCrawler API](https://docs.crawl4ai.com/api/async-webcrawler/) - Class reference
- [arun() Method](https://docs.crawl4ai.com/api/arun/) - Single crawl documentation
- [arun_many() Method](https://docs.crawl4ai.com/api/arun_many/) - Batch crawl documentation
- [Browser & Crawler Config](https://docs.crawl4ai.com/api/parameters/) - Configuration parameters
- [Quick Start](https://docs.crawl4ai.com/core/quickstart/) - Getting started guide

### GitHub Resources
- [GitHub Repository](https://github.com/unclecode/crawl4ai) - Source code and issues
- [Docker README](https://github.com/unclecode/crawl4ai/blob/main/deploy/docker/README.md) - Docker deployment
- [REST API Schema Discussion](https://github.com/unclecode/crawl4ai/discussions/838) - API schema details
- [Releases](https://github.com/unclecode/crawl4ai/releases) - Version history

### Package Repositories
- [PyPI Package](https://pypi.org/project/Crawl4AI/) - Python package distribution
- [Docker Hub](https://hub.docker.com/r/unclecode/crawl4ai) - Docker image

### Tutorials & Guides
- [Pondhouse Data Tutorial](https://www.pondhouse-data.com/blog/webcrawling-with-crawl4ai) - Docker tutorial
- [Apidog Tutorial](https://apidog.com/blog/crawl4ai-tutorial/) - Beginner's guide
- [ScrapingBee Guide](https://www.scrapingbee.com/blog/crawl4ai/) - Hands-on guide

### Third-Party Tools
- [Postman Collection](https://www.postman.com/pixelao/pixel-public-workspace/collection/c26yn3l/crawl4ai-api) - API testing

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-20 | 1.0 | Initial comprehensive comparison |

---

## Notes

1. **Parameter Naming:** REST API uses `extraction_config` while SDK uses `extraction_strategy`. This inconsistency is documented in GitHub Discussion #838.

2. **Deprecated Methods:** The Python SDK previously exposed `aclear_cache()` and `aflush_cache()` methods, but these are now deprecated in favor of `cache_mode` configuration.

3. **Filter Types:** The `/md` endpoint's `f` parameter (`fit`, `raw`, `bm25`, `llm`) is REST API-specific. Python SDK achieves similar results via `content_filter` in `markdown_generator` config.

4. **Streaming:** REST API uses NDJSON format, while Python SDK uses AsyncGenerator pattern.

5. **Version Info:** This comparison is based on Crawl4AI v0.8.0 (January 2026). Check official documentation for updates.

6. **Interactive Documentation:** When running Docker service, access auto-generated API docs at `http://localhost:11235/docs` for most up-to-date schema.

---

**End of Document**
