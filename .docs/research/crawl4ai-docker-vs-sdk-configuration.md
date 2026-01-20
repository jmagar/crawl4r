# Crawl4AI Configuration Differences: Docker API vs Python SDK

**Research Date:** 2026-01-20
**Crawl4AI Version:** 0.8.0 (Latest as of research)
**Purpose:** Document configuration differences between Crawl4AI Docker REST API and Python SDK

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Configuration Objects Overview](#configuration-objects-overview)
3. [Docker API Endpoints](#docker-api-endpoints)
4. [JSON Body Structure](#json-body-structure)
5. [Python SDK Configuration](#python-sdk-configuration)
6. [Parameter Mapping](#parameter-mapping)
7. [Default Values](#default-values)
8. [Key Differences](#key-differences)
9. [Examples](#examples)
10. [Known Issues](#known-issues)
11. [References](#references)

---

## Executive Summary

Crawl4AI provides two primary interfaces:

1. **Docker REST API** - Language-agnostic HTTP endpoints with JSON request bodies
2. **Python SDK** - Native Python library with `AsyncWebCrawler` class

**Core Finding:** Both interfaces share the same underlying configuration schema (`BrowserConfig`, `CrawlerRunConfig`, `LLMConfig`), but differ in:
- How configurations are specified (JSON objects vs Python classes)
- Request structure (JSON serialization requirements)
- Hook support (string-based vs function-based)
- Default behaviors and environment variable handling

---

## Configuration Objects Overview

Crawl4AI uses three main configuration classes:

### 1. BrowserConfig
**Purpose:** Controls browser launch and environment behavior

**Key Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `browser_type` | str | `"chromium"` | Browser engine: chromium, firefox, or webkit |
| `headless` | bool | `True` | Run browser without visible UI |
| `viewport_width` | int | `1080` | Initial page width in pixels |
| `viewport_height` | int | `600` | Initial page height in pixels |
| `proxy_config` | ProxyConfig/dict | `None` | Proxy settings for requests |
| `user_agent` | str | Chrome UA | Custom user agent string |
| `user_agent_mode` | str | `""` | Set to "random" for randomized UAs |
| `use_persistent_context` | bool | `False` | Preserve cookies/sessions across runs |
| `user_data_dir` | str | `None` | Directory for user profiles and cookies |
| `accept_downloads` | bool | `False` | Allow file downloads |
| `ignore_https_errors` | bool | `True` | Continue despite invalid certificates |
| `java_script_enabled` | bool | `True` | Enable JavaScript execution |
| `enable_stealth` | bool | `False` | Bypass bot detection via stealth mode |

### 2. CrawlerRunConfig
**Purpose:** Defines behavior for each crawl operation

**Major Parameter Categories:**

#### A) Content Processing
- `word_count_threshold` (int, ~200): Skip text blocks below threshold words
- `extraction_strategy` (ExtractionStrategy, None): Structured data extraction method
- `css_selector` (str, None): Retain only matching page section
- `excluded_tags` (list, None): Remove specified HTML tags
- `excluded_selector` (str, None): CSS selector for content exclusion
- `remove_forms` (bool, False): Strip all form elements
- `keep_attrs` (list, []): HTML attributes to preserve
- `parser_type` (str, "lxml"): HTML parser engine

#### B) Caching & Session
- `cache_mode` (CacheMode, BYPASS): Control caching behavior (ENABLED, BYPASS, etc.)
- `session_id` (str, None): Reuse browser session across calls

#### C) Page Navigation & Timing
- `wait_until` (str, "domcontentloaded"): Navigation completion condition
- `page_timeout` (int, 60000ms): Timeout for page operations
- `wait_for` (str, None): Wait for CSS or JS condition
- `wait_for_images` (bool, False): Wait for image loading completion
- `mean_delay` (float, 0.1): Average delay between crawls
- `semaphore_count` (int, 5): Max concurrency for batch operations

#### D) Page Interaction
- `js_code` (str/list, None): JavaScript to execute after load
- `scan_full_page` (bool, False): Auto-scroll for dynamic content
- `scroll_delay` (float, 0.2): Delay between scroll steps
- `process_iframes` (bool, False): Inline iframe content
- `remove_overlay_elements` (bool, False): Remove modals/popups
- `simulate_user` (bool, False): Simulate user interactions
- `magic` (bool, False): Auto-handle popups/consent banners

#### E) Media Handling
- `screenshot` (bool, False): Capture page screenshot
- `pdf` (bool, False): Generate PDF output
- `capture_mhtml` (bool, False): Capture MHTML snapshot
- `exclude_all_images` (bool, False): Exclude all images from processing

#### F) Link/Domain Handling
- `exclude_external_links` (bool, False): Remove off-domain links
- `exclude_social_media_links` (bool, False): Strip social media links
- `exclude_domains` (list, []): Custom domain exclusion list

#### G) Debug & Network Monitoring
- `verbose` (bool, True): Print detailed crawling logs
- `log_console` (bool, False): Log page console output
- `capture_network_requests` (bool, False): Record network activity

#### H) Connection & HTTP
- `method` (str, "GET"): HTTP method for requests
- `stream` (bool, False): Enable streaming for batch operations

### 3. LLMConfig
**Purpose:** Manages LLM provider settings for extraction tasks

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | str | `"openai/gpt-4o-mini"` | LLM provider (OpenAI, Groq, Anthropic, Gemini, etc.) |
| `api_token` | str | `None` | API authentication token or env variable reference |
| `base_url` | str | `None` | Custom API endpoint override |
| `backoff_base_delay` | int | `2` | Initial retry delay in seconds |
| `backoff_max_attempts` | int | `3` | Total retry attempts before failure |
| `backoff_exponential_factor` | int | `2` | Exponential backoff multiplier per retry |

---

## Docker API Endpoints

**Base URL:** `http://localhost:11235` (default port)

### Core Endpoints

| Endpoint | Method | Purpose | Response Format |
|----------|--------|---------|-----------------|
| `/crawl` | POST | Synchronous web crawling | JSON (CrawlResult) |
| `/crawl/stream` | POST | Streaming crawl results | NDJSON (line-delimited) |
| `/crawl/job` | POST | Asynchronous crawl job submission | JSON (task_id) |
| `/html` | POST | Preprocessed HTML extraction | JSON (HTML string) |
| `/md` | POST | Markdown extraction with filtering | JSON (markdown + metadata) |
| `/screenshot` | POST | Full-page PNG capture | PNG binary |
| `/pdf` | POST | PDF document generation | PDF binary |
| `/execute_js` | POST | JavaScript execution on pages | JSON (JS result) |
| `/llm/job` | POST | Asynchronous LLM extraction job | JSON (task_id) |
| `/job/{task_id}` | GET | Retrieve async job status/results | JSON (status + result) |

### Specialized Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/playground` | GET | Interactive testing interface (Web UI) |
| `/monitor` | GET | Real-time monitoring dashboard |
| `/monitor/health` | GET | System health snapshot |
| `/monitor/requests` | GET | Active/completed request tracking |
| `/monitor/browsers` | GET | Browser pool details |
| `/monitor/endpoints/stats` | GET | Per-endpoint performance metrics |
| `/monitor/ws` | WebSocket | Real-time updates (2-second intervals) |
| `/mcp/sse` | SSE | Model Context Protocol endpoint |
| `/mcp/ws` | WebSocket | MCP WebSocket connection |

### `/md` Endpoint Filter Options

The `/md` endpoint provides simplified markdown extraction with four filter modes:

| Filter | Parameter | Description | Use Case |
|--------|-----------|-------------|----------|
| `raw` | `f=raw` | Full page markdown (~89K chars) | Complete page capture with nav/footer |
| `fit` | `f=fit` | Filtered main content (~12K chars) | **Recommended** - Clean content extraction |
| `bm25` | `f=bm25` | BM25 relevance ranking | Query-based filtering (requires `q` param) |
| `llm` | `f=llm` | LLM-based extraction | AI-powered content extraction (requires `q` param) |

**Example:**
```bash
# Clean markdown extraction (recommended)
curl -X POST http://localhost:11235/md \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "f": "fit"}'

# BM25 relevance filtering
curl -X POST http://localhost:11235/md \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "f": "bm25", "q": "machine learning tutorials"}'

# LLM extraction with custom provider
curl -X POST http://localhost:11235/md \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "f": "llm",
    "q": "Extract all product names and prices",
    "provider": "groq/mixtral-8x7b",
    "temperature": 0.2
  }'
```

### `/crawl` vs `/md` Endpoint Comparison

| Feature | `/crawl` | `/md` |
|---------|----------|-------|
| **Output** | Complete CrawlResult (HTML, markdown, links, metadata) | Markdown content only |
| **Filtering** | Via `markdown_generator` parameter (complex) | Simple `f` parameter (raw/fit/bm25/llm) |
| **Configuration** | Full BrowserConfig + CrawlerRunConfig support | Simplified interface |
| **Complexity** | High - supports all crawl features | Low - focused on markdown extraction |
| **Use Case** | Comprehensive crawling with extraction strategies | Quick content-to-markdown conversion |
| **Response Size** | Large (includes all metadata) | Smaller (focused output) |

---

## JSON Body Structure

### Basic Request Format

Docker API requests require explicit type wrapping for configuration objects:

```json
{
  "urls": ["https://example.com"],
  "browser_config": {
    "type": "BrowserConfig",
    "params": {
      "headless": true,
      "viewport_width": 1280,
      "viewport_height": 720
    }
  },
  "crawler_config": {
    "type": "CrawlerRunConfig",
    "params": {
      "cache_mode": "bypass",
      "screenshot": true,
      "wait_for": "css:.article-loaded"
    }
  }
}
```

**Important:** Dictionary parameters need `{"type": "dict", "value": {...}}` wrapper.

### Extraction Strategy Example

```json
{
  "urls": ["https://example.com"],
  "crawler_config": {
    "type": "CrawlerRunConfig",
    "params": {
      "cache_mode": "bypass",
      "extraction_strategy": {
        "type": "JsonCssExtractionStrategy",
        "params": {
          "schema": {
            "name": "Articles",
            "baseSelector": "article.post",
            "fields": [
              {
                "name": "title",
                "selector": "h2",
                "type": "text"
              },
              {
                "name": "url",
                "selector": "a",
                "type": "attribute",
                "attribute": "href"
              }
            ]
          }
        }
      }
    }
  }
}
```

### Webhook Configuration (Async Jobs)

```json
{
  "urls": ["https://example.com"],
  "webhook_config": {
    "webhook_url": "https://your-app.com/webhook/crawl-complete",
    "webhook_data_in_payload": true,
    "webhook_headers": {
      "X-Webhook-Secret": "your-secret-token",
      "X-Custom-Header": "value"
    }
  }
}
```

**Webhook Properties:**
- `webhook_url`: Endpoint URL for job completion notifications
- `webhook_data_in_payload`: Include full result data (true) or just task_id (false)
- `webhook_headers`: Custom HTTP headers for authentication/tracking

**Delivery Guarantee:** Exponential backoff retry (5 attempts: 1s → 2s → 4s → 8s → 16s)

### Proxy Configuration

```json
{
  "browser_config": {
    "type": "BrowserConfig",
    "params": {
      "proxy_config": {
        "type": "ProxyConfig",
        "params": {
          "server": "http://myproxy:8080",
          "username": "user",
          "password": "pass"
        }
      }
    }
  }
}
```

**Note:** Prior to v0.7.8, `ProxyConfig` serialization caused JSON errors. This has been fixed.

---

## Python SDK Configuration

### Basic Usage

```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def crawl():
    # Configure browser
    browser_config = BrowserConfig(
        headless=True,
        viewport_width=1280,
        viewport_height=720,
        user_agent_mode="random"
    )

    # Configure crawler
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        screenshot=True,
        wait_for="css:.article-loaded",
        css_selector="main.article",
        excluded_tags=["script", "style"],
        exclude_external_links=True
    )

    # Execute crawl
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://example.com",
            config=crawler_config
        )

        print(result.markdown)
        print(result.screenshot)  # Base64-encoded PNG
```

### Extraction Strategy

```python
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

schema = {
    "name": "Articles",
    "baseSelector": "article.post",
    "fields": [
        {"name": "title", "selector": "h2", "type": "text"},
        {"name": "url", "selector": "a", "type": "attribute", "attribute": "href"}
    ]
}

crawler_config = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS,
    extraction_strategy=JsonCssExtractionStrategy(schema),
    word_count_threshold=15,
    remove_overlay_elements=True
)
```

### Content Filtering

```python
from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# Option 1: Pruning filter (heuristic-based)
pruning_filter = PruningContentFilter(
    threshold=0.48,
    threshold_type="fixed",
    min_word_threshold=0
)

# Option 2: BM25 filter (query-based)
bm25_filter = BM25ContentFilter(
    user_query="machine learning tutorials",
    bm25_threshold=1.0
)

crawler_config = CrawlerRunConfig(
    cache_mode=CacheMode.ENABLED,
    markdown_generator=DefaultMarkdownGenerator(
        content_filter=pruning_filter
    )
)
```

### User-Provided Hooks

**Hook Points:**
- `on_browser_created` - After browser instantiation
- `on_page_context_created` - Authentication, cookies setup
- `before_goto` - Custom headers, pre-navigation logging
- `after_goto` - Verification, element waiting
- `before_retrieve_html` - Scrolling, lazy-loading triggers
- `before_return_html` - Final modifications

```python
async def auth_hook(page, context, **kwargs):
    """Set authentication cookies before navigation"""
    await context.add_cookies([{
        "name": "session_token",
        "value": "abc123",
        "domain": "example.com",
        "path": "/"
    }])
    return page

async def optimize_hook(page, context, **kwargs):
    """Block unnecessary resources"""
    await context.route("**/*.{png,jpg,gif}", lambda r: r.abort())
    await context.route("**/analytics/*", lambda r: r.abort())
    return page

result = await crawler.arun(
    url="https://example.com",
    config=CrawlerRunConfig(
        hooks={
            "on_page_context_created": auth_hook,
            "before_goto": optimize_hook
        },
        hooks_timeout=30
    )
)
```

### Docker Client Usage

```python
from crawl4ai import Crawl4aiDockerClient, BrowserConfig, CrawlerRunConfig

async def use_docker_api():
    async with Crawl4aiDockerClient(
        base_url="http://localhost:11235",
        timeout=60.0,
        verify_ssl=True
    ) as client:
        result = await client.crawl(
            ["https://example.com"],
            browser_config=BrowserConfig(headless=True),
            crawler_config=CrawlerRunConfig(cache_mode="bypass")
        )

        print(result[0].markdown)
```

**SDK Parameters:**
- `base_url`: Server location (default: `http://localhost:8000`)
- `timeout`: Request timeout in seconds (default: 30.0)
- `verify_ssl`: SSL certificate verification (default: True)

---

## Parameter Mapping

### BrowserConfig Mapping

| Python SDK Parameter | Docker API JSON Key | Type | Default |
|---------------------|---------------------|------|---------|
| `browser_type` | `browser_type` | str | `"chromium"` |
| `headless` | `headless` | bool | `True` |
| `viewport_width` | `viewport_width` | int | `1080` |
| `viewport_height` | `viewport_height` | int | `600` |
| `proxy_config` | `proxy_config` | ProxyConfig/dict | `None` |
| `user_agent` | `user_agent` | str | Chrome UA |
| `use_persistent_context` | `use_persistent_context` | bool | `False` |
| `user_data_dir` | `user_data_dir` | str | `None` |
| `enable_stealth` | `enable_stealth` | bool | `False` |

**Example Mapping:**
```python
# Python SDK
browser_config = BrowserConfig(
    headless=True,
    viewport_width=1920,
    viewport_height=1080
)

# Docker API Equivalent
{
  "browser_config": {
    "type": "BrowserConfig",
    "params": {
      "headless": true,
      "viewport_width": 1920,
      "viewport_height": 1080
    }
  }
}
```

### CrawlerRunConfig Mapping

| Python SDK Parameter | Docker API JSON Key | Type | Default |
|---------------------|---------------------|------|---------|
| `cache_mode` | `cache_mode` | str/CacheMode | `"BYPASS"` |
| `session_id` | `session_id` | str | `None` |
| `css_selector` | `css_selector` | str | `None` |
| `screenshot` | `screenshot` | bool | `False` |
| `wait_for` | `wait_for` | str | `None` |
| `excluded_tags` | `excluded_tags` | list | `None` |
| `exclude_external_links` | `exclude_external_links` | bool | `False` |
| `verbose` | `verbose` | bool | `True` |

**Example Mapping:**
```python
# Python SDK
from crawl4ai import CacheMode

crawler_config = CrawlerRunConfig(
    cache_mode=CacheMode.BYPASS,
    screenshot=True,
    wait_for="css:.loaded"
)

# Docker API Equivalent
{
  "crawler_config": {
    "type": "CrawlerRunConfig",
    "params": {
      "cache_mode": "bypass",
      "screenshot": true,
      "wait_for": "css:.loaded"
    }
  }
}
```

### CacheMode Enum Mapping

| Python SDK | Docker API JSON |
|-----------|----------------|
| `CacheMode.ENABLED` | `"enabled"` |
| `CacheMode.DISABLED` | `"disabled"` |
| `CacheMode.BYPASS` | `"bypass"` |
| `CacheMode.READ_ONLY` | `"read_only"` |
| `CacheMode.WRITE_ONLY` | `"write_only"` |

---

## Default Values

### BrowserConfig Defaults

| Parameter | Default Value | Notes |
|-----------|--------------|-------|
| `browser_type` | `"chromium"` | Options: chromium, firefox, webkit |
| `headless` | `True` | Browser runs without visible UI |
| `viewport_width` | `1080` | Initial page width in pixels |
| `viewport_height` | `600` | Initial page height in pixels |
| `proxy_config` | `None` | No proxy configured |
| `user_agent` | Chrome default UA | System-specific |
| `user_agent_mode` | `""` | Empty string (no randomization) |
| `use_persistent_context` | `False` | No session persistence |
| `user_data_dir` | `None` | Temporary profile directory |
| `ignore_https_errors` | `True` | Accept invalid SSL certificates |
| `java_script_enabled` | `True` | JavaScript execution enabled |
| `enable_stealth` | `False` | No stealth mode |

### CrawlerRunConfig Defaults

| Parameter | Default Value | Notes |
|-----------|--------------|-------|
| `cache_mode` | `CacheMode.BYPASS` | **Documentation conflict** - docs claim ENABLED, code uses BYPASS |
| `word_count_threshold` | `~200` | Minimum words per text block |
| `wait_until` | `"domcontentloaded"` | Navigation wait condition |
| `page_timeout` | `60000` | 60 seconds in milliseconds |
| `mean_delay` | `0.1` | 100ms delay between requests |
| `semaphore_count` | `5` | Max concurrent operations |
| `scroll_delay` | `0.2` | 200ms delay between scrolls |
| `verbose` | `True` | Detailed logging enabled |
| `screenshot` | `False` | No screenshots by default |
| `exclude_external_links` | `False` | Include off-domain links |

### LLMConfig Defaults

| Parameter | Default Value | Notes |
|-----------|--------------|-------|
| `provider` | `"openai/gpt-4o-mini"` | Default LLM provider |
| `api_token` | `None` | Must be set via env var or parameter |
| `base_url` | `None` | Uses provider's default endpoint |
| `backoff_base_delay` | `2` | 2 seconds initial retry delay |
| `backoff_max_attempts` | `3` | 3 retry attempts total |
| `backoff_exponential_factor` | `2` | 2x multiplier per retry |

---

## Key Differences

### 1. Request Structure

**Python SDK:**
```python
# Direct object instantiation
browser_config = BrowserConfig(headless=True)
crawler_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

result = await crawler.arun(url, config=crawler_config)
```

**Docker API:**
```json
{
  "urls": ["https://example.com"],
  "browser_config": {
    "type": "BrowserConfig",
    "params": {"headless": true}
  },
  "crawler_config": {
    "type": "CrawlerRunConfig",
    "params": {"cache_mode": "bypass"}
  }
}
```

**Key Difference:** Docker API requires explicit `{"type": "ClassName", "params": {...}}` wrapper.

### 2. Hook Support

**Python SDK:**
```python
# Function-based hooks
async def my_hook(page, context, **kwargs):
    await page.set_viewport_size({"width": 1920, "height": 1080})
    return page

result = await crawler.arun(
    url,
    config=CrawlerRunConfig(
        hooks={"on_page_context_created": my_hook}
    )
)
```

**Docker API:**
```json
{
  "hooks": {
    "on_page_context_created": "async def hook(page, context, **kwargs):\n    await page.set_viewport_size({'width': 1920, 'height': 1080})\n    return page"
  }
}
```

**Key Difference:**
- Python SDK: Functions with full IDE support and automatic conversion
- Docker API: String-based Python code (requires manual formatting)

### 3. Environment Variables

**Python SDK:**
```python
# Configuration hierarchy (highest to lowest):
# 1. Constructor parameters
# 2. Environment variables
# 3. Default values

# Example: API token from env
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

llm_config = LLMConfig(provider="openai/gpt-4")  # Uses env var
```

**Docker API:**
```bash
# Create .llm.env file
OPENAI_API_KEY=sk-your-key
ANTHROPIC_API_KEY=your-anthropic-key
GROQ_API_KEY=your-groq-key

# Mount on container start
docker run -d --env-file .llm.env unclecode/crawl4ai:latest
```

**Configuration Hierarchy (Docker API):**
1. API request parameters (per-request overrides)
2. Provider-specific environment variables (e.g., `OPENAI_TEMPERATURE=0.5`)
3. Global environment variables (e.g., `LLM_PROVIDER=openai/gpt-4o-mini`)
4. `config.yml` defaults

### 4. Simplified vs Full API

**Simplified `/md` Endpoint (Docker only):**
```bash
curl -X POST http://localhost:11235/md \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "f": "fit"}'
```

**Full `/crawl` Endpoint (Both):**
```bash
curl -X POST http://localhost:11235/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com"],
    "crawler_config": {
      "type": "CrawlerRunConfig",
      "params": {
        "markdown_generator": {
          "type": "DefaultMarkdownGenerator",
          "params": {
            "content_filter": {
              "type": "PruningContentFilter",
              "params": {"threshold": 0.48}
            }
          }
        }
      }
    }
  }'
```

**Key Difference:** Docker API provides simplified `/md` endpoint for common use cases; Python SDK always uses full configuration.

### 5. Async Job Handling

**Docker API:**
```json
{
  "urls": ["https://example.com"],
  "webhook_config": {
    "webhook_url": "https://your-app.com/webhook",
    "webhook_data_in_payload": true
  }
}
```

**Python SDK:**
```python
# No webhook support - always awaits result
result = await crawler.arun(url)  # Blocks until complete
```

**Key Difference:** Docker API supports async jobs with webhook notifications; Python SDK is always synchronous (awaits completion).

### 6. Serialization Requirements

**Python SDK:**
```python
# Native Python objects
proxy_config = ProxyConfig(
    server="http://myproxy:8080",
    username="user",
    password="pass"
)

browser_config = BrowserConfig(proxy_config=proxy_config)
```

**Docker API (Pre-v0.7.8):**
```json
{
  "browser_config": {
    "type": "BrowserConfig",
    "params": {
      "proxy_config": {
        "type": "ProxyConfig",
        "params": {
          "server": "http://myproxy:8080",
          "username": "user",
          "password": "pass"
        }
      }
    }
  }
}
```

**Key Difference:** Docker API had JSON serialization bugs with nested objects (fixed in v0.7.8).

---

## Examples

### Example 1: Basic Crawl with Screenshot

**Python SDK:**
```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

async def crawl_with_screenshot():
    browser_config = BrowserConfig(headless=True)
    crawler_config = CrawlerRunConfig(
        screenshot=True,
        wait_for="css:.main-content"
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://example.com",
            config=crawler_config
        )

        # Save screenshot
        import base64
        with open("screenshot.png", "wb") as f:
            f.write(base64.b64decode(result.screenshot))
```

**Docker API:**
```bash
curl -X POST http://localhost:11235/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com"],
    "browser_config": {
      "type": "BrowserConfig",
      "params": {"headless": true}
    },
    "crawler_config": {
      "type": "CrawlerRunConfig",
      "params": {
        "screenshot": true,
        "wait_for": "css:.main-content"
      }
    }
  }' | jq -r '.results[0].screenshot' | base64 -d > screenshot.png
```

### Example 2: CSS Extraction Strategy

**Python SDK:**
```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

schema = {
    "name": "Products",
    "baseSelector": "div.product",
    "fields": [
        {"name": "title", "selector": "h3.title", "type": "text"},
        {"name": "price", "selector": "span.price", "type": "text"},
        {"name": "image", "selector": "img", "type": "attribute", "attribute": "src"}
    ]
}

async def extract_products():
    crawler_config = CrawlerRunConfig(
        extraction_strategy=JsonCssExtractionStrategy(schema),
        cache_mode="bypass"
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com/products",
            config=crawler_config
        )

        import json
        products = json.loads(result.extracted_content)
        print(products)
```

**Docker API:**
```bash
curl -X POST http://localhost:11235/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com/products"],
    "crawler_config": {
      "type": "CrawlerRunConfig",
      "params": {
        "cache_mode": "bypass",
        "extraction_strategy": {
          "type": "JsonCssExtractionStrategy",
          "params": {
            "schema": {
              "name": "Products",
              "baseSelector": "div.product",
              "fields": [
                {"name": "title", "selector": "h3.title", "type": "text"},
                {"name": "price", "selector": "span.price", "type": "text"},
                {"name": "image", "selector": "img", "type": "attribute", "attribute": "src"}
              ]
            }
          }
        }
      }
    }
  }' | jq '.results[0].extracted_content'
```

### Example 3: LLM Extraction with Custom Provider

**Python SDK:**
```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

async def llm_extract():
    llm_config = LLMConfig(
        provider="groq/mixtral-8x7b",
        api_token="gsk_...",
        temperature=0.2
    )

    extraction_strategy = LLMExtractionStrategy(
        llm_config=llm_config,
        instruction="Extract all product names and prices from this page"
    )

    crawler_config = CrawlerRunConfig(
        extraction_strategy=extraction_strategy,
        cache_mode="bypass"
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com/products",
            config=crawler_config
        )

        print(result.extracted_content)
```

**Docker API:**
```bash
curl -X POST http://localhost:11235/md \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/products",
    "f": "llm",
    "q": "Extract all product names and prices from this page",
    "provider": "groq/mixtral-8x7b",
    "temperature": 0.2
  }'
```

### Example 4: Content Filtering

**Python SDK:**
```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

async def filter_content():
    content_filter = BM25ContentFilter(
        user_query="machine learning tutorials",
        bm25_threshold=1.0
    )

    markdown_generator = DefaultMarkdownGenerator(
        content_filter=content_filter
    )

    crawler_config = CrawlerRunConfig(
        cache_mode="enabled",
        markdown_generator=markdown_generator
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com",
            config=crawler_config
        )

        # Filtered markdown
        print(result.markdown.fit_markdown)
```

**Docker API:**
```bash
curl -X POST http://localhost:11235/md \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "f": "bm25",
    "q": "machine learning tutorials"
  }'
```

### Example 5: Async Job with Webhook

**Docker API Only:**
```bash
curl -X POST http://localhost:11235/crawl/job \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com"],
    "crawler_config": {
      "type": "CrawlerRunConfig",
      "params": {
        "cache_mode": "bypass",
        "screenshot": true
      }
    },
    "webhook_config": {
      "webhook_url": "https://your-app.com/webhook/crawl-complete",
      "webhook_data_in_payload": true,
      "webhook_headers": {
        "X-Webhook-Secret": "your-secret-token"
      }
    }
  }'
```

**Response:**
```json
{
  "task_id": "abc123",
  "status": "pending"
}
```

**Webhook Notification:**
```json
{
  "task_id": "abc123",
  "status": "completed",
  "result": {
    "success": true,
    "url": "https://example.com",
    "markdown": "...",
    "screenshot": "base64-encoded-png..."
  }
}
```

### Example 6: Custom Hooks

**Python SDK:**
```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

async def add_auth_cookie(page, context, **kwargs):
    """Add authentication cookie before navigation"""
    await context.add_cookies([{
        "name": "auth_token",
        "value": "abc123",
        "domain": "example.com",
        "path": "/"
    }])
    return page

async def block_resources(page, context, **kwargs):
    """Block images and analytics to speed up crawling"""
    await context.route("**/*.{png,jpg,gif,svg}", lambda r: r.abort())
    await context.route("**/analytics/*", lambda r: r.abort())
    await context.route("**/ads/*", lambda r: r.abort())
    return page

async def crawl_with_hooks():
    crawler_config = CrawlerRunConfig(
        hooks={
            "on_page_context_created": add_auth_cookie,
            "before_goto": block_resources
        },
        hooks_timeout=30
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com/protected",
            config=crawler_config
        )

        print(result.markdown)
```

**Docker API:**
```json
{
  "urls": ["https://example.com/protected"],
  "crawler_config": {
    "type": "CrawlerRunConfig",
    "params": {
      "hooks": {
        "on_page_context_created": "async def hook(page, context, **kwargs):\n    await context.add_cookies([{'name': 'auth_token', 'value': 'abc123', 'domain': 'example.com', 'path': '/'}])\n    return page",
        "before_goto": "async def hook(page, context, **kwargs):\n    await context.route('**/*.{png,jpg,gif,svg}', lambda r: r.abort())\n    await context.route('**/analytics/*', lambda r: r.abort())\n    return page"
      },
      "hooks_timeout": 30
    }
  }
}
```

---

## Known Issues

### 1. Cache Mode Default Inconsistency

**Issue:** Documentation states `cache_mode` defaults to `CacheMode.ENABLED`, but code implementation uses `CacheMode.BYPASS`.

**References:**
- [GitHub Issue #1330](https://github.com/unclecode/crawl4ai/issues/1330)
- [Browser & Crawler Config Docs](https://docs.crawl4ai.com/api/parameters/)

**Workaround:** Always explicitly set `cache_mode` to avoid ambiguity:

```python
# Python SDK
crawler_config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED)

# Docker API
{"crawler_config": {"type": "CrawlerRunConfig", "params": {"cache_mode": "enabled"}}}
```

### 2. ProxyConfig JSON Serialization (Pre-v0.7.8)

**Issue:** `BrowserConfig.to_dict()` failed to serialize `ProxyConfig` objects, causing "Object of type ProxyConfig is not JSON serializable" errors.

**Status:** Fixed in v0.7.8+

**References:**
- [GitHub Issue #1629](https://github.com/unclecode/crawl4ai/issues/1629)
- [Release Notes v0.7.8](https://docs.crawl4ai.com/blog/releases/v0.7.8/)

**Workaround (Pre-v0.7.8):** Manually serialize `ProxyConfig`:

```python
proxy_dict = {
    "type": "ProxyConfig",
    "params": {
        "server": "http://myproxy:8080",
        "username": "user",
        "password": "pass"
    }
}
```

### 3. Webhook Delivery Not Guaranteed

**Issue:** Webhook notifications use exponential backoff with max 5 retries (16 seconds total), but failures are not persisted.

**Workaround:** Poll `/job/{task_id}` endpoint if webhook delivery fails:

```bash
curl http://localhost:11235/job/abc123
```

### 4. String-Based Hooks Error Messages

**Issue:** Docker API string-based hooks provide poor error messages when Python syntax is invalid.

**Workaround:** Test hooks locally with Python SDK before converting to strings for Docker API.

---

## References

### Official Documentation
- [Crawl4AI Documentation (v0.7.x)](https://docs.crawl4ai.com/)
- [Complete SDK Reference](https://docs.crawl4ai.com/complete-sdk-reference/)
- [Browser, Crawler & LLM Config](https://docs.crawl4ai.com/api/parameters/)
- [AsyncWebCrawler API](https://docs.crawl4ai.com/api/async-webcrawler/)
- [Docker Deployment](https://docs.crawl4ai.com/core/docker-deployment/)
- [Self-Hosting Guide](https://docs.crawl4ai.com/core/self-hosting/)
- [Fit Markdown](https://docs.crawl4ai.com/core/fit-markdown/)
- [Markdown Generation](https://docs.crawl4ai.com/core/markdown-generation/)

### GitHub Resources
- [GitHub Repository](https://github.com/unclecode/crawl4ai)
- [Docker Deployment README](https://github.com/unclecode/crawl4ai/blob/main/deploy/docker/README.md)
- [Docker Hub](https://hub.docker.com/r/unclecode/crawl4ai)

### Tutorials and Guides
- [Crawl4AI Tutorial: A Beginner's Guide - Apidog](https://apidog.com/blog/crawl4ai-tutorial/)
- [Crawl4AI Tutorial: Build a Powerful Web Crawler - Pondhouse Data](https://www.pondhouse-data.com/blog/webcrawling-with-crawl4ai)
- [Crawl4AI - a hands-on guide - ScrapingBee](https://www.scrapingbee.com/blog/crawl4ai/)
- [n8n with Crawl4AI Tutorial - OneDollarVPS](https://onedollarvps.com/blogs/n8n-with-crawl4ai-tutorial)

### API Collections
- [Crawl4AI API - Postman](https://www.postman.com/pixelao/pixel-public-workspace/collection/c26yn3l/crawl4ai-api)

### Release Notes
- [Crawl4AI v0.7.8: Stability & Bug Fix Release](https://docs.crawl4ai.com/blog/releases/v0.7.8/)
- [Crawl4AI v0.7.6 Release Notes](https://docs.crawl4ai.com/blog/releases/0.7.6/)

### GitHub Issues
- [Bug: CacheMode is NOT enabled by default #1330](https://github.com/unclecode/crawl4ai/issues/1330)
- [Bug: Docker API JSON serialization fails for ProxyConfig #1629](https://github.com/unclecode/crawl4ai/issues/1629)

---

**Document Version:** 1.0
**Last Updated:** 2026-01-20
**Crawl4AI Version:** 0.8.0
