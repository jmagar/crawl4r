---
spec: web-crawl-cli
phase: research
created: 2026-01-18T00:00:00Z
---

# Research: web-crawl-cli

## Executive Summary

The web-crawl-cli spec aims to add web crawling commands (scrape, crawl, map, extract, screenshot, status) to the existing crawl4r CLI. Research validates that the codebase already contains all necessary components: Crawl4AIReader for web crawling, TEIClient/VectorStoreManager for ingestion, and CircuitBreaker for resilience. The design document is well-aligned with existing patterns. Minor gaps exist around Redis dependency and the CLI entry point mechanism. Implementation is highly feasible with estimated M effort (2-3 sprints).

## External Research

### Typer CLI Best Practices

Based on [Typer documentation](https://typer.tiangolo.com/features/) and [PyTutorial async guide](https://pytutorial.com/python-typer-async-command-support-guide/):

- **Type hint based**: Typer uses Python type hints for command arguments and options - no new syntax needed
- **Subcommand trees**: `@app.command()` decorator creates subcommands; larger apps can use separate modules with `app.add_typer()`
- **Async support**: Native async support exists but has caveats. The [async-typer](https://pypi.org/project/async-typer/) package provides simpler async wrapper, though Typer itself handles event loops internally
- **Best practices**:
  - Use `no_args_is_help=True` to show help when no arguments provided
  - Organize larger CLIs with separate command modules
  - Shell completion is automatic for all OS/shells

**Recommendation**: Use native Typer async support. Wrap async services in sync commands using `asyncio.run()` similar to existing `load_data()` pattern in Crawl4AIReader.

### Rich Progress Display Patterns

Based on [Rich Progress documentation](https://rich.readthedocs.io/en/stable/progress.html):

- **`track()` function**: Simple progress bar for iterating over sequences
- **Multiple concurrent tasks**: Progress display supports multiple tasks with individual bars
- **Custom columns**: SpinnerColumn, TransferSpeedColumn, DownloadColumn available
- **Transient displays**: `transient=True` removes progress on completion
- **Printing during progress**: Output displayed above progress bar
- **Unknown progress**: `console.status()` for spinner animations

**Recommendation**: Use `Progress()` context manager with custom columns showing URL being crawled and chunk counts. For crawl queue operations, show queue position with spinner.

### Redis Queue Coordination Patterns

Based on [Redis Queue documentation](https://redis.io/glossary/redis-queue/) and [Python Redis Queue patterns](https://www.wafermovement.com/2021/redisqueuewithpython/):

- **LPUSH/RPOP pattern**: Standard queue (FIFO) - push left, pop right
- **BRPOP blocking**: Blocks until item available, eliminates polling
- **Reliable queue**: RPOPLPUSH moves item to processing list, LREM removes after completion
- **Lock pattern**: SETNX with TTL for distributed locking
- **Cross-process coordination**: Redis guarantees atomic operations

**Design alignment**: The design document's lock/queue pattern (lines 523-536) aligns with Redis best practices:
- Lock key: `crawl4r:crawl:lock` with TTL (1 hour)
- Queue key: `crawl4r:crawl:queue` using LPUSH/RPOP
- Status keys: `crawl4r:status:{crawl_id}` with 24-hour expiry

**Recommendation**: Use `redis-py` async client with SETNX for locking and LPUSH/BRPOP for queue. Implement reliable queue pattern for crash recovery.

## Codebase Analysis

### Existing CLI Structure

**Current state** (`crawl4r/cli/`):
- `__init__.py` - Empty module docstring
- `commands/__init__.py` - Empty (1 line)
- `main.py` - File watcher orchestration (379 lines)

**Key observation**: The current `main.py` is NOT a Typer CLI - it's the file watcher entry point that runs `asyncio.run(main())`. The design proposes converting this to a Typer-based CLI with subcommands.

**Entry point**: No `[project.scripts]` entry in `pyproject.toml` currently. Design proposes:
```toml
[project.scripts]
crawl4r = "crawl4r.cli.app:app"
```

### Existing Components for Reuse

| Component | Location | Reuse For | Status |
|-----------|----------|-----------|--------|
| `Crawl4AIReader` | `crawl4r/readers/crawl4ai.py` | ScraperService, IngestionService | **Exists** (958 lines) |
| `CircuitBreaker` | `crawl4r/resilience/circuit_breaker.py` | All services | **Exists** (250 lines) |
| `TEIClient` | `crawl4r/storage/tei.py` | IngestionService embeddings | **Exists** (390 lines) |
| `VectorStoreManager` | `crawl4r/storage/qdrant.py` | IngestionService storage | **Exists** (948 lines) |
| `MetadataKeys` | `crawl4r/core/metadata.py` | Consistent field names | **Exists** (45 lines) |
| `Settings` | `crawl4r/core/config.py` | Service configuration | **Exists** (181 lines) |
| `DocumentProcessor` | `crawl4r/processing/processor.py` | Ingestion pipeline | **Exists** (602 lines) |
| `ProcessingResult` | `crawl4r/processing/processor.py` | Result dataclass | **Exists** |

### Pattern Analysis

**1. Async HTTP pattern** (from Crawl4AIReader):
```python
async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
    response = await client.post(f"{self.endpoint_url}/md", json={"url": url, "f": "fit"})
```
This pattern should be used for all Crawl4AI service calls.

**2. Circuit breaker wrapping** (from Crawl4AIReader):
```python
async def _impl() -> Document:
    # actual implementation
    ...
return await self._circuit_breaker.call(_impl)
```

**3. Retry with backoff** (from TEIClient):
```python
for attempt in range(self.max_retries):
    try:
        result = await operation()
        return result
    except transient_errors:
        if attempt == self.max_retries - 1:
            raise
        await asyncio.sleep(2**attempt)
```

**4. Deterministic ID generation** (from VectorStoreManager):
```python
hash_bytes = hashlib.sha256(f"{identifier}".encode()).digest()
return str(uuid.UUID(bytes=hash_bytes[:16]))
```

**5. Deduplication before ingestion** (from Crawl4AIReader):
```python
if self.enable_deduplication and self.vector_store is not None:
    for url in urls:
        await self._deduplicate_url(url)
```

### Validation: Design vs Existing Code

| Design Proposal | Existing Code | Gap |
|-----------------|---------------|-----|
| ScraperService using `/md?f=fit` | Crawl4AIReader uses `/crawl` endpoint | Minor - need separate scraper using `/md` |
| IngestionService with TEI+Qdrant | DocumentProcessor does full pipeline | Minor - adapt for URL source |
| CircuitBreaker in services | Already integrated in TEIClient, Crawl4AIReader | None |
| MetadataKeys constants | Exists in `core/metadata.py` | None |
| Point ID generation | Exists in VectorStoreManager | None |
| Deduplication via delete_by_url | Exists in VectorStoreManager | None |
| Redis queue coordination | **Not implemented** | **Gap** - Need redis dependency |
| Typer CLI framework | **Not implemented** | **Gap** - Need typer dependency |

### Proposed Directory Structure Validation

```
crawl4r/
├── services/           # NEW - Core business logic
│   ├── __init__.py
│   ├── scraper.py     # Uses httpx + /md endpoint
│   ├── mapper.py      # Uses /crawl endpoint links
│   ├── extractor.py   # Uses /llm/job endpoint
│   ├── screenshot.py  # Uses /screenshot endpoint
│   └── ingestion.py   # Wraps DocumentProcessor + Redis queue
│
├── cli/
│   ├── app.py         # NEW - Typer app entry point
│   └── commands/
│       ├── scrape.py  # NEW
│       ├── crawl.py   # NEW
│       ├── map.py     # NEW
│       ├── extract.py # NEW
│       ├── screenshot.py # NEW
│       ├── watch.py   # REFACTOR from main.py
│       └── status.py  # NEW
```

**Conflicts**: None. The `services/` directory is new, and CLI commands are new modules.

### Dependencies Analysis

**Current dependencies** (from `pyproject.toml`):
- `httpx==0.28.1` - HTTP client (exists)
- `pydantic>=2.0.0` - Settings/models (exists)
- `pydantic-settings>=2.0.0` - Environment config (exists)

**Missing dependencies**:
- `typer` - CLI framework (NOT in dependencies)
- `redis` / `redis[hiredis]` - Queue coordination (NOT in dependencies)
- `rich` - Progress display (bundled with typer[all])

**Recommendation**: Add to `pyproject.toml`:
```toml
dependencies = [
    ...
    "typer[all]>=0.12.0",  # Includes rich
    "redis>=5.0.0",
]
```

## Related Specs

| Spec | Relevance | May Need Update |
|------|-----------|-----------------|
| `rag-ingestion` | **High** - Shares ingestion pipeline, TEI, Qdrant components | No - CLI extends, doesn't change |
| `llamaindex-crawl4ai-reader` | **High** - ScraperService may reuse Crawl4AIReader patterns | No - Uses existing reader |

### rag-ingestion Summary
The RAG ingestion spec defines the core pipeline architecture (TEI -> chunking -> Qdrant) that the new CLI will leverage. The DocumentProcessor class provides the template for web document ingestion.

### llamaindex-crawl4ai-reader Summary
This spec implemented the Crawl4AIReader which provides async web crawling with circuit breaker, retry logic, and deduplication. The new ScraperService can reuse patterns but needs to use `/md` endpoint instead of `/crawl` for cleaner markdown extraction.

## Feasibility Assessment

| Aspect | Assessment | Notes |
|--------|------------|-------|
| Technical Viability | **High** | All core components exist, just need orchestration |
| Effort Estimate | **M (2-3 sprints)** | 6-8 services + 7 CLI commands + tests |
| Risk Level | **Low** | Design well-aligned with existing patterns |

### Key Risks

1. **Redis integration complexity**: Queue coordination adds state management. Mitigate with comprehensive integration tests.

2. **Crawl4AI API differences**: Design references `/md` endpoint which differs from Crawl4AIReader's `/crawl` usage. Both are valid - `/md?f=fit` returns cleaner markdown.

3. **Async in sync CLI**: Typer commands are sync; async services need `asyncio.run()` wrapping. Established pattern from Crawl4AIReader.

## Quality Commands

| Type | Command | Source |
|------|---------|--------|
| Lint | `ruff check .` | pyproject.toml [tool.ruff] |
| TypeCheck | `ty check crawl4r/` | pyproject.toml [tool.ty] |
| Unit Test | `pytest tests/unit/` | pyproject.toml [tool.pytest.ini_options] |
| Integration Test | `pytest tests/integration/` | pyproject.toml [tool.pytest.ini_options] |
| Test (all) | `pytest` | pyproject.toml [tool.pytest.ini_options] |
| Test w/ Coverage | `pytest --cov=crawl4r --cov-report=term` | pyproject.toml [tool.coverage] |

**Local CI**: `ruff check . && ty check crawl4r/ && pytest`

## Recommendations for Requirements

1. **Reuse Crawl4AIReader patterns**: Circuit breaker, retry logic, and async HTTP patterns are battle-tested. Services should follow the same structure.

2. **Use /md endpoint for scraping**: The design correctly identifies `/md?f=fit` as the optimal endpoint for clean markdown (12K vs 89K chars).

3. **Add typer and redis dependencies**: These are required but not currently in dependencies.

4. **Implement services as thin wrappers**: Each service should be a thin async wrapper around httpx calls, reusing existing resilience patterns.

5. **Refactor main.py to watch command**: The existing file watcher logic should move to `cli/commands/watch.py` and be exposed as a Typer command.

6. **Test Redis coordination thoroughly**: Cross-process queue coordination is the most complex new feature; needs dedicated integration tests.

7. **Consider progress display patterns**: Use Rich's `Progress()` with custom columns for URL status and chunk counts.

## Open Questions

1. **Should ScraperService extend or wrap Crawl4AIReader?** The reader uses `/crawl` while scraper needs `/md`. Recommend: separate service using same patterns, not inheritance.

2. **Redis connection pooling**: Should services share a Redis connection pool? Recommend: Yes, via dependency injection pattern.

3. **CLI entry point for existing main.py**: How should the transition happen? Recommend: Alias current behavior to `crawl4r watch` command.

4. **Screenshot format support**: Design notes "Only PNG supported by Crawl4AI" - confirm this limitation in implementation.

## Sources

### Web Sources
- [Typer Features](https://typer.tiangolo.com/features/)
- [Typer Commands](https://typer.tiangolo.com/tutorial/commands/)
- [async-typer PyPI](https://pypi.org/project/async-typer/)
- [PyTutorial Typer Async Guide](https://pytutorial.com/python-typer-async-command-support-guide/)
- [Rich Progress Documentation](https://rich.readthedocs.io/en/stable/progress.html)
- [Redis Queue Patterns](https://redis.io/glossary/redis-queue/)
- [Python Redis Queue Tutorial](https://www.wafermovement.com/2021/redisqueuewithpython/)
- [RQ (Redis Queue) for Python](https://python-rq.org/)

### Codebase Sources
- `/home/jmagar/workspace/crawl4r/docs/plans/2026-01-18-cli-web-crawling-design.md` - Design document
- `/home/jmagar/workspace/crawl4r/crawl4r/readers/crawl4ai.py` - Crawl4AIReader implementation
- `/home/jmagar/workspace/crawl4r/crawl4r/resilience/circuit_breaker.py` - CircuitBreaker pattern
- `/home/jmagar/workspace/crawl4r/crawl4r/storage/tei.py` - TEIClient implementation
- `/home/jmagar/workspace/crawl4r/crawl4r/storage/qdrant.py` - VectorStoreManager implementation
- `/home/jmagar/workspace/crawl4r/crawl4r/core/metadata.py` - MetadataKeys constants
- `/home/jmagar/workspace/crawl4r/crawl4r/core/config.py` - Settings class
- `/home/jmagar/workspace/crawl4r/crawl4r/processing/processor.py` - DocumentProcessor
- `/home/jmagar/workspace/crawl4r/crawl4r/cli/main.py` - Current CLI entry point
- `/home/jmagar/workspace/crawl4r/pyproject.toml` - Dependencies and tooling config
