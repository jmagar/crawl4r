# Documentation vs Implementation Gap Analysis

**Generated**: 2026-01-20
**Purpose**: Identify gaps between documented plans and actual implementation to guide development

---

## Executive Summary

Crawl4r has **excellent implementation-first development** with strong test coverage (786 tests, 87%+) and ~9,800 LOC of production code. However, there's a significant gap between aspirational specifications and actual implementation.

**Key Findings:**
1. **Web Crawl CLI**: Fully implemented (7 commands, production-ready)
2. **RAG Ingestion Pipeline**: Partially implemented (core components exist, watch command functional)
3. **Documentation Quality**: README is accurate, CLAUDE.md is comprehensive but slightly outdated on ports
4. **Specification Status**: Design docs are aspirational roadmaps, not implementation guides

**Recommendation**: Documentation accurately reflects current capabilities. Specs should be treated as future roadmaps, not current state.

---

## Documentation Files Review

### Accurate & Current Documentation

#### 1. README.md (13,318 bytes)
**Status**: ✅ Accurate and production-ready

**Strengths:**
- Comprehensive CLI command documentation with examples
- Accurate service configuration (ports, environment variables)
- Real prerequisites (GPU required, Docker setup)
- Working quickstart instructions
- Test coverage accurately reported (87%+, 752 tests - slightly outdated count)

**Minor Issues:**
- Test count is 786 (not 752) - needs update
- Some service ports in README don't match .env.example (see Port Analysis below)

#### 2. CLAUDE.md (15,709 bytes)
**Status**: ✅ Mostly accurate with minor outdated sections

**Strengths:**
- Excellent "Python Environment" section with venv activation patterns
- Accurate "Crawl4AIReader" documentation matches implementation
- Correct service architecture descriptions
- Valid critical implementation notes (port management, embedding dimensions)

**Outdated Sections:**
1. **Port Configuration** (lines 383-389):
   - Says Crawl4AI is on 52004 ✅ (correct)
   - Says TEI is on 52000 ✅ (correct, remote)
   - Says Qdrant HTTP is 52001 ✅ (correct)
   - Says Qdrant gRPC is 52002 ✅ (correct)
   - Says Redis is 53379 ✅ (correct)
   - Says Postgres is 53432 ✅ (correct)
   - **Actually all correct!**

2. **"Current State vs Planned State"** (line 579):
   - Says "no Python code" - FALSE
   - Says "specifications complete" - TRUE (but aspirational)
   - Says "Next Step: Task 1.1.1" - OUTDATED (Phase 1-4 completed)

**Recommendation**: Update "Current State" section to reflect Phase 1-4 completion.

#### 3. .docs/research/codebase-architecture.docs.md (31,824 bytes)
**Status**: ✅ Excellent and comprehensive

**Strengths:**
- Accurate implementation patterns documented
- Real gotchas captured (venv requirement, remote TEI, Docker network)
- Complete architectural patterns (circuit breaker, retry, metadata keys)
- Test organization reflects reality
- Service dependencies correctly mapped

**This is the gold standard for developer onboarding.**

---

### Aspirational Documentation (Roadmaps, Not Reality)

#### 4. specs/rag-ingestion/ (8,210 lines total)
**Status**: ⚠️ Aspirational design, partially implemented

**Files:**
- `requirements.md` (381 lines) - User stories with acceptance criteria
- `design.md` (2,764 lines) - Detailed technical design (100KB+)
- `tasks.md` (1,809 lines) - 47 tasks in 3 phases
- `research.md` (690 lines) - Technical research
- `decisions.md` (438 lines) - Architectural decisions
- `technical-review.md` (1,082 lines) - Design review

**Reality Check:**
| Component | Spec Status | Implementation Status | Gap |
|-----------|-------------|----------------------|-----|
| Configuration Module | Detailed spec (150 lines) | ✅ Implemented (`core/config.py`) | None |
| File Watcher | Detailed spec | ✅ Implemented (`readers/file_watcher.py`) | None |
| Document Processor | Detailed spec | ✅ Implemented (`processing/processor.py`) | None |
| TEI Client | Detailed spec | ✅ Implemented (`storage/tei.py`) | None |
| Qdrant Store | Detailed spec | ✅ Implemented (`storage/qdrant.py`) | None |
| Circuit Breaker | Detailed spec | ✅ Implemented (`resilience/circuit_breaker.py`) | None |
| Watch Command | Detailed spec | ✅ Implemented (`cli/commands/watch.py`) | None |

**Verdict**: Core RAG pipeline is actually implemented! Specs describe an aspirational enhanced version.

**What's Missing from Specs:**
- Web crawl CLI commands (scrape, map, extract, screenshot) - implemented but not in RAG specs
- Redis queue coordination - implemented but not in RAG specs
- Depth-based crawling - implemented but not in RAG specs

#### 5. specs/web-crawl-cli/ (5 files)
**Status**: ✅ Implementation matches specifications

**Files:**
- `requirements.md` - User stories for CLI commands
- `design.md` - Technical design for web crawling
- `tasks.md` - Task breakdown
- `research.md` - Crawl4AI research

**Implementation Status:**
- Phase 1: ScraperService, MapperService ✅ Complete
- Phase 2: ExtractService, ScreenshotService ✅ Complete
- Phase 3: IngestionService with Redis queue ✅ Complete
- Phase 4: Test coverage ✅ Complete (786 tests)

**These specs accurately guided implementation.**

#### 6. docs/plans/complete/ (10 completed plans)
**Status**: ✅ Historical record, accurately reflects implementation journey

**Plans:**
- 2026-01-19-web-crawl-cli-phase-4.md (moved to complete)
- 2026-01-19-web-crawl-cli-phase-1-3.md (completed)
- 2026-01-17-use-simpledirectoryreader.md (completed)
- 2026-01-16-reorganize-rag-ingestion.md (completed)

**These are implementation logs, not specs.**

---

## Port Configuration Analysis

### Documented Ports vs Actual Configuration

| Service | README | .env.example | .env (actual) | docker-compose.yaml | Status |
|---------|--------|--------------|---------------|---------------------|--------|
| Crawl4AI | 52004 | N/A | 52004 | 52004 | ✅ Consistent |
| TEI (remote) | 52000 | N/A | Commented out | Commented out | ✅ Correct (remote) |
| Qdrant HTTP | 52001 | 52001 | 52001 | 52001 | ✅ Consistent |
| Qdrant gRPC | 52002 | 52002 | 52002 | 52002 | ✅ Consistent |
| Postgres | 53432 | 53432 | 53432 | 53432 | ✅ Consistent |
| Redis | 53379 | 53379 | 53379 | 53379 | ✅ Consistent |

**Verdict**: Port documentation is accurate and consistent across all sources.

### .docs/services-ports.md Status
**Status**: ⚠️ Outdated and incomplete

**Current Content** (last updated 2026-01-10):
```
| Service | Purpose | Container Port | Host Port | Notes |
| crawl4r-crawl4ai | Crawl4AI API | 11235 | 52001 | ... |
```

**Issues:**
1. Says Crawl4AI is on 52001 - WRONG (should be 52004)
2. Missing all other services (Qdrant, Redis, Postgres)
3. Last updated 10 days ago

**Recommendation**: Update this file to match current service configuration.

---

## Environment Configuration Analysis

### .env.example vs .env Reality

#### .env.example (66 lines)
**Required Fields:**
- `WATCH_FOLDER` - No default ✅ (correct, user must set)
- `POSTGRES_PASSWORD` - No default ✅ (correct, security)

**Defaults Provided:**
- `TEI_ENDPOINT=http://crawl4r-embeddings:80` - OK (would work if local)
- `QDRANT_URL=http://crawl4r-vectors:6333` - ❌ Should be `:52001` (host port)
- `CRAWL4AI_PORT` - Not in .env.example ❌ (should document)
- `TEI_HTTP_PORT` - Not in .env.example ❌ (should document for future)

#### .env (actual, 115 lines)
**Reality:**
- TEI section fully commented out (lines 23-30) ✅ Correct, uses remote
- All service ports configured (52000-53432 range) ✅
- Environment-aware URLs enabled in config.py ✅
- LLM API keys for extract command ✅

**Gap**: .env.example is minimal, .env has full production config

**Recommendation**: Add commented sections to .env.example showing all available options.

---

## Implementation Status by Component

### Fully Implemented Components ✅

#### CLI Commands (7 commands)
1. **scrape** - Extract markdown from URLs
   - Files: `cli/commands/scrape.py` (3,848 bytes)
   - Service: `services/scraper.py`
   - Tests: `tests/unit/test_scraper_service.py` (44 tests)
   - Status: ✅ Production ready

2. **crawl** - Ingest URLs into vector store
   - Files: `cli/commands/crawl.py` (5,977 bytes)
   - Service: `services/ingestion.py`
   - Tests: `tests/unit/test_ingestion_service.py`
   - Status: ✅ Production ready with Redis queue

3. **map** - Discover URLs from page
   - Files: `cli/commands/map.py` (3,438 bytes)
   - Service: `services/mapper.py`
   - Tests: `tests/unit/test_mapper_service.py`
   - Status: ✅ Production ready

4. **extract** - LLM-powered structured extraction
   - Files: `cli/commands/extract.py` (4,264 bytes)
   - Service: `services/extract.py`
   - Tests: `tests/unit/test_extract_service.py`
   - Status: ✅ Production ready

5. **screenshot** - Capture page screenshots
   - Files: `cli/commands/screenshot.py` (3,588 bytes)
   - Service: `services/screenshot.py`
   - Tests: `tests/unit/test_screenshot_service.py`
   - Status: ✅ Production ready

6. **status** - View crawl status
   - Files: `cli/commands/status.py` (2,554 bytes)
   - Service: `services/queue.py`
   - Tests: `tests/unit/test_queue_manager.py`
   - Status: ✅ Production ready

7. **watch** - Monitor directory for changes
   - Files: `cli/commands/watch.py` (7,892 bytes)
   - Reader: `readers/file_watcher.py`
   - Tests: `tests/unit/test_watch_command.py` (18 tests)
   - Status: ✅ Production ready

#### Core Services
- **ScraperService** ✅ (Crawl4AI HTTP wrapper)
- **MapperService** ✅ (BFS URL discovery)
- **IngestionService** ✅ (Coordinator with Redis queue)
- **QueueManager** ✅ (Redis-backed status tracking)
- **ExtractService** ✅ (LLM extraction)
- **ScreenshotService** ✅ (Page capture)

#### Storage Layer
- **VectorStoreManager** ✅ (Qdrant operations, 1,200+ LOC)
- **TEIClient** ✅ (Embeddings with circuit breaker, 500+ LOC)
- **TEIEmbedding** ✅ (LlamaIndex wrapper)

#### Processing Pipeline
- **DocumentProcessor** ✅ (LlamaIndex integration, 400+ LOC)
- **Crawl4AIReader** ✅ (BasePydanticReader implementation, 800+ LOC)
- **FileWatcher** ✅ (Watchdog handler, 600+ LOC)

#### Resilience
- **CircuitBreaker** ✅ (State machine with timeout)
- **RetryPolicy** ✅ (Exponential backoff with jitter)
- **FailedDocLogger** ✅ (JSONL audit trail)

#### Configuration
- **Settings** ✅ (Pydantic with validation, environment-aware)
- **MetadataKeys** ✅ (Centralized constants)
- **URL Validation** ✅ (Security checks for localhost, private IPs)

---

### Partially Implemented Components ⚠️

#### API Layer (Skeleton Only)
**Files:**
- `api/app.py` (FastAPI skeleton)
- `api/routes/health.py` (Health endpoint only)
- `api/models/responses.py` (Response models)

**Status**: Basic structure exists, not functional
**Gap**: No API endpoints beyond health check
**Documented**: README says "See `specs/` for detailed design docs" but no API spec exists

#### Instrumentation (Defined but Unused)
**Files:**
- `core/instrumentation.py` (custom LlamaIndex events)
- `init_observability()` function exists

**Status**: Code exists but not integrated
**Gap**: OpenTelemetry not configured, events not consumed

---

### Not Implemented (Documented but Missing)

#### From RAG Ingestion Specs
**None** - All core components are implemented!

The specs describe an enhanced version with:
- Advanced circuit breaker strategies (already implemented)
- Comprehensive error handling (already implemented)
- State recovery (already implemented in watch command)
- Batch processing (already implemented)

**Verdict**: Specs were planning documents, implementation exceeded them.

#### From CLAUDE.md "Future State"
1. **REST API endpoints** (beyond health check)
2. **PostgreSQL usage** (service running but unused)
3. **Advanced LLM extraction** (basic version works)
4. **OpenTelemetry integration** (code exists but disabled)

---

## Test Coverage Reality Check

### Documented vs Actual Test Count

| Source | Test Count | Status |
|--------|-----------|--------|
| README.md | 752 tests | ❌ Outdated |
| Git status | N/A | - |
| Pytest collection | 786 tests | ✅ Actual |
| CLAUDE.md | "752 passing tests" | ❌ Outdated |
| codebase-architecture.docs.md | 786 tests | ✅ Correct |

**Actual Test Distribution:**
```
collected 786 items

tests/integration/ - 7 CLI integration tests
tests/integration/ - 5 Crawl4AI reader tests
tests/integration/ - 8 E2E core tests (skipped without services)
tests/integration/ - 1 E2E crawl pipeline (skipped)
tests/integration/ - 7 E2E error tests
tests/unit/ - 758+ unit tests
```

**Coverage**: 87.40% (accurate across all documentation)

**Recommendation**: Update README.md and CLAUDE.md test count to 786.

---

## Service Management Reality

### Running Services (Current State)

**Command**: `docker ps --format "table {{.Names}}\t{{.Status}}"`

**Results** (as of analysis):
```
NAMES             STATUS
crawl4ai          Up 2 hours (healthy)
crawl4r-vectors   Up 2 hours (healthy)
crawl4r-db        Up 2 hours (healthy)
crawl4r-cache     Up 2 hours (healthy)
```

**TEI Service**: Remote at `100.74.16.82:52000` (not local)
- Deployed on: `steamy-wsl:/home/jmagar/compose/crawl4r/`
- GPU: RTX 4070 12GB
- Performance: 59 emb/s (2.8x faster than local RTX 3050 8GB)

**Postgres Status**: Running but unused in current implementation
- Reason: Reserved for future API features (user management, API keys)
- Cost: Minimal (Alpine image, idle CPU/memory)

---

## Critical Gaps to Address

### 1. Outdated Service Port Documentation
**File**: `.docs/services-ports.md`
**Issue**: Last updated 2026-01-10, only lists Crawl4AI with wrong port
**Impact**: Developer confusion when debugging service connections
**Fix**: Update to comprehensive service registry

**Proposed Format:**
```markdown
# Service Ports

Last updated: 2026-01-20

| Service | Container Name | Internal Port | Host Port | Status | Notes |
|---------|---------------|---------------|-----------|--------|-------|
| Crawl4AI | crawl4ai | 11235 | 52004 | Running | Web crawling service |
| Qdrant HTTP | crawl4r-vectors | 6333 | 52001 | Running | Vector database |
| Qdrant gRPC | crawl4r-vectors | 6334 | 52002 | Running | gRPC endpoint |
| Redis | crawl4r-cache | 6379 | 53379 | Running | Queue coordination |
| PostgreSQL | crawl4r-db | 5432 | 53432 | Running | Unused (reserved) |
| TEI | N/A | 80 | 52000 | Remote | steamy-wsl:52000 |
```

### 2. Minimal .env.example
**File**: `.env.example`
**Issue**: Only documents 15 variables, actual .env has 50+
**Impact**: New developers miss optional configuration
**Fix**: Add commented sections for all services

**Proposed Addition:**
```bash
# =============================================================================
# Service Ports (Docker Compose)
# =============================================================================
CRAWL4AI_PORT=52004
QDRANT_HTTP_PORT=52001
QDRANT_GRPC_PORT=52002
POSTGRES_PORT=53432
REDIS_PORT=53379

# =============================================================================
# Service URLs (Auto-detected by Settings class)
# =============================================================================
# Docker: Uses container names (crawl4r-*, crawl4ai)
# Host: Uses localhost with HOST_PORTS above
# Override only if running custom configuration
# CRAWL4AI_BASE_URL=http://localhost:52004
# QDRANT_URL=http://localhost:52001
# REDIS_URL=redis://localhost:53379
```

### 3. README Test Count
**File**: `README.md`
**Issue**: Says 752 tests, actual is 786
**Impact**: Minor credibility issue
**Fix**: Update line 22 and 452

### 4. CLAUDE.md "Current State" Section
**File**: `CLAUDE.md`
**Issue**: Lines 579-586 say "no Python code" and "Next Step: Task 1.1.1"
**Impact**: AI assistants get confused about project maturity
**Fix**: Replace with current reality

**Proposed Update:**
```markdown
## Current Implementation Status

**Completed**: Phase 1-4 of web crawl CLI, core RAG ingestion pipeline
**Active Features**: 7 CLI commands, 786 tests (87%+ coverage), production-ready
**In Development**: API layer (skeleton only), OpenTelemetry integration

Before starting new features:
1. Review `.docs/research/codebase-architecture.docs.md` for patterns
2. Check `specs/` for architectural guidance (aspirational roadmaps)
3. Follow test-driven development (TDD) approach
4. Ensure all services are running (`docker compose ps`)
```

### 5. Deployment Log Maintenance
**File**: `.docs/deployment-log.md`
**Issue**: Last entry 2026-01-10, missing recent service changes
**Impact**: No audit trail for infrastructure changes
**Fix**: Add entries for current service configuration

---

## Specification Usage Guidance

### How to Interpret specs/

#### specs/rag-ingestion/
**Purpose**: Aspirational architecture design (completed 2026-01-14)
**Status**: Core implementation complete, specs describe enhanced version
**Use Case**: Reference for architectural patterns, not step-by-step tasks

**Key Files:**
- `design.md` (2,764 lines) - System architecture, component specs
- `requirements.md` (381 lines) - User stories with acceptance criteria
- `tasks.md` (1,809 lines) - Task breakdown (mostly completed)

**Recommendation**: Use as architectural reference, not implementation plan

#### specs/web-crawl-cli/
**Purpose**: Implementation plan for web crawling features (2026-01-18)
**Status**: Phases 1-4 completed, specs accurately guided implementation
**Use Case**: Historical reference, good example of spec-to-implementation flow

**Recommendation**: Keep as example of successful spec-driven development

#### specs/llamaindex-crawl4ai-reader/
**Purpose**: Design for Crawl4AIReader LlamaIndex integration
**Status**: Implemented in `readers/crawl4ai.py`
**Use Case**: Completed feature, reference for reader patterns

**Recommendation**: Archive as completed spec

---

## Documentation Quality Assessment

### Excellent Documentation ✅

1. **README.md**
   - Score: 9/10
   - Comprehensive CLI examples
   - Accurate prerequisites and setup
   - Minor: Test count outdated

2. **.docs/research/codebase-architecture.docs.md**
   - Score: 10/10
   - Perfect for developer onboarding
   - Accurate implementation patterns
   - Real gotchas documented

3. **CLAUDE.md (Python Environment section)**
   - Score: 10/10
   - Critical venv activation patterns
   - Real-world command examples
   - Saves hours of debugging

### Good Documentation ✅

4. **specs/web-crawl-cli/**
   - Score: 8/10
   - Guided successful implementation
   - Clear requirements and design
   - Good spec-to-code traceability

5. **docs/plans/complete/**
   - Score: 8/10
   - Valuable implementation history
   - Shows evolution of architecture
   - Good for understanding "why" decisions

### Needs Improvement ⚠️

6. **specs/rag-ingestion/**
   - Score: 6/10
   - Excellent design documentation
   - BUT: Reads like unimplemented roadmap
   - Reality: Most of it is already built
   - Fix: Add "Implementation Status" sections

7. **.docs/services-ports.md**
   - Score: 3/10
   - Outdated (10 days old)
   - Incomplete (1 service of 6)
   - Wrong port number
   - Fix: Comprehensive update needed

8. **.docs/deployment-log.md**
   - Score: 4/10
   - Outdated (10 days old)
   - Missing recent deployments
   - Fix: Add current service status

9. **.env.example**
   - Score: 5/10
   - Minimal documentation
   - Missing most optional settings
   - Fix: Add commented sections for all options

### Missing Documentation ❌

10. **API Documentation**
    - Status: None exists
    - Need: OpenAPI spec for future API layer
    - Priority: Low (API is skeleton only)

11. **Service Dependency Graph**
    - Status: Described in prose, no diagram
    - Need: Mermaid diagram showing service interactions
    - Priority: Medium (helpful for debugging)

12. **Troubleshooting Guide**
    - Status: Basic section in README
    - Need: Common errors with solutions
    - Priority: Medium (improves DX)

---

## Practical Information Developers Need

### Essential Onboarding Documents (Priority Order)

1. **README.md** - Start here for CLI usage and setup
2. **.docs/research/codebase-architecture.docs.md** - Comprehensive architecture guide
3. **CLAUDE.md** - Python environment and critical notes
4. **specs/web-crawl-cli/design.md** - Understand CLI architecture
5. **docker-compose.yaml** - Service configuration reference

### Common Tasks and Where to Find Documentation

#### "How do I run the CLI?"
- **CLAUDE.md** lines 1-50 (venv activation patterns)
- **README.md** lines 50-363 (all CLI commands with examples)

#### "How do I add a new CLI command?"
- **codebase-architecture.docs.md** lines 192-230 (CLI command patterns)
- **specs/web-crawl-cli/design.md** (service layer architecture)
- **Example**: `cli/commands/scrape.py` (simple command)

#### "How does the RAG pipeline work?"
- **codebase-architecture.docs.md** lines 360-398 (processing pipeline)
- **CLAUDE.md** lines 13-22 (RAG features overview)
- **specs/rag-ingestion/design.md** (detailed architecture)

#### "Why is my service connection failing?"
- **CLAUDE.md** lines 489-530 (troubleshooting)
- **codebase-architecture.docs.md** lines 566-629 (gotchas)
- **.env.example** (service configuration)

#### "How do I write tests?"
- **codebase-architecture.docs.md** lines 453-512 (testing architecture)
- **Example**: `tests/unit/test_scraper_service.py` (respx mocking)
- **Example**: `tests/integration/test_crawl4ai_reader_integration.py` (real services)

#### "What are the architectural patterns?"
- **codebase-architecture.docs.md** lines 62-190 (10 key patterns)
- **specs/rag-ingestion/design.md** (detailed component specs)
- **CLAUDE.md** lines 95-178 (critical implementation notes)

---

## Recommendations

### Immediate Actions (High Priority)

1. **Update .docs/services-ports.md**
   - Add all 6 services with correct ports
   - Include remote TEI configuration
   - Add timestamp for last update

2. **Update README.md test count**
   - Change 752 to 786 tests (2 occurrences)
   - Verify coverage percentage is still accurate

3. **Update CLAUDE.md "Current State" section**
   - Replace lines 579-586 with implementation status
   - Remove "no Python code" statement
   - Add Phase 1-4 completion notes

4. **Expand .env.example**
   - Add commented sections for all services
   - Document port configuration
   - Show environment-aware URL patterns

### Short-term Actions (Medium Priority)

5. **Add Implementation Status to specs/rag-ingestion/**
   - Annotate requirements.md with ✅/⚠️/❌ status
   - Update design.md intro with "Implementation Status" section
   - Mark completed tasks in tasks.md

6. **Update .docs/deployment-log.md**
   - Add current service configuration
   - Document remote TEI deployment decision
   - Add recent infrastructure changes

7. **Create Troubleshooting Guide**
   - Common errors (venv not activated, service not running)
   - Service health check commands
   - Port conflict resolution

### Long-term Actions (Low Priority)

8. **Create API Specification**
   - OpenAPI/Swagger spec for future REST API
   - Define endpoints, request/response models
   - Authentication strategy

9. **Add Service Dependency Diagram**
   - Mermaid diagram in README or docs/
   - Show CLI → Services → Storage → External dependencies
   - Highlight remote TEI deployment

10. **Archive Completed Specs**
    - Move specs/llamaindex-crawl4ai-reader/ to specs/archive/
    - Add README to specs/ explaining structure
    - Document spec lifecycle (planning → implementation → archive)

---

## Conclusion

### What's Working Well

1. **Implementation Quality**: High test coverage, production-ready code
2. **Architecture Documentation**: Excellent internal docs for developers
3. **README**: Comprehensive user-facing documentation
4. **Spec-Driven Development**: Web crawl CLI specs guided successful implementation

### What Needs Improvement

1. **Spec Status Clarity**: RAG specs read as unimplemented, but they're mostly done
2. **Service Documentation**: Outdated port registry and deployment log
3. **Environment Configuration**: .env.example is too minimal
4. **Test Count**: Minor inconsistency in documented test count

### Key Insight

Crawl4r has **excellent implementation** that exceeded its specifications. The gap is not missing features - it's documentation that doesn't reflect the current mature state of the codebase.

**Priority**: Update documentation to match reality, not implement missing features.

### Next Steps for AI Assistants

When working on this codebase:

1. ✅ Trust the implementation over the specs
2. ✅ Use codebase-architecture.docs.md as primary reference
3. ✅ Refer to specs for architectural patterns, not task lists
4. ✅ Always activate venv before Python commands
5. ✅ Check docker compose ps for service health
6. ✅ Write tests first (TDD approach maintained throughout)
7. ✅ Use MetadataKeys constants (never hardcode metadata keys)
8. ✅ Respect circuit breakers (don't bypass for convenience)
9. ✅ Follow async-first patterns (all I/O is async)
10. ✅ Update documentation when making changes

---

**Document Version**: 1.0
**Last Updated**: 2026-01-20
**Maintainer**: Auto-generated by Claude Code analysis
