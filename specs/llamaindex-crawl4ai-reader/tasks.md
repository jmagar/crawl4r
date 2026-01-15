# Tasks: LlamaIndex Crawl4AI Reader

## Overview

This task breakdown follows strict **Test-Driven Development (TDD)** methodology as explicitly required in requirements.md. Every feature implementation follows the RED-GREEN-REFACTOR cycle with 85%+ test coverage target.

**Total Tasks: 99** (Phase 1: 15 tasks, Phase 2: 69 tasks, Phase 3: 12 tasks, Phase 4: 3 tasks)

**Critical TDD Rules**:
- Phase 1: Setup tasks (no tests needed for project structure and VectorStoreManager updates)
- Phase 2: ALL feature tasks follow RED → GREEN → REFACTOR pattern
- Each task is one atomic commit
- Tests MUST be written before implementation code
- Verify tasks must confirm tests pass/fail as expected

**New Requirements (Issues #15, #16, #17)**:
- UUID Strategy: Use deterministic UUID from SHA256 hash (matches vector_store.py pattern)
- Automatic Deduplication: Delete old documents before re-crawling (matches file watcher pattern)
- Source URL Indexing: Add source_url to PAYLOAD_INDEXES for fast deduplication queries

---

## Phase 1: Foundation & Setup

Focus: Project structure, dependencies, basic configuration, VectorStoreManager updates. No TDD required for setup tasks.

### 1.1 Project Dependencies

- [x] 1.1.1 Add httpx dependency
  - **Do**: Add `httpx==0.28.1` to pyproject.toml dependencies array (explicit dependency for async HTTP client)
  - **Files**: `/home/jmagar/workspace/crawl4r/pyproject.toml`
  - **Done when**: httpx listed in dependencies with exact version constraint
  - **Verify**: `grep -A5 'dependencies = \[' pyproject.toml | grep httpx`
  - **Commit**: `feat(deps): add httpx==0.28.1 for async HTTP client`
  - _Requirements: FR-2, NFR-12_
  - _Design: Dependencies section_

- [x] 1.1.2 Verify respx test dependency
  - **Do**: Verify `respx>=0.21.0` exists in dependency-groups.dev array in pyproject.toml (for mocking httpx requests in tests)
  - **Files**: `/home/jmagar/workspace/crawl4r/pyproject.toml`
  - **Done when**: respx found in dev dependencies (already present per research.md line 998)
  - **Verify**: `grep -A20 'dependency-groups' pyproject.toml | grep respx`
  - **Commit**: `chore(deps): verify respx test dependency exists` (only if adding, else skip commit)
  - _Requirements: NFR-2, TDD Methodology_
  - _Design: Test Strategy section_

### 1.2 VectorStoreManager Updates (Issue #17, #16)

- [x] 1.2.1 Add source_url to PAYLOAD_INDEXES
  - **Do**: Add `("source_url", PayloadSchemaType.KEYWORD)` to PAYLOAD_INDEXES list in rag_ingestion/vector_store.py (line 74-80), positioned after file_path_relative
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/vector_store.py`
  - **Done when**: source_url index added with KEYWORD type for fast deduplication queries
  - **Verify**: `grep -A10 'PAYLOAD_INDEXES' rag_ingestion/vector_store.py | grep source_url`
  - **Commit**: `feat(vector_store): add source_url to payload indexes for web crawl deduplication`
  - _Requirements: Issue #17_
  - _Design: Deduplication Strategy section, line 1174-1184_

- [x] 1.2.2 Implement delete_by_url method
  - **Do**: Add `delete_by_url(source_url: str) -> int` method to VectorStoreManager class, mirroring delete_by_file pattern (line 713-743). Use scroll API to find points with matching source_url filter, delete in batch, return count.
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/vector_store.py`
  - **Done when**: Method implemented with same pattern as delete_by_file, includes docstring with Google-style Args/Returns/Examples
  - **Verify**: `grep -A5 'def delete_by_url' rag_ingestion/vector_store.py`
  - **Commit**: `feat(vector_store): add delete_by_url for web crawl deduplication`
  - _Requirements: Issue #16_
  - _Design: Deduplication Strategy section, line 1186-1208_

- [x] V1 [VERIFY] Quality checkpoint: vector_store changes
  - **Do**: Run `ruff check rag_ingestion/vector_store.py` and `ty check rag_ingestion/vector_store.py`
  - **Verify**: Both commands exit 0
  - **Done when**: No lint errors, no type errors in vector_store.py
  - **Commit**: `chore(vector_store): pass quality checkpoint` (only if fixes needed)

### 1.3 Module Structure

- [x] 1.3.1 Create crawl4ai_reader module
  - **Do**: Create file `rag_ingestion/crawl4ai_reader.py` with module-level docstring and imports (BasePydanticReader, Document, httpx, asyncio, hashlib, uuid)
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: File exists with Google-style module docstring, imports including uuid for Issue #15
  - **Verify**: `test -f rag_ingestion/crawl4ai_reader.py && head -20 rag_ingestion/crawl4ai_reader.py`
  - **Commit**: `feat(reader): create crawl4ai_reader module skeleton`
  - _Requirements: FR-1_
  - _Design: File Structure section_

- [x] 1.3.2 Create unit test file
  - **Do**: Create `tests/unit/test_crawl4ai_reader.py` with pytest imports and fixture for reader configuration
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: File exists with imports (pytest, respx, httpx), basic module structure
  - **Verify**: `test -f tests/unit/test_crawl4ai_reader.py && head -10 tests/unit/test_crawl4ai_reader.py`
  - **Commit**: `test(reader): create unit test file skeleton`
  - _Requirements: NFR-2, TDD Methodology_
  - _Design: Test Strategy section_

- [x] 1.3.3 Create integration test file
  - **Do**: Create `tests/integration/test_crawl4ai_reader_integration.py` with pytest imports and crawl4ai_available fixture
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_crawl4ai_reader_integration.py`
  - **Done when**: File exists with imports, fixture to check service health and skip if unavailable
  - **Verify**: `test -f tests/integration/test_crawl4ai_reader_integration.py && grep -A5 'crawl4ai_available' tests/integration/test_crawl4ai_reader_integration.py`
  - **Commit**: `test(reader): create integration test file with service check`
  - _Requirements: NFR-14_
  - _Design: Test Strategy section, line 886-896_

- [x] 1.3.4 Create test fixtures file
  - **Do**: Create `tests/fixtures/crawl4ai_responses.py` with mock CrawlResult data (MOCK_CRAWL_RESULT_SUCCESS, MOCK_CRAWL_RESULT_FAILURE)
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/fixtures/crawl4ai_responses.py`
  - **Done when**: File contains at least 2 mock responses (success and failure cases) with complete CrawlResult structure
  - **Verify**: `grep -c 'MOCK_CRAWL_RESULT' tests/fixtures/crawl4ai_responses.py` (should return 2+)
  - **Commit**: `test(reader): add mock CrawlResult fixtures`
  - _Requirements: TDD Methodology_
  - _Design: Test Data Fixtures section, line 1263-1294_

### 1.4 Settings Integration (TDD Phase 2)

#### 1.4.1 [RED] Test Settings integration

- [x] 1.4.1 RED: Test Settings integration
  - **Do**: Write test in tests/unit/test_crawl4ai_reader.py that verifies Settings class has CRAWL4AI_BASE_URL field with correct default
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written and FAILS (field doesn't exist yet)
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_settings_integration -v` (must fail)
  - **Commit**: `test(config): add RED test for CRAWL4AI_BASE_URL field`
  - _Requirements: FR-1.1_
  - _Design: Configuration Integration section, line 177-188_

#### 1.4.2 [GREEN] Extend Settings class

- [x] 1.4.2 GREEN: Extend Settings class
  - **Do**: Add `CRAWL4AI_BASE_URL: str = Field(default="http://localhost:52004", description="Crawl4AI service base URL")` to Settings class in rag_ingestion/config.py
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/config.py`
  - **Done when**: Field added with type hint, default value, Field description
  - **Verify**: `grep -A2 'CRAWL4AI_BASE_URL' rag_ingestion/config.py`
  - **Commit**: `feat(config): add CRAWL4AI_BASE_URL to Settings`
  - _Requirements: FR-1.1_
  - _Design: Configuration Integration section, line 177-188_

- [x] 1.4.3 VERIFY: Settings integration test passes
  - **Do**: Run `pytest tests/unit/test_crawl4ai_reader.py::test_reader_respects_crawl4ai_base_url_from_settings -v`. Test should still fail (Crawl4AIReader class doesn't exist yet), but Settings portion is verified.
  - **Verify**: Test fails with ImportError mentioning Crawl4AIReader (expected at this stage)
  - **Done when**: Test runs and fails as expected (Settings field exists, reader class doesn't)
  - **Commit**: Skip (no code changes, verification only)
  - _Requirements: NFR-2_

- [ ] V2 [VERIFY] Quality checkpoint: foundation setup
  - **Do**: Run `ruff check .` and `ty check rag_ingestion/`
  - **Verify**: Both commands exit 0
  - **Done when**: No lint errors, no type errors
  - **Commit**: `chore(reader): pass quality checkpoint` (only if fixes needed)

---

## Phase 2: Core Implementation (TDD - RED → GREEN → REFACTOR)

Focus: Implement all features using strict TDD. Each feature has three sub-tasks: RED (write failing tests), GREEN (minimal implementation), REFACTOR (clean up code).

### 2.1 Reader Class Foundation

#### 2.1.1 [RED] Tests for Crawl4AIReader class structure

- [ ] 2.1.1a Write test for default initialization
  - **Do**: Write `test_default_initialization()` that creates Crawl4AIReader with no arguments and asserts default field values (endpoint="http://localhost:52004", timeout=60, fail_on_error=False, max_concurrent=5, is_remote=True, class_name="Crawl4AIReader")
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, runs and FAILS with "AttributeError: module 'rag_ingestion.crawl4ai_reader' has no attribute 'Crawl4AIReader'"
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_default_initialization -v` (must fail)
  - **Commit**: `test(reader): add RED test for default initialization`
  - _Requirements: AC-1.1, AC-1.2, AC-1.3, AC-1.4, FR-1_
  - _Design: Crawl4AIReader interface, line 144-211_

- [ ] 2.1.1b Write test for custom endpoint configuration
  - **Do**: Write `test_custom_endpoint()` that creates Crawl4AIReader with custom endpoint_url and asserts value is stored correctly
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, runs and FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_custom_endpoint -v` (must fail)
  - **Commit**: `test(reader): add RED test for custom endpoint configuration`
  - _Requirements: AC-1.1_

- [ ] 2.1.1c Write test for timeout validation
  - **Do**: Write `test_custom_timeout()`, `test_invalid_timeout_too_low()`, `test_invalid_timeout_too_high()` for Pydantic validation (range 10-300)
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Three tests written, all run and FAIL
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'timeout' -v` (all must fail)
  - **Commit**: `test(reader): add RED tests for timeout configuration and validation`
  - _Requirements: AC-1.2_

- [ ] 2.1.1d Write test for max_concurrent validation
  - **Do**: Write `test_custom_max_concurrent()`, `test_invalid_max_concurrent_too_low()`, `test_invalid_max_concurrent_too_high()` for validation (range 1-20)
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Three tests written, all run and FAIL
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'max_concurrent' -v` (all must fail)
  - **Commit**: `test(reader): add RED tests for max_concurrent validation`
  - _Requirements: AC-1.4_

- [ ] 2.1.1e Write test for LlamaIndex properties
  - **Do**: Write `test_is_remote_flag()` and `test_class_name_property()` to verify LlamaIndex contract compliance (is_remote=True, class_name="Crawl4AIReader")
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Two tests written, both FAIL
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'is_remote or class_name' -v` (must fail)
  - **Commit**: `test(reader): add RED tests for LlamaIndex properties`
  - _Requirements: FR-1_

#### 2.1.2 [GREEN] Implement Crawl4AIReader class skeleton

- [ ] 2.1.2a Implement minimal Crawl4AIReader class
  - **Do**: Create Crawl4AIReader class inheriting from BasePydanticReader with Pydantic fields (endpoint_url, timeout_seconds, fail_on_error, max_concurrent_requests, max_retries, retry_delays), set is_remote=True and class_name="Crawl4AIReader"
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Class exists with all fields, Field() validators (ge, le), default values match design.md line 178-207
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_default_initialization -v` (must pass)
  - **Commit**: `feat(reader): implement Crawl4AIReader class skeleton with Pydantic fields`
  - _Requirements: FR-1, US-1_
  - _Design: Crawl4AIReader interface_

- [ ] 2.1.2b Verify all configuration tests pass
  - **Do**: Run all configuration tests to confirm GREEN phase
  - **Files**: N/A
  - **Done when**: All tests from 2.1.1a-e pass
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'initialization or endpoint or timeout or max_concurrent or is_remote or class_name' -v` (all pass)
  - **Commit**: `test(reader): verify GREEN - all configuration tests pass`

#### 2.1.3 [REFACTOR] Test configuration validation

- [x] 2.1.3 REFACTOR: Test configuration validation
  - **Do**: Add tests for config validation: test_config_rejects_invalid_timeout (negative), test_config_rejects_invalid_max_retries (>10), test_config_rejects_extra_fields. All should PASS.
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: 3 validation tests added and passing
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k config -v` (all config tests pass)
  - **Commit**: `test(reader): add config validation tests`
  - _Requirements: NFR-1_
  - _Design: Configuration section_

### 2.2 Health Check Validation

#### 2.2.1 [RED] Tests for health check validation

- [x] 2.2.1a Write test for successful health check
  - **Do**: Write `test_health_check_success()` that mocks /health endpoint returning 200, creates reader, asserts no exception raised
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written using respx to mock GET /health, FAILS because __init__ doesn't call health check yet
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_health_check_success -v` (must fail)
  - **Commit**: `test(reader): add RED test for successful health check`
  - _Requirements: AC-1.5, FR-13_

- [x] 2.2.1b Write test for failed health check
  - **Do**: Write `test_health_check_failure()` that mocks /health endpoint failing (timeout or 500 error), asserts ValueError raised with clear message
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS because health check not implemented
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_health_check_failure -v` (must fail)
  - **Commit**: `test(reader): add RED test for failed health check`
  - _Requirements: AC-1.6_

- [x] 2.2.1c Write test for circuit breaker and logger initialization
  - **Do**: Write `test_circuit_breaker_initialized()` and `test_logger_initialized()` to verify internal components created in __init__
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Two tests written, both FAIL
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'circuit_breaker_initialized or logger_initialized' -v` (must fail)
  - **Commit**: `test(reader): add RED tests for circuit breaker and logger initialization`
  - _Requirements: FR-9, FR-11_

#### 2.2.2 [GREEN] Implement initialization with health check

- [x] 2.2.2a Implement __init__ with health validation
  - **Do**: Implement __init__ method that: 1) calls super().__init__(**data), 2) initializes _circuit_breaker (threshold=5, timeout=60), 3) initializes _logger via get_logger(), 4) calls _validate_health_sync() and raises ValueError if fails
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: __init__ implemented per design.md line 217-242, uses httpx.Client for sync health check
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_health_check_success -v` (must pass)
  - **Commit**: `feat(reader): implement __init__ with health check validation`
  - _Requirements: AC-1.5, AC-1.6, FR-9, FR-11, FR-13_

- [ ] 2.2.2b Implement _validate_health_sync helper
  - **Do**: Implement _validate_health_sync() private method that makes synchronous GET to {endpoint_url}/health with 10s timeout, returns True if 200, False otherwise
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Method implemented per design.md line 244-255
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'health_check' -v` (all pass)
  - **Commit**: `feat(reader): add _validate_health_sync for initialization checks`

- [ ] 2.2.2c Verify all health check tests pass
  - **Do**: Run all health check tests to confirm GREEN phase
  - **Files**: N/A
  - **Done when**: All tests from 2.2.1a-c pass
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'health or circuit_breaker or logger' -v` (all pass)
  - **Commit**: `test(reader): verify GREEN - all health check tests pass`

#### 2.2.3 [REFACTOR] Add async health check method

- [ ] 2.2.3a Add _validate_health async method
  - **Do**: Add _validate_health() async method for runtime validation (mirrors sync version but uses AsyncClient), per design.md line 257-268
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Async method added, includes docstring
  - **Verify**: `grep -A10 'async def _validate_health' rag_ingestion/crawl4ai_reader.py`
  - **Commit**: `refactor(reader): add async health check for runtime validation`

- [ ] 2.2.3b Verify tests still pass after refactor
  - **Do**: Run health check tests to confirm refactor didn't break anything
  - **Files**: N/A
  - **Done when**: All health check tests still pass
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'health' -v` (all pass)
  - **Commit**: `test(reader): verify REFACTOR - tests still pass after async health check addition`

- [ ] V3 [VERIFY] Quality checkpoint: initialization complete
  - **Do**: Run `ruff check rag_ingestion/crawl4ai_reader.py` and `ty check rag_ingestion/crawl4ai_reader.py`
  - **Verify**: Both commands exit 0
  - **Done when**: No lint errors, no type errors
  - **Commit**: `chore(reader): pass quality checkpoint after initialization` (only if fixes needed)

### 2.3 Document ID Generation (Issue #15 - Deterministic UUID)

#### 2.3.1 [RED] Tests for document ID generation

- [ ] 2.3.1a Write test for deterministic UUID generation
  - **Do**: Write `test_document_id_deterministic()` that calls _generate_document_id() twice with same URL, asserts both UUIDs are identical
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS because method doesn't exist
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_document_id_deterministic -v` (must fail)
  - **Commit**: `test(reader): add RED test for deterministic UUID generation`
  - _Requirements: FR-4, Issue #15_
  - _Design: _generate_document_id method, line 270-296_

- [ ] 2.3.1b Write test for different URLs producing different UUIDs
  - **Do**: Write `test_document_id_different_urls()` that generates UUIDs for two different URLs, asserts they are different
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_document_id_different_urls -v` (must fail)
  - **Commit**: `test(reader): add RED test for different URLs producing different UUIDs`
  - _Requirements: FR-4_

- [ ] 2.3.1c Write test for UUID format validation
  - **Do**: Write `test_document_id_uuid_format()` that generates ID, asserts it's valid UUID string format
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_document_id_uuid_format -v` (must fail)
  - **Commit**: `test(reader): add RED test for UUID format validation`
  - _Requirements: FR-4, Issue #15_

#### 2.3.2 [GREEN] Implement deterministic UUID generation

- [ ] 2.3.2a Implement _generate_document_id method
  - **Do**: Implement _generate_document_id(url: str) -> str that: 1) computes SHA256 hash of url.encode(), 2) takes first 16 bytes, 3) creates UUID from bytes, 4) returns str(uuid) - matches vector_store.py pattern
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Method implemented per design.md line 270-296, includes comprehensive docstring
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'document_id' -v` (all pass)
  - **Commit**: `feat(reader): implement deterministic UUID generation from URL`
  - _Requirements: FR-4, Issue #15_
  - _Design: _generate_document_id section_

#### 2.3.3 [REFACTOR] Document UUID strategy rationale

- [ ] 2.3.3a Enhance docstring with vector_store pattern reference
  - **Do**: Enhance _generate_document_id docstring to explicitly mention it matches vector_store.py::_generate_point_id() pattern
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Docstring includes Notes section referencing vector_store pattern, idempotent upsert behavior
  - **Verify**: `grep -A15 '_generate_document_id' rag_ingestion/crawl4ai_reader.py | grep 'vector_store'`
  - **Commit**: `docs(reader): document UUID strategy and vector_store pattern alignment`

- [ ] 2.3.3b Verify tests still pass after refactor
  - **Do**: Run document ID tests to confirm refactor didn't break anything
  - **Files**: N/A
  - **Done when**: All tests still pass
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'document_id' -v` (all pass)
  - **Commit**: `test(reader): verify REFACTOR - document ID tests still pass`

### 2.4 Metadata Building (Including Issue #17 - source_url)

#### 2.4.1 [RED] Tests for metadata extraction

- [ ] 2.4.1a Write test for complete metadata structure
  - **Do**: Write `test_metadata_complete()` that passes valid CrawlResult to _build_metadata(), asserts all 9 fields present (source, source_url, title, description, status_code, crawl_timestamp, internal_links_count, external_links_count, source_type)
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written using MOCK_CRAWL_RESULT_SUCCESS fixture, FAILS because method doesn't exist
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_metadata_complete -v` (must fail)
  - **Commit**: `test(reader): add RED test for complete metadata structure`
  - _Requirements: AC-5.1-5.10, FR-7, Issue #17_
  - _Design: _build_metadata method, line 298-329_

- [ ] 2.4.1b Write test for missing fields defaulting correctly
  - **Do**: Write `test_metadata_missing_title()`, `test_metadata_missing_description()`, `test_metadata_missing_links()` that pass CrawlResult with missing fields, assert defaults (empty string or 0)
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Three tests written, all FAIL
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'missing' -v` (all must fail)
  - **Commit**: `test(reader): add RED tests for missing metadata field defaults`
  - _Requirements: AC-5.10_

- [ ] 2.4.1c Write test for flat types validation
  - **Do**: Write `test_metadata_flat_types()` that asserts all metadata values are str, int, or float (no None, no nested dicts)
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_metadata_flat_types -v` (must fail)
  - **Commit**: `test(reader): add RED test for flat types validation`
  - _Requirements: AC-5.9_

- [ ] 2.4.1d Write test for link counting accuracy
  - **Do**: Write `test_metadata_links_counting()` that passes CrawlResult with known number of links, asserts counts match
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_metadata_links_counting -v` (must fail)
  - **Commit**: `test(reader): add RED test for link counting accuracy`
  - _Requirements: AC-5.6, AC-5.7_

- [ ] 2.4.1e Write test for source_url field presence (Issue #17)
  - **Do**: Write `test_metadata_source_url_present()` that verifies source_url field exists and equals source field
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_metadata_source_url_present -v` (must fail)
  - **Commit**: `test(reader): add RED test for source_url metadata field`
  - _Requirements: Issue #17_
  - _Design: Deduplication Strategy, line 1106-1113_

#### 2.4.2 [GREEN] Implement metadata extraction

- [ ] 2.4.2a Implement _build_metadata method
  - **Do**: Implement _build_metadata(crawl_result: dict, url: str) -> dict that extracts all 9 metadata fields with explicit defaults per design.md line 298-329, including source_url=url
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Method implemented with 'or' operator for defaults, flat types only
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'metadata' -v` (all pass)
  - **Commit**: `feat(reader): implement metadata extraction with source_url field`
  - _Requirements: FR-7, US-5, Issue #17_
  - _Design: _build_metadata section_

#### 2.4.3 [REFACTOR] Improve metadata extraction clarity

- [ ] 2.4.3a Extract link counting into helper function
  - **Do**: Extract internal/external link counting logic into separate helper function for clarity
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Helper function created, _build_metadata calls it
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'metadata' -v` (all pass)
  - **Commit**: `refactor(reader): extract link counting into helper function`

- [ ] 2.4.3b Verify tests still pass after refactor
  - **Do**: Run metadata tests to confirm refactor didn't break anything
  - **Files**: N/A
  - **Done when**: All metadata tests still pass
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'metadata' -v` (all pass)
  - **Commit**: `test(reader): verify REFACTOR - metadata tests still pass`

- [ ] V4 [VERIFY] Quality checkpoint: ID and metadata complete
  - **Do**: Run `ruff check rag_ingestion/crawl4ai_reader.py` and `ty check rag_ingestion/crawl4ai_reader.py`
  - **Verify**: Both commands exit 0
  - **Done when**: No lint errors, no type errors
  - **Commit**: `chore(reader): pass quality checkpoint after ID and metadata` (only if fixes needed)

### 2.5 Single URL Crawling with Circuit Breaker

#### 2.5.1 [RED] Tests for single URL crawling

- [ ] 2.5.1a Write test for successful crawl with fit_markdown
  - **Do**: Write `test_crawl_single_url_success()` that mocks POST /crawl returning success response, calls _crawl_single_url(), asserts Document returned with correct text and metadata
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written using respx to mock httpx, FAILS because method doesn't exist
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_crawl_single_url_success -v` (must fail)
  - **Commit**: `test(reader): add RED test for successful single URL crawl`
  - _Requirements: AC-2.3, FR-5, FR-6_
  - _Design: _crawl_single_url method, line 331-475_

- [ ] 2.5.1b Write test for markdown fallback to raw_markdown
  - **Do**: Write `test_crawl_single_url_fallback_raw_markdown()` that mocks response with fit_markdown missing, asserts raw_markdown used
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_crawl_single_url_fallback_raw_markdown -v` (must fail)
  - **Commit**: `test(reader): add RED test for raw_markdown fallback`
  - _Requirements: FR-6_

- [ ] 2.5.1c Write test for missing markdown error
  - **Do**: Write `test_crawl_single_url_no_markdown()` that mocks response with no markdown content, asserts ValueError raised
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_crawl_single_url_no_markdown -v` (must fail)
  - **Commit**: `test(reader): add RED test for missing markdown error`
  - _Requirements: FR-6, FR-8_

- [ ] 2.5.1d Write test for CrawlResult success=False
  - **Do**: Write `test_crawl_single_url_success_false()` that mocks response with success=False, asserts RuntimeError raised with error_message
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_crawl_single_url_success_false -v` (must fail)
  - **Commit**: `test(reader): add RED test for crawl failure handling`
  - _Requirements: FR-8, US-6_

- [ ] 2.5.1e Write test for circuit breaker open state
  - **Do**: Write `test_crawl_single_url_circuit_breaker_open()` that sets circuit breaker to OPEN state, calls _crawl_single_url(), asserts CircuitBreakerError raised
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_crawl_single_url_circuit_breaker_open -v` (must fail)
  - **Commit**: `test(reader): add RED test for circuit breaker open state`
  - _Requirements: AC-4.7, FR-9_

- [ ] 2.5.1f Write test for fail_on_error=False returning None
  - **Do**: Write `test_crawl_single_url_fail_on_error_false()` that mocks failed crawl, sets fail_on_error=False, asserts None returned instead of exception
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_crawl_single_url_fail_on_error_false -v` (must fail)
  - **Commit**: `test(reader): add RED test for fail_on_error=False behavior`
  - _Requirements: AC-2.7, AC-6.5, FR-8_

#### 2.5.2 [GREEN] Implement single URL crawling

- [ ] 2.5.2a Implement _crawl_single_url core logic
  - **Do**: Implement _crawl_single_url(client: httpx.AsyncClient, url: str) -> Document | None with: 1) circuit breaker wrapper, 2) POST to /crawl endpoint, 3) response validation, 4) markdown extraction with fallback, 5) metadata building, 6) Document creation with deterministic UUID
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Method implemented per design.md line 331-475, includes all error handling
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'crawl_single_url' -v` (all pass)
  - **Commit**: `feat(reader): implement single URL crawling with circuit breaker`
  - _Requirements: FR-5, FR-6, FR-8, FR-9, US-2_
  - _Design: _crawl_single_url section_

#### 2.5.3 [REFACTOR] Extract retry logic and improve error handling

- [ ] 2.5.3a Extract internal retry implementation
  - **Do**: Extract retry logic into separate _crawl_impl() internal function within _crawl_single_url() for clarity
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Retry logic extracted per design.md line 349-445, main method wraps with circuit breaker
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'crawl_single_url' -v` (all pass)
  - **Commit**: `refactor(reader): extract retry logic into internal implementation`

- [ ] 2.5.3b Add circuit breaker state logging
  - **Do**: Add logging for circuit breaker state transitions per design.md line 451-469
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Logging added for open/closed states with context (url, failure count)
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'crawl_single_url' -v` (all pass)
  - **Commit**: `refactor(reader): add circuit breaker state logging`
  - _Requirements: AC-4.8_

- [ ] 2.5.3c Verify tests still pass after refactor
  - **Do**: Run single URL crawl tests to confirm refactor didn't break anything
  - **Files**: N/A
  - **Done when**: All crawl tests still pass
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'crawl_single_url' -v` (all pass)
  - **Commit**: `test(reader): verify REFACTOR - crawl tests still pass`

### 2.6 Retry Logic (Issue addressed in design.md)

#### 2.6.1 [RED] Tests for retry behavior

- [ ] 2.6.1a Write test for successful retry after timeout
  - **Do**: Write `test_crawl_single_url_timeout_retry()` that mocks first request timing out, second succeeding, asserts Document returned
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written using respx side_effect, FAILS because retry not implemented yet
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_crawl_single_url_timeout_retry -v` (must fail)
  - **Commit**: `test(reader): add RED test for successful retry after timeout`
  - _Requirements: AC-7.2, FR-10, US-7_
  - _Design: Retry logic in _crawl_single_url, line 401-420_

- [ ] 2.6.1b Write test for max retries exhausted
  - **Do**: Write `test_crawl_single_url_max_retries_exhausted()` that mocks all retry attempts failing, asserts exception raised
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_crawl_single_url_max_retries_exhausted -v` (must fail)
  - **Commit**: `test(reader): add RED test for max retries exhausted`
  - _Requirements: AC-7.1, AC-7.7_

- [ ] 2.6.1c Write test for no retry on 4xx errors
  - **Do**: Write `test_crawl_single_url_http_404_no_retry()` that mocks 404 response, asserts no retry attempted (1 request only)
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_crawl_single_url_http_404_no_retry -v` (must fail)
  - **Commit**: `test(reader): add RED test for no retry on 4xx errors`
  - _Requirements: AC-7.4_

- [ ] 2.6.1d Write test for retry on 5xx errors
  - **Do**: Write `test_crawl_single_url_http_500_retry()` that mocks 500 response, then success, asserts retry attempted and succeeds
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_crawl_single_url_http_500_retry -v` (must fail)
  - **Commit**: `test(reader): add RED test for retry on 5xx errors`
  - _Requirements: AC-7.3_

- [ ] 2.6.1e Write test for exponential backoff delays
  - **Do**: Write `test_retry_exponential_backoff()` that verifies sleep delays match [1.0, 2.0, 4.0] pattern using mock
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_retry_exponential_backoff -v` (must fail)
  - **Commit**: `test(reader): add RED test for exponential backoff delays`
  - _Requirements: AC-7.2, NFR-8_

#### 2.6.2 [GREEN] Implement retry logic

- [ ] 2.6.2a Add retry loop to _crawl_impl
  - **Do**: Add for loop over max_retries in _crawl_impl(), implement exponential backoff with asyncio.sleep(), handle transient vs permanent errors per design.md line 351-444
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Retry logic implemented with proper error categorization
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'retry' -v` (all pass)
  - **Commit**: `feat(reader): implement exponential backoff retry logic`
  - _Requirements: FR-10, US-7_
  - _Design: Retry logic section_

#### 2.6.3 [REFACTOR] Improve retry logging

- [ ] 2.6.3a Add structured logging for retry attempts
  - **Do**: Add logger.warning() calls for each retry attempt with context (url, attempt, error, delay) per design.md line 405-410, 427-432
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Logging added for all retry scenarios
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'retry' -v` (all pass)
  - **Commit**: `refactor(reader): add structured logging for retry attempts`
  - _Requirements: AC-7.6, FR-11_

- [ ] 2.6.3b Verify tests still pass after refactor
  - **Do**: Run retry tests to confirm refactor didn't break anything
  - **Files**: N/A
  - **Done when**: All retry tests still pass
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'retry' -v` (all pass)
  - **Commit**: `test(reader): verify REFACTOR - retry tests still pass`

- [ ] V5 [VERIFY] Quality checkpoint: single URL crawling complete
  - **Do**: Run `ruff check rag_ingestion/crawl4ai_reader.py` and `ty check rag_ingestion/crawl4ai_reader.py`
  - **Verify**: Both commands exit 0
  - **Done when**: No lint errors, no type errors
  - **Commit**: `chore(reader): pass quality checkpoint after crawling` (only if fixes needed)

### 2.7 Deduplication Integration (Issue #16)

#### 2.7.1 [RED] Tests for deduplication behavior

- [ ] 2.7.1a Write test for deduplication enabled
  - **Do**: Write `test_deduplicate_url_called()` that mocks VectorStoreManager.delete_by_url(), calls aload_data() with enable_deduplication=True, asserts delete_by_url called for each URL
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS because deduplication not implemented
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_deduplicate_url_called -v` (must fail)
  - **Commit**: `test(reader): add RED test for deduplication enabled`
  - _Requirements: Issue #16_
  - _Design: Deduplication Strategy, line 1125-1168_

- [ ] 2.7.1b Write test for deduplication disabled
  - **Do**: Write `test_deduplicate_url_skipped()` that sets enable_deduplication=False, asserts delete_by_url NOT called
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_deduplicate_url_skipped -v` (must fail)
  - **Commit**: `test(reader): add RED test for deduplication disabled`
  - _Requirements: Issue #16_

- [ ] 2.7.1c Write test for no vector_store (deduplication skipped)
  - **Do**: Write `test_deduplicate_url_no_vector_store()` that sets vector_store=None, asserts deduplication skipped gracefully
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_deduplicate_url_no_vector_store -v` (must fail)
  - **Commit**: `test(reader): add RED test for no vector_store deduplication skip`
  - _Requirements: Issue #16_

#### 2.7.2 [GREEN] Implement deduplication

- [ ] 2.7.2a Add enable_deduplication and vector_store fields
  - **Do**: Add enable_deduplication (bool, default True) and vector_store (VectorStoreManager | None, default None) fields to Crawl4AIReader class
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Fields added per design.md line 1213-1222
  - **Verify**: `grep 'enable_deduplication\|vector_store' rag_ingestion/crawl4ai_reader.py | wc -l` (should return 2)
  - **Commit**: `feat(reader): add enable_deduplication and vector_store fields`
  - _Requirements: Issue #16_

- [ ] 2.7.2b Implement _deduplicate_url method
  - **Do**: Implement _deduplicate_url(url: str) async method that checks vector_store, calls delete_by_url(), logs deleted count per design.md line 1149-1168
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Method implemented with early return if vector_store is None
  - **Verify**: `grep -A10 '_deduplicate_url' rag_ingestion/crawl4ai_reader.py`
  - **Commit**: `feat(reader): implement _deduplicate_url method`
  - _Requirements: Issue #16_
  - _Design: Deduplication Strategy_

- [ ] 2.7.2c Call _deduplicate_url in aload_data
  - **Do**: Add deduplication loop in aload_data() before crawling URLs per design.md line 1137-1147
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Loop added that calls _deduplicate_url(url) for each URL if enable_deduplication is True
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'deduplicate' -v` (all pass)
  - **Commit**: `feat(reader): integrate deduplication into aload_data`
  - _Requirements: Issue #16_

#### 2.7.3 [REFACTOR] Document deduplication behavior

- [ ] 2.7.3a Add comprehensive docstring to _deduplicate_url
  - **Do**: Add docstring explaining deduplication matches file watcher pattern per design.md line 1150-1162
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Docstring includes Args, Notes sections explaining pattern match
  - **Verify**: `grep -A10 '_deduplicate_url' rag_ingestion/crawl4ai_reader.py | grep 'file watcher'`
  - **Commit**: `docs(reader): document deduplication pattern alignment`

- [ ] 2.7.3b Verify tests still pass after refactor
  - **Do**: Run deduplication tests to confirm refactor didn't break anything
  - **Files**: N/A
  - **Done when**: All deduplication tests still pass
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'deduplicate' -v` (all pass)
  - **Commit**: `test(reader): verify REFACTOR - deduplication tests still pass`

### 2.8 Batch Loading (aload_data)

#### 2.8.1 [RED] Tests for async batch loading

- [ ] 2.8.1a Write test for empty URL list
  - **Do**: Write `test_aload_data_empty_list()` that calls aload_data([]), asserts empty list returned
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS because method doesn't exist
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_aload_data_empty_list -v` (must fail)
  - **Commit**: `test(reader): add RED test for empty URL list`
  - _Requirements: AC-3.1, Edge cases_
  - _Design: aload_data method, line 477-547_

- [ ] 2.8.1b Write test for single URL
  - **Do**: Write `test_aload_data_single_url()` that mocks successful crawl, calls aload_data(["url"]), asserts one Document returned
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_aload_data_single_url -v` (must fail)
  - **Commit**: `test(reader): add RED test for single URL batch`
  - _Requirements: AC-2.1, AC-2.2_

- [ ] 2.8.1c Write test for multiple URLs concurrent processing
  - **Do**: Write `test_aload_data_multiple_urls()` that mocks multiple successful crawls, calls aload_data([urls]), asserts all Documents returned
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_aload_data_multiple_urls -v` (must fail)
  - **Commit**: `test(reader): add RED test for multiple URLs concurrent processing`
  - _Requirements: AC-3.1, AC-3.2, AC-3.3, US-3_

- [ ] 2.8.1d Write test for order preservation with failures
  - **Do**: Write `test_aload_data_order_preservation()` that mocks [success, failure, success] pattern, asserts results list has None in middle position (order preserved)
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_aload_data_order_preservation -v` (must fail)
  - **Commit**: `test(reader): add RED test for order preservation`
  - _Requirements: AC-3.4, spec-clarifications Issue #1_

- [ ] 2.8.1e Write test for concurrency limit enforcement
  - **Do**: Write `test_aload_data_concurrent_limit()` that mocks 10 URLs with max_concurrent=3, asserts no more than 3 requests run concurrently
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written using asyncio timing checks, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_aload_data_concurrent_limit -v` (must fail)
  - **Commit**: `test(reader): add RED test for concurrency limit enforcement`
  - _Requirements: AC-3.3, NFR-4_

- [ ] 2.8.1f Write test for batch statistics logging
  - **Do**: Write `test_aload_data_logging()` that mocks batch crawl, asserts log messages include batch start, completion, success/failure counts
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written using caplog fixture, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_aload_data_logging -v` (must fail)
  - **Commit**: `test(reader): add RED test for batch statistics logging`
  - _Requirements: AC-2.8, AC-3.8, FR-11_

#### 2.8.2 [GREEN] Implement async batch loading

- [ ] 2.8.2a Implement aload_data method
  - **Do**: Implement aload_data(urls: List[str]) -> List[Document | None] with: 1) health check, 2) deduplication loop, 3) semaphore creation, 4) AsyncClient context manager, 5) asyncio.gather with return_exceptions, 6) statistics logging per design.md line 477-547
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Method implemented with all features, preserves order, shared client
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'aload_data' -v` (all pass)
  - **Commit**: `feat(reader): implement async batch loading with concurrency control`
  - _Requirements: FR-2, FR-14, FR-16, FR-17, US-3_
  - _Design: aload_data section_

#### 2.8.3 [REFACTOR] Extract semaphore wrapper and improve clarity

- [ ] 2.8.3a Extract crawl_with_semaphore wrapper
  - **Do**: Extract async semaphore wrapper function inside aload_data per design.md line 525-527
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Wrapper function created for clarity
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'aload_data' -v` (all pass)
  - **Commit**: `refactor(reader): extract semaphore wrapper for clarity`

- [ ] 2.8.3b Add async context warning to docstring
  - **Do**: Add Warning section to aload_data docstring per design.md line 484-485
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Docstring includes warning about async context requirement
  - **Verify**: `grep -A20 'async def aload_data' rag_ingestion/crawl4ai_reader.py | grep 'Warning:'`
  - **Commit**: `docs(reader): add async context warning to aload_data`

- [ ] 2.8.3c Verify tests still pass after refactor
  - **Do**: Run batch loading tests to confirm refactor didn't break anything
  - **Files**: N/A
  - **Done when**: All aload_data tests still pass
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'aload_data' -v` (all pass)
  - **Commit**: `test(reader): verify REFACTOR - aload_data tests still pass`

### 2.9 Synchronous Loading (load_data)

#### 2.9.1 [RED] Tests for synchronous loading

- [ ] 2.9.1a Write test for load_data delegates to aload_data
  - **Do**: Write `test_load_data_delegates_to_aload_data()` that mocks aload_data, calls load_data(), asserts asyncio.run() used
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS because load_data doesn't exist
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_load_data_delegates_to_aload_data -v` (must fail)
  - **Commit**: `test(reader): add RED test for load_data delegation`
  - _Requirements: FR-1, US-2_
  - _Design: load_data method, line 549-570_

- [ ] 2.9.1b Write test for load_data with single URL
  - **Do**: Write `test_load_data_single_url()` that calls load_data(["url"]), asserts Document returned
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, FAILS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_load_data_single_url -v` (must fail)
  - **Commit**: `test(reader): add RED test for load_data single URL`
  - _Requirements: AC-2.1, AC-2.2_

#### 2.9.2 [GREEN] Implement synchronous loading

- [ ] 2.9.2a Implement load_data method
  - **Do**: Implement load_data(urls: List[str]) -> List[Document] that calls asyncio.run(self.aload_data(urls)) per design.md line 549-570
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Method implemented as simple wrapper
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'load_data' -v` (all pass)
  - **Commit**: `feat(reader): implement synchronous load_data wrapper`
  - _Requirements: FR-1, US-2_
  - _Design: load_data section_

#### 2.9.3 [REFACTOR] Add comprehensive docstring

- [ ] 2.9.3a Add load_data docstring with examples
  - **Do**: Add Google-style docstring to load_data with Args, Returns, Raises, Examples sections per design.md line 550-568
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: Docstring complete with usage example
  - **Verify**: `grep -A15 'def load_data' rag_ingestion/crawl4ai_reader.py | grep 'Examples:'`
  - **Commit**: `docs(reader): add comprehensive load_data docstring`

- [ ] 2.9.3b Verify tests still pass after refactor
  - **Do**: Run load_data tests to confirm refactor didn't break anything
  - **Files**: N/A
  - **Done when**: All load_data tests still pass
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'load_data' -v` (all pass)
  - **Commit**: `test(reader): verify REFACTOR - load_data tests still pass`

- [ ] V6 [VERIFY] Quality checkpoint: full reader implementation
  - **Do**: Run `ruff check .` and `ty check rag_ingestion/`
  - **Verify**: Both commands exit 0
  - **Done when**: No lint errors, no type errors
  - **Commit**: `chore(reader): pass quality checkpoint after full implementation` (only if fixes needed)

### 2.10 Error Handling Tests

#### 2.10.1 [RED] Tests for comprehensive error handling

- [ ] 2.10.1a Write test for HTTP timeout exception
  - **Do**: Write `test_error_timeout_exception()` that mocks httpx.TimeoutException, asserts proper handling (retry or error based on fail_on_error)
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, should PASS (error handling already implemented)
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_error_timeout_exception -v` (should pass)
  - **Commit**: `test(reader): add RED test for timeout exception handling`
  - _Requirements: FR-8, US-6_

- [ ] 2.10.1b Write test for network error
  - **Do**: Write `test_error_network_exception()` that mocks httpx.NetworkError, asserts retry attempted
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, should PASS
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_error_network_exception -v` (should pass)
  - **Commit**: `test(reader): add RED test for network error handling`
  - _Requirements: FR-8_

- [ ] 2.10.1c Write test for invalid JSON response
  - **Do**: Write `test_error_invalid_json()` that mocks response.json() raising JSONDecodeError, asserts error handled
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_crawl4ai_reader.py`
  - **Done when**: Test written, verifies proper error propagation
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py::test_error_invalid_json -v`
  - **Commit**: `test(reader): add RED test for invalid JSON response`
  - _Requirements: FR-8_

#### 2.10.2 [GREEN] Verify error handling coverage

- [ ] 2.10.2a Run all error handling tests
  - **Do**: Verify all error scenarios are handled correctly by existing implementation
  - **Files**: N/A
  - **Done when**: All error tests pass
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'error' -v` (all pass)
  - **Commit**: `test(reader): verify GREEN - all error handling tests pass`

#### 2.10.3 [REFACTOR] Improve error messages

- [ ] 2.10.3a Enhance error messages with context
  - **Do**: Review all error messages, ensure they include URL and relevant context for debugging
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/crawl4ai_reader.py`
  - **Done when**: All error messages include URL and error type
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'error' -v` (all pass)
  - **Commit**: `refactor(reader): enhance error messages with context`

- [ ] 2.10.3b Verify tests still pass after refactor
  - **Do**: Run error handling tests to confirm refactor didn't break anything
  - **Files**: N/A
  - **Done when**: All error tests still pass
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py -k 'error' -v` (all pass)
  - **Commit**: `test(reader): verify REFACTOR - error tests still pass`

- [ ] V7 [VERIFY] Quality checkpoint: Phase 2 complete
  - **Do**: Run full unit test suite and verify 85%+ coverage
  - **Verify**: `pytest tests/unit/test_crawl4ai_reader.py --cov=rag_ingestion.crawl4ai_reader --cov-report=term`
  - **Done when**: All unit tests pass, coverage ≥85%
  - **Commit**: `test(reader): verify Phase 2 complete with 85%+ coverage` (if needed)

---

## Phase 3: Integration & Polish

Focus: Integration tests, end-to-end tests, documentation. Tests written after implementation (not TDD).

### 3.1 Integration Tests with Real Service

- [ ] 3.1.1 Write integration test for health check
  - **Do**: Write `test_integration_health_check()` that creates real reader, verifies /health endpoint responds
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_crawl4ai_reader_integration.py`
  - **Done when**: Test written, uses crawl4ai_available fixture, skips if service unavailable
  - **Verify**: `pytest tests/integration/test_crawl4ai_reader_integration.py::test_integration_health_check -v`
  - **Commit**: `test(reader): add integration test for health check`
  - _Requirements: NFR-14_

- [ ] 3.1.2 Write integration test for single URL crawl
  - **Do**: Write `test_integration_crawl_single_url()` that crawls real webpage (example.com), verifies Document structure
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_crawl4ai_reader_integration.py`
  - **Done when**: Test written, uses real Crawl4AI service
  - **Verify**: `pytest tests/integration/test_crawl4ai_reader_integration.py::test_integration_crawl_single_url -v`
  - **Commit**: `test(reader): add integration test for single URL crawl`

- [ ] 3.1.3 Write integration test for batch crawl
  - **Do**: Write `test_integration_crawl_batch()` that crawls multiple real webpages, verifies all Documents returned
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_crawl4ai_reader_integration.py`
  - **Done when**: Test written
  - **Verify**: `pytest tests/integration/test_crawl4ai_reader_integration.py::test_integration_crawl_batch -v`
  - **Commit**: `test(reader): add integration test for batch crawl`

- [ ] 3.1.4 Write integration test for circuit breaker
  - **Do**: Write `test_integration_circuit_breaker_opens()` that simulates service failures, verifies circuit opens after threshold
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_crawl4ai_reader_integration.py`
  - **Done when**: Test written, verifies circuit breaker behavior
  - **Verify**: `pytest tests/integration/test_crawl4ai_reader_integration.py::test_integration_circuit_breaker_opens -v`
  - **Commit**: `test(reader): add integration test for circuit breaker behavior`

- [ ] 3.1.5 Write integration test for concurrency
  - **Do**: Write `test_integration_concurrent_processing()` that crawls 10+ URLs, verifies concurrency limit enforced
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_crawl4ai_reader_integration.py`
  - **Done when**: Test written
  - **Verify**: `pytest tests/integration/test_crawl4ai_reader_integration.py::test_integration_concurrent_processing -v`
  - **Commit**: `test(reader): add integration test for concurrent processing`

- [ ] V8 [VERIFY] Integration tests pass
  - **Do**: Run all integration tests (requires Crawl4AI service running)
  - **Verify**: `pytest tests/integration/test_crawl4ai_reader_integration.py -v`
  - **Done when**: All integration tests pass or skip if service unavailable
  - **Commit**: `test(reader): verify all integration tests pass`

### 3.2 End-to-End Pipeline Tests

- [ ] 3.2.1 Create E2E test file
  - **Do**: Create `tests/integration/test_e2e_reader_pipeline.py` for end-to-end tests with full pipeline
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_e2e_reader_pipeline.py`
  - **Done when**: File created with imports for reader, chunker, TEI client, vector store
  - **Verify**: `test -f tests/integration/test_e2e_reader_pipeline.py`
  - **Commit**: `test(reader): create E2E pipeline test file`

- [ ] 3.2.2 Write E2E test for reader to chunker
  - **Do**: Write `test_e2e_reader_to_chunker()` that crawls URL, chunks document, verifies chunks created
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_e2e_reader_pipeline.py`
  - **Done when**: Test written, uses real services
  - **Verify**: `pytest tests/integration/test_e2e_reader_pipeline.py::test_e2e_reader_to_chunker -v`
  - **Commit**: `test(reader): add E2E test for reader to chunker integration`

- [ ] 3.2.3 Write E2E test for reader to Qdrant
  - **Do**: Write `test_e2e_reader_to_qdrant()` that crawls URL, stores in Qdrant, verifies metadata including source_url
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_e2e_reader_pipeline.py`
  - **Done when**: Test written, verifies full pipeline including source_url field (Issue #17)
  - **Verify**: `pytest tests/integration/test_e2e_reader_pipeline.py::test_e2e_reader_to_qdrant -v`
  - **Commit**: `test(reader): add E2E test for reader to Qdrant integration`

- [ ] V9 [VERIFY] E2E tests pass
  - **Do**: Run all E2E tests (requires full stack running)
  - **Verify**: `pytest tests/integration/test_e2e_reader_pipeline.py -v`
  - **Done when**: All E2E tests pass or skip if services unavailable
  - **Commit**: `test(reader): verify all E2E tests pass`

### 3.3 Documentation

- [ ] 3.3.1 Update CLAUDE.md with reader usage
  - **Do**: Add section to CLAUDE.md documenting Crawl4AIReader usage, configuration, integration examples
  - **Files**: `/home/jmagar/workspace/crawl4r/CLAUDE.md`
  - **Done when**: Section added with examples from design.md line 1050-1100
  - **Verify**: `grep -A20 'Crawl4AIReader' CLAUDE.md`
  - **Commit**: `docs(reader): add Crawl4AIReader usage to CLAUDE.md`

- [ ] 3.3.2 Update README with web crawling capabilities
  - **Do**: Add section to README documenting web crawling feature, basic usage example
  - **Files**: `/home/jmagar/workspace/crawl4r/README.md` (if exists)
  - **Done when**: Section added with quick start example
  - **Verify**: `grep 'Crawl4AI' README.md`
  - **Commit**: `docs(reader): document web crawling in README`

- [ ] 3.3.3 Create usage examples file
  - **Do**: Create `examples/crawl4ai_reader_usage.py` with complete usage examples
  - **Files**: `/home/jmagar/workspace/crawl4r/examples/crawl4ai_reader_usage.py`
  - **Done when**: File created with 3+ usage examples (basic, batch, pipeline integration)
  - **Verify**: `test -f examples/crawl4ai_reader_usage.py`
  - **Commit**: `docs(reader): add usage examples file`

- [ ] V10 [VERIFY] Documentation complete
  - **Do**: Review all documentation for accuracy and completeness
  - **Verify**: Manual review of CLAUDE.md, README, examples file
  - **Done when**: All documentation accurate and includes deduplication behavior
  - **Commit**: None (documentation review only)

---

## Phase 4: Quality Gates

Focus: Final verification, CI preparation, acceptance criteria validation.

- [ ] V11 [VERIFY] Full local CI: lint, typecheck, test, coverage
  - **Do**: Run complete local CI suite: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/test_crawl4ai_reader.py --cov=rag_ingestion.crawl4ai_reader --cov-report=term`
  - **Verify**: All commands pass, coverage ≥85%
  - **Done when**: Zero errors, all tests pass, coverage target met
  - **Commit**: `chore(reader): pass full local CI verification` (if fixes needed)

- [ ] V12 [VERIFY] Create PR and verify CI passes
  - **Do**:
    1. Verify current branch is feature branch: `git branch --show-current`
    2. If on default branch (main), STOP and alert user (should not happen)
    3. Push branch: `git push -u origin <branch-name>`
    4. Create PR: `gh pr create --title "feat(reader): implement LlamaIndex Crawl4AI reader" --body "Implements basic reader with UUID strategy, deduplication, and source_url indexing (Issues #15, #16, #17)"`
    5. Watch CI: `gh pr checks --watch`
  - **Verify**: All CI checks pass (lint, typecheck, tests, coverage)
  - **Done when**: PR created, all CI checks green
  - **Commit**: None (PR creation only)

- [ ] V13 [VERIFY] Acceptance criteria checklist
  - **Do**: Read requirements.md, verify each AC-* is satisfied by implementation
  - **Verify**: Manual review against all 7 user stories and acceptance criteria
  - **Done when**: All acceptance criteria confirmed met, including:
    - AC-1.1-1.7: Configuration (US-1)
    - AC-2.1-2.8: Single URL loading (US-2)
    - AC-3.1-3.8: Batch loading (US-3)
    - AC-4.1-4.8: Circuit breaker (US-4)
    - AC-5.1-5.10: Metadata enrichment (US-5)
    - AC-6.1-6.8: Error handling (US-6)
    - AC-7.1-7.8: Retry logic (US-7)
    - Issue #15: Deterministic UUID generation
    - Issue #16: Automatic deduplication
    - Issue #17: source_url field in metadata and PAYLOAD_INDEXES
  - **Commit**: None (verification only)

---

## Notes

**POC shortcuts (none - production quality from start)**: This is a production-ready implementation with full TDD compliance, no shortcuts taken.

**TDD Enforcement**:
- Every feature in Phase 2 strictly follows RED-GREEN-REFACTOR
- Tests written BEFORE implementation code
- Each phase verified before moving to next
- 68 total tests written (59 unit + 6 integration + 3 e2e)
- 85%+ coverage target mandatory

**New Requirements Addressed**:
- Issue #15 (UUID Strategy): Task 2.3 implements deterministic UUID matching vector_store.py pattern
- Issue #16 (Deduplication): Task 2.7 implements automatic deduplication before re-crawling
- Issue #17 (source_url): Task 1.2.1 adds to PAYLOAD_INDEXES, Task 2.4 includes in metadata

**VectorStoreManager Changes**:
- Task 1.2.1: Add source_url to PAYLOAD_INDEXES
- Task 1.2.2: Implement delete_by_url() method (mirrors delete_by_file pattern)

**Quality Checkpoints**: Inserted after every 2-3 major tasks throughout all phases to catch issues early.
