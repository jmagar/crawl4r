---
spec: rag-ingestion
phase: tasks
total_tasks: 78
created: 2026-01-14T22:30:00Z
tdd: strict
---

# Tasks: RAG Ingestion Pipeline

## Phase 1: Core Infrastructure (TDD)

**Focus**: Establish project foundation, configuration, logging, and service connections with tests first.

### 1.1 Project Setup

- [x] 1.1.1 Create project structure
  - **Do**:
    - Create directory structure: `rag_ingestion/`, `tests/unit/`, `tests/integration/`, `.cache/`
    - Create `__init__.py` files: `rag_ingestion/__init__.py`, `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`
  - **Files**:
    - `/home/jmagar/workspace/crawl4r/rag_ingestion/__init__.py`
    - `/home/jmagar/workspace/crawl4r/tests/__init__.py`
    - `/home/jmagar/workspace/crawl4r/tests/unit/__init__.py`
    - `/home/jmagar/workspace/crawl4r/tests/integration/__init__.py`
    - `/home/jmagar/workspace/crawl4r/.cache/.gitkeep`
  - **Done when**: All directories and init files exist
  - **Verify**: `ls -la rag_ingestion/ tests/unit/ tests/integration/ .cache/`
  - **Commit**: `chore(setup): initialize project structure`
  - _Requirements: NFR-5 (memory management), FR-10 (logging)_
  - _Design: Project Structure_

- [x] 1.1.2 Initialize pyproject.toml with dependencies and tool configs
  - **Do**:
    - Create `pyproject.toml` with project metadata, Python 3.10+ requirement
    - Add dependencies: llama-index-core>=0.14.0, llama-index-vector-stores-qdrant, llama-index-readers-file, qdrant-client>=1.16.0, watchdog>=6.0.0, huggingface-hub, pydantic>=2.0.0, python-dotenv
    - Add dev dependencies: pytest, pytest-asyncio, pytest-cov, ruff, ty (Astral's type checker)
    - Configure ruff: line-length=88, target-version=py310, cache-dir=".cache/.ruff_cache"
    - Configure ty: strict=true, cache-dir=".cache/.ty_cache", disallow_any_explicit=true, disallow_untyped_defs=true
    - Configure pytest: cache_dir=".cache/.pytest_cache", testpaths=["tests"], asyncio_mode="auto"
    - Configure coverage: data_file=".cache/.coverage", omit=["tests/*", ".cache/*"]
  - **Files**: `/home/jmagar/workspace/crawl4r/pyproject.toml`
  - **Done when**: pyproject.toml exists with all dependencies and tool configs
  - **Verify**: `cat pyproject.toml | grep -E "(llama-index|qdrant|watchdog|ruff|ty|pytest)"`
  - **Commit**: `chore(setup): add pyproject.toml with dependencies and tool configs`
  - _Requirements: FR-10 (logging), NFR-12 (dependencies), AC-12.5 (.env support)_
  - _Design: Dependencies, Tool Configuration_

- [x] 1.1.3 Create .env.example and .gitignore
  - **Do**:
    - Create `.env.example` with all configuration parameters documented (WATCH_FOLDER, TEI_ENDPOINT, QDRANT_URL, COLLECTION_NAME, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_PERCENT, MAX_CONCURRENT_DOCS, QUEUE_MAX_SIZE, BATCH_SIZE, LOG_LEVEL, FAILED_DOCS_LOG)
    - Create `.gitignore` with: `.env`, `.cache/`, `__pycache__/`, `*.pyc`, `*.pyo`, `.coverage`, `failed_documents.jsonl`, `*.log`
  - **Files**:
    - `/home/jmagar/workspace/crawl4r/.env.example`
    - `/home/jmagar/workspace/crawl4r/.gitignore`
  - **Done when**: Both files exist and .gitignore covers all cache/temp files
  - **Verify**: `cat .env.example | wc -l` (should be 10+ lines)
  - **Commit**: `chore(setup): add .env.example and .gitignore`
  - _Requirements: AC-12.1-12.6 (configuration), FR-20 (failed docs logging)_
  - _Design: Configuration Management_

### 1.2 Configuration Module (TDD)

- [x] 1.2.1 [RED] Write failing tests for configuration validation
  - **Do**:
    - Create `tests/unit/test_config.py`
    - Write test_config_loads_from_env: Mock environment variables, verify Settings object created
    - Write test_config_requires_watch_folder: Verify ValidationError when WATCH_FOLDER missing
    - Write test_config_default_values: Verify defaults for TEI_ENDPOINT, QDRANT_URL, COLLECTION_NAME, etc.
    - Write test_config_validates_chunk_overlap: Verify ValidationError when CHUNK_OVERLAP_PERCENT > 50 or < 0
    - Write test_config_validates_positive_integers: Verify ValidationError for negative MAX_CONCURRENT_DOCS, QUEUE_MAX_SIZE, BATCH_SIZE
    - Run tests and confirm ALL FAIL
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_config.py`
  - **Done when**: Tests exist and ALL fail with ImportError or assertion failures
  - **Verify**: `pytest tests/unit/test_config.py -v` (expect failures)
  - **Commit**: `test(config): add failing configuration validation tests (RED)`
  - _Requirements: AC-12.1-12.6 (configuration), FR-12 (environment variables)_
  - _Design: Configuration Management, Config Class_

- [x] 1.2.2 [GREEN] Implement configuration module to pass tests
  - **Do**:
    - Create `rag_ingestion/config.py`
    - Define Settings class using Pydantic BaseSettings with all fields from requirements
    - Add field validators for chunk_overlap_percent (0-50 range), positive integers for concurrency/queue/batch
    - Implement __init__ to load from .env file using python-dotenv
    - Add model_config for env_file and validation settings
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/config.py`
  - **Done when**: All tests in test_config.py pass
  - **Verify**: `pytest tests/unit/test_config.py -v` (all pass)
  - **Commit**: `feat(config): implement configuration with pydantic validation (GREEN)`
  - _Requirements: AC-12.1-12.6, FR-12_
  - _Design: Config Class, Pydantic BaseSettings_

- [x] 1.2.3 [REFACTOR] Add type hints and docstrings to config module
  - **Do**:
    - Add comprehensive docstrings to Settings class (Google-style)
    - Add type hints to all validator functions
    - Add inline comments explaining validation logic
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/config.py`
  - **Done when**: ty passes with strict mode, all tests pass
  - **Verify**: `ty check rag_ingestion/config.py && pytest tests/unit/test_config.py -v`
  - **Commit**: `refactor(config): add type hints and docstrings`
  - _Requirements: Code quality standards_
  - _Design: Type Safety_

- [x] V1 [VERIFY] Quality checkpoint after config module
  - **Do**: Run quality commands: ruff check, ty check, pytest
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/test_config.py -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(config): pass quality checkpoint` (if fixes needed)

### 1.3 Logger Module (TDD)

- [x] 1.3.1 [RED] Write failing tests for structured logging
  - **Do**:
    - Create `tests/unit/test_logger.py`
    - Write test_logger_creates_console_handler: Verify console output at INFO level
    - Write test_logger_creates_rotating_file_handler: Verify file handler with 100MB max, 5 backups
    - Write test_logger_formats_human_readable: Verify format includes timestamp, level, module, message
    - Write test_logger_respects_log_level: Test DEBUG, INFO, WARNING, ERROR levels
    - Write test_logger_logs_to_correct_file: Verify log file path from config
    - Run tests and confirm ALL FAIL
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_logger.py`
  - **Done when**: Tests exist and ALL fail
  - **Verify**: `pytest tests/unit/test_logger.py -v` (expect failures)
  - **Commit**: `test(logger): add failing logging tests (RED)`
  - _Requirements: FR-10 (structured logging), NFR-9 (log rotation)_
  - _Design: Logging Module_

- [x] 1.3.2 [GREEN] Implement logger module to pass tests
  - **Do**:
    - Create `rag_ingestion/logger.py`
    - Implement setup_logger function: Create logger with console + rotating file handlers
    - Configure RotatingFileHandler: maxBytes=100*1024*1024 (100MB), backupCount=5
    - Set human-readable format: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    - Support log level from config (DEBUG, INFO, WARNING, ERROR)
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/logger.py`
  - **Done when**: All tests in test_logger.py pass
  - **Verify**: `pytest tests/unit/test_logger.py -v` (all pass)
  - **Commit**: `feat(logger): implement structured logging with rotation (GREEN)`
  - _Requirements: FR-10, NFR-9_
  - _Design: Logger Module_

- [x] 1.3.3 [REFACTOR] Add type hints and improve logger configuration
  - **Do**:
    - Add type hints to setup_logger function
    - Add docstrings explaining parameters and return values
    - Extract format string to constant
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/logger.py`
  - **Done when**: ty passes, all tests pass
  - **Verify**: `ty check rag_ingestion/logger.py && pytest tests/unit/test_logger.py -v`
  - **Commit**: `refactor(logger): add type hints and improve configuration`
  - _Requirements: Code quality_
  - _Design: Type Safety_

- [x] V2 [VERIFY] Quality checkpoint after logger module
  - **Do**: Run quality commands
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/ -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(logger): pass quality checkpoint` (if fixes needed)

## Phase 2: TEI Integration (TDD)

**Focus**: Implement TEI client with embedding generation, dimension validation, and retry logic.

### 2.1 TEI Client Module (TDD)

- [x] 2.1.1 [RED] Write failing tests for TEI client basic operations
  - **Do**:
    - Create `tests/unit/test_embeddings.py`
    - Write test_tei_client_initialization: Verify TEIEmbedding class can be instantiated with endpoint URL
    - Write test_tei_embed_single_text: Mock httpx response, verify single embedding returned as list of 1024 floats
    - Write test_tei_embed_batch_texts: Mock batch response, verify list of embeddings returned
    - Write test_tei_validates_dimension: Mock response with wrong dimensions, verify ValidationError raised
    - Write test_tei_timeout_handling: Mock timeout exception, verify retry logic triggered
    - Run tests and confirm ALL FAIL
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_embeddings.py`
  - **Done when**: Tests exist and ALL fail
  - **Verify**: `pytest tests/unit/test_embeddings.py -v` (expect failures)
  - **Commit**: `test(embeddings): add failing TEI client tests (RED)`
  - _Requirements: FR-5 (TEI integration), AC-6.1-6.6 (embedding generation)_
  - _Design: TEI Embedding Client_

- [ ] 2.1.2 [GREEN] Implement TEI client basic operations
  - **Do**:
    - Create `rag_ingestion/embeddings.py`
    - Implement TEIEmbedding class with __init__(endpoint_url, dimensions=1024)
    - Implement _embed_single(text) method using httpx to POST to /embed endpoint
    - Implement _embed_batch(texts) method for multiple texts
    - Add dimension validation: raise ValueError if embedding length != expected dimensions
    - Add basic error handling for httpx exceptions
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/embeddings.py`
  - **Done when**: All tests in test_embeddings.py pass
  - **Verify**: `pytest tests/unit/test_embeddings.py -v` (all pass)
  - **Commit**: `feat(embeddings): implement basic TEI client (GREEN)`
  - _Requirements: FR-5, AC-6.1-6.3_
  - _Design: TEI Client_

- [ ] 2.1.3 [RED] Write failing tests for TEI retry logic and circuit breaker
  - **Do**:
    - Add to `tests/unit/test_embeddings.py`
    - Write test_tei_retries_on_network_error: Mock 3 consecutive network errors, verify 3 retry attempts with exponential backoff (1s, 2s, 4s)
    - Write test_tei_circuit_breaker_opens: Mock 5 consecutive failures, verify circuit opens
    - Write test_tei_circuit_breaker_half_open: Mock circuit open, wait timeout, verify transitions to half-open
    - Write test_tei_circuit_breaker_closes: In half-open state, mock success, verify circuit closes
    - Run tests and confirm NEW tests fail
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_embeddings.py`
  - **Done when**: New tests exist and fail, old tests still pass
  - **Verify**: `pytest tests/unit/test_embeddings.py::test_tei_retries_on_network_error -v` (expect failure)
  - **Commit**: `test(embeddings): add failing retry and circuit breaker tests (RED)`
  - _Requirements: FR-9 (retry with backoff), FR-12 (circuit breaker), AC-6.5-6.6_
  - _Design: Circuit Breaker Pattern_

- [ ] 2.1.4 [GREEN] Implement TEI retry logic and circuit breaker
  - **Do**:
    - Update `rag_ingestion/embeddings.py`
    - Add retry logic with exponential backoff: tenacity library or manual implementation (3 attempts: 1s, 2s, 4s)
    - Implement CircuitBreaker class: CLOSED, OPEN, HALF_OPEN states
    - Add failure counter and timeout tracking
    - Integrate circuit breaker into TEIEmbedding: check state before requests
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/embeddings.py`
  - **Done when**: All tests in test_embeddings.py pass
  - **Verify**: `pytest tests/unit/test_embeddings.py -v` (all pass)
  - **Commit**: `feat(embeddings): implement retry logic and circuit breaker (GREEN)`
  - _Requirements: FR-9, FR-12, AC-6.5-6.6_
  - _Design: Circuit Breaker_

- [ ] 2.1.5 [REFACTOR] Add type hints and improve TEI client structure
  - **Do**:
    - Add comprehensive type hints to all methods
    - Add docstrings (Google-style) to TEIEmbedding and CircuitBreaker classes
    - Extract constants: MAX_RETRIES=3, RETRY_DELAYS=[1, 2, 4], CIRCUIT_BREAKER_THRESHOLD=5
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/embeddings.py`
  - **Done when**: ty passes with strict mode, all tests pass
  - **Verify**: `ty check rag_ingestion/embeddings.py && pytest tests/unit/test_embeddings.py -v`
  - **Commit**: `refactor(embeddings): add type hints and extract constants`
  - _Requirements: Code quality_
  - _Design: Type Safety_

- [ ] V3 [VERIFY] Quality checkpoint after TEI client
  - **Do**: Run quality commands
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/ -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(embeddings): pass quality checkpoint` (if fixes needed)

### 2.2 LlamaIndex Custom Embedding Integration (TDD)

- [ ] 2.2.1 [RED] Write failing tests for LlamaIndex BaseEmbedding subclass
  - **Do**:
    - Create `tests/unit/test_llama_tei_integration.py`
    - Write test_tei_base_embedding_initialization: Verify LlamaTEIEmbedding inherits from BaseEmbedding
    - Write test_get_query_embedding: Verify returns 1024-dim vector for single query text
    - Write test_get_text_embedding: Verify returns 1024-dim vector for document text
    - Write test_get_text_embeddings_batch: Verify returns list of vectors for multiple texts
    - Write test_async_get_query_embedding: Verify async version works
    - Run tests and confirm ALL FAIL
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_llama_tei_integration.py`
  - **Done when**: Tests exist and ALL fail
  - **Verify**: `pytest tests/unit/test_llama_tei_integration.py -v` (expect failures)
  - **Commit**: `test(llama-tei): add failing LlamaIndex integration tests (RED)`
  - _Requirements: FR-5 (TEI integration with LlamaIndex)_
  - _Design: Custom BaseEmbedding Class_

- [ ] 2.2.2 [GREEN] Implement LlamaIndex BaseEmbedding subclass
  - **Do**:
    - Update `rag_ingestion/embeddings.py`
    - Create LlamaTEIEmbedding class inheriting from llama_index.core.embeddings.BaseEmbedding
    - Implement _get_query_embedding(query: str) -> List[float]: Call TEIEmbedding._embed_single
    - Implement _get_text_embedding(text: str) -> List[float]: Call TEIEmbedding._embed_single
    - Implement _get_text_embedding_batch(texts: List[str]) -> List[List[float]]: Call TEIEmbedding._embed_batch
    - Implement async versions: async_get_query_embedding, async_get_text_embedding, async_get_text_embedding_batch
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/embeddings.py`
  - **Done when**: All tests in test_llama_tei_integration.py pass
  - **Verify**: `pytest tests/unit/test_llama_tei_integration.py -v` (all pass)
  - **Commit**: `feat(llama-tei): implement LlamaIndex BaseEmbedding subclass (GREEN)`
  - _Requirements: FR-5_
  - _Design: Custom BaseEmbedding Class_

- [ ] 2.2.3 [REFACTOR] Add type hints and improve LlamaTEIEmbedding
  - **Do**:
    - Add type hints to all methods
    - Add docstrings explaining LlamaIndex integration
    - Extract common logic between sync/async methods
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/embeddings.py`
  - **Done when**: ty passes, all tests pass
  - **Verify**: `ty check rag_ingestion/embeddings.py && pytest tests/unit/test_llama_tei_integration.py -v`
  - **Commit**: `refactor(llama-tei): add type hints and improve structure`
  - _Requirements: Code quality_
  - _Design: Type Safety_

- [ ] V4 [VERIFY] Quality checkpoint after LlamaIndex integration
  - **Do**: Run quality commands
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/ -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(llama-tei): pass quality checkpoint` (if fixes needed)

## Phase 3: Qdrant Integration (TDD)

**Focus**: Implement Qdrant vector store with lifecycle management, metadata schema, and bulk operations.

### 3.1 Qdrant Client Module (TDD)

- [ ] 3.1.1 [RED] Write failing tests for Qdrant collection setup
  - **Do**:
    - Create `tests/unit/test_vector_store.py`
    - Write test_qdrant_client_initialization: Verify VectorStoreManager can be instantiated with Qdrant URL and collection name
    - Write test_ensure_collection_creates_if_missing: Mock qdrant_client, verify create_collection called with 1024 dimensions, cosine distance
    - Write test_ensure_collection_skips_if_exists: Mock collection already exists, verify create_collection NOT called
    - Write test_collection_configuration_matches: Verify collection has correct vector size and distance metric
    - Run tests and confirm ALL FAIL
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_vector_store.py`
  - **Done when**: Tests exist and ALL fail
  - **Verify**: `pytest tests/unit/test_vector_store.py -v` (expect failures)
  - **Commit**: `test(vector-store): add failing Qdrant setup tests (RED)`
  - _Requirements: FR-6 (Qdrant storage), AC-7.1-7.3 (collection setup)_
  - _Design: Qdrant Vector Store Manager_

- [ ] 3.1.2 [GREEN] Implement Qdrant collection setup
  - **Do**:
    - Create `rag_ingestion/vector_store.py`
    - Implement VectorStoreManager class with __init__(qdrant_url, collection_name, dimensions=1024)
    - Create qdrant_client.QdrantClient instance
    - Implement ensure_collection method: Check if collection exists, create if missing with VectorParams(size=1024, distance=Distance.COSINE)
    - Add error handling for Qdrant connection failures
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/vector_store.py`
  - **Done when**: All tests in test_vector_store.py pass
  - **Verify**: `pytest tests/unit/test_vector_store.py -v` (all pass)
  - **Commit**: `feat(vector-store): implement Qdrant collection setup (GREEN)`
  - _Requirements: FR-6, AC-7.1-7.3_
  - _Design: Vector Store Manager_

- [ ] 3.1.3 [RED] Write failing tests for vector upsert operations
  - **Do**:
    - Add to `tests/unit/test_vector_store.py`
    - Write test_upsert_single_vector: Mock upsert, verify PointStruct with correct id, vector, payload
    - Write test_upsert_batch_vectors: Mock batch upsert, verify 100 points per batch
    - Write test_upsert_generates_deterministic_ids: Verify SHA256(file_path:chunk_index) -> UUID conversion
    - Write test_upsert_includes_full_metadata: Verify payload has file_path_relative, file_path_absolute, filename, modification_date, chunk_index, section_path, heading_level, chunk_text, content_hash
    - Write test_upsert_retries_on_failure: Mock 2 failures then success, verify retry with exponential backoff
    - Run tests and confirm NEW tests fail
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_vector_store.py`
  - **Done when**: New tests exist and fail, old tests still pass
  - **Verify**: `pytest tests/unit/test_vector_store.py::test_upsert_single_vector -v` (expect failure)
  - **Commit**: `test(vector-store): add failing upsert tests (RED)`
  - _Requirements: AC-7.2 (payload), AC-7.4 (deterministic IDs), AC-7.5 (retry), AC-7.6 (bulk upsert)_
  - _Design: Metadata Schema, Point ID Generation_

- [ ] 3.1.4 [GREEN] Implement vector upsert operations
  - **Do**:
    - Update `rag_ingestion/vector_store.py`
    - Implement _generate_point_id(file_path_relative, chunk_index) -> UUID: SHA256 hash, truncate to 128 bits, convert to UUID
    - Implement upsert_vector(vector, metadata): Create PointStruct with deterministic ID and full payload
    - Implement upsert_vectors_batch(vectors_with_metadata): Batch 100 points per upsert call
    - Add retry logic with exponential backoff (3 attempts: 1s, 2s, 4s)
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/vector_store.py`
  - **Done when**: All tests in test_vector_store.py pass
  - **Verify**: `pytest tests/unit/test_vector_store.py -v` (all pass)
  - **Commit**: `feat(vector-store): implement vector upsert with retry (GREEN)`
  - _Requirements: AC-7.2, AC-7.4, AC-7.5, AC-7.6_
  - _Design: Upsert Operations_

- [ ] 3.1.5 [RED] Write failing tests for vector deletion operations
  - **Do**:
    - Add to `tests/unit/test_vector_store.py`
    - Write test_delete_by_file_path: Mock delete, verify filter by file_path_relative field
    - Write test_delete_returns_count: Mock delete response with count, verify returned
    - Write test_delete_handles_missing_file: Mock delete with 0 count, verify no error
    - Run tests and confirm NEW tests fail
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_vector_store.py`
  - **Done when**: New tests exist and fail, old tests still pass
  - **Verify**: `pytest tests/unit/test_vector_store.py::test_delete_by_file_path -v` (expect failure)
  - **Commit**: `test(vector-store): add failing deletion tests (RED)`
  - _Requirements: AC-3.3-3.4 (vector deletion), AC-4.2-4.3 (deletion cleanup)_
  - _Design: Vector Lifecycle Management_

- [ ] 3.1.6 [GREEN] Implement vector deletion operations
  - **Do**:
    - Update `rag_ingestion/vector_store.py`
    - Implement delete_vectors_by_file(file_path_relative): Use qdrant_client.delete with filter matching file_path_relative
    - Return count of deleted vectors
    - Add error handling for deletion failures
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/vector_store.py`
  - **Done when**: All tests in test_vector_store.py pass
  - **Verify**: `pytest tests/unit/test_vector_store.py -v` (all pass)
  - **Commit**: `feat(vector-store): implement vector deletion (GREEN)`
  - _Requirements: AC-3.3-3.4, AC-4.2-4.3_
  - _Design: Deletion Operations_

- [ ] 3.1.7 [REFACTOR] Add type hints and improve vector store structure
  - **Do**:
    - Add comprehensive type hints to all methods
    - Add docstrings (Google-style) to VectorStoreManager class
    - Extract constants: BATCH_SIZE=100, MAX_RETRIES=3
    - Create Metadata TypedDict for payload structure
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/vector_store.py`
  - **Done when**: ty passes with strict mode, all tests pass
  - **Verify**: `ty check rag_ingestion/vector_store.py && pytest tests/unit/test_vector_store.py -v`
  - **Commit**: `refactor(vector-store): add type hints and TypedDict for metadata`
  - _Requirements: Code quality_
  - _Design: Type Safety_

- [ ] V5 [VERIFY] Quality checkpoint after Qdrant integration
  - **Do**: Run quality commands
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/ -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(vector-store): pass quality checkpoint` (if fixes needed)

### 3.2 Qdrant Payload Indexing (TDD)

- [ ] 3.2.1 [RED] Write failing tests for payload index setup
  - **Do**:
    - Add to `tests/unit/test_vector_store.py`
    - Write test_create_payload_index_file_path: Mock create_payload_index call, verify keyword index on file_path_relative
    - Write test_create_payload_index_filename: Verify keyword index on filename
    - Write test_create_payload_index_modification_date: Verify datetime index on modification_date
    - Write test_create_payload_index_tags: Verify keyword array index on tags
    - Write test_ensure_indexes_idempotent: Call ensure_indexes twice, verify create only called once per field
    - Run tests and confirm NEW tests fail
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_vector_store.py`
  - **Done when**: New tests exist and fail
  - **Verify**: `pytest tests/unit/test_vector_store.py::test_create_payload_index_file_path -v` (expect failure)
  - **Commit**: `test(vector-store): add failing payload index tests (RED)`
  - _Requirements: AC-8.6 (payload indexing)_
  - _Design: Qdrant Payload Indexes_

- [ ] 3.2.2 [GREEN] Implement payload indexing
  - **Do**:
    - Update `rag_ingestion/vector_store.py`
    - Implement ensure_payload_indexes method
    - Create keyword indexes for: file_path_relative, filename, tags (array)
    - Create datetime index for: modification_date
    - Check existing indexes before creating to avoid duplicates
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/vector_store.py`
  - **Done when**: All new tests pass
  - **Verify**: `pytest tests/unit/test_vector_store.py -v` (all pass)
  - **Commit**: `feat(vector-store): implement payload indexing (GREEN)`
  - _Requirements: AC-8.6_
  - _Design: Payload Indexes_

- [ ] 3.2.3 [REFACTOR] Extract index configuration to constants
  - **Do**:
    - Create PAYLOAD_INDEXES constant with field names and types
    - Refactor ensure_payload_indexes to iterate over PAYLOAD_INDEXES
    - Add docstrings explaining index purpose
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/vector_store.py`
  - **Done when**: ty passes, all tests pass
  - **Verify**: `ty check rag_ingestion/vector_store.py && pytest tests/unit/test_vector_store.py -v`
  - **Commit**: `refactor(vector-store): extract index configuration to constants`
  - _Requirements: Code quality_

- [ ] V6 [VERIFY] Quality checkpoint after payload indexing
  - **Do**: Run quality commands
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/ -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(vector-store): pass quality checkpoint` (if fixes needed)

## Phase 4: Document Processing (TDD)

**Focus**: Implement markdown-aware chunking, frontmatter parsing, and document processing pipeline.

### 4.1 Chunker Module (TDD)

- [ ] 4.1.1 [RED] Write failing tests for markdown chunking
  - **Do**:
    - Create `tests/unit/test_chunker.py`
    - Write test_chunk_by_headings: Provide markdown with ##, ###, verify chunks split at headings
    - Write test_chunk_preserves_heading_hierarchy: Verify section_path includes parent headings (e.g., "Guide > Installation")
    - Write test_chunk_size_target: Verify chunks target ~512 tokens with 15% overlap (~77 tokens)
    - Write test_chunk_without_headings: Provide markdown without headings, verify paragraph-level splitting with section_path=filename
    - Write test_chunk_includes_metadata: Verify chunk has chunk_index, heading_level, section_path
    - Write test_chunk_preserves_formatting: Verify code blocks, lists preserved in chunk_text
    - Run tests and confirm ALL FAIL
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_chunker.py`
  - **Done when**: Tests exist and ALL fail
  - **Verify**: `pytest tests/unit/test_chunker.py -v` (expect failures)
  - **Commit**: `test(chunker): add failing markdown chunking tests (RED)`
  - _Requirements: FR-4 (chunking), AC-5.1-5.6 (markdown-aware chunking)_
  - _Design: Markdown Chunker_

- [ ] 4.1.2 [GREEN] Implement markdown chunking
  - **Do**:
    - Create `rag_ingestion/chunker.py`
    - Implement MarkdownChunker class with chunk(text, filename) method
    - Use LlamaIndex MarkdownNodeParser with chunk_size=512, chunk_overlap=77
    - Parse heading structure to build section_path (hierarchy of parent headings)
    - Extract heading_level from markdown syntax (#=1, ##=2, etc.)
    - Handle files without headings: Use paragraph splitting, set section_path=filename, heading_level=0
    - Add chunk_index, preserve full chunk_text with formatting
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/chunker.py`
  - **Done when**: All tests in test_chunker.py pass
  - **Verify**: `pytest tests/unit/test_chunker.py -v` (all pass)
  - **Commit**: `feat(chunker): implement markdown-aware chunking (GREEN)`
  - _Requirements: FR-4, AC-5.1-5.6_
  - _Design: Markdown Chunker_

- [ ] 4.1.3 [RED] Write failing tests for frontmatter parsing
  - **Do**:
    - Add to `tests/unit/test_chunker.py`
    - Write test_parse_frontmatter_with_tags: Provide markdown with YAML frontmatter containing tags array, verify extracted
    - Write test_parse_frontmatter_without_tags: Provide frontmatter without tags, verify returns None
    - Write test_parse_frontmatter_invalid_yaml: Provide invalid YAML, verify skipped gracefully with None
    - Write test_parse_frontmatter_missing: Provide markdown without frontmatter, verify returns None
    - Run tests and confirm NEW tests fail
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_chunker.py`
  - **Done when**: New tests exist and fail
  - **Verify**: `pytest tests/unit/test_chunker.py::test_parse_frontmatter_with_tags -v` (expect failure)
  - **Commit**: `test(chunker): add failing frontmatter parsing tests (RED)`
  - _Requirements: FR-13 (frontmatter parsing), AC-8.2 (tags metadata)_
  - _Design: Frontmatter Parser_

- [ ] 4.1.4 [GREEN] Implement frontmatter parsing
  - **Do**:
    - Update `rag_ingestion/chunker.py`
    - Implement parse_frontmatter(text) method using PyYAML
    - Extract YAML between --- delimiters at document start
    - Parse tags field (array of strings), return None if missing
    - Handle invalid YAML gracefully: log warning, return None
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/chunker.py`
  - **Done when**: All tests in test_chunker.py pass
  - **Verify**: `pytest tests/unit/test_chunker.py -v` (all pass)
  - **Commit**: `feat(chunker): implement frontmatter parsing (GREEN)`
  - _Requirements: FR-13, AC-8.2_
  - _Design: Frontmatter Parser_

- [ ] 4.1.5 [REFACTOR] Add type hints and improve chunker structure
  - **Do**:
    - Add comprehensive type hints to all methods
    - Add docstrings (Google-style) to MarkdownChunker class
    - Create Chunk TypedDict for chunk metadata structure
    - Extract constants: CHUNK_SIZE=512, CHUNK_OVERLAP=77
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/chunker.py`
  - **Done when**: ty passes with strict mode, all tests pass
  - **Verify**: `ty check rag_ingestion/chunker.py && pytest tests/unit/test_chunker.py -v`
  - **Commit**: `refactor(chunker): add type hints and TypedDict for chunks`
  - _Requirements: Code quality_
  - _Design: Type Safety_

- [ ] V7 [VERIFY] Quality checkpoint after chunker
  - **Do**: Run quality commands
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/ -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(chunker): pass quality checkpoint` (if fixes needed)

### 4.2 Document Processor Module (TDD)

- [ ] 4.2.1 [RED] Write failing tests for document processing pipeline
  - **Do**:
    - Create `tests/unit/test_processor.py`
    - Write test_load_markdown_file: Verify file loaded, content extracted
    - Write test_process_document_chunks: Verify document chunked, embeddings generated, vectors stored
    - Write test_process_document_extracts_metadata: Verify file_path_relative, file_path_absolute, filename, modification_date extracted
    - Write test_process_document_includes_tags: Mock frontmatter with tags, verify included in payload
    - Write test_process_document_calculates_content_hash: Verify SHA256 hash of chunk_text added to metadata
    - Write test_process_document_handles_errors: Mock chunk failure, verify logged and continues
    - Run tests and confirm ALL FAIL
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_processor.py`
  - **Done when**: Tests exist and ALL fail
  - **Verify**: `pytest tests/unit/test_processor.py -v` (expect failures)
  - **Commit**: `test(processor): add failing document processing tests (RED)`
  - _Requirements: FR-4, FR-5, FR-6 (processing pipeline), AC-8.1-8.3 (metadata)_
  - _Design: Document Processor_

- [ ] 4.2.2 [GREEN] Implement document processor
  - **Do**:
    - Create `rag_ingestion/processor.py`
    - Implement DocumentProcessor class with __init__(chunker, embedder, vector_store, config)
    - Implement load_file(file_path) method: Read file, extract modification date
    - Implement process_document(file_path, watch_folder) method:
      - Load file content
      - Calculate file_path_relative (relative to watch_folder)
      - Parse frontmatter for tags
      - Chunk document
      - Generate embeddings for all chunks
      - Calculate content_hash (SHA256) for each chunk
      - Construct metadata payload
      - Upsert vectors to Qdrant
    - Add error handling: Log failures, continue processing
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/processor.py`
  - **Done when**: All tests in test_processor.py pass
  - **Verify**: `pytest tests/unit/test_processor.py -v` (all pass)
  - **Commit**: `feat(processor): implement document processing pipeline (GREEN)`
  - _Requirements: FR-4, FR-5, FR-6, AC-8.1-8.3_
  - _Design: Document Processor_

- [ ] 4.2.3 [RED] Write failing tests for batch processing
  - **Do**:
    - Add to `tests/unit/test_processor.py`
    - Write test_process_batch: Provide list of 10 file paths, verify all processed
    - Write test_process_batch_respects_concurrency: Mock processing with delays, verify max 10 concurrent (from config)
    - Write test_process_batch_logs_progress: Verify progress logged every N files (e.g., "Processed 50/250")
    - Write test_process_batch_handles_failures: Mock 2 failures, verify logged and other 8 files processed
    - Run tests and confirm NEW tests fail
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_processor.py`
  - **Done when**: New tests exist and fail
  - **Verify**: `pytest tests/unit/test_processor.py::test_process_batch -v` (expect failure)
  - **Commit**: `test(processor): add failing batch processing tests (RED)`
  - _Requirements: FR-3 (batch processing), AC-1.2-1.4 (batch ingestion)_
  - _Design: Batch Processing_

- [ ] 4.2.4 [GREEN] Implement batch processing
  - **Do**:
    - Update `rag_ingestion/processor.py`
    - Implement process_batch(file_paths, watch_folder) method
    - Use asyncio.gather with limit from config.max_concurrent_docs for concurrency control
    - Log progress every 10 files: "Processed 50/250 files"
    - Collect failed files, log at end: "Failed: 2 files (see failed_documents.jsonl)"
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/processor.py`
  - **Done when**: All tests in test_processor.py pass
  - **Verify**: `pytest tests/unit/test_processor.py -v` (all pass)
  - **Commit**: `feat(processor): implement batch processing with concurrency (GREEN)`
  - _Requirements: FR-3, AC-1.2-1.4_
  - _Design: Batch Processing_

- [ ] 4.2.5 [REFACTOR] Add type hints and improve processor structure
  - **Do**:
    - Add comprehensive type hints to all methods
    - Add docstrings (Google-style) to DocumentProcessor class
    - Extract helper methods: _extract_metadata, _calculate_content_hash
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/processor.py`
  - **Done when**: ty passes with strict mode, all tests pass
  - **Verify**: `ty check rag_ingestion/processor.py && pytest tests/unit/test_processor.py -v`
  - **Commit**: `refactor(processor): add type hints and extract helpers`
  - _Requirements: Code quality_
  - _Design: Type Safety_

- [ ] V8 [VERIFY] Quality checkpoint after processor
  - **Do**: Run quality commands
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/ -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(processor): pass quality checkpoint` (if fixes needed)

## Phase 5: File Watching (TDD)

**Focus**: Implement file system monitoring with debouncing, event handling, and lifecycle management.

### 5.1 File Watcher Module (TDD)

- [ ] 5.1.1 [RED] Write failing tests for watchdog event handler
  - **Do**:
    - Create `tests/unit/test_watcher.py`
    - Write test_watcher_detects_create: Simulate file creation event, verify on_created called
    - Write test_watcher_detects_modify: Simulate file modification event, verify on_modified called
    - Write test_watcher_detects_delete: Simulate file deletion event, verify on_deleted called
    - Write test_watcher_filters_markdown: Simulate .txt file event, verify ignored
    - Write test_watcher_ignores_directories: Simulate directory event, verify ignored
    - Write test_watcher_recursive: Simulate event in subdirectory, verify detected
    - Run tests and confirm ALL FAIL
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_watcher.py`
  - **Done when**: Tests exist and ALL fail
  - **Verify**: `pytest tests/unit/test_watcher.py -v` (expect failures)
  - **Commit**: `test(watcher): add failing file watching tests (RED)`
  - _Requirements: FR-1 (file monitoring), AC-2.1-2.2 (detection), AC-2.6 (filtering)_
  - _Design: File Watcher_

- [ ] 5.1.2 [GREEN] Implement watchdog event handler
  - **Do**:
    - Create `rag_ingestion/watcher.py`
    - Implement MarkdownEventHandler class extending PatternMatchingEventHandler
    - Set patterns=["*.md"], ignore_directories=True
    - Implement on_created(event): Log event, queue file path
    - Implement on_modified(event): Log event, queue file path
    - Implement on_deleted(event): Log event, queue file path
    - Configure recursive monitoring
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/watcher.py`
  - **Done when**: All tests in test_watcher.py pass
  - **Verify**: `pytest tests/unit/test_watcher.py -v` (all pass)
  - **Commit**: `feat(watcher): implement watchdog event handler (GREEN)`
  - _Requirements: FR-1, AC-2.1-2.2, AC-2.6_
  - _Design: Event Handler_

- [ ] 5.1.3 [RED] Write failing tests for debouncing
  - **Do**:
    - Add to `tests/unit/test_watcher.py`
    - Write test_debounce_rapid_events: Simulate 5 modify events within 1 second, verify only 1 queued
    - Write test_debounce_uses_timer: Verify threading.Timer used with 1-second delay
    - Write test_debounce_per_file: Simulate events for 2 different files simultaneously, verify both debounced independently
    - Write test_debounce_cancels_previous_timer: Simulate event, then another before timer expires, verify first timer cancelled
    - Run tests and confirm NEW tests fail
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_watcher.py`
  - **Done when**: New tests exist and fail
  - **Verify**: `pytest tests/unit/test_watcher.py::test_debounce_rapid_events -v` (expect failure)
  - **Commit**: `test(watcher): add failing debouncing tests (RED)`
  - _Requirements: FR-2 (debouncing), AC-2.3 (1-second threshold), AC-3.2 (editor saves)_
  - _Design: Debouncing Logic_

- [ ] 5.1.4 [GREEN] Implement debouncing
  - **Do**:
    - Update `rag_ingestion/watcher.py`
    - Add _debounce_timers dict to track timers per file path
    - Implement _debounce(file_path, callback) method:
      - Cancel existing timer for file_path if present
      - Create new threading.Timer(1.0, callback) with 1-second delay
      - Store in _debounce_timers
      - Start timer
    - Update on_created, on_modified, on_deleted to use _debounce
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/watcher.py`
  - **Done when**: All tests in test_watcher.py pass
  - **Verify**: `pytest tests/unit/test_watcher.py -v` (all pass)
  - **Commit**: `feat(watcher): implement event debouncing (GREEN)`
  - _Requirements: FR-2, AC-2.3, AC-3.2_
  - _Design: Debouncing_

- [ ] 5.1.5 [RED] Write failing tests for excluded directories
  - **Do**:
    - Add to `tests/unit/test_watcher.py`
    - Write test_ignore_git_directory: Simulate event in .git/, verify ignored
    - Write test_ignore_hidden_directories: Simulate events in .cache/, .vscode/, verify ignored
    - Write test_ignore_build_directories: Simulate events in __pycache__, node_modules, dist, build, verify ignored
    - Write test_ignore_symlinks: Create symlink, simulate event, verify ignored
    - Run tests and confirm NEW tests fail
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_watcher.py`
  - **Done when**: New tests exist and fail
  - **Verify**: `pytest tests/unit/test_watcher.py::test_ignore_git_directory -v` (expect failure)
  - **Commit**: `test(watcher): add failing exclusion tests (RED)`
  - _Requirements: AC-2.6 (excluded directories)_
  - _Design: File Filtering_

- [ ] 5.1.6 [GREEN] Implement directory exclusions
  - **Do**:
    - Update `rag_ingestion/watcher.py`
    - Add ignore_patterns to PatternMatchingEventHandler: [".git/*", ".*/*", "__pycache__/*", "node_modules/*", "venv/*", "dist/*", "build/*"]
    - Add symlink detection: Check os.path.islink(event.src_path), skip if True
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/watcher.py`
  - **Done when**: All tests in test_watcher.py pass
  - **Verify**: `pytest tests/unit/test_watcher.py -v` (all pass)
  - **Commit**: `feat(watcher): implement directory exclusions and symlink filtering (GREEN)`
  - _Requirements: AC-2.6_
  - _Design: Filtering_

- [ ] 5.1.7 [REFACTOR] Add type hints and improve watcher structure
  - **Do**:
    - Add comprehensive type hints to all methods
    - Add docstrings (Google-style) to MarkdownEventHandler class
    - Extract constants: DEBOUNCE_DELAY=1.0, EXCLUDED_PATTERNS
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/watcher.py`
  - **Done when**: ty passes with strict mode, all tests pass
  - **Verify**: `ty check rag_ingestion/watcher.py && pytest tests/unit/test_watcher.py -v`
  - **Commit**: `refactor(watcher): add type hints and extract constants`
  - _Requirements: Code quality_
  - _Design: Type Safety_

- [ ] V9 [VERIFY] Quality checkpoint after watcher
  - **Do**: Run quality commands
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/ -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(watcher): pass quality checkpoint` (if fixes needed)

### 5.2 Event Queue and Lifecycle Management (TDD)

- [ ] 5.2.1 [RED] Write failing tests for event lifecycle handlers
  - **Do**:
    - Add to `tests/unit/test_watcher.py`
    - Write test_handle_create_event: Mock processor.process_document, verify called for new file
    - Write test_handle_modify_event_deletes_old_vectors: Mock vector_store.delete_vectors_by_file, verify called before re-ingestion
    - Write test_handle_modify_event_reprocesses: Mock processor.process_document, verify called after deletion
    - Write test_handle_delete_event_removes_vectors: Mock vector_store.delete_vectors_by_file, verify called
    - Write test_handle_delete_event_logs_count: Mock deletion returning 15, verify logged "Deleted 15 vectors"
    - Run tests and confirm NEW tests fail
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_watcher.py`
  - **Done when**: New tests exist and fail
  - **Verify**: `pytest tests/unit/test_watcher.py::test_handle_create_event -v` (expect failure)
  - **Commit**: `test(watcher): add failing lifecycle handler tests (RED)`
  - _Requirements: AC-2.4 (async processing), AC-3.3-3.5 (modification), AC-4.2-4.4 (deletion)_
  - _Design: Event Lifecycle Handlers_

- [ ] 5.2.2 [GREEN] Implement event lifecycle handlers
  - **Do**:
    - Update `rag_ingestion/watcher.py`
    - Implement _handle_create(file_path): Call processor.process_document
    - Implement _handle_modify(file_path):
      - Calculate file_path_relative
      - Call vector_store.delete_vectors_by_file(file_path_relative)
      - Call processor.process_document
      - Log "Re-ingested {file_path} (deleted old vectors)"
    - Implement _handle_delete(file_path):
      - Calculate file_path_relative
      - Call vector_store.delete_vectors_by_file(file_path_relative)
      - Log "Deleted {count} vectors for {file_path}"
    - Add error handling: Log failures, don't crash watcher
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/watcher.py`
  - **Done when**: All tests in test_watcher.py pass
  - **Verify**: `pytest tests/unit/test_watcher.py -v` (all pass)
  - **Commit**: `feat(watcher): implement event lifecycle handlers (GREEN)`
  - _Requirements: AC-2.4, AC-3.3-3.5, AC-4.2-4.4_
  - _Design: Lifecycle Handlers_

- [ ] 5.2.3 [RED] Write failing tests for queue integration
  - **Do**:
    - Add to `tests/unit/test_watcher.py`
    - Write test_events_queued_via_callback: Verify debounced events added to asyncio.Queue
    - Write test_queue_non_blocking: Verify queue.put_nowait used (doesn't block watcher thread)
    - Write test_queue_overflow_handling: Fill queue to max size, simulate new event, verify backpressure logged
    - Run tests and confirm NEW tests fail
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_watcher.py`
  - **Done when**: New tests exist and fail
  - **Verify**: `pytest tests/unit/test_watcher.py::test_events_queued_via_callback -v` (expect failure)
  - **Commit**: `test(watcher): add failing queue integration tests (RED)`
  - _Requirements: FR-8 (queue management), AC-10.1-10.2 (non-blocking)_
  - _Design: Queue Integration_

- [ ] 5.2.4 [GREEN] Implement queue integration
  - **Do**:
    - Update `rag_ingestion/watcher.py`
    - Add event_queue parameter to __init__: asyncio.Queue
    - Update debounce callbacks to call queue.put_nowait((event_type, file_path))
    - Add queue overflow check: If queue.qsize() >= config.queue_max_size, log warning "Queue full ({size}/{max}), backpressure active"
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/watcher.py`
  - **Done when**: All tests in test_watcher.py pass
  - **Verify**: `pytest tests/unit/test_watcher.py -v` (all pass)
  - **Commit**: `feat(watcher): implement queue integration with overflow detection (GREEN)`
  - _Requirements: FR-8, AC-10.1-10.2_
  - _Design: Queue Integration_

- [ ] 5.2.5 [REFACTOR] Extract event handling logic
  - **Do**:
    - Create separate methods for create/modify/delete handlers
    - Add comprehensive error handling with try/except in each handler
    - Add docstrings explaining event lifecycle
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/watcher.py`
  - **Done when**: ty passes, all tests pass
  - **Verify**: `ty check rag_ingestion/watcher.py && pytest tests/unit/test_watcher.py -v`
  - **Commit**: `refactor(watcher): extract and document event handlers`
  - _Requirements: Code quality_

- [ ] V10 [VERIFY] Quality checkpoint after event lifecycle
  - **Do**: Run quality commands
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/ -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(watcher): pass quality checkpoint` (if fixes needed)

## Phase 6: State Recovery and Startup (TDD)

**Focus**: Implement startup validation, state recovery, and batch processing of existing files.

### 6.1 Quality Verification Module (TDD)

- [ ] 6.1.1 [RED] Write failing tests for startup validation
  - **Do**:
    - Create `tests/unit/test_quality.py`
    - Write test_validate_tei_connection: Mock successful TEI /embed request, verify passes
    - Write test_validate_tei_retries: Mock 2 failures then success, verify 3 attempts with delays (5s, 10s, 20s)
    - Write test_validate_tei_exits_on_failure: Mock 3 failures, verify sys.exit(1) called
    - Write test_validate_tei_checks_dimensions: Mock response with 1024-dim embedding, verify validated
    - Write test_validate_tei_rejects_wrong_dimensions: Mock 768-dim embedding, verify raises error
    - Run tests and confirm ALL FAIL
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_quality.py`
  - **Done when**: Tests exist and ALL fail
  - **Verify**: `pytest tests/unit/test_quality.py -v` (expect failures)
  - **Commit**: `test(quality): add failing TEI validation tests (RED)`
  - _Requirements: AC-9.1 (TEI validation), FR-11 (dimension validation)_
  - _Design: Quality Verifier_

- [ ] 6.1.2 [GREEN] Implement TEI startup validation
  - **Do**:
    - Create `rag_ingestion/quality.py`
    - Implement QualityVerifier class with validate_tei_connection(embedder) method
    - Send test embedding request with "test text"
    - Validate response has exactly 1024 dimensions
    - Retry 3 times with delays: 5s, 10s, 20s
    - Call sys.exit(1) if all retries fail
    - Log validation steps: "Validating TEI connection...", "TEI validation passed"
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/quality.py`
  - **Done when**: All tests in test_quality.py pass
  - **Verify**: `pytest tests/unit/test_quality.py -v` (all pass)
  - **Commit**: `feat(quality): implement TEI startup validation (GREEN)`
  - _Requirements: AC-9.1, FR-11_
  - _Design: TEI Validation_

- [ ] 6.1.3 [RED] Write failing tests for Qdrant validation
  - **Do**:
    - Add to `tests/unit/test_quality.py`
    - Write test_validate_qdrant_connection: Mock successful Qdrant get_collection, verify passes
    - Write test_validate_qdrant_retries: Mock 2 failures then success, verify 3 attempts with delays
    - Write test_validate_qdrant_exits_on_failure: Mock 3 failures, verify sys.exit(1) called
    - Write test_validate_qdrant_checks_dimensions: Mock collection with 1024 vector size, verify passes
    - Write test_validate_qdrant_rejects_wrong_dimensions: Mock collection with 768 vector size, verify raises error
    - Run tests and confirm NEW tests fail
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_quality.py`
  - **Done when**: New tests exist and fail
  - **Verify**: `pytest tests/unit/test_quality.py::test_validate_qdrant_connection -v` (expect failure)
  - **Commit**: `test(quality): add failing Qdrant validation tests (RED)`
  - _Requirements: AC-9.2 (Qdrant validation)_
  - _Design: Qdrant Validation_

- [ ] 6.1.4 [GREEN] Implement Qdrant startup validation
  - **Do**:
    - Update `rag_ingestion/quality.py`
    - Implement validate_qdrant_connection(vector_store) method
    - Call vector_store.get_collection_info()
    - Validate vector size == 1024, distance metric == COSINE
    - Retry 3 times with delays: 5s, 10s, 20s
    - Call sys.exit(1) if all retries fail
    - Log validation steps
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/quality.py`
  - **Done when**: All tests in test_quality.py pass
  - **Verify**: `pytest tests/unit/test_quality.py -v` (all pass)
  - **Commit**: `feat(quality): implement Qdrant startup validation (GREEN)`
  - _Requirements: AC-9.2_
  - _Design: Qdrant Validation_

- [ ] 6.1.5 [RED] Write failing tests for runtime quality checks
  - **Do**:
    - Add to `tests/unit/test_quality.py`
    - Write test_check_embedding_dimensions: Mock embedding with 1024 dims, verify passes
    - Write test_check_embedding_rejects_wrong_dims: Mock 512-dim embedding, verify raises ValueError
    - Write test_sample_embeddings_for_normalization: Provide 100 embeddings, verify 5% sampled
    - Write test_check_normalization: Provide L2-normalized embedding (norm=1.0), verify passes
    - Write test_check_normalization_with_tolerance: Provide embedding with norm=1.008, verify passes (±0.01 tolerance)
    - Write test_check_normalization_warns: Provide embedding with norm=0.9, verify WARNING logged
    - Run tests and confirm NEW tests fail
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_quality.py`
  - **Done when**: New tests exist and fail
  - **Verify**: `pytest tests/unit/test_quality.py::test_check_embedding_dimensions -v` (expect failure)
  - **Commit**: `test(quality): add failing runtime quality check tests (RED)`
  - _Requirements: AC-9.3-9.5 (runtime checks), FR-16 (sampling)_
  - _Design: Runtime Quality Checks_

- [ ] 6.1.6 [GREEN] Implement runtime quality checks
  - **Do**:
    - Update `rag_ingestion/quality.py`
    - Implement check_embedding_dimensions(embedding, expected_dims=1024): Raise ValueError if mismatch
    - Implement sample_embeddings(embeddings, sample_rate=0.05): Randomly sample 5%
    - Implement check_normalization(embedding, tolerance=0.01): Calculate L2 norm, log WARNING if outside 1.0 ± 0.01
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/quality.py`
  - **Done when**: All tests in test_quality.py pass
  - **Verify**: `pytest tests/unit/test_quality.py -v` (all pass)
  - **Commit**: `feat(quality): implement runtime dimension and normalization checks (GREEN)`
  - _Requirements: AC-9.3-9.5, FR-16_
  - _Design: Runtime Checks_

- [ ] 6.1.7 [REFACTOR] Add type hints and improve quality module
  - **Do**:
    - Add comprehensive type hints to all methods
    - Add docstrings (Google-style) to QualityVerifier class
    - Extract constants: RETRY_DELAYS=[5, 10, 20], NORMALIZATION_TOLERANCE=0.01, SAMPLE_RATE=0.05
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/quality.py`
  - **Done when**: ty passes with strict mode, all tests pass
  - **Verify**: `ty check rag_ingestion/quality.py && pytest tests/unit/test_quality.py -v`
  - **Commit**: `refactor(quality): add type hints and extract constants`
  - _Requirements: Code quality_

- [ ] V11 [VERIFY] Quality checkpoint after quality module
  - **Do**: Run quality commands
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/ -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(quality): pass quality checkpoint` (if fixes needed)

### 6.2 State Recovery Module (TDD)

- [ ] 6.2.1 [RED] Write failing tests for state recovery
  - **Do**:
    - Create `tests/unit/test_recovery.py`
    - Write test_query_existing_files_from_qdrant: Mock scroll API response with 3 files, verify extracted
    - Write test_extract_unique_file_paths: Mock scroll with duplicate file_paths (multiple chunks), verify unique list
    - Write test_extract_modification_dates: Mock scroll with modification_date payloads, verify latest per file
    - Write test_compare_with_filesystem: Provide 5 filesystem files, 3 in Qdrant (2 up-to-date, 1 stale), verify returns 1 stale + 2 new
    - Write test_skip_up_to_date_files: Verify files with Qdrant mod_date >= filesystem mod_date skipped
    - Run tests and confirm ALL FAIL
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_recovery.py`
  - **Done when**: Tests exist and ALL fail
  - **Verify**: `pytest tests/unit/test_recovery.py -v` (expect failures)
  - **Commit**: `test(recovery): add failing state recovery tests (RED)`
  - _Requirements: FR-19 (restart resumption), AC-11.6 (state recovery)_
  - _Design: State Recovery_

- [ ] 6.2.2 [GREEN] Implement state recovery
  - **Do**:
    - Create `rag_ingestion/recovery.py`
    - Implement StateRecovery class with get_files_to_process(watch_folder, vector_store) method
    - Query Qdrant using scroll API to get all file_path_relative + modification_date pairs
    - Build dict of file_path -> latest modification_date
    - Scan filesystem for all .md files recursively
    - Compare modification dates: Include file if Qdrant mod_date < filesystem mod_date OR file not in Qdrant
    - Return list of file paths to process
    - Log skipped files: "Skipped {count} up-to-date files"
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/recovery.py`
  - **Done when**: All tests in test_recovery.py pass
  - **Verify**: `pytest tests/unit/test_recovery.py -v` (all pass)
  - **Commit**: `feat(recovery): implement state recovery with modification date comparison (GREEN)`
  - _Requirements: FR-19, AC-11.6_
  - _Design: State Recovery_

- [ ] 6.2.3 [REFACTOR] Add type hints and improve recovery module
  - **Do**:
    - Add comprehensive type hints to all methods
    - Add docstrings (Google-style) to StateRecovery class
    - Extract helper methods: _query_qdrant_state, _scan_filesystem, _compare_states
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/recovery.py`
  - **Done when**: ty passes with strict mode, all tests pass
  - **Verify**: `ty check rag_ingestion/recovery.py && pytest tests/unit/test_recovery.py -v`
  - **Commit**: `refactor(recovery): add type hints and extract helper methods`
  - _Requirements: Code quality_

- [ ] V12 [VERIFY] Quality checkpoint after recovery module
  - **Do**: Run quality commands
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/ -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(recovery): pass quality checkpoint` (if fixes needed)

### 6.3 Main Entry Point and Orchestration (TDD)

- [ ] 6.3.1 [RED] Write failing tests for main orchestration
  - **Do**:
    - Create `tests/unit/test_main.py`
    - Write test_main_loads_config: Verify Settings loaded from .env
    - Write test_main_validates_services: Verify TEI and Qdrant validation called on startup
    - Write test_main_performs_state_recovery: Verify StateRecovery.get_files_to_process called
    - Write test_main_processes_batch: Verify processor.process_batch called with recovered files
    - Write test_main_starts_watcher: Verify watchdog Observer started after batch processing
    - Write test_main_handles_keyboard_interrupt: Simulate Ctrl+C, verify clean shutdown
    - Run tests and confirm ALL FAIL
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_main.py`
  - **Done when**: Tests exist and ALL fail
  - **Verify**: `pytest tests/unit/test_main.py -v` (expect failures)
  - **Commit**: `test(main): add failing orchestration tests (RED)`
  - _Requirements: FR-3 (batch on startup), AC-1.5 (watch mode after batch)_
  - _Design: Main Entry Point_

- [ ] 6.3.2 [GREEN] Implement main orchestration
  - **Do**:
    - Create `rag_ingestion/main.py`
    - Implement main() function:
      - Load config from Settings()
      - Setup logger
      - Initialize all components (embedder, vector_store, chunker, processor, quality_verifier)
      - Run startup validations: validate_tei_connection, validate_qdrant_connection
      - Ensure collection and indexes exist
      - Perform state recovery: get_files_to_process
      - Process batch if files to process
      - Start watchdog observer
      - Run event processing loop
      - Handle KeyboardInterrupt for clean shutdown
    - Add if __name__ == "__main__": main()
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/main.py`
  - **Done when**: All tests in test_main.py pass
  - **Verify**: `pytest tests/unit/test_main.py -v` (all pass)
  - **Commit**: `feat(main): implement main orchestration and startup sequence (GREEN)`
  - _Requirements: FR-3, AC-1.5_
  - _Design: Main Entry Point_

- [ ] 6.3.3 [RED] Write failing tests for event processing loop
  - **Do**:
    - Add to `tests/unit/test_main.py`
    - Write test_event_loop_processes_queue: Mock queue with 3 events, verify all processed
    - Write test_event_loop_handles_create: Mock create event, verify processor.process_document called
    - Write test_event_loop_handles_modify: Mock modify event, verify delete + reprocess
    - Write test_event_loop_handles_delete: Mock delete event, verify vector deletion
    - Write test_event_loop_logs_queue_depth: Verify queue depth logged periodically
    - Write test_event_loop_handles_exceptions: Mock processing exception, verify logged and loop continues
    - Run tests and confirm NEW tests fail
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_main.py`
  - **Done when**: New tests exist and fail
  - **Verify**: `pytest tests/unit/test_main.py::test_event_loop_processes_queue -v` (expect failure)
  - **Commit**: `test(main): add failing event loop tests (RED)`
  - _Requirements: FR-8 (async processing), AC-10.6 (queue depth logging)_
  - _Design: Event Processing Loop_

- [ ] 6.3.4 [GREEN] Implement event processing loop
  - **Do**:
    - Update `rag_ingestion/main.py`
    - Implement async process_events_loop(event_queue, processor, vector_store, watch_folder):
      - While True: await event_queue.get()
      - Match event_type: "create" -> process_document, "modify" -> delete + process, "delete" -> delete_vectors
      - Log queue depth every 10 events
      - Handle exceptions: Log error, continue loop
      - Mark task done: event_queue.task_done()
    - Integrate into main() with asyncio.create_task
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/main.py`
  - **Done when**: All tests in test_main.py pass
  - **Verify**: `pytest tests/unit/test_main.py -v` (all pass)
  - **Commit**: `feat(main): implement event processing loop (GREEN)`
  - _Requirements: FR-8, AC-10.6_
  - _Design: Event Loop_

- [ ] 6.3.5 [REFACTOR] Add type hints and improve main structure
  - **Do**:
    - Add comprehensive type hints to all functions
    - Add docstrings (Google-style) to main() and helper functions
    - Extract component initialization to setup_components()
    - Extract validation steps to run_startup_validations()
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/main.py`
  - **Done when**: ty passes with strict mode, all tests pass
  - **Verify**: `ty check rag_ingestion/main.py && pytest tests/unit/test_main.py -v`
  - **Commit**: `refactor(main): add type hints and extract setup functions`
  - _Requirements: Code quality_

- [ ] V13 [VERIFY] Quality checkpoint after main orchestration
  - **Do**: Run quality commands
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/ -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(main): pass quality checkpoint` (if fixes needed)

## Phase 7: Integration Testing (TDD)

**Focus**: End-to-end integration tests with real TEI and Qdrant services.

### 7.1 Integration Test Setup

- [ ] 7.1.1 Create integration test fixtures
  - **Do**:
    - Create `tests/integration/conftest.py`
    - Define pytest fixtures: test_config (override with test endpoints), test_collection (unique name per test), cleanup_fixture (delete test collection after test)
    - Add pytest markers: @pytest.mark.integration
    - Configure asyncio for integration tests
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/conftest.py`
  - **Done when**: Fixtures defined and importable
  - **Verify**: `pytest tests/integration/ --collect-only` (lists tests)
  - **Commit**: `test(integration): add integration test fixtures`
  - _Requirements: Testing strategy_
  - _Design: Integration Test Setup_

- [ ] 7.1.2 [RED] Write failing TEI integration test
  - **Do**:
    - Create `tests/integration/test_tei_integration.py`
    - Write test_tei_generates_real_embeddings: Use real TEI service (http://crawl4r-embeddings:80), request embedding for "test text", verify 1024 dimensions
    - Write test_tei_batch_embeddings: Request embeddings for 10 texts, verify all 1024 dims
    - Write test_tei_embeddings_are_normalized: Request embedding, calculate L2 norm, verify ≈ 1.0
    - Run tests and confirm fail if TEI not configured properly
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_tei_integration.py`
  - **Done when**: Tests exist and fail (or skip if TEI not available)
  - **Verify**: `pytest tests/integration/test_tei_integration.py -v -m integration` (expect failures or skips)
  - **Commit**: `test(integration): add failing TEI integration test (RED)`
  - _Requirements: AC-6.1-6.3 (TEI integration)_
  - _Design: TEI Integration_

- [ ] 7.1.3 [GREEN] Configure integration tests to pass with real TEI
  - **Do**:
    - Update test config to use actual TEI endpoint from environment or default to http://crawl4r-embeddings:80
    - Add pytest.mark.skipif for when TEI_ENDPOINT not available
    - Run tests and verify PASS with real TEI service
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_tei_integration.py`
  - **Done when**: Tests pass when run against real TEI service
  - **Verify**: `pytest tests/integration/test_tei_integration.py -v -m integration` (all pass or skip)
  - **Commit**: `test(integration): configure TEI integration tests to pass (GREEN)`
  - _Requirements: AC-6.1-6.3_

- [ ] 7.1.4 [RED] Write failing Qdrant integration test
  - **Do**:
    - Create `tests/integration/test_qdrant_integration.py`
    - Write test_qdrant_collection_lifecycle: Create test collection, verify exists, delete, verify removed
    - Write test_qdrant_upsert_and_retrieve: Upsert 5 vectors with metadata, query by vector, verify retrieved
    - Write test_qdrant_delete_by_file_path: Upsert vectors for 2 files, delete one, verify only remaining vectors exist
    - Write test_qdrant_payload_filtering: Upsert vectors with different metadata, filter by filename, verify correct results
    - Run tests and confirm fail if Qdrant not configured
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_qdrant_integration.py`
  - **Done when**: Tests exist and fail (or skip if Qdrant not available)
  - **Verify**: `pytest tests/integration/test_qdrant_integration.py -v -m integration` (expect failures or skips)
  - **Commit**: `test(integration): add failing Qdrant integration test (RED)`
  - _Requirements: FR-6, FR-7 (Qdrant storage and lifecycle)_
  - _Design: Qdrant Integration_

- [ ] 7.1.5 [GREEN] Configure integration tests to pass with real Qdrant
  - **Do**:
    - Update test config to use actual Qdrant URL from environment or default to http://crawl4r-vectors:6333
    - Add pytest.mark.skipif for when QDRANT_URL not available
    - Use test collection names (rag_test_{uuid}) to isolate tests
    - Ensure cleanup fixture deletes test collections
    - Run tests and verify PASS with real Qdrant service
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_qdrant_integration.py`
  - **Done when**: Tests pass when run against real Qdrant service
  - **Verify**: `pytest tests/integration/test_qdrant_integration.py -v -m integration` (all pass or skip)
  - **Commit**: `test(integration): configure Qdrant integration tests to pass (GREEN)`
  - _Requirements: FR-6, FR-7_

- [ ] V14 [VERIFY] Quality checkpoint after integration test setup
  - **Do**: Run integration tests
  - **Verify**: `pytest tests/integration/ -v -m integration` (all pass or skip)
  - **Done when**: Integration tests execute successfully
  - **Commit**: `chore(integration): pass integration test checkpoint`

### 7.2 End-to-End Pipeline Integration Test

- [ ] 7.2.1 [RED] Write failing end-to-end pipeline test
  - **Do**:
    - Create `tests/integration/test_e2e_pipeline.py`
    - Write test_e2e_document_ingestion:
      - Create temp directory with 3 markdown files
      - Initialize all components with test config
      - Process files using processor.process_batch
      - Query Qdrant for stored vectors
      - Verify: All files processed, correct chunk count, all metadata present, embeddings are 1024 dims
    - Write test_e2e_file_modification:
      - Process file once
      - Modify file content
      - Re-process with vector deletion
      - Query Qdrant
      - Verify: Old vectors removed, new vectors stored, mod_date updated
    - Write test_e2e_file_deletion:
      - Process file
      - Delete vectors
      - Query Qdrant
      - Verify: No vectors remain for file
    - Run tests and confirm fail if pipeline not complete
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_e2e_pipeline.py`
  - **Done when**: Tests exist and fail
  - **Verify**: `pytest tests/integration/test_e2e_pipeline.py -v -m integration` (expect failures)
  - **Commit**: `test(integration): add failing e2e pipeline test (RED)`
  - _Requirements: All FRs, full pipeline validation_
  - _Design: End-to-End Pipeline_

- [ ] 7.2.2 [GREEN] Fix pipeline to pass e2e test
  - **Do**:
    - Run e2e tests and identify failures
    - Fix any integration issues discovered
    - Verify all components work together correctly
    - Run tests and verify ALL PASS
  - **Files**: Multiple (as needed based on test failures)
  - **Done when**: All e2e tests pass
  - **Verify**: `pytest tests/integration/test_e2e_pipeline.py -v -m integration` (all pass)
  - **Commit**: `fix(pipeline): address e2e integration issues (GREEN)`
  - _Requirements: Full pipeline validation_

- [ ] 7.2.3 [REFACTOR] Add e2e test documentation
  - **Do**:
    - Add comprehensive docstrings to all e2e tests explaining what they validate
    - Add inline comments for complex test steps
    - Create test data fixtures for reusable markdown samples
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/integration/test_e2e_pipeline.py`
  - **Done when**: ty passes, all tests pass
  - **Verify**: `ty check tests/ && pytest tests/integration/test_e2e_pipeline.py -v -m integration`
  - **Commit**: `refactor(integration): add e2e test documentation`
  - _Requirements: Code quality_

- [ ] V15 [VERIFY] Quality checkpoint after e2e tests
  - **Do**: Run full test suite
  - **Verify**: `pytest tests/ -v` (all pass)
  - **Done when**: All unit and integration tests pass
  - **Commit**: `chore(testing): pass full test suite checkpoint`

## Phase 8: Quality Gates and Final Verification (TDD)

**Focus**: Final quality checks, documentation, and acceptance criteria verification.

### 8.1 Failed Document Logging (TDD)

- [ ] 8.1.1 [RED] Write failing tests for failed document logging
  - **Do**:
    - Create `tests/unit/test_failed_docs.py`
    - Write test_log_failed_document: Mock file write, verify JSONL entry with schema {file_path, timestamp, error_type, error_message, traceback, retry_count}
    - Write test_failed_docs_log_path_from_config: Verify log path from config.failed_docs_log
    - Write test_failed_docs_append_mode: Write 2 failures, verify both entries in file
    - Write test_failed_docs_skip_after_max_retries: Simulate 3 retry failures, verify logged once with retry_count=3
    - Run tests and confirm ALL FAIL
  - **Files**: `/home/jmagar/workspace/crawl4r/tests/unit/test_failed_docs.py`
  - **Done when**: Tests exist and ALL fail
  - **Verify**: `pytest tests/unit/test_failed_docs.py -v` (expect failures)
  - **Commit**: `test(failed-docs): add failing failed document logging tests (RED)`
  - _Requirements: FR-20 (failed document logging), AC-11.3-11.4 (error logging)_
  - _Design: Failed Document Logging_

- [ ] 8.1.2 [GREEN] Implement failed document logging
  - **Do**:
    - Create `rag_ingestion/failed_docs.py`
    - Implement FailedDocLogger class with log_failure(file_path, error, retry_count) method
    - Format JSONL entry with all required fields
    - Append to config.failed_docs_log file
    - Integrate into processor: Catch exceptions, call log_failure after max retries
    - Run tests and verify ALL PASS
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/failed_docs.py`
  - **Done when**: All tests in test_failed_docs.py pass
  - **Verify**: `pytest tests/unit/test_failed_docs.py -v` (all pass)
  - **Commit**: `feat(failed-docs): implement failed document logging (GREEN)`
  - _Requirements: FR-20, AC-11.3-11.4_
  - _Design: Failed Document Logger_

- [ ] 8.1.3 [REFACTOR] Add type hints and improve failed docs logger
  - **Do**:
    - Add comprehensive type hints to all methods
    - Add docstrings (Google-style)
    - Create FailedDocEntry TypedDict for JSONL schema
    - Run tests and verify still pass
  - **Files**: `/home/jmagar/workspace/crawl4r/rag_ingestion/failed_docs.py`
  - **Done when**: ty passes, all tests pass
  - **Verify**: `ty check rag_ingestion/failed_docs.py && pytest tests/unit/test_failed_docs.py -v`
  - **Commit**: `refactor(failed-docs): add type hints and TypedDict`
  - _Requirements: Code quality_

- [ ] V16 [VERIFY] Quality checkpoint after failed docs logging
  - **Do**: Run quality commands
  - **Verify**: `ruff check . && ty check rag_ingestion/ && pytest tests/unit/ -v`
  - **Done when**: All commands exit 0
  - **Commit**: `chore(failed-docs): pass quality checkpoint` (if fixes needed)

### 8.2 Test Coverage and Final Testing

- [ ] 8.2.1 Run full test suite with coverage
  - **Do**:
    - Run pytest with coverage: `pytest --cov=rag_ingestion --cov-report=term --cov-report=html tests/`
    - Verify coverage >= 85%
    - Identify uncovered critical paths
  - **Files**: N/A (measurement task)
  - **Done when**: Coverage report generated, >= 85% achieved
  - **Verify**: `pytest --cov=rag_ingestion --cov-report=term tests/` (check output)
  - **Commit**: None (measurement only)
  - _Requirements: Testing strategy (85%+ coverage target)_

- [ ] 8.2.2 Add tests for uncovered critical paths
  - **Do**:
    - Review coverage report HTML
    - Identify critical uncovered lines (error paths, edge cases)
    - Write tests to cover gaps
    - Re-run coverage and verify improvement
  - **Files**: Various test files (as needed)
  - **Done when**: Coverage >= 85% including critical paths
  - **Verify**: `pytest --cov=rag_ingestion --cov-report=term tests/` (>= 85%)
  - **Commit**: `test(coverage): add tests for uncovered critical paths`
  - _Requirements: Testing strategy_

- [ ] V17 [VERIFY] Final test suite verification
  - **Do**: Run complete test suite with all markers
  - **Verify**: `pytest tests/ -v --cov=rag_ingestion --cov-report=term`
  - **Done when**: All tests pass, coverage >= 85%
  - **Commit**: None

### 8.3 Local Quality Check

- [ ] V18 [VERIFY] Full local quality suite
  - **Do**: Run all quality commands in sequence
  - **Verify**:
    - `ruff check .` (linting)
    - `ruff format --check .` (formatting)
    - `ty check rag_ingestion/ tests/` (type checking)
    - `pytest tests/ -v` (all tests)
    - `pytest --cov=rag_ingestion --cov-report=term tests/` (coverage >= 85%)
  - **Done when**: All commands exit 0
  - **Commit**: `chore(quality): pass full local quality suite` (if fixes needed)

### 8.4 Integration Test Verification

- [ ] V19 [VERIFY] Full integration test suite with real services
  - **Do**:
    - Verify TEI service running: `curl http://crawl4r-embeddings:80/health || docker ps | grep crawl4r-embeddings`
    - Verify Qdrant service running: `curl http://crawl4r-vectors:6333/health || docker ps | grep crawl4r-vectors`
    - Run integration tests: `pytest tests/integration/ -v -m integration`
  - **Verify**: All integration tests pass or skip gracefully
  - **Done when**: Integration test suite passes against real services
  - **Commit**: None

### 8.5 End-to-End Manual Testing

- [ ] V20 [VERIFY] Manual e2e test with real markdown files
  - **Do**:
    - Create test watch folder with 10 markdown files (various sizes, with/without frontmatter, with/without headings)
    - Set WATCH_FOLDER in .env to test folder
    - Run `python -m rag_ingestion.main`
    - Verify: Startup validation passes, batch processing completes, all files ingested, watch mode starts
    - Create new file, verify detected and processed within 5 seconds
    - Modify file, verify re-ingested with old vectors removed
    - Delete file, verify vectors removed
    - Query Qdrant to verify all metadata present and correct
  - **Verify**: Manual observation of logs and Qdrant queries
  - **Done when**: All manual test scenarios pass
  - **Commit**: None (manual verification)

### 8.6 Acceptance Criteria Verification

The following tasks verify each acceptance criterion from requirements.md.

- [ ] V21 [VERIFY] AC-1: Startup Batch Processing
  - **Do**: Review requirements.md AC-1.1-1.6
  - **Verify**:
    - AC-1.1: System detects all .md files recursively ✓ (test_e2e_pipeline.py, watcher tests)
    - AC-1.2: Files processed in batches ✓ (test_processor.py::test_process_batch)
    - AC-1.3: Progress logging ✓ (processor.py logs every 10 files)
    - AC-1.4: Failed files logged ✓ (test_failed_docs.py)
    - AC-1.5: Watch mode after batch ✓ (main.py orchestration)
    - AC-1.6: Processing status visible ✓ (logger output)
  - **Done when**: All AC-1.x criteria manually verified
  - **Commit**: None

- [ ] V22 [VERIFY] AC-2: Real-Time File Monitoring
  - **Do**: Review requirements.md AC-2.1-2.6
  - **Verify**:
    - AC-2.1: Detects new .md files within 1 second ✓ (watchdog + debouncing)
    - AC-2.2: Recursive monitoring ✓ (watcher.py recursive=True)
    - AC-2.3: Debouncing with 1-second threshold ✓ (test_watcher.py::test_debounce_rapid_events)
    - AC-2.4: Async queuing ✓ (event_queue integration)
    - AC-2.5: Processing status logged ✓ (logger output)
    - AC-2.6: Exclusions and ignore patterns ✓ (test_watcher.py::test_ignore_*)
  - **Done when**: All AC-2.x criteria verified
  - **Commit**: None

- [ ] V23 [VERIFY] AC-3: File Modification Handling
  - **Do**: Review requirements.md AC-3.1-3.6
  - **Verify**:
    - AC-3.1: Detects modifications within 1 second ✓ (watchdog)
    - AC-3.2: Debounced ✓ (debouncing logic)
    - AC-3.3: Old vectors deleted ✓ (test_watcher.py::test_handle_modify_event_deletes_old_vectors)
    - AC-3.4: Deletion by file_path_relative ✓ (vector_store.py::delete_vectors_by_file)
    - AC-3.5: Fresh embeddings generated ✓ (processor re-processes)
    - AC-3.6: Error handling preserves old vectors ✓ (error handling in lifecycle handlers)
  - **Done when**: All AC-3.x criteria verified
  - **Commit**: None

- [ ] V24 [VERIFY] AC-4: File Deletion Cleanup
  - **Do**: Review requirements.md AC-4.1-4.6
  - **Verify**:
    - AC-4.1: Detects deletions within 1 second ✓ (watchdog)
    - AC-4.2: Vectors removed ✓ (test_watcher.py::test_handle_delete_event_removes_vectors)
    - AC-4.3: Filter by file_path_relative ✓ (vector_store.py)
    - AC-4.4: Deletion logged ✓ (logger in lifecycle handler)
    - AC-4.5: Failures logged, don't crash ✓ (error handling)
    - AC-4.6: Confirmation logged ✓ (test_watcher.py::test_handle_delete_event_logs_count)
  - **Done when**: All AC-4.x criteria verified
  - **Commit**: None

- [ ] V25 [VERIFY] AC-5: Markdown-Aware Chunking
  - **Do**: Review requirements.md AC-5.1-5.6
  - **Verify**:
    - AC-5.1: Heading hierarchy respected ✓ (test_chunker.py::test_chunk_by_headings)
    - AC-5.2: 512 tokens with 15% overlap ✓ (chunker.py constants, test_chunker.py::test_chunk_size_target)
    - AC-5.3: Section path in metadata ✓ (test_chunker.py::test_chunk_preserves_heading_hierarchy)
    - AC-5.4: Formatting preserved ✓ (test_chunker.py::test_chunk_preserves_formatting)
    - AC-5.5: Files without headings handled ✓ (test_chunker.py::test_chunk_without_headings)
    - AC-5.6: Chunk metadata includes index, position, heading level ✓ (chunker.py, Chunk TypedDict)
  - **Done when**: All AC-5.x criteria verified
  - **Commit**: None

- [ ] V26 [VERIFY] AC-6: TEI Embedding Generation
  - **Do**: Review requirements.md AC-6.1-6.6
  - **Verify**:
    - AC-6.1: Connects to TEI endpoint ✓ (embeddings.py, test_tei_integration.py)
    - AC-6.2: Requests 1024 dimensions ✓ (embeddings.py dimensions parameter)
    - AC-6.3: Validates 1024 dimensions ✓ (quality.py::check_embedding_dimensions)
    - AC-6.4: Batch embedding requests ✓ (embeddings.py::_embed_batch, max 32 texts)
    - AC-6.5: Retry with exponential backoff ✓ (test_embeddings.py::test_tei_retries_on_network_error)
    - AC-6.6: Circuit breaker pattern ✓ (test_embeddings.py::test_tei_circuit_breaker_*)
  - **Done when**: All AC-6.x criteria verified
  - **Commit**: None

- [ ] V27 [VERIFY] AC-7: Qdrant Vector Storage
  - **Do**: Review requirements.md AC-7.1-7.6
  - **Verify**:
    - AC-7.1: Cosine similarity, "crawl4r" collection ✓ (vector_store.py::ensure_collection)
    - AC-7.2: Full payload metadata ✓ (Metadata TypedDict, test_vector_store.py::test_upsert_includes_full_metadata)
    - AC-7.3: Collection auto-created ✓ (vector_store.py::ensure_collection)
    - AC-7.4: Deterministic UUIDs ✓ (test_vector_store.py::test_upsert_generates_deterministic_ids)
    - AC-7.5: Retry with backoff ✓ (test_vector_store.py::test_upsert_retries_on_failure)
    - AC-7.6: Bulk upsert 100 points ✓ (vector_store.py::upsert_vectors_batch)
  - **Done when**: All AC-7.x criteria verified
  - **Commit**: None

- [ ] V28 [VERIFY] AC-8: Metadata Filtering and Search
  - **Do**: Review requirements.md AC-8.1-8.6
  - **Verify**:
    - AC-8.1: All required metadata fields ✓ (Metadata TypedDict)
    - AC-8.2: Optional tags field ✓ (chunker.py::parse_frontmatter)
    - AC-8.3: chunk_text stored ✓ (processor.py, metadata payload)
    - AC-8.4: Payload filtering support ✓ (test_qdrant_integration.py::test_qdrant_payload_filtering)
    - AC-8.5: Date range filtering ✓ (modification_date field, datetime index)
    - AC-8.6: Payload indexes created ✓ (vector_store.py::ensure_payload_indexes)
  - **Done when**: All AC-8.x criteria verified
  - **Commit**: None

- [ ] V29 [VERIFY] AC-9: Quality Verification
  - **Do**: Review requirements.md AC-9.1-9.6
  - **Verify**:
    - AC-9.1: TEI startup validation with retries ✓ (quality.py::validate_tei_connection)
    - AC-9.2: Qdrant startup validation with retries ✓ (quality.py::validate_qdrant_connection)
    - AC-9.3: 5% sampling for normalization ✓ (quality.py::sample_embeddings)
    - AC-9.4: Dimension validation on every response ✓ (quality.py::check_embedding_dimensions)
    - AC-9.5: Quality check failures logged ✓ (logger in quality.py)
    - AC-9.6: Post-ingestion verification ✓ (can be implemented as optional feature)
  - **Done when**: All AC-9.x criteria verified (AC-9.6 optional)
  - **Commit**: None

- [ ] V30 [VERIFY] AC-10: Async Non-Blocking Processing
  - **Do**: Review requirements.md AC-10.1-10.6
  - **Verify**:
    - AC-10.1: Watcher in separate thread ✓ (watchdog Observer)
    - AC-10.2: Asyncio for processing ✓ (processor async methods)
    - AC-10.3: Queue max size 1000 ✓ (config.py, asyncio.Queue)
    - AC-10.4: Backpressure handling ✓ (watcher.py overflow detection)
    - AC-10.5: Max concurrency configurable ✓ (config.py::max_concurrent_docs)
    - AC-10.6: Queue depth logged ✓ (main.py event loop)
  - **Done when**: All AC-10.x criteria verified
  - **Commit**: None

- [ ] V31 [VERIFY] AC-11: Error Handling and Recovery
  - **Do**: Review requirements.md AC-11.1-11.6
  - **Verify**:
    - AC-11.1: Human-readable logging ✓ (logger.py format)
    - AC-11.2: Retry with exponential backoff ✓ (embeddings.py, vector_store.py)
    - AC-11.3: Failed documents logged to JSONL ✓ (failed_docs.py)
    - AC-11.4: Failed documents recorded ✓ (test_failed_docs.py)
    - AC-11.5: Circuit breaker ✓ (embeddings.py::CircuitBreaker)
    - AC-11.6: Restart resumption ✓ (recovery.py::get_files_to_process)
  - **Done when**: All AC-11.x criteria verified
  - **Commit**: None

- [ ] V32 [VERIFY] AC-12: Configuration Management
  - **Do**: Review requirements.md AC-12.1-12.6
  - **Verify**:
    - AC-12.1: All settings via environment variables ✓ (config.py Settings)
    - AC-12.2: Chunking parameters configurable ✓ (config.py)
    - AC-12.3: Performance parameters configurable ✓ (config.py)
    - AC-12.4: Configuration validation on startup ✓ (Pydantic validation)
    - AC-12.5: .env file support ✓ (python-dotenv)
    - AC-12.6: Configuration logged on startup ✓ (main.py logs config)
  - **Done when**: All AC-12.x criteria verified
  - **Commit**: None

- [ ] V33 [VERIFY] Final acceptance criteria checklist
  - **Do**: Review all 12 user stories (US-1 through US-12) from requirements.md
  - **Verify**: Each user story satisfied by corresponding acceptance criteria verified above
  - **Done when**: All user stories can be demonstrated working
  - **Commit**: None

### 8.7 Documentation

- [ ] 8.7.1 Create project README
  - **Do**:
    - Create `README.md` at project root
    - Include: Project overview, features, prerequisites, installation (uv install), configuration (.env setup), usage (running the pipeline), testing, architecture diagram, troubleshooting, license
    - Document all environment variables with examples
  - **Files**: `/home/jmagar/workspace/crawl4r/README.md`
  - **Done when**: README is comprehensive and accurate
  - **Verify**: `cat README.md | wc -l` (should be 100+ lines)
  - **Commit**: `docs(readme): add comprehensive project README`
  - _Requirements: Documentation_

- [ ] 8.7.2 Update .progress.md with learnings
  - **Do**:
    - Append to `.progress.md` in specs/rag-ingestion/
    - Document: Task planning insights, TDD workflow observations, integration challenges discovered, performance learnings, testing strategy effectiveness
  - **Files**: `/home/jmagar/workspace/crawl4r/specs/rag-ingestion/.progress.md`
  - **Done when**: Learnings section updated with 5+ new insights
  - **Verify**: `tail -20 specs/rag-ingestion/.progress.md`
  - **Commit**: `docs(progress): append task planning and implementation learnings`
  - _Requirements: Progress tracking_

- [ ] V34 [VERIFY] Documentation completeness
  - **Do**: Review all documentation files
  - **Verify**:
    - README.md complete and accurate
    - .env.example has all parameters documented
    - Code has comprehensive docstrings (Google-style)
    - .progress.md updated with learnings
  - **Done when**: Documentation review complete
  - **Commit**: None

## Phase 9: Final Deliverable

### 9.1 Final Quality Gate

- [ ] V35 [VERIFY] Final local CI simulation
  - **Do**: Run complete CI pipeline locally
  - **Verify**: Execute in sequence:
    1. `ruff check .` (lint)
    2. `ruff format --check .` (format check)
    3. `ty check rag_ingestion/ tests/` (type check)
    4. `pytest tests/unit/ -v` (unit tests)
    5. `pytest tests/integration/ -v -m integration` (integration tests)
    6. `pytest --cov=rag_ingestion --cov-report=term tests/` (coverage >= 85%)
  - **Done when**: All commands exit 0 and coverage >= 85%
  - **Commit**: `chore(ci): pass final local CI simulation` (if fixes needed)

### 9.2 Performance Validation

- [ ] V36 [VERIFY] Throughput benchmark
  - **Do**:
    - Create test dataset with 100 markdown files (varying sizes 500-3000 tokens)
    - Run batch processing with timing
    - Calculate documents/minute throughput
    - Verify meets target: >= 50 docs/min on RTX 3050
  - **Verify**: Manual measurement and log analysis
  - **Done when**: Throughput meets or exceeds NFR-1 target
  - **Commit**: None (measurement task)

- [ ] V37 [VERIFY] Latency benchmark
  - **Do**:
    - Process single 2000-token markdown file
    - Measure time from file creation to vector storage complete
    - Verify meets target: < 5 seconds
  - **Verify**: Manual timing
  - **Done when**: Latency meets NFR-2 target
  - **Commit**: None

- [ ] V38 [VERIFY] Memory usage validation
  - **Do**:
    - Run pipeline with 1000 files queued
    - Monitor memory usage during processing
    - Verify meets target: < 4 GB
  - **Verify**: `docker stats` or `htop` during run
  - **Done when**: Memory usage meets NFR-5 target
  - **Commit**: None

### 9.3 Final State

- [ ] V39 [VERIFY] Clean repository state
  - **Do**: Verify repository is clean
  - **Verify**:
    - `git status` shows no uncommitted changes
    - No .cache/ files tracked
    - No .env file tracked
    - No failed_documents.jsonl tracked (if exists)
    - All test collections cleaned up in Qdrant
  - **Done when**: Repository is clean
  - **Commit**: None

- [ ] V40 [VERIFY] Feature branch ready for PR
  - **Do**: Verify branch is ready for merge
  - **Verify**:
    - Current branch: `git branch --show-current` (should be feature branch, not main)
    - All commits follow conventional commit format
    - All tasks marked complete in tasks.md
    - All tests passing
    - Documentation complete
  - **Done when**: Branch ready for PR
  - **Commit**: None

### 9.4 Create Pull Request

- [ ] V41 [VERIFY] Create PR and verify CI passes
  - **Do**:
    1. Verify current branch is feature branch (not main)
    2. Push branch: `git push -u origin $(git branch --show-current)`
    3. Create PR: `gh pr create --title "feat(rag-ingestion): implement RAG ingestion pipeline" --body "Implements complete RAG ingestion pipeline with TEI, Qdrant, and file watching. All 78 tasks completed, 85%+ test coverage, strict TDD methodology."`
    4. Monitor CI: `gh pr checks --watch`
  - **Verify**: All CI checks pass (✓ green)
  - **Done when**: PR created and CI passing
  - **If CI fails**:
    1. Read failure: `gh pr checks`
    2. Fix locally
    3. Push fixes: `git push`
    4. Re-verify: `gh pr checks --watch`

## Notes

### POC Shortcuts (None - Strict TDD)

This implementation follows **strict TDD methodology** from the start. No shortcuts taken.

### Task Summary

**Total Tasks**: 78

**Phase Breakdown**:
- Phase 1 (Core Infrastructure): 11 tasks
- Phase 2 (TEI Integration): 9 tasks
- Phase 3 (Qdrant Integration): 11 tasks
- Phase 4 (Document Processing): 9 tasks
- Phase 5 (File Watching): 10 tasks
- Phase 6 (State Recovery): 9 tasks
- Phase 7 (Integration Testing): 9 tasks
- Phase 8 (Quality Gates): 13 tasks
- Phase 9 (Final Deliverable): 7 tasks

**Verification Tasks**: 41 (V1-V41)
- Quality checkpoints every 2-3 tasks: 16 checkpoints (V1-V16)
- Final verification sequence: 25 tasks (V17-V41)

### TDD Compliance

Every feature follows RED-GREEN-REFACTOR:
1. **RED**: Write failing test first
2. **GREEN**: Implement minimal code to pass
3. **REFACTOR**: Improve code while keeping tests green

No implementation without tests. No skipping RED phase.

### Quality Commands

As discovered from research.md:
- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Type Check**: `ty check rag_ingestion/ tests/`
- **Unit Tests**: `pytest tests/unit/ -v`
- **Integration Tests**: `pytest tests/integration/ -v -m integration`
- **Coverage**: `pytest --cov=rag_ingestion --cov-report=term tests/`

All tools configured in pyproject.toml with `.cache/` directory for artifacts.

### Dependencies

Using **uv** for package management (NOT pip, poetry, or pipenv):
- `uv install` to install dependencies from pyproject.toml
- Python 3.10+ required (Qdrant client constraint)

### Infrastructure

Services already running:
- TEI: http://crawl4r-embeddings:80 (port 52000 external)
- Qdrant: http://crawl4r-vectors:6333 (ports 52001 HTTP, 52002 gRPC)
- Model cached: Qwen3-Embedding-0.6B (1.19GB)
