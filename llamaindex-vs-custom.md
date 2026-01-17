# LlamaIndex vs Custom Pipeline Breakdown

## Verification Scope and Sources

This verification uses the LlamaIndex MCP documentation sources below, plus the LlamaIndex public API reference (web):

- Custom Embeddings example: `/python/examples/embeddings/custom_embeddings`
- Ingestion Pipeline guide: `/python/framework/module_guides/loading/ingestion_pipeline/`
- Node Parser guide: `/python/framework/module_guides/loading/node_parsers/`
- Documents guide: `/python/framework/module_guides/loading/documents_and_nodes/usage_documents`
- Settings guide: `/python/framework/module_guides/supporting_modules/settings`
- Custom modules overview: `/python/framework/optimizing/custom_modules`
- Readers API reference (web): `https://docs.llamaindex.ai/en/stable/api_reference/readers/`

Items without MCP coverage are explicitly marked as **Not Verified (No MCP Doc Match)**.

## LlamaIndex Components In Use (Runtime Code)

- BasePydanticReader — crawl4r/readers/crawl4ai.py
- Document — crawl4r/readers/crawl4ai.py, crawl4r/processing/processor.py
- BaseEmbedding — crawl4r/storage/llama_embeddings.py, crawl4r/processing/processor.py
- IngestionPipeline — crawl4r/processing/processor.py
- DocstoreStrategy — crawl4r/processing/processor.py
- SimpleDocumentStore — crawl4r/processing/processor.py
- QdrantVectorStore — crawl4r/processing/processor.py
- NodeParser — crawl4r/processing/llama_parser.py
- BaseNode — crawl4r/processing/llama_parser.py
- TextNode — crawl4r/processing/llama_parser.py
- Settings — crawl4r/core/llama_settings.py, crawl4r/processing/processor.py
- PrivateAttr — crawl4r/storage/llama_embeddings.py, crawl4r/processing/llama_parser.py
- Instrumentation: get_dispatcher, BaseEvent — crawl4r/core/instrumentation.py

## Alignment Verification (MCP-Backed)

### BaseEmbedding (Custom Embeddings Pattern) — Aligned

**Doc Source:** `/python/examples/embeddings/custom_embeddings`

**Pattern from docs:**
- Subclass `BaseEmbedding`
- Use `PrivateAttr` for non-Pydantic fields
- Implement `_get_query_embedding`, `_get_text_embedding`, and their async counterparts

**Our implementation:**
- `crawl4r/storage/llama_embeddings.py` defines `TEIEmbedding(BaseEmbedding)`
- Uses `PrivateAttr` for the TEI client
- Implements `_get_query_embedding`, `_get_text_embedding`, `_get_text_embeddings`, and async equivalents

**Result:** Aligned with documented custom embedding pattern.

### IngestionPipeline + QdrantVectorStore — Aligned

**Doc Source:** `/python/framework/module_guides/loading/ingestion_pipeline/`

**Pattern from docs:**
- `IngestionPipeline(transformations=[...], vector_store=QdrantVectorStore(...))`
- Embedding stage must be part of transformations when writing to a vector store
- Supports async via `await pipeline.arun(...)`
- Optional `docstore` for document management

**Our implementation:**
- `crawl4r/processing/processor.py` builds `IngestionPipeline`
- `transformations=[CustomMarkdownNodeParser, TEIEmbedding]`
- `vector_store=QdrantVectorStore(...)`
- Uses `await pipeline.arun(...)`
- Attaches `SimpleDocumentStore` with `DocstoreStrategy.UPSERTS`

**Result:** Aligned with ingestion pipeline usage pattern.

### NodeParser / TextNode — Aligned

**Doc Source:** `/python/framework/module_guides/loading/node_parsers/`

**Pattern from docs:**
- Node parsers chunk Documents into Nodes
- Node parsers can be used as transformations in `IngestionPipeline`
- Nodes inherit document metadata

**Our implementation:**
- `crawl4r/processing/llama_parser.py` implements `CustomMarkdownNodeParser(NodeParser)`
- Produces `TextNode` objects with merged metadata from original node
- Used as first transformation in `IngestionPipeline`

**Result:** Aligned with NodeParser usage pattern.

### Document Construction + Metadata — Aligned

**Doc Source:** `/python/framework/module_guides/loading/documents_and_nodes/usage_documents`

**Pattern from docs:**
- Manual construction via `Document(text=..., metadata=...)`
- Metadata is a dict of string keys and flat values
- `id_` can be set explicitly (via `doc_id` / `node_id` / `id_`)

**Our implementation:**
- `crawl4r/processing/processor.py` constructs `Document(text=..., metadata=..., id_=...)`
- Metadata is flat (paths, filename, ISO timestamp strings)
- Deterministic ID is assigned

**Result:** Aligned with document construction and metadata pattern.

### Settings (Global Defaults) — Aligned

**Doc Source:** `/python/framework/module_guides/supporting_modules/settings`

**Pattern from docs:**
- `Settings.embed_model`, `Settings.chunk_size`, `Settings.chunk_overlap`
- `Settings.tokenizer` for token counting
- Global settings used as defaults when local overrides absent

**Our implementation:**
- `crawl4r/core/llama_settings.py` sets `Settings.embed_model`, `chunk_size`, `chunk_overlap`, `tokenizer`
- `crawl4r/processing/processor.py` uses global settings fallback when local embed model not provided

**Result:** Aligned with Settings usage pattern.

### Custom Modules (General Allowance) — Aligned

**Doc Source:** `/python/framework/optimizing/custom_modules`

**Pattern from docs:**
- Core modules are intended to be subclassed and customized
- Custom embeddings and transformations are explicitly supported

**Our implementation:**
- Custom `BaseEmbedding` subclass
- Custom `NodeParser` subclass
- Custom reader (BasePydanticReader) extending data loader behavior

**Result:** Aligned with custom module philosophy.

### DocstoreStrategy (UPSERTS) — Aligned

**Doc Source:** `/python/examples/ingestion/redis_ingestion_pipeline`

**Pattern from docs:**
- `docstore_strategy=DocstoreStrategy.UPSERTS` in `IngestionPipeline`

**Our implementation:**
- `crawl4r/processing/processor.py` uses `docstore_strategy=DocstoreStrategy.UPSERTS`

**Result:** Aligned with documented docstore strategy usage.

### SimpleDocumentStore — Aligned

**Doc Source:** `/python/examples/ingestion/document_management_pipeline`

**Pattern from docs:**
- Attach `docstore=SimpleDocumentStore()` to `IngestionPipeline`

**Our implementation:**
- `crawl4r/processing/processor.py` uses `SimpleDocumentStore()` (default when not provided)

**Result:** Aligned with documented docstore usage pattern.

### QdrantVectorStore — Aligned

**Doc Source:** `/python/examples/vector_stores/QdrantIndexDemo`

**Pattern from docs:**
- `QdrantVectorStore(client=..., collection_name=...)`

**Our implementation:**
- `crawl4r/processing/processor.py` creates `QdrantVectorStore(client=vector_store.client, collection_name=...)`

**Result:** Aligned with documented Qdrant vector store usage.

### Instrumentation (Dispatcher + BaseEvent) — Aligned

**Doc Source:** `/python/framework/module_guides/observability/instrumentation`

**Pattern from docs:**
- Define dispatcher via `get_dispatcher(...)`
- Define custom events by subclassing `BaseEvent`
- Emit events via `dispatcher.event(...)`

**Our implementation:**
- `crawl4r/core/instrumentation.py` uses `get_dispatcher("crawl4r")`
- Custom events `DocumentProcessingStartEvent`, `DocumentProcessingEndEvent` subclass `BaseEvent`
- Events emitted in `crawl4r/processing/processor.py`

**Result:** Aligned with documented instrumentation usage.

### PrivateAttr — Aligned

**Doc Source:** `/python/examples/embeddings/custom_embeddings`

**Pattern from docs:**
- Use `PrivateAttr` for non-Pydantic fields in custom embeddings

**Our implementation:**
- `crawl4r/storage/llama_embeddings.py` uses `PrivateAttr` for TEI client
- `crawl4r/processing/llama_parser.py` uses `PrivateAttr` for chunker

**Result:** Aligned with documented PrivateAttr usage pattern.

### BaseReader / BasePydanticReader — Aligned

**Doc Source (web):** `https://docs.llamaindex.ai/en/stable/api_reference/readers/`

**Pattern from docs:**
- `BaseReader` provides `load_data`, `aload_data`, and lazy variants
- `BasePydanticReader` adds Pydantic config and an `is_remote` flag (default False)

**Our implementation:**
- `crawl4r/readers/crawl4ai.py` implements `load_data(...)` and `aload_data(...)`
- `Crawl4AIReader` sets `is_remote = True` to reflect remote API usage
- Inherits `BasePydanticReader` for serialization/Pydantic behavior

**Result:** Aligned with documented reader class contract.

## Not Verified (No MCP Doc Match Found)

The MCP docs did not return explicit pages for the following components. They are in use, but verification against MCP docs could not be completed:

None.

If you want full verification on these, we can pull LlamaIndex API reference pages (if present in MCP) or inspect upstream source.

## Custom Pipeline Parts (Non-LlamaIndex)

- Crawl4AI reader logic (HTTP, retries, circuit breaker, dedup) — crawl4r/readers/crawl4ai.py
- Document processing orchestration (file IO, metadata, IDs, pipeline invocation) — crawl4r/processing/processor.py
- Markdown chunker — crawl4r/processing/chunker.py
- Custom NodeParser implementation — crawl4r/processing/llama_parser.py
- TEI client + embedding wrapper — crawl4r/storage/tei.py, crawl4r/storage/llama_embeddings.py
- Qdrant manager — crawl4r/storage/qdrant.py
- Circuit breaker — crawl4r/resilience/circuit_breaker.py
- App config + LlamaIndex settings bridge — crawl4r/core/config.py, crawl4r/core/llama_settings.py
- Custom instrumentation events — crawl4r/core/instrumentation.py

## LlamaIndex Equivalents for Custom Pipeline Parts (Excluding Crawl4AI)

**Sources:** MCP docs under `docs-python/src/content/docs/...`

- Markdown chunking + custom node parsing → `MarkdownNodeParser`, `SentenceSplitter`, `TokenTextSplitter`, `HierarchicalNodeParser`  
  Doc: `/python/framework/module_guides/loading/node_parsers/modules`
- Document processing orchestration → `IngestionPipeline` with `transformations` and `vector_store`  
  Doc: `/python/framework/module_guides/loading/ingestion_pipeline/`
- File loading + filesystem metadata → `SimpleDirectoryReader`  
  Doc: `/python/framework/module_guides/loading/simpledirectoryreader`
- Remote embeddings via TEI → `TextEmbeddingsInference` embedding model  
  Doc: `/python/examples/embeddings/text_embedding_inference`
- Vector store management (Qdrant) → `QdrantVectorStore` + vector store layer  
  Docs: `/python/examples/vector_stores/QdrantIndexDemo`, `/python/framework/module_guides/storing/vector_stores`
- Docstore + dedup strategy → `SimpleDocumentStore` + `DocstoreStrategy` in `IngestionPipeline`  
  Docs: `/python/framework/module_guides/storing/docstores`, `/python/examples/ingestion/document_management_pipeline`, `/python/examples/ingestion/redis_ingestion_pipeline`
- Global config bridge → `Settings` (`embed_model`, `chunk_size`, `chunk_overlap`, `tokenizer`)  
  Doc: `/python/framework/module_guides/supporting_modules/settings`
- Instrumentation events → `instrumentation` module (`get_dispatcher`, custom `BaseEvent`)  
  Doc: `/python/framework/module_guides/observability/instrumentation`

**No direct LlamaIndex equivalent found in MCP docs:**

- Circuit breaker / retry policy for external services — custom resilience layer

## Extension Assessment (Are We Extending LlamaIndex Properly?)

**Summary:** Yes. Where we extend LlamaIndex (embeddings, node parsing, ingestion, instrumentation, settings), we follow the documented extension points. The remaining custom parts are service‑level concerns that LlamaIndex does not cover.

### Proper Extensions (Aligned with LlamaIndex Patterns)

- **Custom embeddings (`TEIEmbedding`)** → Subclassing `BaseEmbedding` with `PrivateAttr` and sync/async methods  
  Docs: `/python/examples/embeddings/custom_embeddings`, `/python/framework/optimizing/custom_modules`
- **Custom node parsing (`CustomMarkdownNodeParser`)** → Subclassing `NodeParser`, used as `IngestionPipeline` transformation  
  Docs: `/python/framework/module_guides/loading/node_parsers/`
- **Ingestion orchestration** → `IngestionPipeline` with transformations + vector store  
  Docs: `/python/framework/module_guides/loading/ingestion_pipeline/`
- **Document construction + metadata** → Manual `Document(text=..., metadata=..., id_=...)`  
  Docs: `/python/framework/module_guides/loading/documents_and_nodes/usage_documents`
- **Global defaults** → `Settings.embed_model`, `chunk_size`, `chunk_overlap`, `tokenizer`  
  Docs: `/python/framework/module_guides/supporting_modules/settings`
- **Instrumentation** → `get_dispatcher(...)` + custom `BaseEvent`  
  Docs: `/python/framework/module_guides/observability/instrumentation`

### Custom but Adjacent (Not LlamaIndex Extension Points)

- **Qdrant manager** — adds delete-by-url, payload index management, retry/backoff; sits alongside `QdrantVectorStore`  
  Docs: `/python/framework/module_guides/storing/vector_stores`
- **Circuit breaker / retry policy** — no built‑in LlamaIndex equivalent (service‑level concern)
