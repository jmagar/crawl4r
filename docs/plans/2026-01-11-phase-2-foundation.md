# Phase 2 Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Phase 2 foundation (DB migrations, core abstractions/models, storage adapters, middleware, logging, health checks, app wiring) required before any user story implementation.

**Architecture:** Use handwritten Alembic migrations to define the PostgreSQL schema + indexes per `specs/001-rag-pipeline/data-model.md`. Implement Pydantic domain models and abstract interfaces under `app/core/`, concrete storage adapters under `app/storage/`, middleware and health endpoints under `app/api/`, and app wiring + logging in `app/main.py` with dependency injection.

**Tech Stack:** FastAPI, Pydantic v2, SQLAlchemy async, Alembic, httpx, redis (async), qdrant-client, structlog, pytest

---

**Docstring standard:** All classes, modules, and functions must use XML-style docstrings (e.g., `<summary>`, `<param>`, `<returns>`).

### Task 1: Alembic async configuration baseline (T009)

**Files:**
- Create: `alembic/env.py`
- Modify: `alembic.ini`
- Test: `tests/integration/test_alembic_env.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_alembic_env_exists():
    env_path = Path("alembic/env.py")
    assert env_path.exists(), "alembic/env.py must exist"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_alembic_env.py::test_alembic_env_exists -v`
Expected: FAIL with "alembic/env.py must exist"

**Step 3: Write minimal implementation**

```python
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.config import get_settings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = None


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section) or {},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
```

```ini
[alembic]
script_location = alembic
sqlalchemy.url = driver://user:pass@localhost/dbname

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_alembic_env.py::test_alembic_env_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add alembic/env.py alembic.ini tests/integration/test_alembic_env.py
git commit -m "test: add alembic env baseline"
```

---

### Task 2: Initial schema migration (tables) (T010)

**Files:**
- Create: `alembic/versions/001_initial_schema.py`
- Test: `tests/integration/test_migration_001.py`

**Step 1: Write the failing test**

```python
import subprocess


def test_migration_001_file_exists():
    result = subprocess.run(["test", "-f", "alembic/versions/001_initial_schema.py"], check=False)
    assert result.returncode == 0, "001_initial_schema.py must exist"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_migration_001.py::test_migration_001_file_exists -v`
Expected: FAIL with "001_initial_schema.py must exist"

**Step 3: Write minimal implementation**

```python
"""Initial schema.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-01-11 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("key_hash", sa.String(length=64), nullable=False, unique=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("scopes", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("rate_limit_rpm", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )

    op.create_table(
        "collections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "tags",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=50), nullable=False, unique=True),
    )

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("url", sa.Text(), nullable=False, unique=True),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column("parent_url", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("language", sa.String(length=8), nullable=False, server_default="en"),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("crawled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "fts_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
                "setweight(to_tsvector('english', content), 'B')",
                persisted=True,
            ),
        ),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"]),
    )

    op.create_table(
        "document_tags",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tag_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.PrimaryKeyConstraint("document_id", "tag_id"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["tag_id"], ["tags.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding_model", sa.String(length=200), nullable=False),
        sa.Column("section_header", sa.Text(), nullable=True),
        sa.Column("content_type", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "fts_vector",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("document_id", "chunk_index"),
    )

    op.create_table(
        "crawl_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("extraction_strategy", sa.String(length=50), nullable=False),
        sa.Column("chunking_strategy", sa.String(length=50), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False, server_default="1200"),
        sa.Column("page_timeout_ms", sa.Integer(), nullable=False, server_default="30000"),
        sa.Column("respect_robots_txt", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "crawl_sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("crawl_config_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("url_pattern", sa.Text(), nullable=False),
        sa.Column("schedule_cron", sa.String(length=100), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"]),
        sa.ForeignKeyConstraint(["crawl_config_id"], ["crawl_configs.id"]),
    )

    op.create_table(
        "crawl_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column("crawl_config_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("crawl_source_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("priority", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("result_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("webhook_url", sa.Text(), nullable=True),
        sa.Column("webhook_headers", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["crawl_config_id"], ["crawl_configs.id"]),
        sa.ForeignKeyConstraint(["crawl_source_id"], ["crawl_sources.id"]),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"]),
    )

    op.create_table(
        "deep_crawl_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("root_url", sa.Text(), nullable=False),
        sa.Column("strategy", sa.String(length=20), nullable=False),
        sa.Column("max_depth", sa.Integer(), nullable=False),
        sa.Column("max_pages", sa.Integer(), nullable=False),
        sa.Column("score_threshold", sa.Float(), nullable=False, server_default="0"),
        sa.Column("keywords", postgresql.ARRAY(sa.String()), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("crawl_config_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("collection_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("webhook_url", sa.Text(), nullable=True),
        sa.Column("webhook_headers", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("pages_discovered", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("pages_crawled", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("pages_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("current_depth", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["crawl_config_id"], ["crawl_configs.id"]),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"]),
    )

    op.create_table(
        "deep_crawl_frontier",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("deep_crawl_job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("depth", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("discovered_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["deep_crawl_job_id"], ["deep_crawl_jobs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("deep_crawl_job_id", "url"),
    )

    op.create_table(
        "discovery_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column("sources", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("pattern", sa.Text(), nullable=True),
        sa.Column("max_urls", sa.Integer(), nullable=False, server_default="500"),
        sa.Column("score_query", sa.Text(), nullable=True),
        sa.Column("score_threshold", sa.Float(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("urls_found", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "webhooks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("headers", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("secret", sa.Text(), nullable=True),
        sa.Column("events", postgresql.ARRAY(sa.String()), nullable=False, server_default=sa.text("'{completed,failed}'")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["crawl_sources.id"]),
    )

    op.create_table(
        "webhook_deliveries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("webhook_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("job_type", sa.String(length=50), nullable=True),
        sa.Column("event", sa.String(length=50), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("response_status", sa.Integer(), nullable=True),
        sa.Column("response_body", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["webhook_id"], ["webhooks.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "domain_settings",
        sa.Column("domain", sa.String(length=255), primary_key=True),
        sa.Column("rate_limit_rps", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("circuit_breaker_threshold", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("circuit_breaker_timeout_s", sa.Integer(), nullable=False, server_default="300"),
        sa.Column("is_blocked", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("blocked_reason", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "proxy_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("servers", postgresql.JSONB(), nullable=False),
        sa.Column("rotation_strategy", sa.String(length=50), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "robots_txt_cache",
        sa.Column("domain", sa.String(length=255), primary_key=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "canonical_urls",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_url", sa.Text(), nullable=False, unique=True),
        sa.Column("canonical_url", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("canonical_urls")
    op.drop_table("robots_txt_cache")
    op.drop_table("proxy_configs")
    op.drop_table("domain_settings")
    op.drop_table("webhook_deliveries")
    op.drop_table("webhooks")
    op.drop_table("discovery_jobs")
    op.drop_table("deep_crawl_frontier")
    op.drop_table("deep_crawl_jobs")
    op.drop_table("crawl_jobs")
    op.drop_table("crawl_sources")
    op.drop_table("crawl_configs")
    op.drop_table("chunks")
    op.drop_table("document_tags")
    op.drop_table("documents")
    op.drop_table("tags")
    op.drop_table("collections")
    op.drop_table("api_keys")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_migration_001.py::test_migration_001_file_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add alembic/versions/001_initial_schema.py tests/integration/test_migration_001.py
git commit -m "feat: add initial schema migration"
```

---

### Task 3: Initial schema indexes (T011)

**Files:**
- Modify: `alembic/versions/001_initial_schema.py`
- Test: `tests/integration/test_migration_001_indexes.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_migration_includes_indexes():
    content = Path("alembic/versions/001_initial_schema.py").read_text(encoding="utf-8")
    assert "create_index" in content, "Migration must define indexes"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_migration_001_indexes.py::test_migration_includes_indexes -v`
Expected: FAIL with "Migration must define indexes"

**Step 3: Write minimal implementation**

```python
    op.create_index("idx_api_keys_active", "api_keys", ["is_active"], postgresql_where=sa.text("is_active = true"))
    op.create_index("idx_collections_name", "collections", ["name"], unique=True)
    op.create_index("idx_tags_name", "tags", ["name"], unique=True)
    op.create_index("idx_documents_domain", "documents", ["domain"])
    op.create_index("idx_documents_collection", "documents", ["collection_id"])
    op.create_index("idx_documents_crawled_at", "documents", ["crawled_at"])
    op.create_index("idx_documents_content_hash", "documents", ["content_hash"])
    op.create_index("idx_documents_fts", "documents", ["fts_vector"], postgresql_using="gin")
    op.create_index("idx_documents_active", "documents", ["id"], postgresql_where=sa.text("deleted_at IS NULL"))
    op.create_index("idx_chunks_document", "chunks", ["document_id"])
    op.create_index("idx_chunks_embedding_model", "chunks", ["embedding_model"])
    op.create_index("idx_chunks_fts", "chunks", ["fts_vector"], postgresql_using="gin")
    op.create_index("idx_crawl_jobs_status", "crawl_jobs", ["status"])
    op.create_index("idx_crawl_jobs_domain", "crawl_jobs", ["domain"])
    op.create_index("idx_crawl_jobs_created", "crawl_jobs", ["created_at"])
    op.create_index(
        "idx_crawl_jobs_pending_priority",
        "crawl_jobs",
        ["priority", "created_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index("idx_deep_frontier_pending_score", "deep_crawl_frontier", ["deep_crawl_job_id", sa.text("score DESC")], postgresql_where=sa.text("status = 'pending'"))
    op.create_index("idx_deep_frontier_pending_depth", "deep_crawl_frontier", ["deep_crawl_job_id", "depth", "discovered_at"], postgresql_where=sa.text("status = 'pending'"))
    op.create_index("idx_webhooks_active", "webhooks", ["is_active"], postgresql_where=sa.text("is_active = true"))
    op.create_index("idx_webhook_deliveries_pending", "webhook_deliveries", ["next_attempt_at"], postgresql_where=sa.text("status = 'pending'"))
    op.create_index("idx_domain_settings_blocked", "domain_settings", ["is_blocked"], postgresql_where=sa.text("is_blocked = true"))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_migration_001_indexes.py::test_migration_includes_indexes -v`
Expected: PASS

**Step 5: Commit**

```bash
git add alembic/versions/001_initial_schema.py tests/integration/test_migration_001_indexes.py
git commit -m "feat: add initial indexes"
```

---

### Task 4: Core abstractions (T012)

**Files:**
- Create: `app/core/abstractions.py`
- Test: `tests/unit/test_abstractions.py`

**Step 1: Write the failing test**

```python
from app.core.abstractions import VectorStore, DocumentStore, Cache, Embedder, Crawler


def test_abstractions_are_abstract():
    assert VectorStore.__abstractmethods__
    assert DocumentStore.__abstractmethods__
    assert Cache.__abstractmethods__
    assert Embedder.__abstractmethods__
    assert Crawler.__abstractmethods__
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_abstractions.py::test_abstractions_are_abstract -v`
Expected: FAIL with "No module named 'app'" or missing symbols

**Step 3: Write minimal implementation**

```python
from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from app.core.models import Document, Chunk, SearchConfig, SearchResponse, CrawlResult


class VectorStore(ABC):
    """Interface for vector search storage."""

    @abstractmethod
    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        document: Document,
    ) -> None:
        """Upsert chunk vectors into vector store."""

    @abstractmethod
    async def delete_by_document_id(self, document_id: UUID) -> None:
        """Delete vectors for a document."""

    @abstractmethod
    async def search(self, config: SearchConfig) -> SearchResponse:
        """Run vector search with filters."""


class DocumentStore(ABC):
    """Interface for document persistence."""

    @abstractmethod
    async def create_document(self, document: Document) -> Document:
        """Persist a document."""

    @abstractmethod
    async def get_document(self, document_id: UUID) -> Document | None:
        """Fetch a document by id."""

    @abstractmethod
    async def delete_document(self, document_id: UUID) -> None:
        """Soft delete a document."""

    @abstractmethod
    async def search(self, config: SearchConfig) -> SearchResponse:
        """Keyword search via FTS."""


class Cache(ABC):
    """Interface for cache access."""

    @abstractmethod
    async def get(self, key: str) -> str | None:
        """Get cached value."""

    @abstractmethod
    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """Set cache with TTL."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete cache key."""


class Embedder(ABC):
    """Interface for embedding generation."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch."""


class Crawler(ABC):
    """Interface for crawling content."""

    @abstractmethod
    async def crawl_url(self, url: str, config: dict[str, Any]) -> CrawlResult:
        """Crawl a URL and return result."""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_abstractions.py::test_abstractions_are_abstract -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/core/abstractions.py tests/unit/test_abstractions.py
git commit -m "feat: add core abstractions"
```

---

### Task 5: Core Pydantic models (all 16 entities) (T013)

**Files:**
- Create: `app/core/models.py`
- Test: `tests/unit/test_models.py`

**Step 1: Write the failing test**

```python
import pytest
from app.core.models import Document, DocSource


def test_document_requires_content_hash():
    doc = Document(
        url="https://example.com",
        domain="example.com",
        content="hello",
        content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        source=DocSource.CRAWL,
    )
    assert doc.content_hash


def test_invalid_hash_raises():
    with pytest.raises(ValueError):
        Document(
            url="https://example.com",
            domain="example.com",
            content="hello",
            content_hash="bad",
            source=DocSource.CRAWL,
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_models.py::test_document_requires_content_hash -v`
Expected: FAIL with "No module named 'app'" or missing symbols

**Step 3: Write minimal implementation**

```python
from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator
from urllib.parse import urlparse


class DocSource(str, Enum):
    """Document source types."""

    CRAWL = "crawl"
    UPLOAD = "upload"
    API = "api"


class ChunkContentType(str, Enum):
    """Chunk content types."""

    PROSE = "prose"
    CODE = "code"
    TABLE = "table"
    LIST = "list"


class JobStatus(str, Enum):
    """Job status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Job priority values."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class DeepCrawlStrategy(str, Enum):
    """Deep crawl strategy types."""

    BFS = "bfs"
    DFS = "dfs"
    BEST_FIRST = "best_first"


class DocumentMetadata(BaseModel):
    """Metadata extracted from crawled content."""

    author: str | None = None
    publish_date: datetime | None = None
    description: str | None = None
    keywords: list[str] = Field(default_factory=list)
    og_image: str | None = None
    canonical_url: str | None = None


class ApiKey(BaseModel):
    """API key model."""

    id: UUID = Field(default_factory=uuid4)
    key_hash: str
    name: str
    scopes: list[str] = Field(default_factory=lambda: ["read"])
    rate_limit_rpm: int = 60
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    is_active: bool = True


class Collection(BaseModel):
    """Collection model."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Tag(BaseModel):
    """Tag model."""

    id: UUID = Field(default_factory=uuid4)
    name: str


class Document(BaseModel):
    """Document model."""

    id: UUID = Field(default_factory=uuid4)
    url: str
    domain: str
    parent_url: str | None = None
    title: str | None = None
    content: str
    content_hash: str
    language: str = "en"
    source: DocSource
    collection_id: UUID | None = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    crawled_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: datetime | None = None

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if len(value) != 64:
            raise ValueError("Content hash must be 64-character SHA256 hex")
        return value.lower()

    @field_validator("domain")
    @classmethod
    def normalize_domain(cls, value: str, info):
        if value:
            return value.lower()
        url = info.data.get("url")
        if url:
            return urlparse(url).netloc.lower()
        return value


class Chunk(BaseModel):
    """Chunk model."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int
    embedding_model: str
    section_header: str | None = None
    content_type: ChunkContentType = ChunkContentType.PROSE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CrawlJob(BaseModel):
    """Crawl job model."""

    id: UUID = Field(default_factory=uuid4)
    url: str
    domain: str
    crawl_config_id: UUID | None = None
    crawl_source_id: UUID | None = None
    document_id: UUID | None = None
    collection_id: UUID | None = None
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    result: dict[str, Any] | None = None
    result_expires_at: datetime | None = None
    webhook_url: str | None = None
    webhook_headers: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class DeepCrawlJob(BaseModel):
    """Deep crawl job model."""

    id: UUID = Field(default_factory=uuid4)
    root_url: str
    strategy: DeepCrawlStrategy
    max_depth: int
    max_pages: int
    score_threshold: float = 0.0
    keywords: list[str] = Field(default_factory=list)
    crawl_config_id: UUID
    collection_id: UUID | None = None
    webhook_url: str | None = None
    webhook_headers: dict[str, str] = Field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    pages_discovered: int = 0
    pages_crawled: int = 0
    pages_failed: int = 0
    current_depth: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class DeepCrawlFrontierItem(BaseModel):
    """Frontier item for deep crawl."""

    id: UUID = Field(default_factory=uuid4)
    deep_crawl_job_id: UUID
    url: str
    depth: int
    score: float | None = None
    status: JobStatus = JobStatus.PENDING
    discovered_at: datetime = Field(default_factory=datetime.utcnow)


class DiscoveryJob(BaseModel):
    """Discovery job model."""

    id: UUID = Field(default_factory=uuid4)
    domain: str
    sources: list[str]
    pattern: str | None = None
    max_urls: int = 500
    score_query: str | None = None
    score_threshold: float = 0.0
    status: JobStatus = JobStatus.PENDING
    urls_found: int = 0
    result: list[str] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


class CrawlConfig(BaseModel):
    """Crawl config model."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    extraction_strategy: str
    chunking_strategy: str
    chunk_size: int = 1200
    page_timeout_ms: int = 30000
    respect_robots_txt: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CrawlSource(BaseModel):
    """Crawl source model."""

    id: UUID = Field(default_factory=uuid4)
    collection_id: UUID | None = None
    crawl_config_id: UUID
    name: str
    url_pattern: str
    schedule_cron: str | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DomainSettings(BaseModel):
    """Domain settings model."""

    domain: str
    rate_limit_rps: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_s: int = 300
    is_blocked: bool = False
    blocked_reason: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProxyConfig(BaseModel):
    """Proxy configuration."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    servers: list[dict[str, str]]
    rotation_strategy: str = "round_robin"
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Webhook(BaseModel):
    """Webhook configuration."""

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID | None = None
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    secret: str | None = None
    events: list[str] = Field(default_factory=lambda: ["completed", "failed"])
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WebhookDelivery(BaseModel):
    """Webhook delivery model."""

    id: UUID = Field(default_factory=uuid4)
    webhook_id: UUID
    job_id: UUID | None = None
    job_type: str | None = None
    event: str
    payload: dict[str, Any]
    status: Literal["pending", "delivered", "failed"] = "pending"
    attempts: int = 0
    max_attempts: int = 5
    last_attempt_at: datetime | None = None
    next_attempt_at: datetime | None = None
    response_status: int | None = None
    response_body: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchFilters(BaseModel):
    """Search filter criteria."""

    collection_ids: list[UUID] | None = None
    tag_ids: list[UUID] | None = None
    domains: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    source_types: list[DocSource] | None = None


class SearchConfig(BaseModel):
    """Search configuration."""

    query: str
    filters: SearchFilters = Field(default_factory=SearchFilters)
    query_embedding: list[float] | None = None
    min_score: float = 0.0
    use_reranker: bool = False
    rerank_top_n: int = 20
    expand_chunks: bool = False
    rrf_k: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    limit: int = 10
    cursor: str | None = None


class SearchResult(BaseModel):
    """Search result model."""

    document_id: UUID
    chunk_id: UUID
    url: str
    title: str | None
    content: str
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    source: Literal["vector", "keyword", "fused"]
    highlights: list[tuple[int, int]] | None = None
    section_header: str | None = None
    expanded_chunks: list[str] | None = None


class SearchResponse(BaseModel):
    """Search response model."""

    results: list[SearchResult]
    total_count: int | None = None
    next_cursor: str | None = None
    query_embedding_cached: bool = False
    result_cached: bool = False
    latency_ms: float


class ValidatedUrl(BaseModel):
    """Validated URL model."""

    original: str
    normalized: str
    scheme: str
    host: str
    domain: str
    path: str
    is_valid: bool = True
    rejection_reason: str | None = None


class CrawlResult(BaseModel):
    """Transient crawl result."""

    url: str
    normalized_url: str
    success: bool
    status_code: int | None = None
    content_type: str | None = None
    html: str | None = None
    markdown: str | None = None
    text: str | None = None
    title: str | None = None
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    links: list[str] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    error_type: str | None = None
    crawled_at: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: int

    def to_document(self) -> Document:
        """Convert crawl result to document."""

        content = self.markdown or self.text or ""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        return Document(
            url=self.normalized_url,
            domain=urlparse(self.normalized_url).netloc.lower(),
            title=self.title,
            content=content,
            content_hash=content_hash,
            source=DocSource.CRAWL,
            metadata=self.metadata,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_models.py::test_document_requires_content_hash -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/core/models.py tests/unit/test_models.py
git commit -m "feat: add core pydantic models"
```

---

### Task 6: Settings configuration (T014)

**Files:**
- Create: `app/core/config.py`
- Test: `tests/unit/test_config.py`

**Step 1: Write the failing test**

```python
from app.core.config import get_settings


def test_settings_has_database_url():
    settings = get_settings()
    assert settings.database_url
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py::test_settings_has_database_url -v`
Expected: FAIL with "No module named 'app'" or missing symbols

**Step 3: Write minimal implementation**

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    redis_url: str
    qdrant_url: str
    tei_url: str
    crawl4ai_url: str
    allowed_origins: str = ""
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""

    return Settings()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py::test_settings_has_database_url -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/core/config.py tests/unit/test_config.py
git commit -m "feat: add settings configuration"
```

---

### Task 7: Dependency injection helpers (T015)

**Files:**
- Create: `app/core/deps.py`
- Test: `tests/unit/test_deps.py`

**Step 1: Write the failing test**

```python
from app.core.deps import get_settings


def test_deps_exports_settings():
    assert get_settings
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_deps.py::test_deps_exports_settings -v`
Expected: FAIL with "No module named 'app'" or missing symbols

**Step 3: Write minimal implementation**

```python
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
import redis.asyncio as redis
from qdrant_client import QdrantClient

from app.core.config import get_settings

_engine = None
_session_maker = None
_redis_pool = None
_qdrant_client = None


def _get_engine():
    settings = get_settings()
    return create_async_engine(settings.database_url, pool_pre_ping=True)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async DB session."""

    global _engine, _session_maker
    if _engine is None:
        _engine = _get_engine()
        _session_maker = async_sessionmaker(_engine, expire_on_commit=False)

    async with _session_maker() as session:
        yield session


async def get_redis_client() -> redis.Redis:
    """Provide Redis client."""

    global _redis_pool
    settings = get_settings()
    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool.from_url(settings.redis_url)
    return redis.Redis(connection_pool=_redis_pool)


def get_qdrant_client() -> QdrantClient:
    """Provide Qdrant client."""

    global _qdrant_client
    settings = get_settings()
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url)
    return _qdrant_client
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_deps.py::test_deps_exports_settings -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/core/deps.py tests/unit/test_deps.py
git commit -m "feat: add dependency helpers"
```

---

### Task 8: PostgreSQL DocumentStore adapter with FTS (T016)

**Files:**
- Create: `app/storage/postgres.py`
- Test: `tests/integration/test_postgres_store.py`

**Step 1: Write the failing test**

```python
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.core.models import Document, DocSource, SearchConfig
from app.storage.postgres import PostgresDocumentStore


@pytest.mark.asyncio
async def test_postgres_store_create_and_get(tmp_path):
    engine = create_async_engine("postgresql+asyncpg://postgres:postgres@localhost:53432/crawl4r")
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async with session_maker() as session:
        store = PostgresDocumentStore(session)
        doc = Document(
            url="https://example.com",
            domain="example.com",
            content="hello",
            content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            source=DocSource.CRAWL,
        )
        created = await store.create_document(doc)
        fetched = await store.get_document(created.id)
        assert fetched is not None

    await engine.dispose()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_postgres_store.py::test_postgres_store_create_and_get -v`
Expected: FAIL with missing module or NotImplementedError

**Step 3: Write minimal implementation**

```python
from datetime import datetime
from sqlalchemy import Table, Column, MetaData, String, Text, Integer, DateTime, Boolean, ForeignKey, select, insert, update, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.abstractions import DocumentStore
from app.core.models import Document, SearchConfig, SearchResponse, SearchResult

metadata = MetaData()

documents = Table(
    "documents",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("url", Text, nullable=False),
    Column("domain", String(255), nullable=False),
    Column("parent_url", Text),
    Column("title", Text),
    Column("content", Text, nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("language", String(8), nullable=False),
    Column("source", String(20), nullable=False),
    Column("collection_id", UUID(as_uuid=True)),
    Column("metadata", JSONB, nullable=False),
    Column("crawled_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("deleted_at", DateTime(timezone=True)),
)

chunks = Table(
    "chunks",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("document_id", UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False),
    Column("content", Text, nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("section_header", Text),
)


class PostgresDocumentStore(DocumentStore):
    """PostgreSQL-backed document store."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_document(self, document: Document) -> Document:
        payload = document.model_dump()
        payload["metadata"] = payload["metadata"]
        payload["updated_at"] = datetime.utcnow()
        await self._session.execute(insert(documents).values(**payload))
        await self._session.commit()
        return document

    async def get_document(self, document_id):
        result = await self._session.execute(select(documents).where(documents.c.id == document_id))
        row = result.mappings().first()
        return Document(**row) if row else None

    async def delete_document(self, document_id) -> None:
        await self._session.execute(
            update(documents)
            .where(documents.c.id == document_id)
            .values(deleted_at=datetime.utcnow())
        )
        await self._session.commit()

    async def search(self, config: SearchConfig) -> SearchResponse:
        query = text(
            """
            SELECT d.id AS document_id, c.id AS chunk_id, d.url, d.title, c.content,
                   ts_rank_cd(to_tsvector('english', c.content), to_tsquery('english', :q)) AS keyword_score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE to_tsvector('english', c.content) @@ to_tsquery('english', :q)
            ORDER BY keyword_score DESC
            LIMIT :limit
            """
        )
        result = await self._session.execute(query, {"q": config.query, "limit": config.limit})
        rows = result.mappings().all()
        items = [
            SearchResult(
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
                url=row["url"],
                title=row["title"],
                content=row["content"],
                score=float(row["keyword_score"]),
                keyword_score=float(row["keyword_score"]),
                source="keyword",
            )
            for row in rows
        ]
        return SearchResponse(results=items, latency_ms=0.0)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_postgres_store.py::test_postgres_store_create_and_get -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/storage/postgres.py tests/integration/test_postgres_store.py
git commit -m "feat: implement postgres document store"
```

---

### Task 9: Qdrant VectorStore adapter with INT8 config (T017)

**Files:**
- Create: `app/storage/qdrant.py`
- Test: `tests/integration/test_qdrant_store.py`

**Step 1: Write the failing test**

```python
from app.storage.qdrant import QdrantVectorStore


def test_qdrant_store_exposes_search():
    store = QdrantVectorStore(None, "crawl4r")
    assert callable(store.search)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_qdrant_store.py::test_qdrant_store_exposes_search -v`
Expected: FAIL with missing module or errors

**Step 3: Write minimal implementation**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    ScalarQuantizationConfig,
    ScalarType,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
)

from app.core.abstractions import VectorStore
from app.core.models import Chunk, Document, SearchConfig, SearchResponse, SearchResult


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store."""

    def __init__(self, client: QdrantClient | None, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    def ensure_collection(self) -> None:
        if self._client is None:
            return
        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE, on_disk=True),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                quantization_config=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True),
            )

    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        document: Document,
    ) -> None:
        if self._client is None:
            return
        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            payload = {
                "document_id": str(chunk.document_id),
                "chunk_id": str(chunk.id),
                "chunk_index": chunk.chunk_index,
                "collection_id": str(document.collection_id) if document.collection_id else None,
                "domain": document.domain,
                "url": document.url,
                "title": document.title,
                "source": document.source.value,
                "tags": [],
                "section_header": chunk.section_header,
                "content_type": chunk.content_type.value,
                "crawled_at": document.crawled_at.isoformat(),
            }
            points.append(PointStruct(id=str(chunk.id), vector=vector, payload=payload))

        self._client.upsert(collection_name=self._collection_name, points=points)

    async def delete_by_document_id(self, document_id) -> None:
        if self._client is None:
            return
        selector = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=str(document_id)))])
        self._client.delete(collection_name=self._collection_name, points_selector=selector)

    async def search(self, config: SearchConfig) -> SearchResponse:
        if self._client is None:
            return SearchResponse(results=[], latency_ms=0.0)
        if not config.query_embedding:
            raise ValueError("query_embedding required for vector search")

        must: list[FieldCondition] = []
        if config.filters.collection_ids:
            must.append(
                FieldCondition(
                    key="collection_id",
                    match=MatchAny(values=[str(cid) for cid in config.filters.collection_ids]),
                )
            )
        if config.filters.domains:
            must.append(FieldCondition(key="domain", match=MatchAny(values=config.filters.domains)))
        if config.filters.source_types:
            must.append(
                FieldCondition(
                    key="source",
                    match=MatchAny(values=[s.value for s in config.filters.source_types]),
                )
            )
        search_filter = Filter(must=must) if must else None

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=config.query_embedding,
            query_filter=search_filter,
            limit=config.limit,
        )
        items = [
            SearchResult(
                document_id=r.payload.get("document_id"),
                chunk_id=r.payload.get("chunk_id"),
                url=r.payload.get("url"),
                title=r.payload.get("title"),
                content="",
                score=float(r.score),
                vector_score=float(r.score),
                source="vector",
            )
            for r in results
        ]
        return SearchResponse(results=items, latency_ms=0.0)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_qdrant_store.py::test_qdrant_store_exposes_search -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/storage/qdrant.py tests/integration/test_qdrant_store.py
git commit -m "feat: add qdrant vector store"
```

---

### Task 10: Redis cache adapter (T018)

**Files:**
- Create: `app/storage/redis_cache.py`
- Test: `tests/integration/test_redis_cache.py`

**Step 1: Write the failing test**

```python
from app.storage.redis_cache import RedisCache


def test_cache_has_set():
    cache = RedisCache(None)
    assert callable(cache.set)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_redis_cache.py::test_cache_has_set -v`
Expected: FAIL with missing module

**Step 3: Write minimal implementation**

```python
import redis.asyncio as redis

from app.core.abstractions import Cache


class RedisCache(Cache):
    """Redis-backed cache adapter."""

    def __init__(self, client: redis.Redis | None) -> None:
        self._client = client

    async def get(self, key: str) -> str | None:
        if self._client is None:
            return None
        return await self._client.get(key)

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        if self._client is None:
            return
        await self._client.setex(key, ttl_seconds, value)

    async def delete(self, key: str) -> None:
        if self._client is None:
            return
        await self._client.delete(key)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_redis_cache.py::test_cache_has_set -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/storage/redis_cache.py tests/integration/test_redis_cache.py
git commit -m "feat: add redis cache"
```

---

### Task 11: Middleware (auth + expiry + rate limiting + CORS + exceptions) (T019–T023)

**Files:**
- Create: `app/api/middleware.py`
- Test: `tests/integration/test_middleware.py`

**Step 1: Write the failing test**

```python
from app.api.middleware import AuthMiddleware


def test_auth_middleware_exists():
    assert AuthMiddleware
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_middleware.py::test_auth_middleware_exists -v`
Expected: FAIL with missing module

**Step 3: Write minimal implementation**

```python
import hashlib
import time
from datetime import datetime, timezone

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import text
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS

from app.core.deps import get_db_session, get_redis_client

RATE_LIMIT_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local tokens = tonumber(redis.call('GET', key) or capacity)
local last_refill = tonumber(redis.call('GET', key .. ':last') or now)

local elapsed = now - last_refill
local refill_amount = elapsed * refill_rate
local new_tokens = math.min(capacity, tokens + refill_amount)

if new_tokens >= cost then
    new_tokens = new_tokens - cost
    redis.call('SET', key, new_tokens)
    redis.call('SET', key .. ':last', now)
    redis.call('EXPIRE', key, 60)
    redis.call('EXPIRE', key .. ':last', 60)
    return 1
else
    return 0
end
"""


class AuthMiddleware(BaseHTTPMiddleware):
    """Bearer token authentication middleware."""

    async def dispatch(self, request: Request, call_next):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"error": "Unauthorized"}, status_code=HTTP_401_UNAUTHORIZED)

        token = auth.replace("Bearer ", "").strip()
        key_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

        async for session in get_db_session():
            result = await session.execute(
                text(
                    """
                    SELECT id, scopes, rate_limit_rpm, expires_at, is_active
                    FROM api_keys
                    WHERE key_hash = :key_hash
                    """
                ),
                {"key_hash": key_hash},
            )
            row = result.mappings().first()

        if not row or not row["is_active"]:
            return JSONResponse({"error": "Unauthorized"}, status_code=HTTP_401_UNAUTHORIZED)

        expires_at = row["expires_at"]
        if expires_at and expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
            return JSONResponse(
                {"error": "Unauthorized", "details": {"expires_at": expires_at.isoformat()}},
                status_code=HTTP_401_UNAUTHORIZED,
            )

        await enforce_rate_limit(key_hash, int(row["rate_limit_rpm"]))
        request.state.api_key_hash = key_hash
        request.state.scopes = row["scopes"]

        return await call_next(request)


async def enforce_rate_limit(api_key_hash: str, rpm: int) -> None:
    """Token bucket using Redis Lua script."""

    redis_client = await get_redis_client()
    capacity = rpm
    refill_rate = rpm / 60.0
    cost = 1
    now = int(time.time())

    allowed = await redis_client.eval(RATE_LIMIT_LUA, 1, f"crawl4r:rate:api:{api_key_hash}", capacity, refill_rate, cost, now)
    if int(allowed) != 1:
        raise HTTPException(status_code=HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")


def register_exception_handlers(app) -> None:
    """Register global exception handlers."""

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        return JSONResponse(status_code=422, content={"error": "Validation failed", "details": exc.errors()})

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})


def get_cors_config(allowed_origins: str) -> dict[str, object]:
    """Build CORS middleware configuration."""

    origins = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]
    return {
        "allow_origins": origins,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_middleware.py::test_auth_middleware_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/api/middleware.py tests/integration/test_middleware.py
git commit -m "feat: implement core middleware"
```

---

### Task 12: Structured logging setup (T022)

**Files:**
- Create: `app/core/logging.py`
- Test: `tests/unit/test_logging.py`

**Step 1: Write the failing test**

```python
from app.core.logging import configure_logging


def test_configure_logging_exists():
    assert configure_logging
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_logging.py::test_configure_logging_exists -v`
Expected: FAIL with missing module

**Step 3: Write minimal implementation**

```python
import structlog


def configure_logging() -> None:
    """Configure structlog for JSON output."""

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_logging.py::test_configure_logging_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/core/logging.py tests/unit/test_logging.py
git commit -m "feat: add structured logging config"
```

---

### Task 13: Health endpoints (T024)

**Files:**
- Create: `app/api/v1/admin.py`
- Test: `tests/integration/test_health_endpoints.py`

**Step 1: Write the failing test**

```python
from fastapi import FastAPI
from app.api.v1.admin import router


def test_admin_router_mounts():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_health_endpoints.py::test_admin_router_mounts -v`
Expected: FAIL with missing module

**Step 3: Write minimal implementation**

```python
import httpx
from fastapi import APIRouter
from starlette.responses import JSONResponse

from app.core.config import get_settings
from app.core.deps import get_redis_client, get_db_session, get_qdrant_client

router = APIRouter(prefix="", tags=["admin"])


@router.get("/health")
async def health():
    """Liveness probe."""

    return {"status": "healthy"}


@router.get("/health/ready")
async def readiness():
    """Readiness probe."""

    settings = get_settings()

    checks = {"postgres": False, "redis": False, "qdrant": False, "tei": False, "crawl4ai": False}

    async for session in get_db_session():
        try:
            await session.execute("SELECT 1")
            checks["postgres"] = True
        except Exception:
            checks["postgres"] = False

    try:
        redis_client = await get_redis_client()
        checks["redis"] = await redis_client.ping()
    except Exception:
        checks["redis"] = False

    try:
        client = get_qdrant_client()
        _ = client.get_collections()
        checks["qdrant"] = True
    except Exception:
        checks["qdrant"] = False

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            tei = await client.get(f"{settings.tei_url}/health")
            checks["tei"] = tei.status_code == 200
        except Exception:
            checks["tei"] = False

        try:
            crawl = await client.get(f"{settings.crawl4ai_url}/health")
            checks["crawl4ai"] = crawl.status_code == 200
        except Exception:
            checks["crawl4ai"] = False

    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    status = "ready" if all_healthy else "degraded"

    return JSONResponse(status_code=status_code, content={"status": status, "checks": checks})
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_health_endpoints.py::test_admin_router_mounts -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/api/v1/admin.py tests/integration/test_health_endpoints.py
git commit -m "feat: add health endpoints"
```

---

### Task 14: Main FastAPI app + router wiring (T025–T026)

**Files:**
- Create: `app/api/v1/router.py`
- Create: `app/main.py`
- Test: `tests/integration/test_app_boot.py`

**Step 1: Write the failing test**

```python
from app.main import app


def test_app_exists():
    assert app
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_app_boot.py::test_app_exists -v`
Expected: FAIL with missing module

**Step 3: Write minimal implementation**

```python
from fastapi import APIRouter

from app.api.v1.admin import router as admin_router

router = APIRouter()
router.include_router(admin_router)
```

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import AuthMiddleware, register_exception_handlers, get_cors_config
from app.api.v1.router import router as api_v1_router
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    """Create FastAPI application."""

    configure_logging()
    settings = get_settings()
    app = FastAPI()

    app.add_middleware(AuthMiddleware)
    cors_config = get_cors_config(settings.allowed_origins)
    app.add_middleware(CORSMiddleware, **cors_config)

    app.include_router(api_v1_router, prefix="/api/v1")
    register_exception_handlers(app)
    return app


app = create_app()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_app_boot.py::test_app_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/api/v1/router.py app/main.py tests/integration/test_app_boot.py
git commit -m "feat: add app wiring"
```

---

## Verification Checklist (Phase 2)

Run these after completing all tasks:

```bash
pytest tests/unit tests/integration -v
```

Expected: All tests PASS

---

## Notes / Follow-ups for Phase 3

- Extend storage adapters to full CRUD and hybrid search logic; add Qdrant upsert/search and Redis cache compression.
- Add SQLAlchemy ORM models if required for more complex queries; keep migrations handwritten until models are stable.
- Add structured logging with correlation IDs in request context.
