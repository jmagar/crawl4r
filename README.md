# crawl4r

## Overview

crawl4r is a self-hosted RAG pipeline that crawls web content, extracts clean text, stores documents in PostgreSQL with full-text search, and indexes embeddings in Qdrant for hybrid retrieval.

## Quick Start

1. Copy `.env.example` to `.env` and fill in values.
2. Start local services (Postgres, Redis, Crawl4AI): `docker compose up -d`
3. Ensure Qdrant and TEI are running on your external host and set `QDRANT_URL` and `TEI_URL` in `.env`.
4. Create a virtual environment: `uv venv`
5. Install dependencies: `uv pip install -e ".[dev]"`
6. Run the API: `uvicorn app.main:app --reload`

## Development

- Lint: `ruff check app tests`
- Format: `ruff format app tests`
- Type check: `mypy app --strict`
- Test: `pytest`
