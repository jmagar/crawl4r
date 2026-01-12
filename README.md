# crawl4r

## Overview

crawl4r is a self-hosted RAG pipeline that crawls web content, extracts clean text, stores documents in PostgreSQL with full-text search, and indexes embeddings in Qdrant for hybrid retrieval.

## Quick Start

1. Copy `.env.example` to `.env` and fill in values.
2. Start services: `docker compose up -d`
3. Create a virtual environment: `uv venv`
4. Install dependencies: `uv pip install -e ".[dev]"`
5. Run the API: `uvicorn app.main:app --reload`

## Development

- Lint: `ruff check app tests`
- Format: `ruff format app tests`
- Type check: `mypy app --strict`
- Test: `pytest`
