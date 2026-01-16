"""FastAPI application for Crawl4r REST API.

Provides REST endpoints for crawling, ingestion status, and health monitoring.

Example:
    uvicorn crawl4r.api.app:app --host 0.0.0.0 --port 8000
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from crawl4r.api.routes.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle events.

    Handles startup and shutdown operations for the API:
    - Startup: Initialize services, connections
    - Shutdown: Clean up resources, close connections

    Args:
        app: FastAPI application instance

    Yields:
        None during application runtime

    Example:
        app = FastAPI(lifespan=lifespan)
    """
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="Crawl4r API",
    description="REST API for RAG ingestion pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(health_router)
