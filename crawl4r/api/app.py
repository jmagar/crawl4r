"""FastAPI application for Crawl4r REST API.

Provides REST endpoints for crawling, ingestion status, and health monitoring.

Example:
    uvicorn crawl4r.api.app:app --host 0.0.0.0 --port 8000
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from crawl4r.api.routes.health import router as health_router

# API key authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """Verify API key from request header.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        Validated API key

    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    expected_key = os.getenv("CRAWL4R_API_KEY")

    # Allow unauthenticated access if no API key is configured
    if not expected_key:
        return ""

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


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
# Note: Health endpoint is public. Add dependencies=[Depends(verify_api_key)]
# to protected routers when adding new endpoints
app.include_router(health_router)
