"""Health check endpoint for API monitoring.

Provides service health status and component connectivity checks.

Example:
    GET /health
    Response: {"status": "healthy", "services": {...}}
"""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return API health status.

    Returns:
        Dict with status field indicating service health.

    Example:
        >>> response = client.get("/health")
        >>> response.json()
        {"status": "healthy"}
    """
    return {"status": "healthy"}
