"""Response models for API endpoints.

Pydantic models defining the structure of API responses.

Example:
    from crawl4r.api.models.responses import HealthResponse

    response = HealthResponse(status="healthy")
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response model.

    Attributes:
        status: Current health status ('healthy' or 'unhealthy')

    Example:
        >>> response = HealthResponse(status="healthy")
        >>> response.model_dump()
        {"status": "healthy"}
    """

    status: str
