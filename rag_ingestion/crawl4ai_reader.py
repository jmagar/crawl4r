"""LlamaIndex reader for Crawl4AI web crawling service.

This module provides a production-ready LlamaIndex reader that integrates with the
Crawl4AI Docker service to fetch web content as markdown-formatted documents.
It extends BasePydanticReader for serialization support and implements async-first
HTTP operations with circuit breaker pattern, retry logic, and structured logging.

The Crawl4AIReader class provides:
- Async batch web crawling with configurable concurrency
- Deterministic Document IDs via SHA256 URL hashing
- Rich metadata extraction for Qdrant compatibility
- Circuit breaker pattern for service failures
- Exponential backoff retry logic
- Health validation before batch operations
- Order-preserving results with error handling

Examples:
    Basic single URL crawling:

        >>> from rag_ingestion.crawl4ai_reader import Crawl4AIReader
        >>> reader = Crawl4AIReader(
        ...     endpoint_url="http://localhost:52004",
        ...     timeout_seconds=60
        ... )
        >>> documents = await reader.aload_data(["https://example.com"])
        >>> print(documents[0].text[:100])

    Batch crawling with custom configuration:

        >>> urls = [
        ...     "https://example.com/page1",
        ...     "https://example.com/page2",
        ...     "https://example.com/page3"
        ... ]
        >>> reader = Crawl4AIReader(
        ...     endpoint_url="http://localhost:52004",
        ...     timeout_seconds=60,
        ...     fail_on_error=False,
        ...     max_concurrent_requests=5
        ... )
        >>> documents = await reader.aload_data(urls)
        >>> successful = [d for d in documents if d is not None]
"""

from logging import Logger
from typing import Any

import httpx
from llama_index.core.readers.base import BasePydanticReader
from pydantic import BaseModel, ConfigDict, Field

from rag_ingestion.circuit_breaker import CircuitBreaker
from rag_ingestion.logger import get_logger


class Crawl4AIReaderConfig(BaseModel):
    """Configuration for Crawl4AI reader.

    This class defines all configuration parameters for the Crawl4AIReader,
    including HTTP settings, retry behavior, circuit breaker thresholds,
    and concurrency limits.

    Attributes:
        base_url: Crawl4AI service endpoint URL
        timeout: HTTP request timeout in seconds
        max_retries: Maximum retry attempts for transient errors
        retry_delays: Exponential backoff delays in seconds
        circuit_breaker_threshold: Number of failures before opening circuit
        circuit_breaker_timeout: Seconds to wait before closing circuit
        concurrency_limit: Maximum concurrent requests for batch processing

    Examples:
        Default configuration:
            >>> config = Crawl4AIReaderConfig()
            >>> assert config.base_url == "http://localhost:52004"
            >>> assert config.timeout == 30

        Custom configuration:
            >>> config = Crawl4AIReaderConfig(
            ...     base_url="http://crawl4ai:11235",
            ...     timeout=60,
            ...     concurrency_limit=10
            ... )
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    base_url: str = Field(
        default="http://localhost:52004",
        description="Crawl4AI service endpoint URL",
    )
    timeout: int = Field(
        default=30,
        ge=10,
        le=300,
        description="HTTP request timeout in seconds (10-300)",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for transient errors (0-10)",
    )
    retry_delays: list[float] = Field(
        default=[1.0, 2.0, 4.0],
        description="Exponential backoff delays in seconds",
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of failures before opening circuit (1-20)",
    )
    circuit_breaker_timeout: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Seconds to wait before closing circuit (10-600)",
    )
    concurrency_limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent requests for batch processing (1-20)",
    )


class Crawl4AIReader(BasePydanticReader):
    """LlamaIndex reader for crawling web pages via Crawl4AI service.

    This reader integrates with the Crawl4AI Docker service to fetch web content
    as markdown-formatted documents. It supports both synchronous and asynchronous
    loading, concurrent processing with configurable limits, circuit breaker
    protection, and exponential backoff retry logic.

    Attributes:
        endpoint_url: Crawl4AI service endpoint URL (default: http://localhost:52004)
        timeout_seconds: HTTP request timeout in seconds (default: 60, range: 10-300)
        fail_on_error: Raise exception on first error vs. continue (default: False)
        max_concurrent_requests: Concurrency limit (default: 5, range: 1-20)
        max_retries: Maximum retry attempts for transient errors (default: 3)
        retry_delays: Exponential backoff delays in seconds (default: [1, 2, 4])
        is_remote: LlamaIndex flag for remote data source (always True)
        class_name: LlamaIndex serialization class name (always "Crawl4AIReader")

    Examples:
        Basic usage with defaults:
            >>> reader = Crawl4AIReader()
            >>> docs = await reader.aload_data(["https://example.com"])
            >>> print(docs[0].metadata["title"])

        Custom configuration:
            >>> reader = Crawl4AIReader(
            ...     endpoint_url="http://crawl4ai:11235",
            ...     timeout_seconds=90,
            ...     fail_on_error=True,
            ...     max_concurrent_requests=10
            ... )
            >>> docs = reader.load_data(["https://site1.com", "https://site2.com"])
    """

    endpoint_url: str = Field(
        default="http://localhost:52004",
        description="Crawl4AI service endpoint URL",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="HTTP request timeout in seconds (10-300)",
    )
    fail_on_error: bool = Field(
        default=False,
        description="Raise exception on first error (True) or skip failures (False)",
    )
    max_concurrent_requests: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent requests for batch processing (1-20)",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for transient errors (0-10)",
    )
    retry_delays: list[float] = Field(
        default=[1.0, 2.0, 4.0],
        description="Exponential backoff delays in seconds",
    )

    # LlamaIndex required properties
    is_remote: bool = True

    @classmethod
    def class_name(cls) -> str:
        """Return the class name for LlamaIndex serialization."""
        return "Crawl4AIReader"

    # Internal components (not serialized)
    _circuit_breaker: CircuitBreaker | None = None
    _logger: Logger | None = None

    def __init__(self, **data: Any) -> None:
        """Initialize reader and validate Crawl4AI service health.

        Args:
            **data: Pydantic field values (endpoint_url, timeout_seconds, etc.)

        Raises:
            ValueError: If endpoint URL is invalid or service is unreachable
        """
        super().__init__(**data)

        # Initialize circuit breaker for fault tolerance
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,  # Project standard
            reset_timeout=60.0,  # Project standard
        )

        # Initialize structured logger
        self._logger = get_logger("rag_ingestion.crawl4ai_reader", log_level="INFO")

        # Validate service health on initialization
        # This is blocking, but necessary to fail fast on misconfiguration
        if not self._validate_health_sync():
            raise ValueError(
                f"Crawl4AI service unreachable at {self.endpoint_url}/health"
            )

    def _validate_health_sync(self) -> bool:
        """Synchronous health check for initialization.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.endpoint_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    async def _validate_health(self) -> bool:
        """Asynchronous health check for runtime validation.

        This method mirrors _validate_health_sync() but uses httpx.AsyncClient
        for non-blocking operations. Used in aload_data() before batch processing
        to ensure service availability.

        Returns:
            True if service is healthy (HTTP 200), False otherwise

        Examples:
            >>> reader = Crawl4AIReader()
            >>> is_healthy = await reader._validate_health()
            >>> assert is_healthy is True
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.endpoint_url}/health")
                return response.status_code == 200
        except Exception:
            return False
