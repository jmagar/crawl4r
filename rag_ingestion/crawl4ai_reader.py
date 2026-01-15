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

import asyncio
import hashlib
import uuid
from typing import Any

import httpx
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from pydantic import BaseModel, ConfigDict, Field


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
