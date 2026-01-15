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
