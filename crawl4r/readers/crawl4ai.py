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

        >>> from crawl4r.readers.crawl4ai import Crawl4AIReader
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
import ipaddress
import re
import uuid
from collections.abc import AsyncIterator, Iterable
from logging import Logger
from typing import Annotated, Any
from urllib.parse import urlparse

import httpx
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import SkipValidation

from crawl4r.core.logger import get_logger
from crawl4r.resilience.circuit_breaker import CircuitBreaker
from crawl4r.storage.qdrant import VectorStoreManager


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
        max_concurrent_requests: Concurrency limit (default: 5, range: 1-100)
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
        le=100,
        description="Maximum concurrent requests for batch processing (1-100)",
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
    enable_deduplication: bool = Field(
        default=True,
        description="Auto-delete old versions before crawl (prevents duplicates)",
    )
    vector_store: Annotated[VectorStoreManager | None, SkipValidation] = Field(
        default=None,
        description="Optional vector store for deduplication (None = skip)",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # LlamaIndex required properties
    is_remote: bool = True

    @classmethod
    def class_name(cls) -> str:
        """Return the class name for LlamaIndex serialization."""
        return "Crawl4AIReader"

    # Internal components (not serialized, always initialized in __init__)
    _circuit_breaker: CircuitBreaker
    _logger: Logger

    def __init__(self, settings: Any = None, **data: Any) -> None:
        """Initialize reader and validate Crawl4AI service health.

        Args:
            settings: Optional Settings instance to read CRAWL4AI_BASE_URL from
            **data: Pydantic field values (endpoint_url, timeout_seconds, etc.)

        Raises:
            ValueError: If endpoint URL is invalid or service is unreachable
        """
        # If Settings provided, use its CRAWL4AI_BASE_URL as default endpoint_url
        if settings is not None and "endpoint_url" not in data:
            if hasattr(settings, "CRAWL4AI_BASE_URL"):
                data["endpoint_url"] = settings.CRAWL4AI_BASE_URL

        super().__init__(**data)

        # Initialize circuit breaker for fault tolerance
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,  # Project standard
            reset_timeout=60.0,  # Project standard
        )

        # Initialize structured logger
        self._logger = get_logger("crawl4r.readers.crawl4ai", log_level="INFO")

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

    def validate_url(self, url: str) -> bool:
        """Validate URL is safe for crawling (SSRF prevention).

        Prevents Server-Side Request Forgery (SSRF) attacks by validating URLs
        before forwarding them to the Crawl4AI service. Rejects:
        - Private IP ranges (10.x, 172.16-31.x, 192.168.x, 127.x)
        - Cloud metadata endpoints (169.254.169.254)
        - Non-HTTP(S) schemes (file://, ftp://, gopher://)
        - Localhost hostnames
        - IP addresses in alternate notations (decimal, hex)

        Args:
            url: URL to validate. Must be a well-formed URL string.

        Returns:
            True if URL is safe for crawling, False otherwise.

        Examples:
            Safe URLs:
                >>> reader.validate_url("https://example.com/")
                True

            Blocked URLs:
                >>> reader.validate_url("http://192.168.1.1/admin")
                False
                >>> reader.validate_url("http://169.254.169.254/")
                False
        """
        # Blocked hostnames (case-insensitive)
        blocked_hostnames = {
            "localhost",
            "metadata.google.internal",
            "metadata",
        }

        # Allowed schemes
        allowed_schemes = {"http", "https"}

        try:
            parsed = urlparse(url)

            # Check scheme
            if not parsed.scheme or parsed.scheme.lower() not in allowed_schemes:
                return False

            # Check hostname exists
            hostname = parsed.hostname
            if not hostname:
                return False

            hostname_lower = hostname.lower()

            # Check blocked hostnames
            if hostname_lower in blocked_hostnames:
                return False

            # Check if hostname is an IP address
            try:
                # Handle IPv6 addresses (already unwrapped by urlparse)
                ip = ipaddress.ip_address(hostname)

                # Block private IP ranges
                if ip.is_private:
                    return False

                # Block loopback (127.x.x.x, ::1)
                if ip.is_loopback:
                    return False

                # Block link-local (169.254.x.x - AWS metadata)
                if ip.is_link_local:
                    return False

                # Block reserved ranges
                if ip.is_reserved:
                    return False

            except ValueError:
                # Not an IP address, check for decimal/hex IP notation
                # Decimal: 2130706433 = 127.0.0.1
                # Hex: 0x7f000001 = 127.0.0.1
                if re.match(r"^(0x[0-9a-fA-F]+|\d{8,})$", hostname):
                    return False

            return True

        except Exception:
            # Malformed URL
            return False

    def _generate_document_id(self, url: str) -> str:
        """Generate deterministic UUID from URL.

        Creates a deterministic document ID by hashing the URL. This ensures
        that the same URL always gets the same document ID, enabling idempotent
        re-ingestion and deduplication across multiple crawls.

        The implementation follows the pattern from vector_store.py:
        1. Compute SHA256 hash of URL (UTF-8 encoded)
        2. Take first 16 bytes of hash (128-bit collision resistance)
        3. Create UUID from bytes
        4. Return string representation

        Args:
            url: URL to hash. Must be a non-empty string. Same URL always
                produces same UUID (deterministic).

        Returns:
            UUID string derived from SHA256 hash of URL. Format is standard
            UUID with hyphens (e.g., "550e8400-e29b-41d4-a716-446655440000").

        Examples:
            Generate document ID for URL:
                >>> reader = Crawl4AIReader()
                >>> doc_id = reader._generate_document_id("https://example.com")
                >>> assert isinstance(doc_id, str)

            Deterministic property (same input → same output):
                >>> url = "https://example.com/page"
                >>> id1 = reader._generate_document_id(url)
                >>> id2 = reader._generate_document_id(url)
                >>> assert id1 == id2

            Different URLs produce different IDs:
                >>> id1 = reader._generate_document_id("https://example.com/page1")
                >>> id2 = reader._generate_document_id("https://example.com/page2")
                >>> assert id1 != id2

        Notes:
            - Uses SHA256 for cryptographic-quality hash
            - Converts hash to UUID format for LlamaIndex compatibility
            - Same inputs always produce same UUID (deterministic)
            - Pattern matches vector_store.py::_generate_point_id() for consistency
            - Enables idempotent upsert behavior when documents are re-crawled
            - When combined with chunking, each chunk gets deterministic point ID
            - Prevents duplicate documents in vector database across crawl runs
        """
        # Create deterministic hash from URL
        hash_bytes = hashlib.sha256(url.encode()).digest()
        # Convert first 16 bytes to UUID (128-bit collision resistance)
        return str(uuid.UUID(bytes=hash_bytes[:16]))

    def _count_links(self, links: dict) -> tuple[int, int]:
        """Count internal and external links from CrawlResult links structure.

        Args:
            links: Links dictionary from CrawlResult (with "internal" and
                "external" keys)

        Returns:
            Tuple of (internal_count, external_count)
        """
        internal_count = len(links.get("internal", []))
        external_count = len(links.get("external", []))
        return internal_count, external_count

    def _build_metadata(self, crawl_result: dict, url: str) -> dict:
        """Build comprehensive metadata from CrawlResult.

        Extracts metadata fields from Crawl4AI response and enforces
        Qdrant compatibility (flat types only: str, int, float).

        Args:
            crawl_result: Parsed CrawlResult JSON from Crawl4AI
            url: Source URL

        Returns:
            Metadata dictionary with flat types only
        """
        # Extract page metadata from CrawlResult
        page_metadata = crawl_result.get("metadata", {}) or {}
        links = crawl_result.get("links", {}) or {}

        # Count links using helper function
        internal_count, external_count = self._count_links(links)

        # Build flat metadata structure with explicit defaults
        # Use 'or' operator to handle None/empty values naturally
        metadata = {
            "source": url,  # Always present from function arg
            "source_url": url,  # Same as source (indexed for deduplication queries)
            "title": page_metadata.get("title") or "",  # str default
            "description": page_metadata.get("description") or "",  # str default
            "status_code": crawl_result.get("status_code") or 0,  # int default
            "crawl_timestamp": crawl_result.get("crawl_timestamp") or "",  # str default
            "internal_links_count": internal_count,  # Count from helper
            "external_links_count": external_count,  # Count from helper
            "source_type": "web_crawl",  # Always present
        }

        return metadata

    async def _deduplicate_url(self, url: str) -> None:
        """Delete existing documents for URL before ingesting new crawl.

        This method prevents duplicate documents in the vector store by deleting
        all existing chunks for a URL before crawling it again. This matches the
        file watcher behavior and ensures each URL has only one version in the
        vector database at any time.

        The method queries Qdrant by the source_url metadata field and deletes
        all matching points (all chunks for the URL). If no vector store is
        configured, the method returns early with no action.

        Args:
            url: Source URL to deduplicate. This is matched against the
                source_url metadata field in Qdrant to find all chunks
                from previous crawls of this URL.

        Returns:
            None. Side effects: deletes points from Qdrant and logs results.

        Examples:
            Delete old versions before crawling:
                >>> reader = Crawl4AIReader(
                ...     enable_deduplication=True,
                ...     vector_store=vector_store_instance
                ... )
                >>> await reader._deduplicate_url("https://example.com/page")
                # Logs: "Deleted 5 old vectors for https://example.com/page"

            No-op when vector store not configured:
                >>> reader = Crawl4AIReader(enable_deduplication=True)
                >>> await reader._deduplicate_url("https://example.com/page")
                # Returns immediately, no deletion

        Notes:
            - Requires enable_deduplication=True (default)
            - Requires vector_store to be set (not None)
            - Deletes all points matching source_url metadata field
            - Uses structured logging with url and deleted_count
            - Called automatically in aload_data() before crawling
            - Prevents duplicate documents across multiple crawl runs
            - Matches file watcher behavior for consistency
        """
        if self.vector_store is None:
            return  # No deduplication if vector store not configured

        deleted_count = self.vector_store.delete_by_url(url)
        self._logger.info(
            f"Deleted {deleted_count} old vectors for {url}",
            extra={"url": url, "deleted_count": deleted_count},
        )

    async def _crawl_single_url(
        self, client: httpx.AsyncClient, url: str
    ) -> Document | None:
        """Crawl a single URL with circuit breaker and retry logic.

        This method wraps the HTTP request with:
        1. Circuit breaker protection (prevents cascading failures)
        2. Exponential backoff retry (handles transient errors)
        3. Error logging (structured logging for observability)

        Args:
            client: Shared httpx AsyncClient for connection pooling
            url: URL to crawl

        Returns:
            Document object on success, None on failure (when fail_on_error=False)

        Raises:
            Exception: On failure when fail_on_error=True
        """

        async def _crawl_impl() -> Document:
            """Internal implementation with retry logic."""
            for attempt in range(self.max_retries + 1):
                try:
                    # Make request to Crawl4AI /crawl endpoint (using shared client)
                    response = await client.post(
                        f"{self.endpoint_url}/crawl",
                        json={"urls": [url]},  # API expects urls array
                    )

                    # Check HTTP status
                    response.raise_for_status()

                    # Parse response
                    response_data = response.json()

                    # Check batch success
                    if not response_data.get("success", False):
                        error_msg = response_data.get("error_message", "Unknown error")
                        raise RuntimeError(f"Crawl failed for {url}: {error_msg}")

                    # Extract first result (single URL batch)
                    results = response_data.get("results", [])
                    if not results:
                        raise ValueError(f"No results returned for {url}")

                    crawl_result = results[0]

                    # Check individual crawl success
                    if not crawl_result.get("success", False):
                        error_msg = crawl_result.get("error_message", "Unknown error")
                        raise RuntimeError(f"Crawl failed for {url}: {error_msg}")

                    # Extract markdown content
                    markdown_data = crawl_result.get("markdown", {})
                    if isinstance(markdown_data, dict):
                        # Prefer fit_markdown (pre-filtered), fallback to raw_markdown
                        text = markdown_data.get("fit_markdown") or markdown_data.get(
                            "raw_markdown", ""
                        )
                    else:
                        # Markdown data is string (older API version)
                        text = markdown_data or ""

                    if not text:
                        raise ValueError(
                            f"No markdown content in response for {url}"
                        )

                    # Build metadata
                    metadata = self._build_metadata(crawl_result, url)

                    # Generate deterministic ID
                    doc_id = self._generate_document_id(url)

                    # Create Document
                    return Document(text=text, metadata=metadata, id_=doc_id)

                except (
                    httpx.TimeoutException,
                    httpx.NetworkError,
                    httpx.ConnectError,
                ) as e:
                    # Transient errors - retry with backoff
                    if attempt < self.max_retries:
                        delay = self.retry_delays[
                            min(attempt, len(self.retry_delays) - 1)
                        ]
                        self._logger.warning(
                            f"Crawl attempt {attempt + 1} failed for {url}, "
                            f"retrying in {delay}s",
                            extra={
                                "url": url,
                                "attempt": attempt + 1,
                                "error": str(e),
                                "delay": delay,
                            },
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Max retries exhausted
                        self._logger.error(
                            f"Crawl failed after {self.max_retries + 1} attempts",
                            extra={"url": url, "error": str(e)},
                        )
                        raise

                except httpx.HTTPStatusError as e:
                    # HTTP errors (4xx, 5xx) - do not retry 4xx
                    if e.response.status_code >= 500 and attempt < self.max_retries:
                        # Retry 5xx errors
                        delay = self.retry_delays[
                            min(attempt, len(self.retry_delays) - 1)
                        ]
                        self._logger.warning(
                            f"Server error {e.response.status_code} for {url}, "
                            f"retrying in {delay}s",
                            extra={
                                "url": url,
                                "status_code": e.response.status_code,
                                "attempt": attempt + 1,
                                "delay": delay,
                            },
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # 4xx errors or max retries exhausted
                        self._logger.error(
                            f"HTTP error {e.response.status_code} for {url}",
                            extra={"url": url, "status_code": e.response.status_code},
                        )
                        raise

            # Should never reach here
            raise RuntimeError(f"Failed to crawl {url} after all retries")

        # Wrap with circuit breaker
        try:
            result = await self._circuit_breaker.call(_crawl_impl)

            # Log circuit breaker state after successful call
            if self._circuit_breaker.state == "open":
                self._logger.warning(
                    "Circuit breaker opened after failures",
                    extra={
                        "url": url,
                        "failures": self._circuit_breaker.failure_count,
                        "state": self._circuit_breaker.state,
                    },
                )

            return result
        except Exception as e:
            # Log circuit breaker state on failure
            if self._circuit_breaker.state == "open":
                self._logger.error(
                    "Circuit breaker open, rejecting request",
                    extra={"url": url, "state": "open"},
                )

            if self.fail_on_error:
                raise
            else:
                self._logger.error(
                    f"Skipping failed URL {url}", extra={"url": url, "error": str(e)}
                )
                return None

    async def _aload_batch(
        self, urls: list[str]
    ) -> list[Document | None]:
        """Internal batch loader that preserves order and returns None for failures.

        See aload_data() for public API documentation.
        """
        if not urls:
            return []

        # Validate service health before batch processing
        if not await self._validate_health():
            raise RuntimeError(
                f"Crawl4AI service unhealthy at {self.endpoint_url}/health"
            )

        self._logger.info(
            f"Starting batch crawl of {len(urls)} URLs",
            extra={
                "url_count": len(urls),
                "max_concurrent": self.max_concurrent_requests,
            },
        )

        # Deduplicate each URL before crawling (if enabled and vector_store configured)
        if self.enable_deduplication and self.vector_store is not None:
            for url in urls:
                await self._deduplicate_url(url)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        # Shared AsyncClient for connection pooling across all URLs
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:

            async def crawl_with_semaphore(url: str) -> Document | None:
                """Wrapper to enforce concurrency limit via semaphore."""
                async with semaphore:
                    return await self._crawl_single_url(client, url)

            # Process URLs concurrently, preserving order
            raw_results = await asyncio.gather(
                *[crawl_with_semaphore(url) for url in urls],
                return_exceptions=not self.fail_on_error,
            )

        # Convert exceptions to None, filter to Document | None
        results: list[Document | None] = [
            r if isinstance(r, Document) else None for r in raw_results
        ]

        # Count successes
        success_count = sum(1 for r in results if r is not None)
        failure_count = len(urls) - success_count

        # Log batch statistics
        self._logger.info(
            f"Batch crawl complete: {success_count} succeeded, {failure_count} failed",
            extra={
                "total": len(urls),
                "succeeded": success_count,
                "failed": failure_count,
            },
        )

        return results

    async def aload_data_with_results(self, urls: list[str]) -> list[Document | None]:
        """Load documents asynchronously, preserving order and returning None for failures.

        Use this method when you need to know which specific URLs failed.
        """
        return await self._aload_batch(urls)

    async def aload_data(  # type: ignore[override]
        self, urls: list[str]
    ) -> list[Document]:
        """Load documents asynchronously from URLs.

        This is the primary async method for loading web content. It processes
        URLs concurrently and returns only successfully crawled Documents.
        Failed URLs are filtered out.

        Args:
            urls: List of URLs to crawl.

        Returns:
            List of successfully crawled Document objects.
        """
        results = await self._aload_batch(urls)
        return [doc for doc in results if doc is not None]

    def load_data(self, urls: list[str]) -> list[Document]:
        """Load documents synchronously from URLs.

        Returns only successfully crawled documents (filters out failures).
        """
        try:
            asyncio.get_running_loop()
            # If we get here, a loop is running
            raise RuntimeError(
                "Cannot use sync load_data() inside running event loop. "
                "Use await aload_data() instead."
            )
        except RuntimeError as e:
            if "no running event loop" not in str(e):
                raise
            # No running loop - safe to proceed with asyncio.run()
        
        results = asyncio.run(self.aload_data(urls))
        return results

    def load_data_with_errors(self, urls: list[str]) -> list[Document | None]:
        """Load documents with error tracking (order-preserving).

        Returns list matching input length, with None for failed URLs.
        """
        try:
            asyncio.get_running_loop()
            # If we get here, a loop is running
            raise RuntimeError(
                "Cannot use sync load_data_with_errors() inside running event loop. "
                "Use await aload_data_with_results() instead."
            )
        except RuntimeError as e:
            if "no running event loop" not in str(e):
                raise
            # No running loop - safe to proceed with asyncio.run()
            
        return asyncio.run(self.aload_data_with_results(urls))

    async def alazy_load_data(  # type: ignore[override]
        self, urls: list[str]
    ) -> AsyncIterator[Document]:
        """Lazily load documents from URLs as an async generator.

        This method yields documents as they are successfully crawled,
        enabling memory-efficient processing of large URL lists. Failed
        URLs are silently skipped (not yielded) when fail_on_error=False.

        Unlike aload_data() which returns a list of all documents at once,
        this method yields documents one at a time as they become available,
        reducing peak memory usage when processing many URLs.

        Args:
            urls: List of URLs to crawl. Each URL is processed sequentially
                and yielded immediately upon successful crawl.

        Yields:
            Document objects for successfully crawled URLs. Failed URLs are
            skipped unless fail_on_error=True, in which case an exception
            is raised immediately.

        Raises:
            RuntimeError: If Crawl4AI service is unhealthy before iteration
            Exception: On crawl failure when fail_on_error=True

        Examples:
            Memory-efficient async processing:
                >>> reader = Crawl4AIReader()
                >>> async for doc in reader.alazy_load_data(urls):
                ...     await process_document(doc)  # Process one at a time

            Streaming to external storage:
                >>> async for doc in reader.alazy_load_data(urls):
                ...     await vector_store.add(doc)  # Stream to storage

        Notes:
            - Uses sequential processing (one URL at a time) for predictable
              memory usage, unlike aload_data() which processes concurrently
            - Deduplication is NOT performed in lazy mode (caller's responsibility)
            - Circuit breaker protection is applied per-URL
            - Retry logic with exponential backoff is applied per-URL
            - For concurrent processing, use aload_data() instead
        """
        if not urls:
            return

        # Validate service health before starting lazy iteration
        if not await self._validate_health():
            raise RuntimeError(
                f"Crawl4AI service unhealthy at {self.endpoint_url}/health"
            )

        # Use shared client for connection pooling
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for url in urls:
                try:
                    result = await self._crawl_single_url(client, url)
                    if result is not None:
                        yield result
                except Exception as e:
                    self._logger.warning(
                        f"Lazy load failed for {url}: {e}",
                        extra={"url": url, "error": str(e)},
                    )
                    if self.fail_on_error:
                        raise
                    # Skip failed URLs in lazy mode (don't yield None)
                    continue

    def lazy_load_data(self, urls: list[str]) -> Iterable[Document]:
        """Lazily load documents synchronously (wrapper over async).

        This provides a synchronous interface by collecting results from
        the async generator. While this collects all results before returning,
        it still benefits from the sequential processing of the async version.

        For true memory efficiency in sync contexts, consider using
        alazy_load_data() directly with an event loop.

        Args:
            urls: List of URLs to crawl

        Returns:
            Iterable of Document objects for successfully crawled URLs.
            Failed URLs are excluded (not represented as None).

        Examples:
            Synchronous lazy loading:
                >>> reader = Crawl4AIReader()
                >>> for doc in reader.lazy_load_data(urls):
                ...     process_document(doc)

        Notes:
            - Wraps alazy_load_data() with asyncio.run()
            - Collects all documents into a list before returning
            - For true streaming behavior, use alazy_load_data() with asyncio
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot use sync lazy_load_data() inside running event loop. "
                "Use async for doc in alazy_load_data() instead."
            )
        except RuntimeError as e:
            if "running event loop" in str(e):
                raise
            # No running loop - safe to proceed with asyncio.run()
        return asyncio.run(self._collect_lazy_results(urls))

    async def _collect_lazy_results(self, urls: list[str]) -> list[Document]:
        """Collect results from async generator into a list.

        Internal helper method to convert async generator results
        to a list for the synchronous lazy_load_data wrapper.

        Args:
            urls: List of URLs to crawl

        Returns:
            List of successfully crawled Documents
        """
        return [doc async for doc in self.alazy_load_data(urls)]
