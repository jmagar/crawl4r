"""TEI (Text Embeddings Inference) client for generating embeddings.

This module provides an async HTTP client for the TEI service running
Qwen3-Embedding-0.6B. It handles:
- Single and batch text embedding generation
- Connection error handling with exponential backoff retries
- Timeout error handling with retries
- Invalid response validation
- Embedding dimension validation
- Batch size limit enforcement
- Circuit breaker pattern for fault tolerance and preventing cascading failures

Circuit Breaker Integration:
    The TEIClient integrates a circuit breaker to protect against cascading
    failures when the TEI service becomes unavailable. The circuit breaker
    monitors consecutive failures and can temporarily reject requests (fail-fast)
    to allow the service time to recover.

    - CLOSED state: Normal operation, all requests allowed
    - OPEN state: Service failing, requests rejected immediately
    - HALF_OPEN state: Testing recovery, limited requests allowed

    Configuration:
        circuit_breaker_threshold: Failures before opening (default: 5)
        circuit_breaker_timeout: Seconds before recovery test (default: 60.0)

Example:
    >>> client = TEIClient("http://crawl4r-embeddings:80")
    >>> embedding = await client.embed_single("Hello world")
    >>> assert len(embedding) == 1024
    >>> embeddings = await client.embed_batch(["text1", "text2"])
    >>> assert len(embeddings) == 2
"""

import asyncio

import httpx

from crawl4r.resilience.circuit_breaker import CircuitBreaker


class TEIClient:
    """Async client for Text Embeddings Inference (TEI) service.

    This client communicates with a TEI server to generate text embeddings using
    the Qwen3-Embedding-0.6B model. It provides both single and batch embedding
    generation with comprehensive error handling and validation.

    Circuit Breaker Protection:
        The client integrates a circuit breaker pattern to prevent cascading failures.
        When the TEI service experiences consecutive failures (network errors, timeouts,
        HTTP errors), the circuit breaker transitions through states:

        1. CLOSED: Normal operation. All requests proceed with retry logic.
        2. OPEN: After reaching failure_threshold consecutive failures, the circuit
           opens and immediately rejects new requests (fail-fast) for reset_timeout
           seconds. This prevents overwhelming a failing service.
        3. HALF_OPEN: After reset_timeout expires, the circuit allows one test request.
           If successful, transitions to CLOSED. If failed, returns to OPEN.

        The circuit breaker operates at the method level, wrapping embed_single()
        and embed_batch() calls. Retry logic (exponential backoff) runs within
        the circuit breaker protection.

    Attributes:
        endpoint_url: Base URL of the TEI service
        expected_dimensions: Expected dimension count for embeddings (default: 1024)
        timeout: HTTP request timeout in seconds (default: 30.0)
        max_retries: Maximum number of retry attempts for failures (default: 3)
        batch_size_limit: Maximum number of texts in a batch request (default: 100)
        circuit_breaker: Circuit breaker instance for fault tolerance

    Example:
        >>> client = TEIClient(
        ...     "http://crawl4r-embeddings:80",
        ...     dimensions=1024,
        ...     circuit_breaker_threshold=5,
        ...     circuit_breaker_timeout=60.0
        ... )
        >>> embedding = await client.embed_single("Sample text")
        >>> assert len(embedding) == 1024
    """

    def __init__(
        self,
        endpoint_url: str,
        dimensions: int = 1024,
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size_limit: int = 100,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
    ) -> None:
        """Initialize TEI client with endpoint and configuration.

        Args:
            endpoint_url: Base URL of the TEI service (e.g., "http://crawl4r-embeddings:80")
            dimensions: Expected dimension count for embeddings (default: 1024)
            timeout: HTTP request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retry attempts for failures (default: 3)
            batch_size_limit: Maximum number of texts in a batch request (default: 100)
            circuit_breaker_threshold: Number of consecutive failures before opening
                circuit (default: 5)
            circuit_breaker_timeout: Seconds to wait before attempting recovery
                (default: 60.0)

        Raises:
            ValueError: If endpoint_url is invalid or batch_size_limit is not positive

        Example:
            >>> client = TEIClient("http://crawl4r-embeddings:80", dimensions=512)
            >>> assert client.expected_dimensions == 512
        """
        # Validate endpoint URL format
        if not endpoint_url or not endpoint_url.startswith(("http://", "https://")):
            raise ValueError("Invalid endpoint URL")

        # Validate batch size limit is positive
        if batch_size_limit <= 0:
            raise ValueError("Batch size limit must be positive")

        self.endpoint_url = endpoint_url.rstrip("/")
        self.expected_dimensions = dimensions
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size_limit = batch_size_limit

        # Initialize circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_timeout,
        )

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed (cannot be empty)

        Returns:
            List of floats representing the embedding vector with expected dimensions

        Raises:
            ValueError: If text is empty, response is invalid, or dimensions don't match
            httpx.ConnectError: If connection to TEI service fails after retries
            httpx.TimeoutException: If request times out after retries
            httpx.HTTPStatusError: If TEI service returns HTTP error status
            CircuitBreakerError: If circuit breaker is OPEN and rejects the call

        Example:
            >>> embedding = await client.embed_single("Hello world")
            >>> assert len(embedding) == 1024
            >>> assert all(isinstance(x, float) for x in embedding)
        """
        # Validate input before checking circuit breaker
        if not text:
            raise ValueError("Text cannot be empty")

        # Wrap the actual implementation with circuit breaker
        async def _impl() -> list[float]:
            return await self._embed_single_impl(text)

        return await self.circuit_breaker.call(_impl)

    async def _embed_single_impl(self, text: str) -> list[float]:
        """Internal implementation of embed_single without circuit breaker.

        This method contains the actual HTTP request logic with retry handling.
        It is called by embed_single() after the circuit breaker check passes.
        All exceptions raised by this method will be caught by the circuit breaker,
        which will record the failure and potentially open the circuit.

        Retry Logic:
            On network/connection/timeout errors, retries up to max_retries times
            with exponential backoff (1s, 2s, 4s). Other errors (validation,
            HTTP status) fail immediately without retries.

        Args:
            text: Text to embed (pre-validated)

        Returns:
            List of floats representing the embedding vector

        Raises:
            ValueError: If response is invalid or dimensions don't match
            httpx.ConnectError: If connection to TEI service fails after retries
            httpx.TimeoutException: If request times out after retries
            httpx.HTTPStatusError: If TEI service returns HTTP error status
        """
        # Prepare request payload
        payload = {"inputs": [text]}

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.endpoint_url}/embed",
                        json=payload,
                    )

                    # Check for HTTP errors
                    response.raise_for_status()

                    # Parse response JSON
                    try:
                        data = response.json()
                    except ValueError as e:
                        raise ValueError("Invalid JSON") from e

                    # Validate response structure
                    if not isinstance(data, list) or not data:
                        raise ValueError("Invalid response structure")

                    # Extract embedding from response
                    # TEI returns [embedding] for single input
                    if not isinstance(data[0], list) or not data[0]:
                        raise ValueError("Invalid response structure")

                    embedding = data[0]

                    # Validate dimensions
                    if len(embedding) != self.expected_dimensions:
                        raise ValueError(
                            f"Expected {self.expected_dimensions} dimensions, "
                            f"got {len(embedding)}"
                        )

                    return embedding

            except (httpx.ConnectError, httpx.NetworkError, httpx.TimeoutException):
                # Retry on network/connection/timeout errors
                if attempt == self.max_retries - 1:
                    # Last attempt failed, re-raise
                    raise
                # Exponential backoff: 1s, 2s, 4s
                await asyncio.sleep(2**attempt)
                continue

        # This should never be reached, but satisfies type checker
        raise RuntimeError("Unexpected error in _embed_single_impl")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed (cannot be empty)

        Returns:
            List of embedding vectors, one per input text

        Raises:
            ValueError: If texts is empty, exceeds batch size limit,
                       response is invalid, dimensions don't match, or
                       response count doesn't match request count
            httpx.ConnectError: If connection to TEI service fails after retries
            httpx.TimeoutException: If request times out after retries
            httpx.HTTPStatusError: If TEI service returns HTTP error status
            CircuitBreakerError: If circuit breaker is OPEN and rejects the call

        Example:
            >>> embeddings = await client.embed_batch(["text1", "text2", "text3"])
            >>> assert len(embeddings) == 3
            >>> assert all(len(emb) == 1024 for emb in embeddings)
        """
        # Validate input before checking circuit breaker
        if not texts:
            raise ValueError("Batch cannot be empty")

        if len(texts) > self.batch_size_limit:
            raise ValueError(f"Batch size exceeds limit of {self.batch_size_limit}")

        # Wrap the actual implementation with circuit breaker
        async def _impl() -> list[list[float]]:
            return await self._embed_batch_impl(texts)

        return await self.circuit_breaker.call(_impl)

    async def _embed_batch_impl(self, texts: list[str]) -> list[list[float]]:
        """Internal implementation of embed_batch without circuit breaker.

        This method contains the actual HTTP request logic with retry handling.
        It is called by embed_batch() after the circuit breaker check passes.
        All exceptions raised by this method will be caught by the circuit breaker,
        which will record the failure and potentially open the circuit.

        Retry Logic:
            On network/connection/timeout errors, retries up to max_retries times
            with exponential backoff (1s, 2s, 4s). Other errors (validation,
            HTTP status) fail immediately without retries.

        Args:
            texts: List of texts to embed (pre-validated)

        Returns:
            List of embedding vectors, one per input text

        Raises:
            ValueError: If response is invalid, dimensions don't match, or
                       response count doesn't match request count
            httpx.ConnectError: If connection to TEI service fails after retries
            httpx.TimeoutException: If request times out after retries
            httpx.HTTPStatusError: If TEI service returns HTTP error status
        """
        # Prepare request payload
        payload = {"inputs": texts}

        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.endpoint_url}/embed",
                        json=payload,
                    )

                    # Check for HTTP errors
                    response.raise_for_status()

                    # Parse response JSON
                    try:
                        data = response.json()
                    except ValueError as e:
                        raise ValueError("Invalid JSON") from e

                    # Validate response structure
                    if not isinstance(data, list) or not data:
                        raise ValueError("Invalid response structure")

                    # Extract embeddings from response
                    # TEI returns [embedding1, embedding2, ...] for batch
                    embeddings = data

                    # Validate count matches request
                    if len(embeddings) != len(texts):
                        raise ValueError("Response count does not match request count")

                    # Validate dimensions for all embeddings
                    for embedding in embeddings:
                        if not isinstance(embedding, list):
                            raise ValueError("Invalid response structure")

                        if len(embedding) != self.expected_dimensions:
                            raise ValueError(
                                f"Expected {self.expected_dimensions} dimensions, "
                                f"got {len(embedding)}"
                            )

                    return embeddings

            except (httpx.ConnectError, httpx.NetworkError, httpx.TimeoutException):
                # Retry on network/connection/timeout errors
                if attempt == self.max_retries - 1:
                    # Last attempt failed, re-raise
                    raise
                # Exponential backoff: 1s, 2s, 4s
                await asyncio.sleep(2**attempt)
                continue

        # This should never be reached, but satisfies type checker
        raise RuntimeError("Unexpected error in _embed_batch_impl")
