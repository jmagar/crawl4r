"""Extractor service for structured data extraction using Crawl4AI.

This module provides structured data extraction functionality using the Crawl4AI
/llm/job endpoint. It supports both JSON schema-based extraction and natural
language prompt-based extraction.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from crawl4r.core.url_validation import validate_url
from crawl4r.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerError
from crawl4r.services.models import ExtractResult


class ExtractorService:
    """Service for extracting structured data from web pages using LLMs.

    The ExtractorService leverages Crawl4AI's /llm/job endpoint to extract
    structured data from web pages. It supports two extraction modes:

    1. Schema-based extraction: Provide a JSON schema defining the expected
       structure, and the LLM extracts data matching that schema.

    2. Prompt-based extraction: Provide a natural language prompt describing
       what to extract, and the LLM returns the relevant data.

    Attributes:
        endpoint_url: Base URL for the Crawl4AI service.

    Example:
        >>> service = ExtractorService(endpoint_url="http://localhost:52004")
        >>> result = await service.extract_with_schema(
        ...     "https://example.com/product",
        ...     schema={"type": "object", "properties": {"title": {"type": "string"}}}
        ... )
        >>> print(result.data)
        {'title': 'Example Product'}
    """

    def __init__(
        self,
        endpoint_url: str,
        timeout: float = 60.0,
        health_endpoint: str = "/health",
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        default_provider: str | None = None,
    ) -> None:
        """Initialize the extractor service.

        Args:
            endpoint_url: Base URL for the Crawl4AI service.
            timeout: Request timeout in seconds.
            health_endpoint: Path to health check endpoint.
            circuit_breaker_threshold: Failures before opening circuit.
            circuit_breaker_timeout: Seconds before allowing recovery.
            default_provider: Default LLM provider (e.g., 'openai/gpt-4o-mini').
        """
        self._endpoint_url = endpoint_url.rstrip("/")
        self._health_endpoint = health_endpoint
        self._client = httpx.AsyncClient(base_url=self._endpoint_url, timeout=timeout)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_timeout,
        )
        self._default_provider = default_provider

    async def extract_with_schema(
        self,
        url: str,
        schema: dict[str, Any],
        provider: str | None = None,
    ) -> ExtractResult:
        """Extract structured data from a URL using a JSON schema.

        Sends the URL and JSON schema to the Crawl4AI /llm/job endpoint,
        which uses an LLM to extract data matching the schema structure.

        Args:
            url: URL to extract data from.
            schema: JSON schema defining the expected data structure.
            provider: LLM provider to use (e.g., 'openai/gpt-4o-mini').
                     Uses default_provider if not specified.

        Returns:
            ExtractResult with extracted data or error details.

        Example:
            >>> result = await service.extract_with_schema(
            ...     "https://example.com/product",
            ...     schema={
            ...         "type": "object",
            ...         "properties": {
            ...             "title": {"type": "string"},
            ...             "price": {"type": "number"}
            ...         }
            ...     }
            ... )
            >>> print(result.data)
            {'title': 'Widget', 'price': 29.99}
        """
        # Validate URL before processing
        if not validate_url(url):
            return ExtractResult(
                url=url,
                success=False,
                error="Invalid URL",
                extraction_method="schema",
            )

        # Validate schema is not empty
        if not schema:
            return ExtractResult(
                url=url,
                success=False,
                error="Schema cannot be empty",
                extraction_method="schema",
            )

        try:
            result = await self._circuit_breaker.call(
                lambda: self._fetch_extraction(
                    url=url,
                    schema=schema,
                    prompt=None,
                    provider=provider or self._default_provider,
                )
            )
            return result
        except CircuitBreakerError as exc:
            return ExtractResult(
                url=url,
                success=False,
                error=str(exc),
                extraction_method="schema",
            )
        except Exception as exc:  # noqa: BLE001
            return ExtractResult(
                url=url,
                success=False,
                error=str(exc),
                extraction_method="schema",
            )

    async def extract_with_prompt(
        self,
        url: str,
        prompt: str,
        provider: str | None = None,
    ) -> ExtractResult:
        """Extract data from a URL using a natural language prompt.

        Sends the URL and prompt to the Crawl4AI /llm/job endpoint,
        which uses an LLM to extract data based on the prompt instructions.

        Args:
            url: URL to extract data from.
            prompt: Natural language description of what to extract.
            provider: LLM provider to use (e.g., 'openai/gpt-4o-mini').
                     Uses default_provider if not specified.

        Returns:
            ExtractResult with extracted data or error details.

        Example:
            >>> result = await service.extract_with_prompt(
            ...     "https://example.com/products",
            ...     prompt="Extract all product names and their prices as a list"
            ... )
            >>> print(result.data)
            {'products': [{'name': 'Widget', 'price': 29.99}]}
        """
        # Validate URL before processing
        if not validate_url(url):
            return ExtractResult(
                url=url,
                success=False,
                error="Invalid URL",
                extraction_method="prompt",
            )

        # Validate prompt is not empty
        if not prompt or not prompt.strip():
            return ExtractResult(
                url=url,
                success=False,
                error="Prompt cannot be empty",
                extraction_method="prompt",
            )

        try:
            result = await self._circuit_breaker.call(
                lambda: self._fetch_extraction(
                    url=url,
                    schema=None,
                    prompt=prompt,
                    provider=provider or self._default_provider,
                )
            )
            return result
        except CircuitBreakerError as exc:
            return ExtractResult(
                url=url,
                success=False,
                error=str(exc),
                extraction_method="prompt",
            )
        except Exception as exc:  # noqa: BLE001
            return ExtractResult(
                url=url,
                success=False,
                error=str(exc),
                extraction_method="prompt",
            )

    async def extract_batch(
        self,
        urls: list[str],
        schema: dict[str, Any] | None = None,
        prompt: str | None = None,
        provider: str | None = None,
        max_concurrent: int = 5,
    ) -> list[ExtractResult]:
        """Extract structured data from multiple URLs.

        Processes multiple URLs concurrently with a configurable concurrency limit.
        Each URL is extracted independently, and partial failures do not affect
        other URLs in the batch.

        Args:
            urls: List of URLs to extract data from.
            schema: JSON schema for extraction (mutually exclusive with prompt).
            prompt: Natural language prompt (mutually exclusive with schema).
            provider: LLM provider to use.
            max_concurrent: Maximum concurrent extraction requests.

        Returns:
            List of ExtractResult for each URL, in the same order as input.

        Example:
            >>> results = await service.extract_batch(
            ...     urls=["https://example.com/1", "https://example.com/2"],
            ...     schema={"type": "object"}
            ... )
            >>> print(len(results))
            2
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded_extract(target_url: str) -> ExtractResult:
            async with semaphore:
                if schema is not None:
                    return await self.extract_with_schema(
                        target_url, schema=schema, provider=provider
                    )
                elif prompt is not None:
                    return await self.extract_with_prompt(
                        target_url, prompt=prompt, provider=provider
                    )
                else:
                    return ExtractResult(
                        url=target_url,
                        success=False,
                        error="Either schema or prompt must be provided",
                    )

        tasks = [asyncio.create_task(_bounded_extract(url)) for url in urls]
        return await asyncio.gather(*tasks)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> ExtractorService:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager and cleanup resources."""
        await self.close()

    async def _fetch_extraction(
        self,
        url: str,
        schema: dict[str, Any] | None,
        prompt: str | None,
        provider: str | None,
    ) -> ExtractResult:
        """Fetch extraction results from Crawl4AI /llm/job endpoint.

        Args:
            url: URL to extract data from.
            schema: JSON schema for extraction (optional).
            prompt: Natural language prompt (optional).
            provider: LLM provider to use (optional).

        Returns:
            ExtractResult with extracted data or error details.
        """
        extraction_method = "schema" if schema is not None else "prompt"

        # Build request payload
        payload: dict[str, Any] = {"url": url}
        if schema is not None:
            payload["schema"] = schema
        if prompt is not None:
            payload["prompt"] = prompt
            payload["instruction"] = prompt  # Some APIs use 'instruction'
        if provider is not None:
            payload["provider"] = provider

        backoff_seconds = [1.0, 2.0, 4.0]

        for attempt in range(len(backoff_seconds) + 1):
            try:
                response = await self._client.post("/llm/job", json=payload)

                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        "Server error from Crawl4AI",
                        request=response.request,
                        response=response,
                    )

                if response.status_code >= 400:
                    # Client error, no retry
                    return ExtractResult(
                        url=url,
                        success=False,
                        error=f"Request failed with status {response.status_code}",
                        extraction_method=extraction_method,
                    )

                return self._parse_response(url, response, extraction_method)

            except httpx.TimeoutException:
                if attempt >= len(backoff_seconds):
                    return ExtractResult(
                        url=url,
                        success=False,
                        error="Request timeout during extraction",
                        extraction_method=extraction_method,
                    )
                await asyncio.sleep(backoff_seconds[attempt])

            except httpx.ConnectError:
                if attempt >= len(backoff_seconds):
                    return ExtractResult(
                        url=url,
                        success=False,
                        error="Connection error during extraction",
                        extraction_method=extraction_method,
                    )
                await asyncio.sleep(backoff_seconds[attempt])

            except httpx.RequestError:
                if attempt >= len(backoff_seconds):
                    return ExtractResult(
                        url=url,
                        success=False,
                        error="Request error during extraction",
                        extraction_method=extraction_method,
                    )
                await asyncio.sleep(backoff_seconds[attempt])

            except httpx.HTTPStatusError:
                if attempt >= len(backoff_seconds):
                    return ExtractResult(
                        url=url,
                        success=False,
                        error="Server error during extraction",
                        extraction_method=extraction_method,
                    )
                await asyncio.sleep(backoff_seconds[attempt])

        return ExtractResult(
            url=url,
            success=False,
            error="Unexpected extraction failure",
            extraction_method=extraction_method,
        )

    def _parse_response(
        self,
        url: str,
        response: httpx.Response,
        extraction_method: str,
    ) -> ExtractResult:
        """Parse extraction response from Crawl4AI.

        Args:
            url: Source URL for the extraction.
            response: HTTP response from Crawl4AI.
            extraction_method: Method used ('schema' or 'prompt').

        Returns:
            ExtractResult with parsed data or error details.
        """
        try:
            data = response.json()
        except Exception:
            return ExtractResult(
                url=url,
                success=False,
                error="Invalid JSON response from extraction endpoint",
                extraction_method=extraction_method,
            )

        # Check for error in response body
        if data.get("success") is False:
            return ExtractResult(
                url=url,
                success=False,
                error=data.get("error", "Extraction failed"),
                extraction_method=extraction_method,
            )

        # Extract the data field
        extracted_data = data.get("data")
        if extracted_data is None and "error" in data:
            return ExtractResult(
                url=url,
                success=False,
                error=data.get("error", "No data extracted"),
                extraction_method=extraction_method,
            )

        return ExtractResult(
            url=url,
            success=True,
            data=extracted_data,
            extraction_method=extraction_method,
        )
