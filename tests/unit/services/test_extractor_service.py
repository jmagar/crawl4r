"""Unit tests for ExtractorService structured data extraction functionality.

These tests verify the ExtractorService correctly:
- Extracts structured data using JSON schemas via Crawl4AI /llm/job endpoint
- Extracts data using natural language prompts
- Handles LLM provider configuration
- Reports extraction errors and validation failures

This is the RED phase of TDD - tests should fail because
crawl4r.services.extractor module does not yet exist.
"""

import httpx
import pytest
import respx

from crawl4r.services.extractor import ExtractorService

# =============================================================================
# Schema-based extraction tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_extract_with_schema() -> None:
    """Test schema-based extraction from URL.

    This is the primary test for schema-based extraction functionality.
    Verifies that the service correctly extracts structured data from a URL
    using a JSON schema.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(200, json={"data": {"name": "Test"}})
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "https://example.com",
        schema={"type": "object", "properties": {"name": {"type": "string"}}},
    )

    assert result.success is True
    assert result.data["name"] == "Test"


@respx.mock
@pytest.mark.asyncio
async def test_extract_with_schema_calls_llm_job() -> None:
    """Verify extract_with_schema calls /llm/job endpoint with schema.

    The service should POST to /llm/job with the URL and JSON schema,
    returning the extracted structured data.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(200, json={"data": {"title": "Example"}})
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "https://example.com", schema={"type": "object"}
    )

    assert result.success is True
    assert result.data == {"title": "Example"}


@respx.mock
@pytest.mark.asyncio
async def test_extract_with_schema_passes_schema_to_endpoint() -> None:
    """Verify the JSON schema is correctly passed in the request body.

    The /llm/job endpoint should receive the schema for structured extraction.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "price": {"type": "number"},
        },
        "required": ["title"],
    }

    route = respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(
            200, json={"data": {"title": "Product", "price": 29.99}}
        )
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "https://example.com/product", schema=schema
    )

    assert result.success is True
    assert result.data == {"title": "Product", "price": 29.99}
    # Verify schema was sent in request
    assert route.called
    request_body = route.calls.last.request.content
    assert b"schema" in request_body


@respx.mock
@pytest.mark.asyncio
async def test_extract_with_schema_returns_nested_data() -> None:
    """Verify extraction handles nested JSON structures correctly.

    Complex schemas with nested objects should be properly extracted
    and returned in the result.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "product": {
                        "name": "Widget",
                        "details": {"sku": "WGT-001", "weight": 1.5},
                    }
                }
            },
        )
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "https://example.com",
        schema={
            "type": "object",
            "properties": {
                "product": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "details": {"type": "object"},
                    },
                }
            },
        },
    )

    assert result.success is True
    assert result.data["product"]["name"] == "Widget"
    assert result.data["product"]["details"]["sku"] == "WGT-001"


# =============================================================================
# Prompt-based extraction tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_extract_with_prompt_calls_llm_job() -> None:
    """Verify extract_with_prompt calls /llm/job endpoint with prompt.

    The service should POST to /llm/job with the URL and natural language
    prompt, returning the extracted data.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(200, json={"data": {"heading": "Hello"}})
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_prompt(
        "https://example.com", prompt="extract heading"
    )

    assert result.success is True
    assert result.data == {"heading": "Hello"}


@respx.mock
@pytest.mark.asyncio
async def test_extract_with_prompt_passes_prompt_to_endpoint() -> None:
    """Verify the prompt is correctly passed in the request body.

    The /llm/job endpoint should receive the natural language prompt
    for LLM-based extraction.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    prompt = "Extract all product names and their prices as a list"
    route = respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(
            200, json={"data": {"products": [{"name": "A", "price": 10}]}}
        )
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_prompt(
        "https://example.com/products", prompt=prompt
    )

    assert result.success is True
    assert route.called
    request_body = route.calls.last.request.content
    assert b"prompt" in request_body or b"instruction" in request_body


# =============================================================================
# LLM provider configuration tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_extract_with_custom_provider() -> None:
    """Verify custom LLM provider can be specified for extraction.

    The service should allow configuring different LLM providers
    (e.g., openai, anthropic, ollama) for the extraction task.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    route = respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(200, json={"data": {"result": "extracted"}})
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "https://example.com",
        schema={"type": "object"},
        provider="ollama/llama3",
    )

    assert result.success is True
    assert route.called
    request_body = route.calls.last.request.content
    assert b"provider" in request_body or b"ollama" in request_body


@respx.mock
@pytest.mark.asyncio
async def test_extract_uses_default_provider_when_not_specified() -> None:
    """Verify default LLM provider is used when none specified.

    When no provider is explicitly set, the service should use
    a sensible default (configured at service level).
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(200, json={"data": {"result": "ok"}})
    )

    service = ExtractorService(
        endpoint_url="http://localhost:52004",
        default_provider="openai/gpt-4o-mini",
    )
    result = await service.extract_with_schema(
        "https://example.com", schema={"type": "object"}
    )

    assert result.success is True


# =============================================================================
# Error handling tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_extract_returns_failure_on_service_error() -> None:
    """Verify ExtractResult indicates failure when service returns error."""
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(500, json={"error": "LLM service unavailable"})
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "https://example.com", schema={"type": "object"}
    )

    assert result.success is False
    assert result.error is not None


@respx.mock
@pytest.mark.asyncio
async def test_extract_handles_timeout() -> None:
    """Verify ExtractResult indicates failure on request timeout."""
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        side_effect=httpx.TimeoutException("LLM extraction timeout")
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "https://example.com", schema={"type": "object"}
    )

    assert result.success is False
    assert result.error is not None
    assert "timeout" in result.error.lower()


@respx.mock
@pytest.mark.asyncio
async def test_extract_handles_connection_error() -> None:
    """Verify ExtractResult indicates failure on connection error."""
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "https://example.com", schema={"type": "object"}
    )

    assert result.success is False
    assert result.error is not None


@respx.mock
@pytest.mark.asyncio
async def test_extract_handles_invalid_json_response() -> None:
    """Verify ExtractResult handles malformed JSON from endpoint.

    If the /llm/job endpoint returns invalid JSON, the service should
    gracefully handle the error and return a failure result.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(200, content=b"not valid json")
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "https://example.com", schema={"type": "object"}
    )

    assert result.success is False
    assert result.error is not None


@respx.mock
@pytest.mark.asyncio
async def test_extract_handles_llm_extraction_failure() -> None:
    """Verify ExtractResult handles LLM extraction failure response.

    When the LLM fails to extract data matching the schema, the endpoint
    may return a success HTTP status but with an error in the response body.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": False,
                "error": "Failed to extract data matching schema",
                "data": None,
            },
        )
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "https://example.com", schema={"type": "object"}
    )

    assert result.success is False
    assert result.error is not None


# =============================================================================
# URL validation tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_extract_validates_invalid_url() -> None:
    """Verify invalid URLs return failure without making requests."""
    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "not-a-valid-url", schema={"type": "object"}
    )

    assert result.success is False
    assert "Invalid URL" in result.error or "url" in result.error.lower()


@respx.mock
@pytest.mark.asyncio
async def test_extract_validates_empty_schema() -> None:
    """Verify empty schema returns failure.

    A schema must be provided for schema-based extraction.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema("https://example.com", schema={})

    assert result.success is False
    assert result.error is not None


@respx.mock
@pytest.mark.asyncio
async def test_extract_validates_empty_prompt() -> None:
    """Verify empty prompt returns failure.

    A prompt must be provided for prompt-based extraction.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_prompt("https://example.com", prompt="")

    assert result.success is False
    assert result.error is not None


# =============================================================================
# Result metadata tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_extract_result_includes_source_url() -> None:
    """Verify ExtractResult includes the source URL in metadata.

    The result should track which URL was extracted from for traceability.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(200, json={"data": {"title": "Test"}})
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")
    result = await service.extract_with_schema(
        "https://example.com/page", schema={"type": "object"}
    )

    assert result.success is True
    assert result.source_url == "https://example.com/page"


@respx.mock
@pytest.mark.asyncio
async def test_extract_result_includes_extraction_method() -> None:
    """Verify ExtractResult indicates the extraction method used.

    The result should indicate whether schema or prompt extraction was used.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(200, json={"data": {"title": "Test"}})
    )

    service = ExtractorService(endpoint_url="http://localhost:52004")

    schema_result = await service.extract_with_schema(
        "https://example.com", schema={"type": "object"}
    )
    assert schema_result.extraction_method == "schema"

    prompt_result = await service.extract_with_prompt(
        "https://example.com", prompt="extract title"
    )
    assert prompt_result.extraction_method == "prompt"


# =============================================================================
# Batch extraction tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_extract_batch_processes_multiple_urls() -> None:
    """Verify batch extraction processes multiple URLs.

    The service should support extracting from multiple URLs in a single
    call, returning results for each.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    route = respx.post("http://localhost:52004/llm/job")
    route.side_effect = [
        httpx.Response(200, json={"data": {"title": "Page 1"}}),
        httpx.Response(200, json={"data": {"title": "Page 2"}}),
    ]

    service = ExtractorService(endpoint_url="http://localhost:52004")
    results = await service.extract_batch(
        urls=["https://example.com/1", "https://example.com/2"],
        schema={"type": "object"},
    )

    assert len(results) == 2
    assert results[0].success is True
    assert results[0].data == {"title": "Page 1"}
    assert results[1].success is True
    assert results[1].data == {"title": "Page 2"}


@respx.mock
@pytest.mark.asyncio
async def test_extract_batch_handles_partial_failures() -> None:
    """Verify batch extraction continues on individual URL failures.

    If one URL fails during batch extraction, the service should continue
    processing remaining URLs and return partial results.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    route = respx.post("http://localhost:52004/llm/job")
    route.side_effect = [
        httpx.Response(200, json={"data": {"title": "Page 1"}}),
        httpx.Response(500, json={"error": "Failed"}),
        httpx.Response(200, json={"data": {"title": "Page 3"}}),
    ]

    service = ExtractorService(endpoint_url="http://localhost:52004")
    results = await service.extract_batch(
        urls=[
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ],
        schema={"type": "object"},
    )

    assert len(results) == 3
    assert results[0].success is True
    assert results[1].success is False
    assert results[2].success is True


@respx.mock
@pytest.mark.asyncio
async def test_extract_batch_requires_schema_or_prompt() -> None:
    """Verify batch extraction fails when neither schema nor prompt provided.

    If neither schema nor prompt is provided, each URL should return
    a failure result with an appropriate error message.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    service = ExtractorService(endpoint_url="http://localhost:52004")
    results = await service.extract_batch(
        urls=["https://example.com/1", "https://example.com/2"]
    )

    assert len(results) == 2
    assert results[0].success is False
    assert "schema or prompt" in results[0].error.lower()
    assert results[1].success is False
    assert "schema or prompt" in results[1].error.lower()


# =============================================================================
# Service lifecycle tests
# =============================================================================


@respx.mock
@pytest.mark.asyncio
async def test_extractor_service_close() -> None:
    """Verify service close method cleans up resources.

    The service should have a close method to release HTTP client resources.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))

    service = ExtractorService(endpoint_url="http://localhost:52004")
    await service.close()

    # Should not raise any errors


@respx.mock
@pytest.mark.asyncio
async def test_extractor_service_context_manager() -> None:
    """Verify service can be used as async context manager.

    The service should support async with syntax for automatic cleanup.
    """
    respx.get("http://localhost:52004/health").mock(return_value=httpx.Response(200))
    respx.post("http://localhost:52004/llm/job").mock(
        return_value=httpx.Response(200, json={"data": {"title": "Test"}})
    )

    async with ExtractorService(endpoint_url="http://localhost:52004") as service:
        result = await service.extract_with_schema(
            "https://example.com", schema={"type": "object"}
        )
        assert result.success is True

    # Service should be closed automatically
