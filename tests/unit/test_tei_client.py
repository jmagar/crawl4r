"""Unit tests for TEI client module.

TDD RED Phase: All tests should FAIL initially (no implementation exists).

This test suite covers:
- TEI client initialization
- Single text embedding generation
- Batch text embedding generation
- Network error handling (connection failures)
- Timeout error handling
- Invalid response handling
- Embedding dimension validation (1024)
- Batch size limit validation
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# This import will fail initially - that's expected in RED phase
from crawl4r.storage.tei import TEIClient


class TestTEIClientInitialization:
    """Test TEI client initialization and configuration."""

    def test_tei_client_initialization_with_endpoint(self) -> None:
        """Test that TEIClient can be instantiated with endpoint URL."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        assert client.endpoint_url == endpoint
        assert client.expected_dimensions == 1024

    def test_tei_client_initialization_with_custom_dimensions(self) -> None:
        """Test that TEIClient accepts custom embedding dimensions."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, dimensions=512)

        assert client.expected_dimensions == 512

    def test_tei_client_initialization_validates_endpoint_url(self) -> None:
        """Test that TEIClient validates endpoint URL format."""
        with pytest.raises(ValueError, match="Invalid endpoint URL"):
            TEIClient(endpoint_url="not-a-valid-url")

    def test_tei_client_initialization_defaults(self) -> None:
        """Test that TEIClient sets sensible defaults."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        assert client.timeout == 30.0
        assert client.max_retries == 3
        assert client.batch_size_limit == 100

    def test_tei_client_initialization_validates_dimensions(self) -> None:
        """Test that TEIClient validates embedding dimensions."""
        endpoint = "http://crawl4r-embeddings:80"

        with pytest.raises(ValueError, match="Dimensions must be a positive integer"):
            TEIClient(endpoint_url=endpoint, dimensions=0)

        with pytest.raises(ValueError, match="Dimensions must be a positive integer"):
            TEIClient(endpoint_url=endpoint, dimensions=-5)

    def test_tei_client_initialization_validates_timeout(self) -> None:
        """Test that TEIClient validates timeout configuration."""
        endpoint = "http://crawl4r-embeddings:80"

        with pytest.raises(ValueError, match="Timeout must be a positive number"):
            TEIClient(endpoint_url=endpoint, timeout=0)

        with pytest.raises(ValueError, match="Timeout must be a positive number"):
            TEIClient(endpoint_url=endpoint, timeout=-1.0)

    def test_tei_client_initialization_validates_max_retries(self) -> None:
        """Test that TEIClient validates max retries configuration."""
        endpoint = "http://crawl4r-embeddings:80"

        with pytest.raises(ValueError, match="Max retries must be an integer >= 1"):
            TEIClient(endpoint_url=endpoint, max_retries=0)

        with pytest.raises(ValueError, match="Max retries must be an integer >= 1"):
            TEIClient(endpoint_url=endpoint, max_retries=-2)


class TestTEIClientSingleTextEmbedding:
    """Test single text embedding generation."""

    @pytest.mark.asyncio
    async def test_embed_single_text_returns_1024_dim_vector(self) -> None:
        """Test that embedding single text returns 1024-dimensional vector."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        # Mock httpx response with 1024-dim embedding
        mock_embedding = [0.1] * 1024
        mock_response = MagicMock()
        mock_response.json.return_value = [mock_embedding]
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            embedding = await client.embed_single("test text")

            assert isinstance(embedding, list)
            assert len(embedding) == 1024
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_single_text_sends_correct_request(self) -> None:
        """Test that embed_single sends correct POST request to TEI endpoint."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        mock_embedding = [0.1] * 1024
        mock_response = MagicMock()
        mock_response.json.return_value = [mock_embedding]
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.post", return_value=mock_response) as mock_post:
            await client.embed_single("test text")

            # Verify POST was called with correct endpoint and payload
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/embed" in str(call_args)
            assert "test text" in str(call_args)

    @pytest.mark.asyncio
    async def test_embed_single_handles_empty_text(self) -> None:
        """Test that embed_single raises error for empty text."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        with pytest.raises(ValueError, match="Text cannot be empty"):
            await client.embed_single("")


class TestTEIClientBatchTextEmbedding:
    """Test batch text embedding generation."""

    @pytest.mark.asyncio
    async def test_embed_batch_returns_list_of_vectors(self) -> None:
        """Test that embedding batch of texts returns list of 1024-dim vectors."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        # Mock httpx response with batch of embeddings
        texts = ["text1", "text2", "text3"]
        mock_embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        mock_response = MagicMock()
        mock_response.json.return_value = mock_embeddings
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            embeddings = await client.embed_batch(texts)

            assert isinstance(embeddings, list)
            assert len(embeddings) == 3
            assert all(len(emb) == 1024 for emb in embeddings)

    @pytest.mark.asyncio
    async def test_embed_batch_validates_batch_size_limit(self) -> None:
        """Test that embed_batch enforces batch size limit."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, batch_size_limit=50)

        # Try to embed more than batch size limit
        texts = [f"text{i}" for i in range(100)]

        with pytest.raises(ValueError, match="Batch size exceeds limit"):
            await client.embed_batch(texts)

    @pytest.mark.asyncio
    async def test_embed_batch_handles_empty_list(self) -> None:
        """Test that embed_batch raises error for empty list."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        with pytest.raises(ValueError, match="Batch cannot be empty"):
            await client.embed_batch([])

    @pytest.mark.asyncio
    async def test_embed_batch_rejects_empty_texts(self) -> None:
        """Test that embed_batch validates empty text entries."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        with pytest.raises(ValueError, match="Batch contains empty text at index 1"):
            await client.embed_batch(["valid", ""])

    @pytest.mark.asyncio
    async def test_embed_batch_handles_single_item(self) -> None:
        """Test that embed_batch works with single item."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        mock_embedding = [[0.1] * 1024]
        mock_response = MagicMock()
        mock_response.json.return_value = mock_embedding
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            embeddings = await client.embed_batch(["single text"])

            assert len(embeddings) == 1
            assert len(embeddings[0]) == 1024


class TestTEIClientConnectionErrors:
    """Test handling of network connection errors."""

    @pytest.mark.asyncio
    async def test_embed_single_handles_connection_error(self) -> None:
        """Test that connection errors are handled with retries."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, max_retries=3)

        # Mock connection error
        with patch(
            "httpx.AsyncClient.post",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(httpx.ConnectError):
                await client.embed_single("test text")

    @pytest.mark.asyncio
    async def test_embed_single_retries_on_network_error(self) -> None:
        """Test that network errors trigger retry logic."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, max_retries=3)

        # Mock 2 failures then success
        mock_embedding = [0.1] * 1024
        mock_success_response = MagicMock()
        mock_success_response.json.return_value = [mock_embedding]
        mock_success_response.status_code = 200

        with patch(
            "httpx.AsyncClient.post",
            side_effect=[
                httpx.NetworkError("Network error"),
                httpx.NetworkError("Network error"),
                mock_success_response,
            ],
        ):
            embedding = await client.embed_single("test text")

            assert len(embedding) == 1024

    @pytest.mark.asyncio
    async def test_embed_single_backoff_adds_jitter(self) -> None:
        """Test that retry backoff includes jitter for network errors."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, max_retries=2)

        with (
            patch(
                "httpx.AsyncClient.post",
                side_effect=httpx.NetworkError("Network error"),
            ),
            patch("crawl4r.storage.tei.random.uniform", return_value=0.1),
            patch("crawl4r.storage.tei.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(httpx.NetworkError):
                await client._embed_single_impl("test text")

        mock_sleep.assert_awaited_once_with(1.1)

    @pytest.mark.asyncio
    async def test_embed_batch_handles_connection_error(self) -> None:
        """Test that batch embedding handles connection errors."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        with patch(
            "httpx.AsyncClient.post",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(httpx.ConnectError):
                await client.embed_batch(["text1", "text2"])


class TestTEIClientPersistentClient:
    """Test persistent AsyncClient usage."""

    @pytest.mark.asyncio
    async def test_embed_single_reuses_async_client(self) -> None:
        """TEIClient should reuse a single AsyncClient instance."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        mock_embedding = [0.1] * 1024
        mock_response = MagicMock()
        mock_response.json.return_value = [mock_embedding]
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_http_client) as mock_factory:
            await client.embed_single("text1")
            await client.embed_single("text2")

            assert mock_factory.call_count == 1
            assert mock_http_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_close_closes_persistent_client(self) -> None:
        """TEIClient.close should close the underlying AsyncClient."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        mock_embedding = [0.1] * 1024
        mock_response = MagicMock()
        mock_response.json.return_value = [mock_embedding]
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.aclose = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            await client.embed_single("text1")
            await client.close()

        mock_http_client.aclose.assert_called_once()


class TestTEIClientTimeoutErrors:
    """Test handling of timeout errors."""

    @pytest.mark.asyncio
    async def test_embed_single_handles_timeout(self) -> None:
        """Test that timeout errors are handled properly."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, timeout=5.0)

        with patch(
            "httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Timeout")
        ):
            with pytest.raises(httpx.TimeoutException):
                await client.embed_single("test text")

    @pytest.mark.asyncio
    async def test_embed_single_retries_on_timeout(self) -> None:
        """Test that timeout errors trigger retry logic."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, max_retries=3, timeout=5.0)

        # Mock 1 timeout then success
        mock_embedding = [0.1] * 1024
        mock_success_response = MagicMock()
        mock_success_response.json.return_value = [mock_embedding]
        mock_success_response.status_code = 200

        with patch(
            "httpx.AsyncClient.post",
            side_effect=[httpx.TimeoutException("Timeout"), mock_success_response],
        ):
            embedding = await client.embed_single("test text")

            assert len(embedding) == 1024

    @pytest.mark.asyncio
    async def test_embed_batch_handles_timeout(self) -> None:
        """Test that batch embedding handles timeout errors."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, timeout=5.0)

        with patch(
            "httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Timeout")
        ):
            with pytest.raises(httpx.TimeoutException):
                await client.embed_batch(["text1", "text2"])


class TestTEIClientInvalidResponses:
    """Test handling of invalid responses from TEI server."""

    @pytest.mark.asyncio
    async def test_embed_single_handles_invalid_json(self) -> None:
        """Test that invalid JSON response raises error."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            with pytest.raises(ValueError, match="Invalid JSON"):
                await client.embed_single("test text")

    @pytest.mark.asyncio
    async def test_embed_single_handles_http_error_status(self) -> None:
        """Test that HTTP error status codes are handled."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=MagicMock(), response=mock_response
        )

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            with pytest.raises(httpx.HTTPStatusError):
                await client.embed_single("test text")

    @pytest.mark.asyncio
    async def test_embed_single_handles_malformed_response_structure(self) -> None:
        """Test that malformed response structure raises error."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        # Mock response with wrong structure (not a list of lists)
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "unexpected format"}
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            with pytest.raises(ValueError, match="Invalid response structure"):
                await client.embed_single("test text")

    @pytest.mark.asyncio
    async def test_embed_batch_handles_partial_response(self) -> None:
        """Test that partial batch response (missing embeddings) raises error."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        # Request 3 embeddings but only get 2
        texts = ["text1", "text2", "text3"]
        mock_embeddings = [[0.1] * 1024, [0.2] * 1024]  # Only 2 embeddings
        mock_response = MagicMock()
        mock_response.json.return_value = mock_embeddings
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            with pytest.raises(
                ValueError, match="Response count does not match request count"
            ):
                await client.embed_batch(texts)


class TestTEIClientDimensionValidation:
    """Test validation of embedding dimensions."""

    @pytest.mark.asyncio
    async def test_embed_single_validates_dimension_1024(self) -> None:
        """Test that embeddings are validated to have exactly 1024 dimensions."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, dimensions=1024)

        # Mock response with wrong dimensions (512 instead of 1024)
        mock_embedding = [0.1] * 512
        mock_response = MagicMock()
        mock_response.json.return_value = [mock_embedding]
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            with pytest.raises(ValueError, match="Expected 1024 dimensions, got 512"):
                await client.embed_single("test text")

    @pytest.mark.asyncio
    async def test_embed_batch_validates_all_dimensions(self) -> None:
        """Test that all embeddings in batch are validated for dimensions."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, dimensions=1024)

        # Mock response with mixed dimensions (one wrong)
        mock_embeddings = [
            [0.1] * 1024,
            [0.2] * 512,  # Wrong dimension
            [0.3] * 1024,
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = mock_embeddings
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            with pytest.raises(ValueError, match="Expected 1024 dimensions"):
                await client.embed_batch(["text1", "text2", "text3"])

    @pytest.mark.asyncio
    async def test_embed_single_accepts_custom_dimensions(self) -> None:
        """Test that client accepts custom embedding dimensions."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, dimensions=512)

        # Mock response with 512 dimensions (matching custom setting)
        mock_embedding = [0.1] * 512
        mock_response = MagicMock()
        mock_response.json.return_value = [mock_embedding]
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            embedding = await client.embed_single("test text")

            assert len(embedding) == 512

    @pytest.mark.asyncio
    async def test_embed_single_rejects_zero_dimension_vector(self) -> None:
        """Test that zero-dimension vectors are rejected."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, dimensions=1024)

        # Mock response with empty embedding
        mock_response = MagicMock()
        mock_response.json.return_value = [[]]
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            with pytest.raises(ValueError, match="Invalid response structure"):
                await client.embed_single("test text")


class TestTEIClientBatchSizeLimits:
    """Test validation of batch size limits."""

    @pytest.mark.asyncio
    async def test_embed_batch_enforces_default_limit(self) -> None:
        """Test that default batch size limit (100) is enforced."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint)

        # Create batch larger than default limit (100)
        texts = [f"text{i}" for i in range(150)]

        with pytest.raises(ValueError, match="Batch size exceeds limit of 100"):
            await client.embed_batch(texts)

    @pytest.mark.asyncio
    async def test_embed_batch_enforces_custom_limit(self) -> None:
        """Test that custom batch size limit is enforced."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, batch_size_limit=10)

        # Create batch larger than custom limit (10)
        texts = [f"text{i}" for i in range(20)]

        with pytest.raises(ValueError, match="Batch size exceeds limit of 10"):
            await client.embed_batch(texts)

    @pytest.mark.asyncio
    async def test_embed_batch_accepts_limit_size_batch(self) -> None:
        """Test that batch exactly at limit size is accepted."""
        endpoint = "http://crawl4r-embeddings:80"
        client = TEIClient(endpoint_url=endpoint, batch_size_limit=50)

        # Create batch exactly at limit
        texts = [f"text{i}" for i in range(50)]
        mock_embeddings = [[0.1] * 1024 for _ in range(50)]
        mock_response = MagicMock()
        mock_response.json.return_value = mock_embeddings
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            embeddings = await client.embed_batch(texts)

            assert len(embeddings) == 50

    def test_client_initialization_validates_batch_size_limit(self) -> None:
        """Test that client validates batch size limit on initialization."""
        endpoint = "http://crawl4r-embeddings:80"

        # Batch size limit must be positive
        with pytest.raises(ValueError, match="Batch size limit must be positive"):
            TEIClient(endpoint_url=endpoint, batch_size_limit=0)

        with pytest.raises(ValueError, match="Batch size limit must be positive"):
            TEIClient(endpoint_url=endpoint, batch_size_limit=-10)
