"""Quality verification module for startup validation.

Validates TEI and Qdrant connections during application startup to ensure
all dependencies are available and configured correctly.

Features:
- TEI connection validation with dimension checking
- Qdrant connection validation with collection verification
- Exponential backoff retry logic (5s, 10s, 20s)
- Graceful exit on validation failure

Example:
    from rag_ingestion.config import Settings
    from rag_ingestion.tei_client import TEIClient
    from rag_ingestion.quality import QualityVerifier

    config = Settings()
    tei_client = TEIClient(config.tei_url)
    verifier = QualityVerifier()

    # Validate TEI connection on startup
    await verifier.validate_tei_connection(tei_client)
"""

import asyncio
import logging
import sys
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from rag_ingestion.tei_client import TEIClient


class VectorStoreProtocol(Protocol):
    """Protocol defining expected interface for vector store operations."""

    async def get_collection_info(self) -> dict[str, Any]:
        """Get collection metadata including vector_size and distance."""
        ...


# Retry configuration
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAYS = [5, 10, 20]  # seconds

# Quality check defaults
DEFAULT_SAMPLE_RATE = 0.05  # 5% sampling
DEFAULT_NORMALIZATION_TOLERANCE = 0.01  # ±1% from norm=1.0


class QualityVerifier:
    """Startup validation for TEI and Qdrant connections.

    Performs health checks on critical dependencies during application startup.
    Implements retry logic with exponential backoff and exits gracefully on
    persistent failures.

    Attributes:
        logger: Logger instance for validation messages
        expected_dimensions: Expected embedding vector dimensions (default: 1024)

    Example:
        verifier = QualityVerifier()
        await verifier.validate_tei_connection(tei_client)
        await verifier.validate_qdrant_connection(vector_store)
    """

    def __init__(self, expected_dimensions: int = 1024) -> None:
        """Initialize quality verifier with expected dimensions.

        Args:
            expected_dimensions: Expected embedding vector size (default: 1024)

        Example:
            verifier = QualityVerifier(expected_dimensions=1024)
        """
        self.logger = logging.getLogger(__name__)
        self.expected_dimensions = expected_dimensions

    async def validate_tei_connection(self, tei_client: "TEIClient") -> bool:
        """Validate TEI connection with retry logic.

        Sends test embedding request to verify TEI service is available and
        returns vectors with correct dimensions. Retries up to 3 times with
        exponential backoff (5s, 10s, 20s delays).

        Args:
            tei_client: TEIClient instance to validate

        Returns:
            True if validation succeeds

        Raises:
            ValueError: If embedding dimensions don't match expected
            SystemExit: If all retry attempts fail (exits with code 1)

        Example:
            from rag_ingestion.tei_client import TEIClient

            tei_client = TEIClient("http://localhost:8080")
            verifier = QualityVerifier()
            await verifier.validate_tei_connection(tei_client)

        Notes:
            - Retry delays: 5s (1st retry), 10s (2nd retry), 20s (3rd retry)
            - Calls sys.exit(1) after max retries
            - Logs all validation steps and failures
        """
        self.logger.info("Validating TEI connection...")

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # Send test embedding request
                embedding = await tei_client.embed_single("test text")

                # Validate dimensions (raise immediately, no retry)
                if len(embedding) != self.expected_dimensions:
                    raise ValueError(
                        f"Expected {self.expected_dimensions} dimensions, "
                        f"got {len(embedding)}"
                    )

                # Validation succeeded
                self.logger.info("TEI validation passed")
                return True

            except ValueError:
                # Dimension mismatch - raise immediately without retry
                raise

            except Exception as e:
                # Connection/network error - retry with backoff
                self.logger.warning(
                    f"TEI validation attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS} "
                    f"failed: {e}"
                )

                # If this was the last attempt, exit
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    self.logger.error(
                        f"TEI validation failed after {MAX_RETRY_ATTEMPTS} attempts"
                    )
                    sys.exit(1)

                # Wait before retry (no delay after last attempt)
                delay = RETRY_DELAYS[attempt]
                self.logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

        # Unreachable (sys.exit called above), but satisfies type checker
        return False

    async def validate_qdrant_connection(
        self, vector_store: VectorStoreProtocol
    ) -> bool:
        """Validate Qdrant connection with retry logic.

        Retrieves collection info to verify Qdrant service is available and
        collection has correct vector size and distance metric. Retries up to
        3 times with exponential backoff (5s, 10s, 20s delays).

        Args:
            vector_store: VectorStoreManager instance to validate

        Returns:
            True if validation succeeds

        Raises:
            ValueError: If vector dimensions or distance metric don't match expected
            SystemExit: If all retry attempts fail (exits with code 1)

        Example:
            from rag_ingestion.vector_store import VectorStoreManager

            vector_store = VectorStoreManager("http://localhost:6333", "docs")
            verifier = QualityVerifier()
            await verifier.validate_qdrant_connection(vector_store)

        Notes:
            - Retry delays: 5s (1st retry), 10s (2nd retry), 20s (3rd retry)
            - Calls sys.exit(1) after max retries
            - Logs all validation steps and failures
            - Expected distance metric: "Cosine" (case-sensitive)
        """
        self.logger.info("Validating Qdrant connection...")

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                # Get collection info directly from qdrant client
                # Use client directly since get_collection_info not implemented
                if not hasattr(vector_store, "client"):
                    raise AttributeError("VectorStore missing 'client' attribute")

                try:
                    collection_info = vector_store.client.get_collection(  # type: ignore[attr-defined]
                        collection_name=vector_store.collection_name  # type: ignore[attr-defined]
                    )
                    info = {
                        "vector_size": collection_info.config.params.vectors.size,
                        "distance": collection_info.config.params.vectors.distance.name,
                    }
                except Exception as e:
                    # Collection might not exist yet - that's OK
                    err_msg = str(e).lower()
                    if "doesn't exist" in err_msg or "not found" in err_msg:
                        col_name = vector_store.collection_name  # type: ignore[attr-defined]
                        msg = (
                            f"Collection '{col_name}' doesn't exist yet, "
                            "will be created on first use"
                        )
                        self.logger.info(msg)
                        return True
                    raise

                # Validate vector size (raise immediately, no retry)
                vector_size = info.get("vector_size")
                if vector_size != self.expected_dimensions:
                    raise ValueError(
                        f"Expected {self.expected_dimensions} dimensions, "
                        f"got {vector_size}"
                    )

                # Validate distance metric (case-insensitive)
                distance = info.get("distance", "").upper()
                if distance != "COSINE":
                    raise ValueError(f"Expected COSINE distance metric, got {distance}")

                # Validation succeeded
                self.logger.info("Qdrant validation passed")
                return True

            except ValueError:
                # Configuration mismatch - raise immediately without retry
                raise

            except Exception as e:
                # Connection/network error - retry with backoff
                self.logger.warning(
                    f"Qdrant validation attempt {attempt + 1}/{MAX_RETRY_ATTEMPTS} "
                    f"failed: {e}"
                )

                # If this was the last attempt, exit
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    self.logger.error(
                        f"Qdrant validation failed after {MAX_RETRY_ATTEMPTS} attempts"
                    )
                    sys.exit(1)

                # Wait before retry (no delay after last attempt)
                delay = RETRY_DELAYS[attempt]
                self.logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

        # Unreachable (sys.exit called above), but satisfies type checker
        return False

    def check_embedding_dimensions(
        self, embedding: list[float], expected_dims: int | None = None
    ) -> None:
        """Check embedding has expected dimensions.

        Validates that embedding vector has the correct number of dimensions.
        Raises ValueError immediately if dimensions don't match.

        Args:
            embedding: Embedding vector to validate
            expected_dims: Expected dimension count (default: self.expected_dimensions)

        Raises:
            ValueError: If embedding dimensions don't match expected

        Example:
            verifier = QualityVerifier(expected_dimensions=1024)
            embedding = [0.1] * 1024
            verifier.check_embedding_dimensions(embedding)

        Notes:
            - This is a runtime check, not a startup validation
            - No retry logic - raises immediately on mismatch
            - Use for validating embeddings during processing
        """
        if expected_dims is None:
            expected_dims = self.expected_dimensions

        actual_dims = len(embedding)
        if actual_dims != expected_dims:
            raise ValueError(f"Expected {expected_dims} dimensions, got {actual_dims}")

    def sample_embeddings(
        self, embeddings: list[list[float]], sample_rate: float = DEFAULT_SAMPLE_RATE
    ) -> list[list[float]]:
        """Randomly sample embeddings for quality checks.

        Samples a percentage of embeddings to reduce overhead of quality checks.
        Uses random sampling to get representative subset.

        Args:
            embeddings: List of embedding vectors
            sample_rate: Percentage to sample (default: 0.05 = 5%)

        Returns:
            List of sampled embedding vectors

        Example:
            verifier = QualityVerifier()
            embeddings = [[0.1] * 1024 for _ in range(100)]
            sampled = verifier.sample_embeddings(embeddings)
            # Returns 5 embeddings (5% of 100)

        Notes:
            - Uses random.sample for unbiased selection
            - Sample size is max(1, int(len(embeddings) * sample_rate))
            - Returns at least 1 embedding if input is non-empty
        """
        import random

        if not embeddings:
            return []

        sample_size = max(1, int(len(embeddings) * sample_rate))
        return random.sample(embeddings, sample_size)

    def check_normalization(
        self, embedding: list[float], tolerance: float = DEFAULT_NORMALIZATION_TOLERANCE
    ) -> None:
        """Check if embedding is L2-normalized.

        Calculates L2 norm of embedding and logs warning if it's outside
        the expected range of 1.0 ± tolerance.

        Args:
            embedding: Embedding vector to check
            tolerance: Acceptable deviation from 1.0 (default: 0.01)

        Example:
            import math
            verifier = QualityVerifier()
            n = 1024
            embedding = [1.0 / math.sqrt(n)] * n  # L2-normalized
            verifier.check_normalization(embedding)

        Notes:
            - L2 norm = sqrt(sum of squares)
            - Expected range: [1.0 - tolerance, 1.0 + tolerance]
            - Logs WARNING if outside range, does not raise exception
            - Common issue: embeddings not normalized by model
        """
        import math

        # Calculate L2 norm
        norm = math.sqrt(sum(x * x for x in embedding))

        # Check if within tolerance
        if abs(norm - 1.0) > tolerance:
            self.logger.warning(
                f"Embedding not L2-normalized: norm={norm:.4f}, "
                f"expected 1.0 ± {tolerance}"
            )
