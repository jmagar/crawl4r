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

    async def validate_tei_connection(self, tei_client) -> bool:
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

        # Retry logic: attempt 3 times with exponential backoff
        max_attempts = 3
        retry_delays = [5, 10, 20]  # seconds

        for attempt in range(max_attempts):
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
                    f"TEI validation attempt {attempt + 1}/{max_attempts} failed: {e}"
                )

                # If this was the last attempt, exit
                if attempt == max_attempts - 1:
                    self.logger.error(
                        f"TEI validation failed after {max_attempts} attempts"
                    )
                    sys.exit(1)

                # Wait before retry (no delay after last attempt)
                delay = retry_delays[attempt]
                self.logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

        # Unreachable (sys.exit called above), but satisfies type checker
        return False

    async def validate_qdrant_connection(self, vector_store) -> bool:
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

        # Retry logic: attempt 3 times with exponential backoff
        max_attempts = 3
        retry_delays = [5, 10, 20]  # seconds

        for attempt in range(max_attempts):
            try:
                # Get collection info
                info = await vector_store.get_collection_info()

                # Validate vector size (raise immediately, no retry)
                vector_size = info.get("vector_size")
                if vector_size != self.expected_dimensions:
                    raise ValueError(
                        f"Expected {self.expected_dimensions} dimensions, "
                        f"got {vector_size}"
                    )

                # Validate distance metric
                distance = info.get("distance")
                if distance != "Cosine":
                    raise ValueError(
                        f"Expected Cosine distance metric, got {distance}"
                    )

                # Validation succeeded
                self.logger.info("Qdrant validation passed")
                return True

            except ValueError:
                # Configuration mismatch - raise immediately without retry
                raise

            except Exception as e:
                # Connection/network error - retry with backoff
                self.logger.warning(
                    f"Qdrant validation attempt {attempt + 1}/{max_attempts} "
                    f"failed: {e}"
                )

                # If this was the last attempt, exit
                if attempt == max_attempts - 1:
                    self.logger.error(
                        f"Qdrant validation failed after {max_attempts} attempts"
                    )
                    sys.exit(1)

                # Wait before retry (no delay after last attempt)
                delay = retry_delays[attempt]
                self.logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)

        # Unreachable (sys.exit called above), but satisfies type checker
        return False
