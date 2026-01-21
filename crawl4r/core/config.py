"""Configuration module for RAG ingestion service.

Provides Pydantic-based configuration management with environment variable support
and comprehensive field validation.

Example:
    >>> from crawl4r.core.config import Settings
    >>> settings = Settings(watch_folder="/path/to/docs")
    >>> print(settings.tei_endpoint)
    'http://100.74.16.82:52000'  # Host URL
"""

import os
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def is_running_in_docker() -> bool:
    """Detect if code is running inside a Docker container.

    Checks for Docker-specific files and environment markers.

    Returns:
        True if running inside Docker container, False otherwise.
    """
    # Check for .dockerenv file (exists in Docker containers)
    if Path("/.dockerenv").exists():
        return True

    # Check cgroup for docker indicators
    try:
        with Path("/proc/1/cgroup").open() as f:
            return "docker" in f.read()
    except (FileNotFoundError, PermissionError):
        pass

    # Check for explicit environment variable
    return os.getenv("RUN_IN_DOCKER", "").lower() in ("true", "1", "yes")



class Settings(BaseSettings):
    """RAG ingestion service configuration.

    Environment-aware configuration that automatically uses:
    - Docker network URLs when running inside containers
    - Localhost URLs when running on host machine (CLI)

    Attributes:
        watch_folder: Directory to monitor for new documents (required)
        tei_endpoint: Text Embeddings Inference service endpoint
        tei_model_name: TEI embedding model name (default: "Qwen/Qwen3-Embedding-0.6B")
        qdrant_url: Qdrant vector database URL
        collection_name: Qdrant collection name for storing embeddings
        chunk_size_tokens: Maximum tokens per chunk
        chunk_overlap_percent: Percentage overlap between chunks (0-50)
        max_concurrent_docs: Maximum documents to process concurrently
        queue_max_size: Maximum size of the processing queue
        batch_size: Number of chunks to batch for embedding
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        failed_docs_log: Path to failed documents log file

    Raises:
        ValidationError: If required fields are missing or values are invalid

    Example:
        >>> settings = Settings(
        ...     watch_folder="/data/documents",
        ...     chunk_size_tokens=512,
        ...     chunk_overlap_percent=15
        ... )
        >>> print(settings.collection_name)
        'crawl4r'
    """

    # Required fields
    watch_folder: Path

    # Service endpoints (will be set by model_validator based on environment)
    tei_endpoint: str = ""
    tei_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimensions: int = 1024  # Qwen3-Embedding-0.6B dimension size
    qdrant_url: str = ""
    collection_name: str = "crawl4r"
    crawl4ai_base_url: str = ""
    redis_url: str = ""

    # Chunking configuration
    chunk_size_tokens: int = 512
    chunk_overlap_percent: int = 15

    # Concurrency and batching
    max_concurrent_docs: int = 10
    queue_max_size: int = 1000
    batch_size: int = 50

    # Logging
    log_level: str = "INFO"
    failed_docs_log: Path = Path("failed_documents.jsonl")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_default=True,
        extra="ignore",
    )

    @model_validator(mode="after")
    def set_environment_aware_defaults(self) -> "Settings":
        """Set service URLs based on environment if not explicitly configured.

        Auto-detects if running in Docker container vs host machine and uses
        appropriate service URLs. Environment variable overrides take precedence.

        Returns:
            Settings instance with environment-aware URLs.
        """
        in_docker = is_running_in_docker()

        # Set defaults only if not explicitly configured via env vars
        if not self.redis_url:
            self.redis_url = (
                "redis://crawl4r-cache:6379"
                if in_docker
                else "redis://localhost:53379"
            )

        if not self.qdrant_url:
            self.qdrant_url = (
                "http://crawl4r-vectors:6333"
                if in_docker
                else "http://localhost:52001"
            )

        if not self.tei_endpoint:
            self.tei_endpoint = (
                "http://crawl4r-embeddings:80"
                if in_docker
                else "http://100.74.16.82:52000"  # Remote GPU machine
            )

        if not self.crawl4ai_base_url:
            self.crawl4ai_base_url = (
                "http://crawl4ai:11235"
                if in_docker
                else "http://localhost:52004"
            )

        return self

    @field_validator("chunk_overlap_percent")
    @classmethod
    def validate_chunk_overlap(cls: type["Settings"], v: int) -> int:
        """Validate chunk overlap percentage is between 0 and 50.

        Overlap percentages above 50% would cause excessive redundancy in chunks,
        while negative values are invalid. The recommended range is 10-20% for
        optimal balance between context preservation and processing efficiency.

        Args:
            cls: The Settings class (provided by Pydantic)
            v: Chunk overlap percentage value

        Returns:
            Validated chunk overlap percentage

        Raises:
            ValueError: If chunk_overlap_percent is not in range [0, 50]
        """
        # Enforce overlap bounds: 0% (no overlap) to 50% (maximum recommended)
        if v < 0 or v > 50:
            raise ValueError("chunk_overlap_percent must be between 0 and 50")
        return v

    @field_validator("max_concurrent_docs")
    @classmethod
    def validate_max_concurrent_docs(cls: type["Settings"], v: int) -> int:
        """Validate max_concurrent_docs is positive.

        Controls the number of documents processed concurrently. Higher values
        increase throughput but consume more memory and CPU. Recommended values:
        10-20 for development systems, 50-100 for production with adequate resources.

        Args:
            cls: The Settings class (provided by Pydantic)
            v: Maximum concurrent documents value

        Returns:
            Validated max_concurrent_docs

        Raises:
            ValueError: If max_concurrent_docs is not positive
        """
        # Must have at least one concurrent document to make progress
        if v <= 0:
            raise ValueError("max_concurrent_docs must be positive")
        return v

    @field_validator("queue_max_size")
    @classmethod
    def validate_queue_max_size(cls: type["Settings"], v: int) -> int:
        """Validate queue_max_size is positive.

        Defines the maximum number of items that can be queued for processing.
        When the queue reaches this size, backpressure is applied to prevent
        memory exhaustion. Typical values: 100-1000 depending on available memory.

        Args:
            cls: The Settings class (provided by Pydantic)
            v: Queue maximum size value

        Returns:
            Validated queue_max_size

        Raises:
            ValueError: If queue_max_size is not positive
        """
        # Queue must hold at least one item to function
        if v <= 0:
            raise ValueError("queue_max_size must be positive")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls: type["Settings"], v: int) -> int:
        """Validate batch_size is positive.

        Controls how many document chunks are sent to TEI in a single embedding
        request. Larger batches improve throughput but require more memory.
        Optimal values depend on GPU memory: 10-50 for 8GB, 50-200 for 16GB+.

        Args:
            cls: The Settings class (provided by Pydantic)
            v: Batch size value

        Returns:
            Validated batch_size

        Raises:
            ValueError: If batch_size is not positive
        """
        # Batch size must be at least 1 to process any chunks
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v
