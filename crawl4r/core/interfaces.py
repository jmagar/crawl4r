"""Core protocol definitions for crawl4r components.

This module provides canonical protocol definitions used across the codebase
for type checking and dependency injection.
"""

from typing import Any, Protocol


class VectorStoreProtocol(Protocol):
    """Protocol defining expected interface for vector store operations.

    This protocol is implemented by VectorStoreManager and used by components
    that need vector store capabilities (QualityGate, StateRecovery).

    Methods required:
    - get_collection_info: Returns collection metadata (vector_size, distance)
    - scroll: Returns all points in collection for batch processing
    """

    async def get_collection_info(self) -> dict[str, Any]:
        """Get collection metadata including vector_size and distance.

        Returns:
            Dictionary with keys:
            - vector_size: Dimension of vectors (e.g., 1024)
            - distance: Distance metric (e.g., "Cosine")
        """
        ...

    async def scroll(self) -> list[dict[str, Any]]:
        """Scroll through all points in the collection.

        Returns:
            List of point dictionaries with payload data containing metadata
            like file_path, chunk_index, etc.
        """
        ...
