"""CRITICAL GATE: Verify point IDs are stable with absolute paths.

This test ensures point ID generation is deterministic and consistent
when using absolute file paths. The _generate_point_id method must produce
identical IDs for the same absolute path and chunk index.

This is critical for idempotent upserts - re-processing the same file
must update existing vectors rather than creating duplicates.
"""

from unittest.mock import MagicMock, patch

import pytest

from crawl4r.storage.qdrant import VectorStoreManager


@pytest.fixture
def vector_store() -> VectorStoreManager:
    """Create a VectorStoreManager with mocked Qdrant client."""
    with patch("crawl4r.storage.qdrant.QdrantClient") as mock_client:
        mock_client.return_value = MagicMock()
        return VectorStoreManager(
            qdrant_url="http://localhost:6333",
            collection_name="test_stability_gate",
        )


class TestPointIdStabilityGate:
    """Gate tests for point ID stability with absolute paths.

    These tests verify that point ID generation is deterministic and
    produces consistent IDs for the same absolute file path.
    """

    def test_point_id_deterministic_for_absolute_path(
        self, vector_store: VectorStoreManager
    ) -> None:
        """Point IDs must be deterministic for the same absolute path.

        Same absolute path + chunk index must always produce the same point ID.
        """
        abs_path = "/home/user/docs/guide/install.md"
        chunk_index = 0

        id1 = vector_store._generate_point_id(abs_path, chunk_index)
        id2 = vector_store._generate_point_id(abs_path, chunk_index)

        assert id1 == id2, (
            f"Point ID not deterministic!\n"
            f"  First call: {id1}\n"
            f"  Second call: {id2}"
        )

    def test_point_id_unique_across_chunk_indices(
        self, vector_store: VectorStoreManager
    ) -> None:
        """Different chunk indices must produce different IDs.

        Each chunk in a file must have a unique ID to avoid collisions.
        """
        file_path = "/home/user/docs/guide.md"

        id_chunk_0 = vector_store._generate_point_id(file_path, 0)
        id_chunk_1 = vector_store._generate_point_id(file_path, 1)
        id_chunk_2 = vector_store._generate_point_id(file_path, 2)

        assert id_chunk_0 != id_chunk_1, "Different chunks must have different IDs"
        assert id_chunk_1 != id_chunk_2, "Different chunks must have different IDs"
        assert id_chunk_0 != id_chunk_2, "Different chunks must have different IDs"

    def test_point_id_unique_across_files(
        self, vector_store: VectorStoreManager
    ) -> None:
        """Different files must produce different IDs for same chunk index.

        Each file must have unique IDs even at the same chunk position.
        """
        chunk_index = 0

        id_file_a = vector_store._generate_point_id("/home/user/docs/file_a.md", chunk_index)
        id_file_b = vector_store._generate_point_id("/home/user/docs/file_b.md", chunk_index)

        assert id_file_a != id_file_b, "Different files must have different IDs"

    def test_point_id_with_deeply_nested_paths(
        self, vector_store: VectorStoreManager
    ) -> None:
        """Point ID stability must work with deeply nested paths.

        Deeply nested absolute paths must produce consistent IDs.
        """
        deep_path = "/home/user/documents/projects/crawl4r/docs/api/v2/endpoints/users.md"
        chunk_index = 5

        id1 = vector_store._generate_point_id(deep_path, chunk_index)
        id2 = vector_store._generate_point_id(deep_path, chunk_index)

        assert id1 == id2, (
            f"Deeply nested path ID not deterministic!\n"
            f"  Path: {deep_path}\n"
            f"  First: {id1}\n"
            f"  Second: {id2}"
        )

    def test_point_id_valid_uuid_format(
        self, vector_store: VectorStoreManager
    ) -> None:
        """Generated point IDs must be valid UUID format.

        Qdrant requires valid UUIDs for point IDs.
        """
        import uuid

        file_path = "/home/user/docs/test.md"
        chunk_index = 0

        point_id = vector_store._generate_point_id(file_path, chunk_index)

        # This will raise ValueError if not a valid UUID
        try:
            uuid.UUID(point_id)
        except ValueError:
            pytest.fail(f"Point ID is not a valid UUID: {point_id}")

    def test_point_id_length_is_36_chars(
        self, vector_store: VectorStoreManager
    ) -> None:
        """Point IDs must be 36 characters (standard UUID format)."""
        file_path = "/home/user/docs/readme.md"
        chunk_index = 0

        point_id = vector_store._generate_point_id(file_path, chunk_index)

        assert len(point_id) == 36, f"Must return valid UUID format (36 chars), got {len(point_id)}"

    def test_point_id_different_for_paths_differing_only_in_case(
        self, vector_store: VectorStoreManager
    ) -> None:
        """Paths differing only in case should produce different IDs.

        File paths are case-sensitive on most systems, so different casing
        should produce different IDs.
        """
        chunk_index = 0

        id_lower = vector_store._generate_point_id("/home/user/docs/readme.md", chunk_index)
        id_upper = vector_store._generate_point_id("/home/user/docs/README.md", chunk_index)

        assert id_lower != id_upper, "Case-different paths should have different IDs"
