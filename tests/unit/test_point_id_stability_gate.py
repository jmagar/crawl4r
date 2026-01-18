"""CRITICAL GATE: Verify point IDs remain stable after Task 0.5 changes.

This test MUST pass before proceeding to Task 1 (SimpleDirectoryReader integration).
If this test fails, point ID generation is broken and the migration will corrupt data.

The _generate_point_id method must produce identical IDs whether called with:
1. A relative path (existing behavior)
2. An absolute path + watch_folder (new behavior from Task 0.5)

This ensures that re-ingesting documents with SimpleDirectoryReader (which provides
absolute paths) will correctly overwrite existing vectors instead of creating duplicates.
"""

from pathlib import Path
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
    """Gate tests that MUST pass before Task 1 (SimpleDirectoryReader migration).

    These tests verify that point ID generation is path-agnostic and backward
    compatible. If any test fails, DO NOT proceed to Task 1 - the migration
    will corrupt existing data by creating duplicate vectors.
    """

    def test_point_id_same_for_relative_and_absolute_paths(
        self, vector_store: VectorStoreManager, tmp_path: Path
    ) -> None:
        """Point IDs must be identical whether computed from relative or absolute path.

        This is the CRITICAL gate test. SimpleDirectoryReader provides absolute paths,
        but existing data was indexed with relative paths. The IDs must match to
        enable proper upsert/overwrite behavior.
        """
        watch_folder = tmp_path / "docs"
        watch_folder.mkdir()

        rel_path = "guide/install.md"
        abs_path = str(watch_folder / rel_path)
        chunk_index = 0

        # ID from relative path (OLD behavior - current production)
        id_from_relative = vector_store._generate_point_id(rel_path, chunk_index)

        # ID from absolute path (NEW behavior - with watch_folder)
        id_from_absolute = vector_store._generate_point_id(
            abs_path, chunk_index, watch_folder=watch_folder
        )

        assert id_from_relative == id_from_absolute, (
            f"CRITICAL GATE FAILURE: Point ID changed!\n"
            f"  Relative path ID: {id_from_relative}\n"
            f"  Absolute path ID: {id_from_absolute}\n"
            f"  DO NOT PROCEED TO TASK 1 - fix Task 0.5 first."
        )

    def test_backward_compatibility_without_watch_folder(
        self, vector_store: VectorStoreManager
    ) -> None:
        """Existing code without watch_folder must still work.

        This ensures backward compatibility with existing callers that don't
        provide the watch_folder parameter.
        """
        rel_path = "docs/readme.md"
        chunk_index = 0

        # This is how existing code calls the method (no watch_folder)
        point_id = vector_store._generate_point_id(rel_path, chunk_index)

        assert point_id is not None, "Must work without watch_folder parameter"
        assert len(point_id) == 36, f"Must return valid UUID format (36 chars), got {len(point_id)}"

    def test_point_id_deterministic(self, vector_store: VectorStoreManager) -> None:
        """Same inputs must always produce same ID.

        Point IDs must be deterministic to enable idempotent upsert operations.
        Re-processing the same file must produce the same IDs.
        """
        file_path = "docs/api.md"
        chunk_index = 3

        id1 = vector_store._generate_point_id(file_path, chunk_index)
        id2 = vector_store._generate_point_id(file_path, chunk_index)

        assert id1 == id2, "Point IDs must be deterministic"

    def test_point_id_unique_across_chunk_indices(
        self, vector_store: VectorStoreManager
    ) -> None:
        """Different chunk indices must produce different IDs.

        Each chunk in a file must have a unique ID to avoid collisions.
        """
        file_path = "docs/guide.md"

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

        id_file_a = vector_store._generate_point_id("docs/file_a.md", chunk_index)
        id_file_b = vector_store._generate_point_id("docs/file_b.md", chunk_index)

        assert id_file_a != id_file_b, "Different files must have different IDs"

    def test_point_id_with_nested_paths(
        self, vector_store: VectorStoreManager, tmp_path: Path
    ) -> None:
        """Point ID stability must work with deeply nested paths.

        SimpleDirectoryReader can return deeply nested absolute paths.
        These must produce the same IDs as the equivalent relative paths.
        """
        watch_folder = tmp_path / "docs"
        (watch_folder / "deeply" / "nested" / "folder").mkdir(parents=True)

        rel_path = "deeply/nested/folder/document.md"
        abs_path = str(watch_folder / rel_path)
        chunk_index = 5

        id_from_relative = vector_store._generate_point_id(rel_path, chunk_index)
        id_from_absolute = vector_store._generate_point_id(
            abs_path, chunk_index, watch_folder=watch_folder
        )

        assert id_from_relative == id_from_absolute, (
            f"Nested path ID mismatch!\n"
            f"  Relative: {rel_path} -> {id_from_relative}\n"
            f"  Absolute: {abs_path} -> {id_from_absolute}"
        )

    def test_point_id_with_none_watch_folder_explicit(
        self, vector_store: VectorStoreManager
    ) -> None:
        """Explicitly passing watch_folder=None must behave same as omitting it.

        This tests that the default parameter behavior is correct.
        """
        file_path = "docs/test.md"
        chunk_index = 0

        id_omitted = vector_store._generate_point_id(file_path, chunk_index)
        id_explicit_none = vector_store._generate_point_id(
            file_path, chunk_index, watch_folder=None
        )

        assert id_omitted == id_explicit_none, (
            "watch_folder=None must behave same as omitting the parameter"
        )

    def test_point_id_valid_uuid_format(
        self, vector_store: VectorStoreManager
    ) -> None:
        """Generated point IDs must be valid UUID format.

        Qdrant requires valid UUIDs for point IDs.
        """
        import uuid

        file_path = "docs/test.md"
        chunk_index = 0

        point_id = vector_store._generate_point_id(file_path, chunk_index)

        # This will raise ValueError if not a valid UUID
        try:
            uuid.UUID(point_id)
        except ValueError:
            pytest.fail(f"Point ID is not a valid UUID: {point_id}")

    def test_point_id_path_outside_watch_folder_uses_fallback(
        self, vector_store: VectorStoreManager, tmp_path: Path
    ) -> None:
        """Paths outside watch_folder should use full path as fallback.

        When a file is not under watch_folder, we cannot compute a relative path.
        The method should handle this gracefully by using the full path.
        """
        watch_folder = tmp_path / "docs"
        watch_folder.mkdir()

        # Path NOT under watch_folder
        external_path = "/some/other/location/file.md"
        chunk_index = 0

        # Should not raise an error
        point_id = vector_store._generate_point_id(
            external_path, chunk_index, watch_folder=watch_folder
        )

        assert point_id is not None, "Should handle paths outside watch_folder"
        assert len(point_id) == 36, "Should return valid UUID format"
