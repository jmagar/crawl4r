"""Tests for centralized metadata key definitions."""

from crawl4r.core.metadata import MetadataKeys


def test_metadata_keys_has_file_path():
    """Verify MetadataKeys defines FILE_PATH constant."""
    assert hasattr(MetadataKeys, "FILE_PATH")
    assert MetadataKeys.FILE_PATH == "file_path"


def test_metadata_keys_has_file_name():
    """Verify MetadataKeys defines FILE_NAME constant."""
    assert hasattr(MetadataKeys, "FILE_NAME")
    assert MetadataKeys.FILE_NAME == "file_name"


def test_metadata_keys_has_chunk_index():
    """Verify MetadataKeys defines CHUNK_INDEX constant."""
    assert hasattr(MetadataKeys, "CHUNK_INDEX")
    assert MetadataKeys.CHUNK_INDEX == "chunk_index"


def test_metadata_keys_all_values_are_strings():
    """Verify all MetadataKeys values are strings."""
    for attr in dir(MetadataKeys):
        if not attr.startswith("_"):
            value = getattr(MetadataKeys, attr)
            assert isinstance(value, str), f"{attr} should be a string"


def test_metadata_keys_has_simple_directory_reader_keys():
    """Verify MetadataKeys defines all SimpleDirectoryReader default keys."""
    expected_keys = {
        "FILE_PATH": "file_path",
        "FILE_NAME": "file_name",
        "FILE_TYPE": "file_type",
        "FILE_SIZE": "file_size",
        "CREATION_DATE": "creation_date",
        "LAST_MODIFIED_DATE": "last_modified_date",
    }
    for const_name, expected_value in expected_keys.items():
        assert hasattr(MetadataKeys, const_name), f"Missing {const_name}"
        assert getattr(MetadataKeys, const_name) == expected_value


def test_metadata_keys_has_chunking_keys():
    """Verify MetadataKeys defines all chunking-related keys."""
    expected_keys = {
        "CHUNK_INDEX": "chunk_index",
        "CHUNK_TEXT": "chunk_text",
        "SECTION_PATH": "section_path",
        "TOTAL_CHUNKS": "total_chunks",
    }
    for const_name, expected_value in expected_keys.items():
        assert hasattr(MetadataKeys, const_name), f"Missing {const_name}"
        assert getattr(MetadataKeys, const_name) == expected_value


def test_metadata_keys_has_web_crawl_keys():
    """Verify MetadataKeys defines all web crawl metadata keys."""
    expected_keys = {
        "SOURCE_URL": "source_url",
        "SOURCE_TYPE": "source_type",
        "TITLE": "title",
        "DESCRIPTION": "description",
        "STATUS_CODE": "status_code",
        "CRAWL_TIMESTAMP": "crawl_timestamp",
    }
    for const_name, expected_value in expected_keys.items():
        assert hasattr(MetadataKeys, const_name), f"Missing {const_name}"
        assert getattr(MetadataKeys, const_name) == expected_value


def test_metadata_keys_has_legacy_keys():
    """Verify MetadataKeys defines legacy keys for migration compatibility."""
    expected_keys = {
        "FILE_PATH_RELATIVE": "file_path_relative",
        "FILE_PATH_ABSOLUTE": "file_path_absolute",
    }
    for const_name, expected_value in expected_keys.items():
        assert hasattr(MetadataKeys, const_name), f"Missing {const_name}"
        assert getattr(MetadataKeys, const_name) == expected_value
