"""Centralized metadata key definitions for the crawl4r pipeline.

This module provides a single source of truth for all metadata keys used
across document processing, storage, and retrieval. Using these constants
instead of hardcoded strings enables safe refactoring and IDE support.

Usage:
    from crawl4r.core.metadata import MetadataKeys

    doc.metadata[MetadataKeys.FILE_PATH]  # Instead of doc.metadata["file_path"]
"""


class MetadataKeys:
    """Constants for document metadata keys.

    These keys align with LlamaIndex SimpleDirectoryReader defaults where applicable.
    Custom keys (CHUNK_*, DETECTED_LANGUAGE, LANGUAGE_CONFIDENCE) are crawl4r-specific additions.
    """
    __slots__ = ()

    # SimpleDirectoryReader defaults
    FILE_PATH = "file_path"  # Absolute path from SimpleDirectoryReader
    FILE_NAME = "file_name"  # Base filename with extension
    FILE_TYPE = "file_type"  # MIME type (e.g., "text/markdown")
    FILE_SIZE = "file_size"  # Size in bytes
    CREATION_DATE = "creation_date"  # File creation timestamp
    LAST_MODIFIED_DATE = "last_modified_date"  # Last modification timestamp
    FILE_PATH_RELATIVE = "file_path_relative"  # Deprecated: relative path
    FILE_PATH_ABSOLUTE = "file_path_absolute"  # Deprecated: absolute path alias

    # Crawl4r chunking metadata
    CHUNK_INDEX = "chunk_index"  # Position of chunk in document
    CHUNK_TEXT = "chunk_text"  # Raw text content of chunk
    SECTION_PATH = "section_path"  # Heading hierarchy (e.g., "Guide > Install")
    TOTAL_CHUNKS = "total_chunks"  # Total chunks in document
    HEADING_LEVEL = "heading_level"  # Markdown heading level
    CONTENT_HASH = "content_hash"  # Hash of full content for change detection

    # Web crawl metadata (from Crawl4AIReader)
    SOURCE_URL = "source_url"  # Original URL
    SOURCE_TYPE = "source_type"  # "web_crawl" or "local_file"
    TITLE = "title"  # Page/document title
    DESCRIPTION = "description"  # Page description
    STATUS_CODE = "status_code"  # HTTP status code
    CRAWL_TIMESTAMP = "crawl_timestamp"  # When crawled
    DETECTED_LANGUAGE = "detected_language"  # ISO 639-1 code or "unknown"
    LANGUAGE_CONFIDENCE = "language_confidence"  # 0.0-1.0

    # Common optional metadata
    TAGS = "tags"  # Tag list from frontmatter
