"""Unit tests for configuration module.

TDD RED Phase: All tests should FAIL initially (no implementation exists).
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

# This import will fail initially - that's expected in RED phase
from rag_ingestion.config import Settings


class TestConfigLoading:
    """Test configuration loading from environment variables."""

    def test_config_loads_from_env(self) -> None:
        """Test that Settings object is created from environment variables."""
        env_vars = {
            "WATCH_FOLDER": "/tmp/test_watch",
            "TEI_ENDPOINT": "http://localhost:52000",
            "QDRANT_URL": "http://localhost:52001",
            "COLLECTION_NAME": "test_collection",
            "CHUNK_SIZE_TOKENS": "512",
            "CHUNK_OVERLAP_PERCENT": "15",
            "MAX_CONCURRENT_DOCS": "10",
            "QUEUE_MAX_SIZE": "1000",
            "BATCH_SIZE": "50",
            "LOG_LEVEL": "INFO",
            "FAILED_DOCS_LOG": "failed_documents.jsonl",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()

            assert settings.watch_folder == Path("/tmp/test_watch")
            assert settings.tei_endpoint == "http://localhost:52000"
            assert settings.qdrant_url == "http://localhost:52001"
            assert settings.collection_name == "test_collection"
            assert settings.chunk_size_tokens == 512
            assert settings.chunk_overlap_percent == 15
            assert settings.max_concurrent_docs == 10
            assert settings.queue_max_size == 1000
            assert settings.batch_size == 50
            assert settings.log_level == "INFO"
            assert settings.failed_docs_log == Path("failed_documents.jsonl")


class TestConfigValidation:
    """Test configuration validation rules."""

    def test_config_requires_watch_folder(self) -> None:
        """Test that ValidationError is raised when WATCH_FOLDER is missing."""
        env_vars: dict[str, str] = {}

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                # Create config without loading .env file
                class TestSettings(Settings):
                    model_config = SettingsConfigDict(env_file=None)

                TestSettings()

            # Verify that the error is about the missing WATCH_FOLDER
            errors = exc_info.value.errors()
            assert any(error["loc"] == ("watch_folder",) for error in errors), (
                "Expected validation error for watch_folder"
            )

    def test_config_validates_chunk_overlap(self) -> None:
        """Test that ValidationError is raised for invalid chunk overlap values."""
        # Test overlap > 50
        env_vars_high = {
            "WATCH_FOLDER": "/tmp/test",
            "CHUNK_OVERLAP_PERCENT": "55",
        }

        with patch.dict(os.environ, env_vars_high, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            errors = exc_info.value.errors()
            assert any(
                error["loc"] == ("chunk_overlap_percent",) for error in errors
            ), "Expected validation error for chunk_overlap_percent > 50"

        # Test overlap < 0
        env_vars_low = {
            "WATCH_FOLDER": "/tmp/test",
            "CHUNK_OVERLAP_PERCENT": "-5",
        }

        with patch.dict(os.environ, env_vars_low, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            errors = exc_info.value.errors()
            assert any(
                error["loc"] == ("chunk_overlap_percent",) for error in errors
            ), "Expected validation error for chunk_overlap_percent < 0"

    def test_config_validates_positive_integers(self) -> None:
        """Test that ValidationError is raised for negative integer values."""
        base_env = {
            "WATCH_FOLDER": "/tmp/test",
        }

        # Test negative MAX_CONCURRENT_DOCS
        env_vars_concurrent = {**base_env, "MAX_CONCURRENT_DOCS": "-5"}
        with patch.dict(os.environ, env_vars_concurrent, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            errors = exc_info.value.errors()
            assert any(error["loc"] == ("max_concurrent_docs",) for error in errors), (
                "Expected validation error for negative max_concurrent_docs"
            )

        # Test negative QUEUE_MAX_SIZE
        env_vars_queue = {**base_env, "QUEUE_MAX_SIZE": "-100"}
        with patch.dict(os.environ, env_vars_queue, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            errors = exc_info.value.errors()
            assert any(error["loc"] == ("queue_max_size",) for error in errors), (
                "Expected validation error for negative queue_max_size"
            )

        # Test negative BATCH_SIZE
        env_vars_batch = {**base_env, "BATCH_SIZE": "-10"}
        with patch.dict(os.environ, env_vars_batch, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            errors = exc_info.value.errors()
            assert any(error["loc"] == ("batch_size",) for error in errors), (
                "Expected validation error for negative batch_size"
            )


class TestConfigDefaults:
    """Test configuration default values."""

    def test_config_default_values(self) -> None:
        """Test that default values are applied when env vars are not set."""
        env_vars = {
            "WATCH_FOLDER": "/tmp/test_watch",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Create config without loading .env file
            class TestSettings(Settings):
                model_config = SettingsConfigDict(env_file=None)

            settings = TestSettings()

            # Test defaults from .env.example
            assert settings.tei_endpoint == "http://crawl4r-embeddings:80"
            assert settings.qdrant_url == "http://crawl4r-vectors:6333"
            assert settings.collection_name == "crawl4r"
            assert settings.chunk_size_tokens == 512
            assert settings.chunk_overlap_percent == 15
            assert settings.max_concurrent_docs == 10
            assert settings.queue_max_size == 1000
            assert settings.batch_size == 50
            assert settings.log_level == "INFO"
            assert settings.failed_docs_log == Path("failed_documents.jsonl")


class TestConfigTypeValidation:
    """Test that configuration validates types correctly."""

    def test_config_validates_types(self) -> None:
        """Test that type validation works for integer fields."""
        env_vars = {
            "WATCH_FOLDER": "/tmp/test",
            "CHUNK_SIZE_TOKENS": "not_an_integer",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            errors = exc_info.value.errors()
            assert any(error["loc"] == ("chunk_size_tokens",) for error in errors), (
                "Expected validation error for non-integer chunk_size_tokens"
            )

    def test_config_converts_path_strings(self) -> None:
        """Test that string paths are converted to Path objects."""
        env_vars = {
            "WATCH_FOLDER": "/tmp/test_watch",
            "FAILED_DOCS_LOG": "/var/log/failed.jsonl",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()

            assert isinstance(settings.watch_folder, Path)
            assert isinstance(settings.failed_docs_log, Path)
            assert settings.watch_folder == Path("/tmp/test_watch")
            assert settings.failed_docs_log == Path("/var/log/failed.jsonl")
