"""Tests for crawl4r.readers.crawl.language_detector module.

This module tests the LanguageDetector component for fast, accurate language
detection in RAG pipelines. Tests cover:

- FR-1: fast-langdetect integration for thread-safe detection
- FR-9: Short text handling (skip detection for text < min_text_length)
- FR-10: Error handling with fail-open behavior (returns "unknown" on failure)

Requirements coverage:
- US-1: Default English-only filtering (detection accuracy)
- US-2: Multi-language configuration support (detection for multiple languages)
- US-3: Confidence threshold tuning (confidence score validation)
- US-4: Language metadata enrichment (LanguageResult structure)
- US-5: Edge case handling (empty text, short text, detection failures)

Test Categories:
1. Initialization and configuration
2. Language detection accuracy
3. Short text handling
4. Error handling and fail-open behavior
5. Edge cases (empty, whitespace, multi-language)
"""

import pytest

from crawl4r.readers.crawl.language_detector import LanguageDetector, LanguageResult


# Basic language detection tests
def test_detect_english_text():
    """Test detection of English text with high confidence."""
    detector = LanguageDetector()
    result = detector.detect("This is English text")

    assert result.language == "en"
    assert result.confidence > 0.9


def test_detect_spanish_text():
    """Test detection of Spanish text with high confidence."""
    detector = LanguageDetector()
    result = detector.detect("Esto es texto en español")

    assert result.language == "es"
    assert result.confidence > 0.9


def test_detect_french_text():
    """Test detection of French text with high confidence."""
    detector = LanguageDetector()
    result = detector.detect("Ceci est du texte en français")

    assert result.language == "fr"
    assert result.confidence > 0.9


def test_detect_german_text():
    """Test detection of German text with high confidence."""
    detector = LanguageDetector()
    result = detector.detect("Das ist deutscher Text")

    assert result.language == "de"
    assert result.confidence > 0.9
