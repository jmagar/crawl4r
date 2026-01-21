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


# Edge case tests
def test_detect_empty_text():
    """Test detection of empty text returns unknown with 0.0 confidence."""
    detector = LanguageDetector()
    result = detector.detect("")

    assert result.language == "unknown"
    assert result.confidence == 0.0


def test_detect_whitespace_only():
    """Test detection of whitespace-only text returns unknown with 0.0 confidence."""
    detector = LanguageDetector()
    result = detector.detect("   \n\t  ")

    assert result.language == "unknown"
    assert result.confidence == 0.0


def test_detect_short_text():
    """Test detection of text below min_text_length returns unknown with 0.0 confidence."""
    detector = LanguageDetector(min_text_length=50)
    result = detector.detect("Short text")  # 10 chars, below 50 threshold

    assert result.language == "unknown"
    assert result.confidence == 0.0


def test_detect_exact_threshold():
    """Test detection of text at exact min_text_length threshold is detected."""
    detector = LanguageDetector(min_text_length=50)
    # Create 50-char English text
    text = "This is English text for language detection tests."  # 50 chars
    result = detector.detect(text)

    assert result.language == "en"
    assert result.confidence > 0.0


def test_min_text_length_configurable():
    """Test min_text_length threshold is configurable."""
    detector = LanguageDetector(min_text_length=20)
    # Text is 15 chars, below custom threshold of 20
    result_below = detector.detect("Short text here")

    assert result_below.language == "unknown"
    assert result_below.confidence == 0.0

    # Text is 25 chars, above custom threshold of 20
    result_above = detector.detect("This is above the limit!")

    assert result_above.language == "en"
    assert result_above.confidence > 0.0
