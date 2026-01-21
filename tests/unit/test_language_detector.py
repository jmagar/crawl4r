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


# Error handling and performance tests
def test_detect_library_error():
    """Test detection library error results in fail-open behavior."""
    import unittest.mock as mock

    detector = LanguageDetector()

    # Mock fast_langdetect.detect to raise exception
    with mock.patch("crawl4r.readers.crawl.language_detector.detect") as mock_detect:
        mock_detect.side_effect = RuntimeError("Simulated detection error")
        result = detector.detect("This should fail gracefully")

        assert result.language == "unknown"
        assert result.confidence == 0.0


def test_detect_deterministic():
    """Test detection is deterministic - same input produces same output."""
    detector = LanguageDetector()
    text = "This is a test of deterministic behavior in language detection"

    # Detect same text 10 times
    results = [detector.detect(text) for _ in range(10)]

    # All results should be identical
    languages = [r.language for r in results]
    confidences = [r.confidence for r in results]

    assert len(set(languages)) == 1, "All languages should be identical"
    assert len(set(confidences)) == 1, "All confidences should be identical"
    assert languages[0] == "en", "Should detect English"


def test_detect_performance():
    """Test detection performance - 100 docs < 500ms (avg < 5ms)."""
    import time

    detector = LanguageDetector()
    text = "This is a performance test for language detection. We want to ensure detection is fast enough for production use."

    # Detect 100 times
    start = time.perf_counter()
    for _ in range(100):
        detector.detect(text)
    elapsed = time.perf_counter() - start

    # Total time should be < 500ms (avg < 5ms per detection)
    assert elapsed < 0.5, f"100 detections took {elapsed:.3f}s, expected < 0.5s"


def test_detect_large_document():
    """Test detection of large document (1MB) completes without error."""
    detector = LanguageDetector()

    # Create 1MB of English text (~1,000,000 chars)
    sentence = "This is a test sentence for large document detection. "
    text = sentence * 20000  # ~1.1 MB

    result = detector.detect(text)

    assert result.language == "en"
    assert result.confidence > 0.9


def test_detect_multilingual():
    """Test detection of multilingual text returns primary language."""
    detector = LanguageDetector()

    # Mixed English and Spanish - English dominant
    text_en_dominant = (
        "This is English text. This is more English text. This is even more English text. "
        "Esto es texto en español."
    )

    result_en = detector.detect(text_en_dominant)
    assert result_en.language == "en", "Should detect English as primary language"

    # Mixed Spanish and English - Spanish dominant
    text_es_dominant = (
        "Esto es texto en español. Más texto en español aquí. Y más texto en español. "
        "This is English text."
    )

    result_es = detector.detect(text_es_dominant)
    assert result_es.language == "es", "Should detect Spanish as primary language"
