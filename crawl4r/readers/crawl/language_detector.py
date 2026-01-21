"""Language detection component using fast-langdetect.

This module provides fast, accurate language detection for markdown content
using the fast-langdetect library. It's designed for RAG pipelines that need
to filter content by language before ingestion.

The LanguageDetector class provides a thread-safe, fail-open interface that
skips detection for very short text and returns "unknown" on errors rather
than crashing the pipeline.

Examples:
    Basic usage for content filtering:
        >>> from crawl4r.readers.crawl.language_detector import LanguageDetector
        >>> detector = LanguageDetector(min_text_length=50)
        >>> result = detector.detect("This is English content")
        >>> if result.language == "en" and result.confidence > 0.5:
        ...     print("English document detected")

    Integration with crawl pipeline:
        >>> detector = LanguageDetector()
        >>> for doc in crawled_documents:
        ...     result = detector.detect(doc.text)
        ...     doc.metadata["detected_language"] = result.language
        ...     doc.metadata["language_confidence"] = result.confidence
"""

import logging
from dataclasses import dataclass

try:
    from fast_langdetect import detect
except ImportError:
    raise ImportError(
        "fast-langdetect is required for language detection. "
        "Install it with: uv pip install fast-langdetect>=0.4.0"
    )


@dataclass
class LanguageResult:
    """Result of language detection.

    Attributes:
        language: ISO 639-1 language code ("en", "es", "fr", etc.) or "unknown"
        confidence: Confidence score 0.0-1.0, where 0.0 = unknown/error
    """

    language: str
    confidence: float


class LanguageDetector:
    """Fast language detection using fast-langdetect library.

    Thread-safe, fail-open on errors (returns unknown), skips short text.

    Attributes:
        min_text_length: Minimum text length for detection (default: 10 chars)
    """

    def __init__(self, min_text_length: int = 10) -> None:
        """Initialize detector with minimum text length.

        Args:
            min_text_length: Skip detection for text shorter than this (default: 10)
        """
        self.min_text_length = min_text_length
        self._logger = logging.getLogger(__name__)

    def detect(self, text: str) -> LanguageResult:
        """Detect primary language and confidence from markdown content.

        Analyzes text using fast-langdetect library to identify the primary
        language and return a confidence score. The method is fail-open,
        returning "unknown" on errors rather than raising exceptions.

        Args:
            text: Markdown content to analyze. Can be empty, whitespace-only,
                or any UTF-8 string. No preprocessing required - the method
                handles edge cases internally.

        Returns:
            LanguageResult with two fields:
                - language: ISO 639-1 code ("en", "es", "fr", etc.) or "unknown"
                - confidence: Float 0.0-1.0, where higher values indicate
                  stronger confidence in the detected language. Returns 0.0
                  for "unknown" results.

        Edge Cases:
            - Empty text: LanguageResult(language="unknown", confidence=0.0)
            - Whitespace-only: LanguageResult(language="unknown", confidence=0.0)
            - Short text (< min_text_length):
              LanguageResult(language="unknown", confidence=0.0)
            - Detection error: LanguageResult(language="unknown", confidence=0.0)
              (logs warning with error details)
            - Multi-language: Primary language (highest confidence) returned
            - Non-UTF8 text: Handled by fast-langdetect, may return "unknown"

        Examples:
            Basic English detection:
                >>> detector = LanguageDetector()
                >>> result = detector.detect("This is English text")
                >>> assert result.language == "en"
                >>> assert result.confidence > 0.5

            Empty text handling:
                >>> result = detector.detect("")  # Empty text
                >>> assert result.language == "unknown"
                >>> assert result.confidence == 0.0

            Short text handling:
                >>> detector = LanguageDetector(min_text_length=50)
                >>> result = detector.detect("Hello")  # Too short
                >>> assert result.language == "unknown"

            Multi-language content:
                >>> text = "English text. Texto en español. Texte en français."
                >>> result = detector.detect(text)
                >>> # Returns primary language based on proportion

        Notes:
            - Thread-safe (uses fast-langdetect which is thread-safe)
            - Fail-open design: never raises exceptions
            - Logs warnings on detection errors for debugging
            - Performance: ~1000 detections/second on typical documents
        """
        # Skip detection for empty or whitespace-only text
        if not text or not text.strip():
            return LanguageResult(language="unknown", confidence=0.0)

        # Skip detection for text shorter than minimum length
        if len(text) < self.min_text_length:
            return LanguageResult(language="unknown", confidence=0.0)

        # Perform detection with error handling
        try:
            result = detect(text)
            # fast-langdetect returns a list, take first result (primary language)
            if not result:
                return LanguageResult(language="unknown", confidence=0.0)

            primary = result[0]
            return LanguageResult(
                language=primary["lang"],
                confidence=primary["score"],
            )
        except Exception as e:
            # Fail-open: log warning and return unknown
            self._logger.warning(
                f"Language detection failed: {type(e).__name__}: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback": "unknown",
                    "text_length": len(text),
                },
            )
            return LanguageResult(language="unknown", confidence=0.0)
