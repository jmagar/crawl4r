"""Language detection component using fast-langdetect."""

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
        """Detect primary language and confidence.

        Args:
            text: Markdown content to analyze

        Returns:
            LanguageResult with language code and confidence

        Edge Cases:
            - Empty text: LanguageResult(language="unknown", confidence=0.0)
            - Short text (< min_text_length): LanguageResult(language="unknown", confidence=0.0)
            - Detection error: LanguageResult(language="unknown", confidence=0.0)
            - Multi-language: Primary language (highest confidence)

        Examples:
            >>> detector = LanguageDetector()
            >>> result = detector.detect("This is English text")
            >>> assert result.language == "en"
            >>> assert result.confidence > 0.5

            >>> result = detector.detect("")  # Empty text
            >>> assert result.language == "unknown"
            >>> assert result.confidence == 0.0
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
                f"Language detection failed: {e}",
                extra={
                    "error": str(e),
                    "fallback": "unknown",
                    "text_length": len(text),
                },
            )
            return LanguageResult(language="unknown", confidence=0.0)
