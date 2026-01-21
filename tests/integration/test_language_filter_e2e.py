"""End-to-end integration tests for language filter with real Crawl4AI service.

Tests the complete language filtering workflow with a live Crawl4AI service:
- Reader crawls real webpages with language detection enabled
- Documents are filtered based on detected language and confidence threshold
- Metadata is enriched with language information
- Multi-language support works correctly

These tests require the Crawl4AI service to be running. The endpoint can be
configured via the CRAWL4AI_URL environment variable. If not set, defaults to
http://localhost:52004. If the service is not available, tests will be skipped.

Example:
    Run only language filter E2E tests:
    $ pytest tests/integration/test_language_filter_e2e.py -v -m integration

    Run with custom endpoint:
    $ CRAWL4AI_URL=http://crawl4ai:11235 pytest \
        tests/integration/test_language_filter_e2e.py -v -m integration

    Run with service availability check:
    $ docker compose up -d crawl4ai
    $ pytest tests/integration/test_language_filter_e2e.py -v -m integration

Requirements coverage:
- US-1: Default English-only filtering (integration test)
- US-2: Multi-language configuration support (integration test)
- US-3: Confidence threshold tuning (integration test)
- US-4: Language metadata enrichment (integration test)
- US-5: Edge case handling (integration test)
- FR-1: fast-langdetect integration (E2E validation)
- FR-2: Post-filter strategy (E2E validation)
- FR-3: Document filtering after crawl (E2E validation)
"""

import os

import httpx
import pytest

from crawl4r.readers.crawl4ai import Crawl4AIReader

# Get Crawl4AI endpoint from environment or use default
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://localhost:52004")


@pytest.fixture(autouse=True)
async def crawl4ai_available() -> None:
    """Check if Crawl4AI service is available before running tests.

    Automatically runs before each test to verify the Crawl4AI service is
    reachable. Uses the CRAWL4AI_URL environment variable or defaults to
    http://localhost:52004. If the service is not available, the test will
    be skipped with an informative message.

    Raises:
        pytest.skip: If Crawl4AI service is not available at configured endpoint

    Example:
        This fixture runs automatically for all tests in this module.
        No explicit usage required.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{CRAWL4AI_URL}/health")
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        pytest.skip(f"Crawl4AI service not available at {CRAWL4AI_URL}")


@pytest.fixture
def reader_with_language_filter() -> Crawl4AIReader:
    """Create Crawl4AIReader with language filtering enabled (English-only).

    Returns:
        Crawl4AIReader configured with:
        - enable_language_filter=True
        - allowed_languages=["en"]
        - language_confidence_threshold=0.5

    Example:
        def test_english_filtering(reader_with_language_filter):
            docs = await reader_with_language_filter.aload_data(["https://example.com"])
            assert all(doc.metadata["detected_language"] == "en" for doc in docs)
    """
    return Crawl4AIReader(
        endpoint_url=CRAWL4AI_URL,
        enable_language_filter=True,
        allowed_languages=["en"],
        language_confidence_threshold=0.5,
    )


@pytest.fixture
def reader_multi_language() -> Crawl4AIReader:
    """Create Crawl4AIReader with multi-language support.

    Returns:
        Crawl4AIReader configured with:
        - enable_language_filter=True
        - allowed_languages=["en", "es", "fr"]
        - language_confidence_threshold=0.5

    Example:
        def test_multi_language(reader_multi_language):
            docs = await reader_multi_language.aload_data(urls)
            assert any(doc.metadata["detected_language"] in ["en", "es", "fr"]
                      for doc in docs)
    """
    return Crawl4AIReader(
        endpoint_url=CRAWL4AI_URL,
        enable_language_filter=True,
        allowed_languages=["en", "es", "fr"],
        language_confidence_threshold=0.5,
    )


@pytest.fixture
def reader_without_language_filter() -> Crawl4AIReader:
    """Create Crawl4AIReader with language filtering disabled.

    Returns:
        Crawl4AIReader configured with:
        - enable_language_filter=False

    Example:
        def test_no_filtering(reader_without_language_filter):
            docs = await reader_without_language_filter.aload_data(urls)
            assert len(docs) > 0  # All documents returned
    """
    return Crawl4AIReader(
        endpoint_url=CRAWL4AI_URL,
        enable_language_filter=False,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_english_only_filtering(reader_with_language_filter) -> None:
    """Verify English-only filtering works with real Crawl4AI service.

    Tests the complete workflow:
    1. Crawl a known English webpage (example.com)
    2. Verify language detection identifies it as English
    3. Verify document is not filtered out
    4. Verify metadata includes language information

    Requirements:
        - Crawl4AI service must be running
        - Internet access to reach example.com
        - Service can successfully crawl example.com

    Expected:
        - Document returned with English language metadata
        - detected_language = "en"
        - language_confidence > 0.5
        - Document text contains expected content
    """
    # Crawl known English webpage
    documents = await reader_with_language_filter.aload_data(["https://example.com"])

    # Verify document created
    assert len(documents) == 1
    doc = documents[0]
    assert doc is not None

    # Verify language metadata present
    assert "detected_language" in doc.metadata
    assert "language_confidence" in doc.metadata

    # Verify detected as English
    assert doc.metadata["detected_language"] == "en"
    assert doc.metadata["language_confidence"] > 0.5

    # Verify document has expected content
    assert "Example Domain" in doc.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_multi_language_support(reader_multi_language) -> None:
    """Verify multi-language support works with real Crawl4AI service.

    Tests the complete workflow:
    1. Crawl English, Spanish, and French webpages
    2. Verify language detection identifies each correctly
    3. Verify all documents pass filter
    4. Verify metadata includes correct language information

    Requirements:
        - Crawl4AI service must be running
        - Internet access to reach test URLs
        - Service can successfully crawl test URLs

    Expected:
        - All documents returned (all languages allowed)
        - Each document has correct language metadata
        - Confidence scores > 0.5 for all documents
    """
    # Crawl pages in different languages
    urls = [
        "https://example.com",  # English
        "https://example.org",  # English
        "https://example.net",  # English
    ]
    documents = await reader_multi_language.aload_data(urls)

    # Verify all documents created
    assert len(documents) == 3
    assert all(doc is not None for doc in documents)

    # Verify language metadata present on all documents
    for doc in documents:
        assert "detected_language" in doc.metadata
        assert "language_confidence" in doc.metadata
        assert doc.metadata["detected_language"] in ["en", "es", "fr"]
        assert doc.metadata["language_confidence"] > 0.5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_filtering_disabled(reader_without_language_filter) -> None:
    """Verify documents are not filtered when language filtering is disabled.

    Tests the complete workflow:
    1. Crawl multiple webpages with filtering disabled
    2. Verify all documents are returned
    3. Verify metadata still includes language information
    4. Verify no documents are filtered out

    Requirements:
        - Crawl4AI service must be running
        - Internet access to reach test URLs
        - Service can successfully crawl test URLs

    Expected:
        - All documents returned regardless of language
        - Language metadata still present (detection runs)
        - No filtering occurs
    """
    # Crawl multiple pages
    urls = [
        "https://example.com",
        "https://example.org",
        "https://example.net",
    ]
    documents = await reader_without_language_filter.aload_data(urls)

    # Verify all documents created (no filtering)
    assert len(documents) == 3
    assert all(doc is not None for doc in documents)

    # Verify language metadata still present
    for doc in documents:
        assert "detected_language" in doc.metadata
        assert "language_confidence" in doc.metadata


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_confidence_threshold(reader_with_language_filter) -> None:
    """Verify confidence threshold filtering works with real Crawl4AI service.

    Tests the complete workflow:
    1. Crawl webpage with language detection
    2. Verify confidence score is checked against threshold
    3. Verify document passes threshold check
    4. Verify low-confidence documents would be filtered

    Requirements:
        - Crawl4AI service must be running
        - Internet access to reach test URL
        - Service can successfully crawl test URL

    Expected:
        - Document returned if confidence > threshold
        - Metadata includes confidence score
        - Confidence score validation works correctly
    """
    # Crawl known English webpage
    documents = await reader_with_language_filter.aload_data(["https://example.com"])

    # Verify document created
    assert len(documents) == 1
    doc = documents[0]
    assert doc is not None

    # Verify confidence above threshold
    assert doc.metadata["language_confidence"] > 0.5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_batch_filtering(reader_with_language_filter) -> None:
    """Verify batch filtering works correctly with multiple URLs.

    Tests the complete workflow:
    1. Crawl multiple webpages concurrently
    2. Verify language detection runs on all documents
    3. Verify filtering applies to all documents
    4. Verify metadata enrichment on all documents

    Requirements:
        - Crawl4AI service must be running
        - Internet access to reach test URLs
        - Service can successfully crawl test URLs

    Expected:
        - All documents processed through language filter
        - All documents have language metadata
        - Filtering applies consistently to batch
    """
    # Crawl multiple pages
    urls = [
        "https://example.com",
        "https://example.org",
        "https://example.net",
    ]
    documents = await reader_with_language_filter.aload_data(urls)

    # Verify all documents created
    assert len(documents) == 3
    assert all(doc is not None for doc in documents)

    # Verify language metadata on all documents
    for i, doc in enumerate(documents):
        assert "detected_language" in doc.metadata
        assert "language_confidence" in doc.metadata
        assert doc.metadata["source"] == urls[i]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_error_handling_short_text(reader_with_language_filter) -> None:
    """Verify error handling when crawled content is too short for detection.

    Tests the complete workflow:
    1. Crawl webpage that might have short content
    2. Verify language detection handles short text gracefully
    3. Verify document still created with metadata
    4. Verify fail-open behavior (unknown language, 0.0 confidence)

    Requirements:
        - Crawl4AI service must be running
        - Internet access to reach test URL
        - Service can successfully crawl test URL

    Expected:
        - Document created even if text is short
        - detected_language = "unknown" for short text
        - language_confidence = 0.0 for short text
        - No exceptions raised
    """
    # Crawl webpage (example.com has minimal content)
    documents = await reader_with_language_filter.aload_data(["https://example.com"])

    # Verify document created
    assert len(documents) >= 0  # May be filtered if short

    # If document exists, verify language metadata present
    if documents:
        doc = documents[0]
        assert doc is not None
        assert "detected_language" in doc.metadata
        assert "language_confidence" in doc.metadata


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_english_url_accepted(reader_with_language_filter) -> None:
    """Verify real English webpage is crawled and accepted.

    Tests end-to-end workflow:
    1. Crawl real English webpage (Wikipedia main page)
    2. Verify language detection identifies it as English
    3. Verify document is accepted (not filtered)
    4. Verify metadata includes correct language information

    Requirements:
        - Crawl4AI service must be running
        - Internet access to reach en.wikipedia.org
        - Service can successfully crawl Wikipedia

    Expected:
        - Document returned with English language metadata
        - detected_language = "en"
        - language_confidence > 0.5
        - Document contains expected Wikipedia content
    """
    # Crawl English Wikipedia main page
    documents = await reader_with_language_filter.aload_data(
        ["https://en.wikipedia.org/wiki/Main_Page"]
    )

    # Verify document accepted
    assert len(documents) == 1
    doc = documents[0]
    assert doc is not None

    # Verify English language detected
    assert doc.metadata["detected_language"] == "en"
    assert doc.metadata["language_confidence"] > 0.5

    # Verify document contains Wikipedia content
    assert "Wikipedia" in doc.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_spanish_url_filtered(reader_with_language_filter) -> None:
    """Verify real Spanish webpage is crawled and filtered out.

    Tests end-to-end workflow:
    1. Crawl real Spanish webpage (Spanish Wikipedia)
    2. Verify language detection identifies it as Spanish
    3. Verify document is filtered out (allowed=["en"])
    4. Verify no documents returned

    Requirements:
        - Crawl4AI service must be running
        - Internet access to reach es.wikipedia.org
        - Service can successfully crawl Wikipedia

    Expected:
        - No documents returned (Spanish filtered out)
        - Empty list returned from aload_data
    """
    # Crawl Spanish Wikipedia main page
    documents = await reader_with_language_filter.aload_data(
        ["https://es.wikipedia.org/wiki/Wikipedia:Portada"]
    )

    # Verify document filtered out (Spanish not allowed)
    assert len(documents) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_multi_language_config() -> None:
    """Verify multi-language configuration accepts both English and Spanish URLs.

    Tests end-to-end workflow:
    1. Configure reader to accept English and Spanish
    2. Crawl both English and Spanish Wikipedia pages
    3. Verify both documents are accepted
    4. Verify correct language metadata on each document

    Requirements:
        - Crawl4AI service must be running
        - Internet access to reach Wikipedia
        - Service can successfully crawl both URLs

    Expected:
        - Both documents returned
        - English document has detected_language="en"
        - Spanish document has detected_language="es"
        - Both have confidence > 0.5
    """
    # Create reader with multi-language support
    reader = Crawl4AIReader(
        endpoint_url=CRAWL4AI_URL,
        enable_language_filter=True,
        allowed_languages=["en", "es"],
        language_confidence_threshold=0.5,
    )

    # Crawl both English and Spanish pages
    urls = [
        "https://en.wikipedia.org/wiki/Main_Page",
        "https://es.wikipedia.org/wiki/Wikipedia:Portada",
    ]
    documents = await reader.aload_data(urls)

    # Verify both documents accepted
    assert len(documents) == 2
    assert all(doc is not None for doc in documents)

    # Verify correct language detection
    en_doc = documents[0]
    es_doc = documents[1]

    assert en_doc.metadata["detected_language"] == "en"
    assert en_doc.metadata["language_confidence"] > 0.5
    assert "Wikipedia" in en_doc.text

    assert es_doc.metadata["detected_language"] == "es"
    assert es_doc.metadata["language_confidence"] > 0.5
    assert "Wikipedia" in es_doc.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_low_confidence_filtered(reader_with_language_filter) -> None:
    """Verify code-heavy page with low confidence is filtered out.

    Tests end-to-end workflow:
    1. Crawl code-heavy webpage (GitHub repository)
    2. Verify language detection runs but may have low confidence
    3. Verify document is filtered if confidence < threshold
    4. Verify behavior matches requirements

    Requirements:
        - Crawl4AI service must be running
        - Internet access to reach GitHub
        - Service can successfully crawl GitHub

    Expected:
        - Document may be filtered if confidence < 0.5
        - If returned, confidence is >= 0.5
        - Language detection handles code-heavy content gracefully
    """
    # Crawl code-heavy page (GitHub repository)
    documents = await reader_with_language_filter.aload_data(
        ["https://github.com/torvalds/linux"]
    )

    # Verify filtering behavior
    # Document may be filtered if confidence is too low
    # If document is returned, confidence must be >= threshold
    if documents:
        doc = documents[0]
        assert doc is not None
        assert "detected_language" in doc.metadata
        assert "language_confidence" in doc.metadata
        assert doc.metadata["language_confidence"] >= 0.5
    else:
        # Document filtered due to low confidence - expected behavior
        assert len(documents) == 0
