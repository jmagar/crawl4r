"""HTTP client for Crawl4AI service communication."""

import httpx

from crawl4r.readers.crawl.language_detector import LanguageDetector
from crawl4r.readers.crawl.models import CrawlResult


class HttpCrawlClient:
    """HTTP client for Crawl4AI service.

    Handles HTTP communication with the Crawl4AI service, converting
    responses into CrawlResult objects for downstream processing.

    Note: Retry logic is handled at the orchestration level by
    Crawl4AIReader._crawl_single_url with circuit breaker pattern.

    Args:
        endpoint_url: Crawl4AI service URL
        timeout: Request timeout in seconds
        language_detector: Optional LanguageDetector for content language detection
    """

    def __init__(
        self,
        endpoint_url: str,
        timeout: float = 60.0,
        language_detector: LanguageDetector | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.timeout = timeout
        self.language_detector = language_detector

    async def crawl(self, url: str) -> CrawlResult:
        """Crawl URL using Crawl4AI service.

        Makes a POST request to the /md endpoint with fit filter for
        clean markdown extraction.

        Args:
            url: URL to crawl

        Returns:
            CrawlResult with markdown and metadata on success,
            or CrawlResult with error details on failure
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.endpoint_url}/md",
                    json={"url": url, "f": "fit"},
                )

                if response.status_code == 200:
                    data = response.json()
                    markdown = data.get("markdown", "")

                    detected_language = None
                    language_confidence = None
                    if self.language_detector is not None:
                        lang_result = self.language_detector.detect(markdown)
                        detected_language = lang_result.language
                        language_confidence = lang_result.confidence

                    return CrawlResult(
                        url=url,
                        markdown=markdown,
                        title=data.get("title"),
                        description=data.get("description"),
                        status_code=200,
                        success=True,
                        detected_language=detected_language,
                        language_confidence=language_confidence,
                    )
                else:
                    return CrawlResult(
                        url=url,
                        markdown="",
                        status_code=response.status_code,
                        success=False,
                        error=f"HTTP {response.status_code}",
                    )
        except Exception as e:
            return CrawlResult(
                url=url,
                markdown="",
                status_code=0,
                success=False,
                error=str(e),
            )
