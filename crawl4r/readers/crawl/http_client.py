"""HTTP client for Crawl4AI service communication."""

import httpx

from crawl4r.readers.crawl.models import CrawlResult


class HttpCrawlClient:
    """HTTP client for Crawl4AI service.

    Handles HTTP communication with the Crawl4AI service, converting
    responses into CrawlResult objects for downstream processing.

    Args:
        endpoint_url: Crawl4AI service URL
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
    """

    def __init__(
        self,
        endpoint_url: str,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

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
                    return CrawlResult(
                        url=url,
                        markdown=data.get("markdown", ""),
                        title=data.get("title"),
                        description=data.get("description"),
                        status_code=200,
                        success=True,
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
