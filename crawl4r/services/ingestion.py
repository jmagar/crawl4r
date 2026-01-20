"""Ingestion service for crawling and indexing URLs."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

from crawl4r.core.config import Settings
from crawl4r.core.metadata import MetadataKeys
from crawl4r.services.models import (
    CrawlStatus,
    CrawlStatusInfo,
    IngestResult,
    ScrapeResult,
)
from crawl4r.services.queue import QueueManager
from crawl4r.services.scraper import ScraperService
from crawl4r.storage.qdrant import VectorStoreManager
from crawl4r.storage.tei import TEIClient


def generate_crawl_id() -> str:
    """Generate a unique crawl identifier.

    Returns:
        Crawl identifier prefixed with "crawl_"
    """
    timestamp = int(time.time() * 1000)
    nonce = random.randint(1000, 9999)
    return f"crawl_{timestamp}_{nonce}"


class IngestionService:
    """Coordinate scraping, embedding, and vector store ingestion."""

    def __init__(
        self,
        scraper: ScraperService | None = None,
        embeddings: TEIClient | Any | None = None,
        vector_store: VectorStoreManager | Any | None = None,
        queue_manager: QueueManager | Any | None = None,
        node_parser: MarkdownNodeParser | None = None,
    ) -> None:
        """Initialize ingestion service dependencies.

        Args:
            scraper: Scraper service for Crawl4AI requests
            embeddings: TEI client for embedding generation
            vector_store: Vector store manager for Qdrant
            queue_manager: Redis-backed queue manager
            node_parser: Markdown node parser for chunking
        """
        if scraper and embeddings and vector_store and queue_manager:
            self.scraper = scraper
            self.embeddings = embeddings
            self.vector_store = vector_store
            self.queue_manager = queue_manager
        else:
            settings = Settings(watch_folder=Path("."))
            self.scraper = scraper or ScraperService(
                endpoint_url=settings.CRAWL4AI_BASE_URL
            )
            self.embeddings = embeddings or TEIClient(settings.tei_endpoint)
            self.vector_store = vector_store or VectorStoreManager(
                qdrant_url=settings.qdrant_url,
                collection_name=settings.collection_name,
            )
            self.queue_manager = queue_manager or QueueManager(settings.REDIS_URL)

        self.node_parser = node_parser or MarkdownNodeParser()

    async def ingest_urls(
        self, urls: list[str], max_concurrent: int = 5
    ) -> IngestResult:
        """Ingest URLs into the vector store.

        Args:
            urls: List of URLs to crawl
            max_concurrent: Maximum concurrent scrape requests

        Returns:
            IngestResult summarizing the ingestion outcome
        """
        crawl_id = generate_crawl_id()
        urls_total = len(urls)
        urls_failed = 0
        chunks_created = 0

        if not urls:
            return IngestResult(
                crawl_id=crawl_id,
                success=False,
                error="No URLs provided",
                urls_total=0,
                urls_failed=0,
                chunks_created=0,
                queued=False,
            )

        lock_owner = crawl_id
        lock_acquired = await self.queue_manager.acquire_lock(lock_owner)
        if not lock_acquired:
            await self.queue_manager.enqueue_crawl(crawl_id, urls)
            await self.queue_manager.set_status(
                CrawlStatusInfo(crawl_id=crawl_id, status=CrawlStatus.QUEUED)
            )
            return IngestResult(
                crawl_id=crawl_id,
                success=True,
                error=None,
                urls_total=urls_total,
                urls_failed=0,
                chunks_created=0,
                queued=True,
            )

        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await self.queue_manager.set_status(
            CrawlStatusInfo(
                crawl_id=crawl_id,
                status=CrawlStatus.RUNNING,
                started_at=started_at,
            )
        )

        try:
            results = await self.scraper.scrape_urls(
                urls, max_concurrent=max_concurrent
            )
            for result in results:
                if not result.success or not result.markdown:
                    urls_failed += 1
                    continue
                await self._ingest_result(result)
                chunks_created += self._count_chunks(result)

            finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            status = CrawlStatus.COMPLETED if urls_failed == 0 else CrawlStatus.FAILED
            await self.queue_manager.set_status(
                CrawlStatusInfo(
                    crawl_id=crawl_id,
                    status=status,
                    started_at=started_at,
                    finished_at=finished_at,
                    error=None if urls_failed == 0 else "One or more URLs failed",
                )
            )

            return IngestResult(
                crawl_id=crawl_id,
                success=urls_failed == 0,
                error=None if urls_failed == 0 else "One or more URLs failed",
                urls_total=urls_total,
                urls_failed=urls_failed,
                chunks_created=chunks_created,
                queued=False,
            )
        finally:
            await self.queue_manager.release_lock(lock_owner)

    async def _ingest_result(self, result: ScrapeResult) -> None:
        document = Document(
            text=result.markdown or "",
            metadata=self._document_metadata(result),
        )
        nodes = self.node_parser.get_nodes_from_documents([document])
        if not nodes:
            return

        texts = [self._node_text(node) for node in nodes]
        vectors = await self.embeddings.embed_batch(texts)

        await self.vector_store.delete_by_url(result.url)
        vectors_with_metadata = []
        for index, (text, vector, node) in enumerate(zip(texts, vectors, nodes)):
            metadata = self._vector_metadata(result, text, index, node.metadata)
            vectors_with_metadata.append({"vector": vector, "metadata": metadata})

        await self.vector_store.upsert_vectors_batch(vectors_with_metadata)

    def _node_text(self, node: Any) -> str:
        if hasattr(node, "get_content"):
            return node.get_content()
        return getattr(node, "text", "")

    def _document_metadata(self, result: ScrapeResult) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            MetadataKeys.FILE_PATH: result.url,
            MetadataKeys.SOURCE_URL: result.url,
            MetadataKeys.SOURCE_TYPE: "web_crawl",
        }
        if result.metadata:
            if "title" in result.metadata:
                metadata[MetadataKeys.TITLE] = result.metadata["title"]
            if "description" in result.metadata:
                metadata[MetadataKeys.DESCRIPTION] = result.metadata["description"]
            if "status_code" in result.metadata:
                metadata[MetadataKeys.STATUS_CODE] = result.metadata["status_code"]
        return metadata

    def _vector_metadata(
        self,
        result: ScrapeResult,
        text: str,
        index: int,
        node_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            MetadataKeys.FILE_PATH: result.url,
            MetadataKeys.CHUNK_INDEX: index,
            MetadataKeys.CHUNK_TEXT: text,
            MetadataKeys.SOURCE_URL: result.url,
            MetadataKeys.SOURCE_TYPE: "web_crawl",
        }
        if MetadataKeys.SECTION_PATH in node_metadata:
            metadata[MetadataKeys.SECTION_PATH] = node_metadata[
                MetadataKeys.SECTION_PATH
            ]
        if MetadataKeys.HEADING_LEVEL in node_metadata:
            metadata[MetadataKeys.HEADING_LEVEL] = node_metadata[
                MetadataKeys.HEADING_LEVEL
            ]
        return metadata

    def _count_chunks(self, result: ScrapeResult) -> int:
        document = Document(text=result.markdown or "")
        nodes = self.node_parser.get_nodes_from_documents([document])
        return len(nodes)
