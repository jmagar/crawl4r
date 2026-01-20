"""Instrumentation and observability for crawl4r pipeline.

This module provides:
- Custom events for document processing lifecycle
- Span wrappers for performance profiling
- OpenTelemetry integration for production tracing
- Development-friendly logging handlers

Usage:
    # Initialize observability (call once at startup)
    from crawl4r.core.instrumentation import init_observability
    init_observability(enable_otel=True, otel_endpoint="http://localhost:4317")

    # Use span decorator for profiling
    from crawl4r.core.instrumentation import span

    @span("my_operation")
    async def my_async_function():
        ...

    # Dispatch custom events
    from crawl4r.core.instrumentation import dispatcher, DocumentProcessingStartEvent
    dispatcher.event(DocumentProcessingStartEvent(file_path="/path/to/doc.md"))
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.base import BaseEvent
from llama_index.core.instrumentation.span.base import BaseSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from pydantic import Field, PrivateAttr

logger = logging.getLogger(__name__)

# Type variables for generic decorators
P = ParamSpec("P")
T = TypeVar("T")

# Create dispatcher for crawl4r namespace
dispatcher = get_dispatcher("crawl4r")

# Track initialization state (thread-safe)
_initialized = False
_init_lock = threading.Lock()
_added_handlers: list[BaseEventHandler | BaseSpanHandler] = []


# =============================================================================
# Custom Events
# =============================================================================


class DocumentProcessingStartEvent(BaseEvent):
    """Event fired when document processing starts."""

    file_path: str


class DocumentProcessingEndEvent(BaseEvent):
    """Event fired when document processing ends."""

    file_path: str
    success: bool
    error: str | None = None


class ChunkingStartEvent(BaseEvent):
    """Event fired when document chunking starts."""

    file_path: str
    document_length: int


class ChunkingEndEvent(BaseEvent):
    """Event fired when document chunking completes."""

    file_path: str
    num_chunks: int
    duration_ms: float


class EmbeddingBatchEvent(BaseEvent):
    """Event fired after embedding batch completes."""

    batch_size: int
    duration_ms: float
    tokens_processed: int | None = None


class VectorStoreUpsertEvent(BaseEvent):
    """Event fired after vector store upsert."""

    collection_name: str
    point_count: int
    duration_ms: float


class PipelineStartEvent(BaseEvent):
    """Event fired when ingestion pipeline starts."""

    total_documents: int


class PipelineEndEvent(BaseEvent):
    """Event fired when ingestion pipeline completes."""

    total_documents: int
    successful: int
    failed: int
    duration_seconds: float


# =============================================================================
# Span Handler for Performance Profiling
# =============================================================================


class Crawl4rSpan(BaseSpan):
    """Custom span for crawl4r operations."""

    name: str
    start_time: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize span with name and start time."""
        start_time = time.time()
        super().__init__(
            id_=f"crawl4r-{name}-{time.time_ns()}",
            name=name,
            start_time=start_time,
            **kwargs,
        )
        self.name = name
        self.start_time = start_time


class PerformanceSpanHandler(BaseSpanHandler[Crawl4rSpan]):
    """Span handler that logs performance metrics.

    This handler tracks span durations and logs them for performance analysis.
    In production, it can also export to OpenTelemetry.
    """

    log_level: int = logging.DEBUG
    _active_spans: dict[str, list[Crawl4rSpan]] = PrivateAttr(default_factory=dict)
    _active_spans_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def class_name(self) -> str:
        """Return handler class name."""
        return "PerformanceSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Any | None = None,
        parent_span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Crawl4rSpan | None:
        """Create a new span for tracking."""
        new_span_obj = Crawl4rSpan(name=id_)
        if tags:
            new_span_obj.metadata.update(tags)
        with self._active_spans_lock:
            self._active_spans.setdefault(new_span_obj.name, []).append(new_span_obj)
        logger.log(self.log_level, f"[SPAN START] {id_}")
        return new_span_obj

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Any | None = None,
        result: Any | None = None,
        **kwargs: Any,
    ) -> Crawl4rSpan | None:
        """Called when span is about to exit - return and remove the span."""
        with self._active_spans_lock:
            span_stack = self._active_spans.get(id_)
            if not span_stack:
                return None
            span_obj = span_stack.pop()
            if not span_stack:
                del self._active_spans[id_]
        duration_ms = (time.time() - span_obj.start_time) * 1000
        logger.log(self.log_level, f"[SPAN END] {id_} duration={duration_ms:.2f}ms")
        return span_obj

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Any | None = None,
        err: BaseException | None = None,
        **kwargs: Any,
    ) -> Crawl4rSpan | None:
        """Called when span exits with error."""
        with self._active_spans_lock:
            span_stack = self._active_spans.get(id_)
            if not span_stack:
                return None
            span_obj = span_stack.pop()
            if not span_stack:
                del self._active_spans[id_]
        duration_ms = (time.time() - span_obj.start_time) * 1000
        logger.warning(
            f"[SPAN ERROR] {id_} duration={duration_ms:.2f}ms error={err}"
        )
        return span_obj


# =============================================================================
# Event Handler for Logging
# =============================================================================


class LoggingEventHandler(BaseEventHandler):
    """Event handler that logs all events for debugging.

    Useful during development to trace the flow of events through the pipeline.
    """

    log_level: int = logging.DEBUG

    def class_name(self) -> str:
        """Return handler class name."""
        return "LoggingEventHandler"

    def handle(self, event: BaseEvent, **kwargs: Any) -> Any:
        """Log the event with its details."""
        event_name = event.__class__.__name__
        event_data = event.model_dump(exclude={"timestamp", "id_"})
        logger.log(self.log_level, f"[EVENT] {event_name}: {event_data}")


# =============================================================================
# Span Decorator for Easy Profiling
# =============================================================================


def asyncio_iscoroutinefunction(func: Callable[..., Any]) -> bool:
    """Check if function is async, handling wrapped functions."""
    # Check the function itself
    if asyncio.iscoroutinefunction(func):
        return True
    # Check wrapped function
    if hasattr(func, "__wrapped__"):
        return asyncio.iscoroutinefunction(func.__wrapped__)
    # Check for async generator
    if inspect.isasyncgenfunction(func):
        return True
    return False


@contextmanager
def span_context(name: str, **metadata: Any):
    """Context manager for creating manual spans.

    Args:
        name: Name of the span
        **metadata: Additional metadata to attach to the span

    Yields:
        Dict that can be updated with additional metadata during execution

    Example:
        with span_context("process_chunk", chunk_id=1) as ctx:
            result = process_chunk(data)
            ctx["result_size"] = len(result)
    """
    start_time = time.time()
    ctx: dict[str, Any] = {"name": name, **metadata}
    had_exception = False
    logger.debug(f"[SPAN START] {name} metadata={metadata}")
    try:
        yield ctx
    except Exception as e:
        had_exception = True
        duration_ms = (time.time() - start_time) * 1000
        logger.warning(f"[SPAN ERROR] {name} duration={duration_ms:.2f}ms error={e}")
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        ctx["duration_ms"] = duration_ms
        if not had_exception:
            logger.debug(f"[SPAN END] {name} duration={duration_ms:.2f}ms")


def span(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to wrap a function in a performance span.

    Works with both sync and async functions.

    Args:
        name: Name of the span (operation being tracked)

    Returns:
        Decorated function with automatic span tracking

    Example:
        @span("embed_documents")
        async def embed_documents(texts: list[str]) -> list[list[float]]:
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if asyncio_iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                start_time = time.time()
                logger.debug(f"[SPAN START] {name}")
                try:
                    result = await func(*args, **kwargs)  # ty: ignore[invalid-await]
                    duration_ms = (time.time() - start_time) * 1000
                    logger.debug(f"[SPAN END] {name} duration={duration_ms:.2f}ms")
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.warning(
                        f"[SPAN ERROR] {name} duration={duration_ms:.2f}ms error={e}"
                    )
                    raise

            return async_wrapper  # ty: ignore[invalid-return-type]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                start_time = time.time()
                logger.debug(f"[SPAN START] {name}")
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    logger.debug(f"[SPAN END] {name} duration={duration_ms:.2f}ms")
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.warning(
                        f"[SPAN ERROR] {name} duration={duration_ms:.2f}ms error={e}"
                    )
                    raise

            return sync_wrapper

    return decorator


# =============================================================================
# Initialization Functions
# =============================================================================


def init_observability(
    *,
    enable_otel: bool = False,
    otel_endpoint: str | None = None,
    otel_service_name: str = "crawl4r",
    enable_logging_handler: bool = True,
    enable_performance_spans: bool = True,
    log_level: int = logging.DEBUG,
) -> None:
    """Initialize observability for the crawl4r pipeline.

    This function should be called once at application startup to configure
    event handlers, span handlers, and optionally OpenTelemetry export.

    Args:
        enable_otel: Enable OpenTelemetry tracing export
        otel_endpoint: OTLP endpoint (e.g., "http://localhost:4317").
            If None, uses OTEL_EXPORTER_OTLP_ENDPOINT env var or localhost:4317
        otel_service_name: Service name for OTEL traces
        enable_logging_handler: Enable debug logging of all events
        enable_performance_spans: Enable span tracking for performance profiling
        log_level: Logging level for handlers (default: DEBUG)

    Environment Variables:
        CRAWL4R_ENABLE_OTEL: Set to "true" to enable OTEL (alternative to param)
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
        OTEL_SERVICE_NAME: Override service name

    Example:
        # Development setup (logging only)
        init_observability(enable_logging_handler=True)

        # Production setup with OTEL
        init_observability(
            enable_otel=True,
            otel_endpoint="http://jaeger:4317",
            otel_service_name="crawl4r-prod"
        )
    """
    global _initialized, _added_handlers

    with _init_lock:
        if _initialized:
            logger.warning("Observability already initialized, skipping")
            return

        # Check environment for OTEL enable flag
        if os.getenv("CRAWL4R_ENABLE_OTEL", "").lower() == "true":
            enable_otel = True

        # Add logging event handler
        if enable_logging_handler:
            logging_handler = LoggingEventHandler(log_level=log_level)
            dispatcher.add_event_handler(logging_handler)
            _added_handlers.append(logging_handler)
            logger.info("Added LoggingEventHandler to dispatcher")

        # Add performance span handler
        if enable_performance_spans:
            span_handler_obj = PerformanceSpanHandler(log_level=log_level)
            dispatcher.add_span_handler(span_handler_obj)
            _added_handlers.append(span_handler_obj)
            logger.info("Added PerformanceSpanHandler to dispatcher")

        # Initialize OpenTelemetry if requested
        if enable_otel:
            _init_opentelemetry(
                endpoint=otel_endpoint,
                service_name=otel_service_name,
            )

        _initialized = True
        logger.info(
            f"Observability initialized: otel={enable_otel}, "
            f"logging={enable_logging_handler}, spans={enable_performance_spans}"
        )


def _init_opentelemetry(
    endpoint: str | None = None,
    service_name: str = "crawl4r",
) -> None:
    """Initialize OpenTelemetry tracing with OTLP exporter.

    This sets up the global handler for LlamaIndex OTEL integration,
    which automatically traces LLM calls, embeddings, and retrieval.

    Args:
        endpoint: OTLP endpoint URL (uses env var or localhost if None)
        service_name: Service name for traces
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Get endpoint from env if not provided
        if endpoint is None:
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

        # Get service name from env if set
        service_name = os.getenv("OTEL_SERVICE_NAME", service_name)

        # Get package version from metadata
        try:
            from importlib.metadata import version as get_version

            service_version = get_version("crawl4r")
        except Exception:
            service_version = "0.0.0"  # Fallback if package not installed

        # Create resource with service info
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": service_version,
            }
        )

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add OTLP exporter
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
        processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(processor)

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Try to use LlamaIndex OTEL integration if available
        try:
            from llama_index.observability.otel import LlamaIndexOpenTelemetry

            otel_instrumentor = LlamaIndexOpenTelemetry()
            otel_instrumentor.start_registering()
            logger.info(
                f"LlamaIndex OpenTelemetry instrumentation enabled, endpoint={endpoint}"
            )
        except ImportError:
            logger.warning(
                "llama-index-observability-otel not available, "
                "using basic OTEL setup only"
            )

        logger.info(f"OpenTelemetry initialized: endpoint={endpoint}")

    except ImportError as e:
        logger.warning(f"OpenTelemetry packages not available: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")


def shutdown_observability() -> None:
    """Shutdown observability and flush any pending traces.

    Call this before application exit to ensure all traces are exported.
    Removes handlers added by init_observability to support clean re-initialization.
    """
    global _initialized, _added_handlers

    with _init_lock:
        if not _initialized:
            return

        # Remove handlers added during initialization
        for handler in _added_handlers:
            if isinstance(handler, BaseEventHandler):
                try:
                    dispatcher.event_handlers.remove(handler)
                except ValueError:
                    pass  # Handler already removed
            elif isinstance(handler, BaseSpanHandler):
                try:
                    dispatcher.span_handlers.remove(handler)
                except ValueError:
                    pass  # Handler already removed
        _added_handlers.clear()

        # Shutdown OpenTelemetry
        try:
            from opentelemetry import trace

            provider = trace.get_tracer_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
                logger.info("OpenTelemetry tracer provider shut down")
        except Exception as e:
            logger.warning(f"Error shutting down OpenTelemetry: {e}")

        _initialized = False


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    # Dispatcher
    "dispatcher",
    # Events
    "DocumentProcessingStartEvent",
    "DocumentProcessingEndEvent",
    "ChunkingStartEvent",
    "ChunkingEndEvent",
    "EmbeddingBatchEvent",
    "VectorStoreUpsertEvent",
    "PipelineStartEvent",
    "PipelineEndEvent",
    # Handlers
    "LoggingEventHandler",
    "PerformanceSpanHandler",
    # Span utilities
    "span",
    "span_context",
    # Initialization
    "init_observability",
    "shutdown_observability",
]
