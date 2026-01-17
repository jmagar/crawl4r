from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.events.base import BaseEvent

# Create dispatcher for crawl4r namespace
dispatcher = get_dispatcher("crawl4r")

class DocumentProcessingStartEvent(BaseEvent):
    """Event fired when document processing starts."""
    file_path: str

class DocumentProcessingEndEvent(BaseEvent):
    """Event fired when document processing ends."""
    file_path: str
    success: bool
    error: str | None = None
