import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from watchdog.observers import Observer

from crawl4r.core.config import Settings
from crawl4r.readers.file_watcher import FileWatcher


@pytest.mark.integration
@pytest.mark.watcher
async def test_file_watcher_integration_with_fix(tmp_path):
    """
    Verify that FileWatcher correctly triggers the processor when a file is modified,
    confirming the fix where the asyncio loop is passed to the watcher.
    """
    # 1. Setup Mocks
    config = Mock(spec=Settings)
    config.watch_folder = tmp_path

    processor = AsyncMock()
    # Mock process_document to return a success result or just finish
    processor.process_document.return_value = None

    vector_store = Mock()
    vector_store.delete_by_file.return_value = 0

    # 2. Initialize FileWatcher WITH the loop (The Fix)
    loop = asyncio.get_running_loop()

    # We use an event queue to synchronize the test
    # FileWatcher puts events into this queue AFTER processing if provided
    event_queue = asyncio.Queue()

    watcher = FileWatcher(
        config=config,
        processor=processor,
        vector_store=vector_store,
        event_queue=event_queue,
        loop=loop
    )

    # 3. Start Observer
    observer = Observer()
    observer.schedule(watcher, str(tmp_path), recursive=True)
    observer.start()

    try:
        # Give observer a moment to start
        await asyncio.sleep(0.1)

        # 4. Create a file
        file_path = tmp_path / "test_integration.md"
        file_path.write_text("# Test Content")

        # 5. Wait for "created" event in the queue
        # The FileWatcher logic puts the event in the queue *after* processor returns
        try:
            event_type, path = await asyncio.wait_for(event_queue.get(), timeout=3.0)
            # Debouncing might cancel 'created' and send 'modified'
            # if they happen close together
            assert event_type in ("created", "modified")
            assert path == file_path
        except asyncio.TimeoutError:
            pytest.fail(
                "Timeout waiting for 'created' event processing. "
                "The loop fix might be missing or not working."
            )

        # Verify processor called
        processor.process_document.assert_called_with(file_path)
        processor.process_document.reset_mock()

        # 6. Modify the file
        file_path.write_text("# Modified Content")

        # 7. Wait for "modified" event
        try:
            event_type, path = await asyncio.wait_for(event_queue.get(), timeout=3.0)
            assert event_type == "modified"
            assert path == file_path
        except asyncio.TimeoutError:
            pytest.fail("Timeout waiting for 'modified' event processing")

        # Verify processor called again
        processor.process_document.assert_called_with(file_path)

    finally:
        observer.stop()
        observer.join()
