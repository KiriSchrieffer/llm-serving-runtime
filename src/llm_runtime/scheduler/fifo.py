import asyncio
from time import perf_counter

from llm_runtime.core.response import ScheduledRequest
from llm_runtime.scheduler.base import Scheduler
from llm_runtime.scheduler.batching import Batch


class FIFOScheduler(Scheduler):
    """First-in, first-out request scheduler."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[ScheduledRequest] = asyncio.Queue()

    async def submit(self, item: ScheduledRequest) -> None:
        item.request.mark_enqueued()
        await self._queue.put(item)

    async def next_request(self) -> ScheduledRequest:
        item = await self._queue.get()
        item.request.mark_started()
        return item

    async def next_batch(self, max_batch_size: int, batch_timeout_ms: int) -> Batch:
        first = await self._queue.get()
        items = [first]
        timeout_s = batch_timeout_ms / 1000
        deadline = perf_counter() + timeout_s

        while len(items) < max_batch_size:
            if timeout_s == 0:
                try:
                    items.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
                continue

            remaining_s = deadline - perf_counter()
            if remaining_s <= 0:
                break
            try:
                items.append(
                    await asyncio.wait_for(self._queue.get(), timeout=remaining_s)
                )
            except TimeoutError:
                break

        for item in items:
            item.request.mark_started()
        return Batch(items=items)

    def size(self) -> int:
        return self._queue.qsize()
