import asyncio
from time import perf_counter

from llm_runtime.core.response import ScheduledRequest
from llm_runtime.scheduler.base import Scheduler
from llm_runtime.scheduler.batching import Batch


class PriorityScheduler(Scheduler):
    """Priority-ordered request scheduler with FIFO within each priority level.

    Lower priority values are treated as higher priority (0 is highest).
    Within the same priority level, requests are dequeued in FIFO order
    using a monotonic sequence counter assigned at submit time.
    """

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, int, ScheduledRequest]] = (
            asyncio.PriorityQueue()
        )
        self._counter = 0

    async def submit(self, item: ScheduledRequest) -> None:
        item.request.mark_enqueued()
        self._counter += 1
        await self._queue.put((item.request.priority, self._counter, item))

    async def next_request(self) -> ScheduledRequest:
        _, _, item = await self._queue.get()
        item.request.mark_started()
        return item

    async def next_batch(self, max_batch_size: int, batch_timeout_ms: int) -> Batch:
        _, _, first = await self._queue.get()
        items = [first]
        timeout_s = batch_timeout_ms / 1000
        deadline = perf_counter() + timeout_s

        while len(items) < max_batch_size:
            if timeout_s == 0:
                try:
                    _, _, item = self._queue.get_nowait()
                    items.append(item)
                except asyncio.QueueEmpty:
                    break
                continue

            remaining_s = deadline - perf_counter()
            if remaining_s <= 0:
                break
            try:
                _, _, item = await asyncio.wait_for(
                    self._queue.get(), timeout=remaining_s
                )
                items.append(item)
            except TimeoutError:
                break

        for item in items:
            item.request.mark_started()
        return Batch(items=items)

    def size(self) -> int:
        return self._queue.qsize()
