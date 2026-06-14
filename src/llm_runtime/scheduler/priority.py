import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from math import floor
from time import perf_counter

from llm_runtime.core.response import ScheduledRequest
from llm_runtime.scheduler.base import Scheduler
from llm_runtime.scheduler.batching import Batch


@dataclass(slots=True)
class _QueuedRequest:
    priority: int
    sequence: int
    queued_at: float
    item: ScheduledRequest


class PriorityScheduler(Scheduler):
    """Priority-ordered request scheduler with optional aging.

    Lower priority values are treated as higher priority (0 is highest).
    Within the same effective priority level, requests are dequeued in FIFO
    order using a monotonic sequence counter assigned at submit time.

    If aging_boost_interval_s is greater than zero, a queued request's effective
    priority improves by one level for each interval it waits. This lets old
    low-priority requests eventually compete with new high-priority traffic.
    """

    def __init__(
        self,
        aging_boost_interval_s: float = 0.0,
        time_fn: Callable[[], float] = perf_counter,
    ) -> None:
        self.aging_boost_interval_s = aging_boost_interval_s
        self._time_fn = time_fn
        self._items: list[_QueuedRequest] = []
        self._counter = 0
        self._condition = asyncio.Condition()

    async def submit(self, item: ScheduledRequest) -> None:
        item.request.mark_enqueued()
        async with self._condition:
            self._counter += 1
            self._items.append(
                _QueuedRequest(
                    priority=item.request.priority,
                    sequence=self._counter,
                    queued_at=self._time_fn(),
                    item=item,
                )
            )
            self._condition.notify()

    async def next_request(self) -> ScheduledRequest:
        async with self._condition:
            item = await self._pop_next_when_available()
        item.request.mark_started()
        return item

    async def next_batch(self, max_batch_size: int, batch_timeout_ms: int) -> Batch:
        async with self._condition:
            first = await self._pop_next_when_available()
            items = [first]
            timeout_s = batch_timeout_ms / 1000
            deadline = self._time_fn() + timeout_s

            while len(items) < max_batch_size:
                if self._items:
                    items.append(self._pop_next_locked())
                    continue
                if timeout_s == 0:
                    break

                remaining_s = deadline - self._time_fn()
                if remaining_s <= 0:
                    break
                try:
                    await asyncio.wait_for(
                        self._condition.wait(),
                        timeout=remaining_s,
                    )
                except TimeoutError:
                    break

        for item in items:
            item.request.mark_started()
        return Batch(items=items)

    def size(self) -> int:
        return len(self._items)

    async def _pop_next_when_available(self) -> ScheduledRequest:
        while not self._items:
            await self._condition.wait()
        return self._pop_next_locked()

    def _pop_next_locked(self) -> ScheduledRequest:
        now = self._time_fn()
        best_index = min(
            range(len(self._items)),
            key=lambda index: (
                self._effective_priority(self._items[index], now),
                self._items[index].sequence,
            ),
        )
        return self._items.pop(best_index).item

    def _effective_priority(self, queued: _QueuedRequest, now: float) -> int:
        if self.aging_boost_interval_s <= 0:
            return queued.priority
        waited_s = max(0.0, now - queued.queued_at)
        boost = floor(waited_s / self.aging_boost_interval_s)
        return max(0, queued.priority - boost)
