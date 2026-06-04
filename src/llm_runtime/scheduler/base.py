from abc import ABC, abstractmethod

from llm_runtime.core.response import ScheduledRequest
from llm_runtime.scheduler.batching import Batch


class Scheduler(ABC):
    """Common async scheduler interface."""

    @abstractmethod
    async def submit(self, item: ScheduledRequest) -> None:
        """Submit a request to the scheduler."""

    @abstractmethod
    async def next_request(self) -> ScheduledRequest:
        """Wait for and return the next scheduled request."""

    @abstractmethod
    async def next_batch(self, max_batch_size: int, batch_timeout_ms: int) -> Batch:
        """Wait for the first request, then collect a bounded micro-batch."""

    @abstractmethod
    def size(self) -> int:
        """Return the current queue size."""
