from dataclasses import dataclass, field
from time import perf_counter

from llm_runtime.core.response import ScheduledRequest


@dataclass(slots=True)
class Batch:
    """A bounded group of scheduled requests executed together."""

    items: list[ScheduledRequest]
    created_at: float = field(default_factory=perf_counter)

    @property
    def size(self) -> int:
        return len(self.items)


def build_batch(
    items: list[ScheduledRequest],
    max_batch_size: int,
) -> Batch:
    """Select a bounded batch from waiting requests."""

    return Batch(items=items[:max_batch_size])
