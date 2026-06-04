from collections import deque

from llm_runtime.core.request import RuntimeRequest


class RequestQueue:
    """Small testable FIFO queue wrapper for scheduler internals."""

    def __init__(self) -> None:
        self._items: deque[RuntimeRequest] = deque()

    def push(self, request: RuntimeRequest) -> None:
        self._items.append(request)

    def pop(self) -> RuntimeRequest | None:
        if not self._items:
            return None
        return self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)

