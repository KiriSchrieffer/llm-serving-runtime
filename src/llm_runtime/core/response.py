import asyncio
from dataclasses import dataclass

from llm_runtime.core.request import RuntimeRequest


@dataclass(slots=True)
class GenerationResult:
    """Complete token generation result returned to a non-streaming caller."""

    request_id: str
    tokens: list[str]
    finish_reason: str = "stop"


@dataclass(slots=True)
class TokenEvent:
    """One token delivered to a streaming caller."""

    token: str


@dataclass(slots=True)
class CompletedEvent:
    """Signal that streaming generation finished normally."""

    finish_reason: str = "stop"


@dataclass(slots=True)
class ErrorEvent:
    """Signal that streaming generation failed."""

    message: str


StreamEvent = TokenEvent | CompletedEvent | ErrorEvent


@dataclass(slots=True)
class CompletionHandle:
    """Per-request completion channel for a JSON response."""

    future: asyncio.Future[GenerationResult]
    cancelled: bool = False

    @classmethod
    def create(cls) -> "CompletionHandle":
        loop = asyncio.get_running_loop()
        return cls(future=loop.create_future())

    def cancel(self) -> bool:
        if self.cancelled:
            return False
        self.cancelled = True
        if not self.future.done():
            self.future.cancel()
        return True


@dataclass(slots=True)
class StreamingHandle:
    """Per-request event channel for an SSE response."""

    queue: asyncio.Queue[StreamEvent]
    cancelled: bool = False
    completed: bool = False

    @classmethod
    def create(cls) -> "StreamingHandle":
        return cls(queue=asyncio.Queue())

    def cancel(self) -> bool:
        if self.cancelled or self.completed:
            return False
        self.cancelled = True
        return True

    def mark_completed(self) -> None:
        self.completed = True


ResponseHandle = CompletionHandle | StreamingHandle


@dataclass(slots=True)
class ScheduledRequest:
    """A runtime request paired with its caller-specific response channel."""

    request: RuntimeRequest
    handle: ResponseHandle

    @property
    def cancelled(self) -> bool:
        return self.handle.cancelled
