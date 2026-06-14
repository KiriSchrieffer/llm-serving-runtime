from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Flag, auto

from llm_runtime.core.request import RuntimeRequest


class BackendCapability(Flag):
    """Capabilities declared by a backend to guide scheduling and metrics collection."""

    STREAMING = auto()
    NATIVE_BATCHING = auto()
    GPU_METRICS = auto()


@dataclass(slots=True)
class BatchToken:
    """One generated token associated with its request inside a batch."""

    request_id: str
    token: str


@dataclass(slots=True)
class BatchCompleted:
    """Signal that one request in a batch finished generation."""

    request_id: str


BatchEvent = BatchToken | BatchCompleted


class Backend(ABC):
    """Common backend interface for token generation."""

    @property
    def capabilities(self) -> BackendCapability:
        """Bitfield of capabilities this backend supports."""
        return BackendCapability(0)

    @abstractmethod
    def generate(self, request: RuntimeRequest) -> AsyncIterator[str]:
        """Yield generated tokens for a request."""

    async def generate_batch(
        self,
        requests: list[RuntimeRequest],
    ) -> AsyncIterator[BatchEvent]:
        """Yield events for a batch using single-request compatibility mode.

        Backends with native or simulated batch support should override this
        method so one batch step advances multiple requests.
        """

        for request in requests:
            async for token in self.generate(request):
                yield BatchToken(request_id=request.request_id, token=token)
            yield BatchCompleted(request_id=request.request_id)
