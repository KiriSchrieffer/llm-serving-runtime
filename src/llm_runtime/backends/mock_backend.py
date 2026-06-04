import asyncio
from collections.abc import AsyncIterator

from llm_runtime.backends.base import Backend, BackendCapability, BatchCompleted, BatchEvent, BatchToken
from llm_runtime.core.request import RuntimeRequest


class MockBackend(Backend):
    """Async backend that simulates prefill and decode latency."""

    def __init__(self, prefill_latency_ms: int = 25, decode_latency_ms: int = 10) -> None:
        self.prefill_latency_s = prefill_latency_ms / 1000
        self.decode_latency_s = decode_latency_ms / 1000

    @property
    def capabilities(self) -> BackendCapability:
        return BackendCapability.STREAMING

    async def generate(self, request: RuntimeRequest) -> AsyncIterator[str]:
        await asyncio.sleep(self.prefill_latency_s)
        for index in range(request.sampling.max_tokens):
            await asyncio.sleep(self.decode_latency_s)
            yield f"tok{index} "

    async def generate_batch(
        self,
        requests: list[RuntimeRequest],
    ) -> AsyncIterator[BatchEvent]:
        """Simulate batched prefill and decode steps for waiting requests."""

        if not requests:
            return
        await asyncio.sleep(self.prefill_latency_s)
        max_tokens = max(request.sampling.max_tokens for request in requests)
        for index in range(max_tokens):
            await asyncio.sleep(self.decode_latency_s)
            for request in requests:
                if index >= request.sampling.max_tokens:
                    continue
                yield BatchToken(request_id=request.request_id, token=f"tok{index} ")
                if index + 1 == request.sampling.max_tokens:
                    yield BatchCompleted(request_id=request.request_id)
