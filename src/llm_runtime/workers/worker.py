from collections.abc import AsyncIterator

from llm_runtime.backends.base import Backend
from llm_runtime.core.request import RuntimeRequest


class Worker:
    """Executes one request against a backend."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    async def stream(self, request: RuntimeRequest) -> AsyncIterator[str]:
        async for token in self.backend.generate(request):
            request.mark_first_token()
            yield token
        request.mark_completed()

    async def collect(self, request: RuntimeRequest) -> list[str]:
        tokens: list[str] = []
        async for token in self.stream(request):
            tokens.append(token)
        return tokens

