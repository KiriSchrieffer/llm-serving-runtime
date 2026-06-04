import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from llm_runtime.backends.base import Backend, BackendCapability
from llm_runtime.core.request import RuntimeRequest


_DEFAULT_VLLM_COMMAND = "vllm"


class VLLMBackend(Backend):
    """Async backend that manages a `vllm serve` subprocess.

    The subprocess is started / stopped via :meth:start / :meth:stop,
    which are called by `RuntimeServices` during the FastAPI lifespan.
    Generation uses the OpenAI-compatible HTTP API exposed by vLLM.
    """

    def __init__(
        self,
        model_path: str,
        vllm_command: str = _DEFAULT_VLLM_COMMAND,
        port: int = 8082,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        verbose: bool = False,
    ) -> None:
        self._model_path = model_path
        self._vllm_command = vllm_command
        self._port = port
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._verbose = verbose
        self._process: asyncio.subprocess.Process | None = None
        self._client: httpx.AsyncClient | None = None
        self._base_url = f"http://127.0.0.1:{port}"

    @property
    def capabilities(self) -> BackendCapability:
        return (
            BackendCapability.STREAMING
            | BackendCapability.NATIVE_BATCHING
            | BackendCapability.GPU_METRICS
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start `vllm serve` and wait for it to become healthy."""
        if self._client is not None:
            return

        args = [
            self._vllm_command,
            "serve",
            self._model_path,
            "--host", "127.0.0.1",
            "--port", str(self._port),
            "--gpu-memory-utilization", str(self._gpu_memory_utilization),
        ]
        if self._max_model_len is not None:
            args.extend(["--max-model-len", str(self._max_model_len)])

        self._process = await asyncio.create_subprocess_exec(*args)

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(120.0, connect=3.0),
        )

        # vLLM may take longer to load a model; allow up to 10 minutes
        for attempt in range(600):
            try:
                r = await self._client.get("/health")
                if r.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(1)

        raise RuntimeError(
            f"vllm serve failed to start on {self._base_url} "
            f"(model={self._model_path})"
        )

    async def stop(self) -> None:
        """Shut down `vllm serve` and release the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10.0)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()
            self._process = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(self, request: RuntimeRequest) -> AsyncIterator[str]:
        client = self._client
        if client is None:
            raise RuntimeError("vllm serve not started; call start() first")

        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]

        payload = {
            "messages": messages,
            "max_tokens": request.sampling.max_tokens,
            "temperature": request.sampling.temperature,
            "top_p": request.sampling.top_p,
            "stream": True,
        }

        async with client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line or line.startswith(":") or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    for choice in data.get("choices", []):
                        token = choice.get("delta", {}).get("content", "")
                        if token:
                            yield token