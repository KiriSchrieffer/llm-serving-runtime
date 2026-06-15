import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from llm_runtime.backends.base import Backend, BackendCapability
from llm_runtime.core.request import RuntimeRequest


_DEFAULT_SERVER = str(Path(r"E:\llama.cpp\llama-server.exe"))


class LlamaCppBackend(Backend):
    """Async backend that manages a `llama-server` subprocess.

    The subprocess is started / stopped via :meth:start / :meth:stop,
    which are called by `RuntimeServices` during the FastAPI lifespan.
    Generation uses the OpenAI-compatible HTTP API exposed by `llama-server`.
    """

    def __init__(
        self,
        model_path: str,
        llama_server_path: str = _DEFAULT_SERVER,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        port: int = 8081,
        verbose: bool = False,
    ) -> None:
        self._model_path = model_path
        self._llama_server_path = llama_server_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._port = port
        self._verbose = verbose
        self._process: asyncio.subprocess.Process | None = None
        self._client: httpx.AsyncClient | None = None
        self._base_url = f"http://127.0.0.1:{port}"

    @property
    def capabilities(self) -> BackendCapability:
        return BackendCapability.STREAMING | BackendCapability.GPU_METRICS

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start `llama-server` and wait for it to become healthy."""
        if self._client is not None:
            return

        args = [
            self._llama_server_path,
            "-m", self._model_path,
            "--port", str(self._port),
            "--host", "127.0.0.1",
            "-c", str(self._n_ctx),
            "-ngl", str(self._n_gpu_layers),
            "--no-webui",
        ]
        if not self._verbose:
            args.append("--log-disable")

        self._process = await asyncio.create_subprocess_exec(*args)

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(120.0, connect=3.0),
        )

        for attempt in range(120):
            try:
                r = await self._client.get("/health")
                if r.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(1)

        raise RuntimeError(
            f"llama-server failed to start on {self._base_url} "
            f"(model={self._model_path})"
        )

    async def stop(self) -> None:
        """Shut down `llama-server` and release the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
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
            raise RuntimeError("llama-server not started; call start() first")

        async with client.stream(
            "POST", "/v1/chat/completions", json=self._chat_payload(request)
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

    def _chat_payload(self, request: RuntimeRequest) -> dict[str, object]:
        return {
            "model": self._model_path,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ],
            "max_tokens": request.sampling.max_tokens,
            "temperature": request.sampling.temperature,
            "top_p": request.sampling.top_p,
            "stream": True,
        }
