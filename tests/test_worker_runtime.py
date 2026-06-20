import asyncio
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from fastapi.testclient import TestClient

from llm_runtime.backends.base import (
    Backend,
    BackendCapability,
    BatchCompleted,
    BatchEvent,
    BatchToken,
)
from llm_runtime.config import Settings
from llm_runtime.core.lifecycle import RuntimeServices
from llm_runtime.core.request import RuntimeRequest
from llm_runtime.main import create_app


class PromptTokenBackend(Backend):
    """Backend that exposes request identity in generated tokens."""

    async def generate(self, request: RuntimeRequest) -> AsyncIterator[str]:
        label = request.messages[-1].content
        await asyncio.sleep(0.001)
        for index in range(request.sampling.max_tokens):
            yield f"{label}_tok{index} "


class FailingBackend(Backend):
    async def generate(self, request: RuntimeRequest) -> AsyncIterator[str]:
        raise RuntimeError("simulated backend failure")
        yield ""


class SlowBackend(Backend):
    async def generate(self, request: RuntimeRequest) -> AsyncIterator[str]:
        await asyncio.sleep(0.05)
        yield "late_token "


class SlowNativeBackend(Backend):
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self._lock = asyncio.Lock()

    @property
    def capabilities(self) -> BackendCapability:
        return BackendCapability.NATIVE_BATCHING

    async def generate(self, request: RuntimeRequest) -> AsyncIterator[str]:
        async with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.05)
        label = request.messages[-1].content
        yield f"{label}_tok0 "
        async with self._lock:
            self.active -= 1


class PromptBatchBackend(PromptTokenBackend):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def generate_batch(
        self,
        requests: list[RuntimeRequest],
    ) -> AsyncIterator[BatchEvent]:
        self.batch_sizes.append(len(requests))
        for index in range(max(request.sampling.max_tokens for request in requests)):
            await asyncio.sleep(0)
            for request in requests:
                if index < request.sampling.max_tokens:
                    label = request.messages[-1].content
                    yield BatchToken(request.request_id, f"{label}_tok{index} ")
                    if index + 1 == request.sampling.max_tokens:
                        yield BatchCompleted(request.request_id)


class FailingBatchBackend(PromptTokenBackend):
    async def generate_batch(
        self,
        requests: list[RuntimeRequest],
    ) -> AsyncIterator[BatchEvent]:
        raise RuntimeError("simulated batch failure")
        yield BatchCompleted(requests[0].request_id)


def test_concurrent_requests_receive_their_own_results() -> None:
    services = RuntimeServices.create(backend=PromptTokenBackend())
    app = create_app(services)

    def send(client: TestClient, label: str) -> str:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": label}],
                "max_tokens": 2,
            },
        )
        assert response.status_code == 200
        return response.json()["choices"][0]["message"]["content"]

    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(send, client, "A")
            future_b = executor.submit(send, client, "B")
            result_a = future_a.result()
            result_b = future_b.result()

    assert result_a == "A_tok0 A_tok1 "
    assert result_b == "B_tok0 B_tok1 "
    assert services.metrics.completed_count == 2
    assert len(services.metrics.queue_wait_times) == 2
    assert len(services.metrics.ttft_times) == 2


def test_backend_failure_reaches_api_and_metrics() -> None:
    services = RuntimeServices.create(backend=FailingBackend())

    with TestClient(create_app(services)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "fail"}],
                "max_tokens": 1,
            },
        )
        metrics = client.get("/metrics").json()

    assert response.status_code == 500
    assert response.json()["detail"] == "simulated backend failure"
    assert metrics["failed_count"] == 1
    assert metrics["active_requests"] == 0


def test_non_streaming_request_timeout_cancels_late_worker_completion() -> None:
    services = RuntimeServices.create(
        settings=Settings(request_timeout_s=0.01),
        backend=SlowBackend(),
    )

    with TestClient(create_app(services)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "too slow"}],
                "max_tokens": 1,
            },
        )
        immediate_metrics = client.get("/metrics").json()
        time.sleep(0.08)
        final_metrics = client.get("/metrics").json()

    assert response.status_code == 504
    assert response.json()["detail"] == "request timed out"
    assert immediate_metrics["failed_count"] == 1
    assert immediate_metrics["active_requests"] == 0
    assert final_metrics["completed_count"] == 0
    assert final_metrics["failed_count"] == 1
    assert final_metrics["active_requests"] == 0
    assert final_metrics["generated_tokens_total"] == 0


def test_dynamic_batch_routes_non_streaming_results_and_records_metrics() -> None:
    backend = PromptBatchBackend()
    settings = Settings(max_batch_size=3, batch_timeout_ms=100)
    services = RuntimeServices.create(settings=settings, backend=backend)
    barrier = Barrier(3)

    def send(client: TestClient, label: str) -> str:
        barrier.wait()
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": label}],
                "max_tokens": 2,
            },
        )
        assert response.status_code == 200
        return response.json()["choices"][0]["message"]["content"]

    with TestClient(create_app(services)) as client:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(send, client, label) for label in ("A", "B", "C")]
            results = [future.result() for future in futures]
        metrics = client.get("/metrics").json()

    assert set(results) == {"A_tok0 A_tok1 ", "B_tok0 B_tok1 ", "C_tok0 C_tok1 "}
    assert backend.batch_sizes == [3]
    assert metrics["batch_count"] == 1
    assert metrics["batch_size_max"] == 3
    assert metrics["batch_size_distribution"] == {"3": 1}


def test_dynamic_batch_routes_streaming_results() -> None:
    backend = PromptBatchBackend()
    services = RuntimeServices.create(
        settings=Settings(max_batch_size=2, batch_timeout_ms=100),
        backend=backend,
    )
    barrier = Barrier(2)

    def stream(client: TestClient, label: str) -> str:
        barrier.wait()
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": label}],
                "max_tokens": 2,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            return "".join(response.iter_text())

    with TestClient(create_app(services)) as client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(stream, client, "A")
            future_b = executor.submit(stream, client, "B")
            body_a = future_a.result()
            body_b = future_b.result()

    assert '"content": "A_tok0 "' in body_a
    assert '"content": "B_tok0 "' not in body_a
    assert '"content": "B_tok0 "' in body_b
    assert '"content": "A_tok0 "' not in body_b
    assert "data: [DONE]" in body_a
    assert "data: [DONE]" in body_b
    assert backend.batch_sizes == [2]


def test_disabling_batching_preserves_single_request_baseline() -> None:
    backend = PromptBatchBackend()
    services = RuntimeServices.create(
        settings=Settings(
            enable_batching=False,
            max_batch_size=8,
            batch_timeout_ms=25,
        ),
        backend=backend,
    )

    def send(client: TestClient, label: str) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": label}], "max_tokens": 1},
        )
        assert response.status_code == 200

    with TestClient(create_app(services)) as client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(send, client, label) for label in ("A", "B")]
            for future in futures:
                future.result()

    assert backend.batch_sizes == [1, 1]
    assert services.metrics.batch_sizes == [1, 1]


def test_native_backend_requests_can_execute_concurrently() -> None:
    backend = SlowNativeBackend()
    services = RuntimeServices.create(
        settings=Settings(native_backend_concurrency=2),
        backend=backend,
    )
    barrier = Barrier(2)

    def send(client: TestClient, label: str) -> str:
        barrier.wait()
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": label}], "max_tokens": 1},
        )
        assert response.status_code == 200
        return response.json()["choices"][0]["message"]["content"]

    with TestClient(create_app(services)) as client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(send, client, "A")
            future_b = executor.submit(send, client, "B")
            results = {future_a.result(), future_b.result()}

    assert results == {"A_tok0 ", "B_tok0 "}
    assert backend.max_active == 2
    assert services.manager.worker_count == 2


def test_batch_failure_fails_all_requests() -> None:
    services = RuntimeServices.create(
        settings=Settings(max_batch_size=2, batch_timeout_ms=100),
        backend=FailingBatchBackend(),
    )
    barrier = Barrier(2)

    def send(client: TestClient, label: str):
        barrier.wait()
        return client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": label}], "max_tokens": 1},
        )

    with TestClient(create_app(services)) as client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = [
                future.result()
                for future in [
                    executor.submit(send, client, "A"),
                    executor.submit(send, client, "B"),
                ]
            ]
        metrics = client.get("/metrics").json()

    assert [response.status_code for response in responses] == [500, 500]
    assert metrics["failed_count"] == 2
    assert metrics["active_requests"] == 0
    assert metrics["batch_size_max"] == 2
