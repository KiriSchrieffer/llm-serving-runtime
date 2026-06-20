"""Tests for backend capability model and vLLM backend adapter."""

import pytest

from llm_runtime.backends.base import BackendCapability
from llm_runtime.backends.mock_backend import MockBackend
from llm_runtime.backends.llama_cpp_backend import LlamaCppBackend
from llm_runtime.backends.vllm_backend import VLLMBackend
from llm_runtime.config import Settings
from llm_runtime.core.lifecycle import (
    _build_backend,
    _build_scheduler,
    _dispatch_worker_count,
)
from llm_runtime.scheduler.priority import PriorityScheduler


def test_mock_backend_capabilities():
    backend = MockBackend()
    caps = backend.capabilities
    assert BackendCapability.STREAMING in caps
    assert BackendCapability.NATIVE_BATCHING not in caps
    assert BackendCapability.GPU_METRICS not in caps


def test_llama_cpp_backend_capabilities():
    backend = LlamaCppBackend(model_path="/nonexistent/model.gguf")
    caps = backend.capabilities
    assert BackendCapability.STREAMING in caps
    assert BackendCapability.GPU_METRICS in caps
    assert BackendCapability.NATIVE_BATCHING not in caps


def test_vllm_backend_capabilities():
    backend = VLLMBackend(model_path="meta-llama/Llama-2-7b-hf")
    caps = backend.capabilities
    assert BackendCapability.STREAMING in caps
    assert BackendCapability.NATIVE_BATCHING in caps
    assert BackendCapability.GPU_METRICS in caps


def test_capability_combinations_are_bitfields():
    caps = BackendCapability.STREAMING | BackendCapability.NATIVE_BATCHING
    assert BackendCapability.STREAMING in caps
    assert BackendCapability.NATIVE_BATCHING in caps
    assert BackendCapability.GPU_METRICS not in caps


# --- vLLM Backend ---

def test_vllm_backend_constructor_defaults():
    backend = VLLMBackend(model_path="meta-llama/Llama-2-7b-hf")
    assert backend._model_path == "meta-llama/Llama-2-7b-hf"
    assert backend._port == 8082
    assert backend._gpu_memory_utilization == 0.9
    assert backend._max_model_len is None
    assert backend._client is None
    assert backend._process is None


def test_vllm_backend_constructor_custom():
    backend = VLLMBackend(
        model_path="mistralai/Mistral-7B-v0.1",
        vllm_command="/usr/local/bin/vllm",
        port=9000,
        gpu_memory_utilization=0.7,
        max_model_len=4096,
    )
    assert backend._port == 9000
    assert backend._gpu_memory_utilization == 0.7
    assert backend._max_model_len == 4096


def test_vllm_chat_payload_includes_served_model():
    from llm_runtime.api.schemas import ChatCompletionRequest
    from llm_runtime.core.request import RuntimeRequest

    backend = VLLMBackend(model_path="Qwen/Qwen2.5-1.5B-Instruct")
    request = RuntimeRequest.from_chat_request(
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=8,
            temperature=0.0,
        )
    )

    payload = backend._chat_payload(request)

    assert payload["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert payload["stream"] is True
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_llama_cpp_chat_payload_includes_model_path():
    from llm_runtime.api.schemas import ChatCompletionRequest
    from llm_runtime.core.request import RuntimeRequest

    backend = LlamaCppBackend(model_path="/models/qwen.gguf")
    request = RuntimeRequest.from_chat_request(
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=8,
            temperature=0.0,
        )
    )

    payload = backend._chat_payload(request)

    assert payload["model"] == "/models/qwen.gguf"
    assert payload["stream"] is True
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_vllm_generate_raises_when_not_started():
    backend = VLLMBackend(model_path="meta-llama/Llama-2-7b-hf")
    from llm_runtime.api.schemas import ChatCompletionRequest
    from llm_runtime.core.request import RuntimeRequest
    request = RuntimeRequest.from_chat_request(
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1,
        )
    )
    with pytest.raises(RuntimeError, match="not started"):
        async for _ in backend.generate(request):
            pass


# --- Lifecycle wiring ---

def test_build_backend_returns_mock_by_default():
    settings = Settings()
    backend = _build_backend(settings)
    assert isinstance(backend, MockBackend)


def test_build_backend_returns_vllm_when_configured():
    settings = Settings(backend="vllm", model_path="meta-llama/Llama-2-7b-hf")
    backend = _build_backend(settings)
    assert isinstance(backend, VLLMBackend)
    assert backend._port == 8082


def test_build_backend_returns_llama_cpp_when_configured():
    settings = Settings(backend="llama.cpp", model_path="/tmp/model.gguf")
    backend = _build_backend(settings)
    assert isinstance(backend, LlamaCppBackend)


def test_build_scheduler_configures_priority_aging():
    settings = Settings(
        scheduler="priority",
        priority_aging_boost_interval_s=0.25,
    )
    scheduler = _build_scheduler(settings)

    assert isinstance(scheduler, PriorityScheduler)
    assert scheduler.aging_boost_interval_s == 0.25


def test_dispatch_respects_native_batching():
    from llm_runtime.backends.vllm_backend import VLLMBackend
    from llm_runtime.config import Settings
    from llm_runtime.core.lifecycle import _dispatch_batch_params
    backend = VLLMBackend(model_path="test")
    settings = Settings(enable_batching=True, max_batch_size=8, batch_timeout_ms=10)
    size, timeout = _dispatch_batch_params(settings, backend)
    assert size == 1
    assert timeout == 0


def test_dispatch_uses_config_when_no_native_batching():
    from llm_runtime.backends.mock_backend import MockBackend
    from llm_runtime.config import Settings
    from llm_runtime.core.lifecycle import _dispatch_batch_params
    backend = MockBackend()
    settings = Settings(enable_batching=True, max_batch_size=4, batch_timeout_ms=50)
    size, timeout = _dispatch_batch_params(settings, backend)
    assert size == 4
    assert timeout == 50


def test_dispatch_disabled_batching_overrides_all():
    from llm_runtime.backends.vllm_backend import VLLMBackend
    from llm_runtime.config import Settings
    from llm_runtime.core.lifecycle import _dispatch_batch_params
    backend = VLLMBackend(model_path="test")
    settings = Settings(enable_batching=False, max_batch_size=16, batch_timeout_ms=100)
    size, timeout = _dispatch_batch_params(settings, backend)
    assert size == 1
    assert timeout == 0


def test_native_backend_worker_count_uses_configured_concurrency():
    backend = VLLMBackend(model_path="test")
    settings = Settings(native_backend_concurrency=8)

    assert _dispatch_worker_count(settings, backend) == 8


def test_non_native_backend_worker_count_remains_single_worker():
    settings = Settings(native_backend_concurrency=8)

    assert _dispatch_worker_count(settings, MockBackend()) == 1
    assert _dispatch_worker_count(settings, LlamaCppBackend(model_path="/tmp/model.gguf")) == 1
