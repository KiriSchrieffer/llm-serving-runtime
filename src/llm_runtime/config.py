import os

from pydantic import BaseModel, Field


_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__),  # noqa: PTH120
    "..", "..", "..",
    "..", "llama.cpp", "models", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
)


class Settings(BaseModel):
    """Runtime configuration loaded from environment variables."""

    backend: str = "mock"
    scheduler: str = "fifo"
    priority_aging_boost_interval_s: float = Field(default=0.0, ge=0)
    enable_batching: bool = True
    prefill_latency_ms: int = Field(default=25, ge=0)
    decode_latency_ms: int = Field(default=10, ge=0)
    max_batch_size: int = Field(default=8, ge=1)
    batch_timeout_ms: int = Field(default=10, ge=0)
    request_timeout_s: float = Field(default=120.0, gt=0)
    max_queue_size: int = Field(default=0, ge=0)
    request_rate_limit_per_s: float = Field(default=0.0, ge=0)
    request_rate_limit_burst: int = Field(default=0, ge=0)
    native_backend_concurrency: int = Field(default=4, ge=1)
    model_path: str = _DEFAULT_MODEL
    n_ctx: int = Field(default=2048, ge=128)
    n_gpu_layers: int = 0
    vllm_port: int = 8082
    vllm_gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0)
    vllm_max_model_len: int | None = None
    vllm_command: str = "vllm"


def get_settings() -> Settings:
    return Settings(
        backend=os.getenv("LLM_RUNTIME_BACKEND", "mock"),
        scheduler=os.getenv("LLM_RUNTIME_SCHEDULER", "fifo"),
        priority_aging_boost_interval_s=float(
            os.getenv("LLM_RUNTIME_PRIORITY_AGING_BOOST_INTERVAL_S", "0")
        ),
        enable_batching=os.getenv("LLM_RUNTIME_ENABLE_BATCHING", "true").lower()
        in {"1", "true", "yes", "on"},
        prefill_latency_ms=int(os.getenv("LLM_RUNTIME_PREFILL_LATENCY_MS", "25")),
        decode_latency_ms=int(os.getenv("LLM_RUNTIME_DECODE_LATENCY_MS", "10")),
        max_batch_size=int(os.getenv("LLM_RUNTIME_MAX_BATCH_SIZE", "8")),
        batch_timeout_ms=int(os.getenv("LLM_RUNTIME_BATCH_TIMEOUT_MS", "10")),
        request_timeout_s=float(os.getenv("LLM_RUNTIME_REQUEST_TIMEOUT_S", "120")),
        max_queue_size=int(os.getenv("LLM_RUNTIME_MAX_QUEUE_SIZE", "0")),
        request_rate_limit_per_s=float(
            os.getenv("LLM_RUNTIME_REQUEST_RATE_LIMIT_PER_S", "0")
        ),
        request_rate_limit_burst=int(
            os.getenv("LLM_RUNTIME_REQUEST_RATE_LIMIT_BURST", "0")
        ),
        native_backend_concurrency=int(
            os.getenv("LLM_RUNTIME_NATIVE_BACKEND_CONCURRENCY", "4")
        ),
        model_path=os.getenv("LLM_RUNTIME_MODEL_PATH", _DEFAULT_MODEL),
        n_ctx=int(os.getenv("LLM_RUNTIME_N_CTX", "2048")),
        n_gpu_layers=int(os.getenv("LLM_RUNTIME_N_GPU_LAYERS", "0")),
        vllm_port=int(os.getenv("LLM_RUNTIME_VLLM_PORT", "8082")),
        vllm_gpu_memory_utilization=float(
            os.getenv("LLM_RUNTIME_VLLM_GPU_MEMORY_UTILIZATION", "0.9")
        ),
        vllm_max_model_len=(
            int(v) if (v := os.getenv("LLM_RUNTIME_VLLM_MAX_MODEL_LEN", "")) else None
        ),
        vllm_command=os.getenv("LLM_RUNTIME_VLLM_COMMAND", "vllm"),
    )
