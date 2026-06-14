from dataclasses import dataclass

from llm_runtime.backends.base import Backend, BackendCapability
from llm_runtime.backends.llama_cpp_backend import LlamaCppBackend
from llm_runtime.backends.mock_backend import MockBackend
from llm_runtime.backends.vllm_backend import VLLMBackend
from llm_runtime.config import Settings, get_settings
from llm_runtime.core.admission import AdmissionController
from llm_runtime.metrics.collector import MetricsCollector
from llm_runtime.scheduler.base import Scheduler
from llm_runtime.scheduler.fifo import FIFOScheduler
from llm_runtime.scheduler.priority import PriorityScheduler
from llm_runtime.utils.logging import RequestLogger
from llm_runtime.workers.manager import WorkerManager


def _build_scheduler(settings: Settings) -> Scheduler:
    if settings.scheduler == "priority":
        return PriorityScheduler(
            aging_boost_interval_s=settings.priority_aging_boost_interval_s
        )
    return FIFOScheduler()


def _build_backend(settings: Settings) -> Backend:
    if settings.backend == "llama.cpp":
        return LlamaCppBackend(
            model_path=settings.model_path,
            n_ctx=settings.n_ctx,
            n_gpu_layers=settings.n_gpu_layers,
        )
    if settings.backend == "vllm":
        return VLLMBackend(
            model_path=settings.model_path,
            vllm_command=settings.vllm_command,
            port=settings.vllm_port,
            gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
            max_model_len=settings.vllm_max_model_len,
        )
    return MockBackend(
        prefill_latency_ms=settings.prefill_latency_ms,
        decode_latency_ms=settings.decode_latency_ms,
    )


def _dispatch_batch_params(
    configured: Settings, backend: Backend
) -> tuple[int, int]:
    """Resolve batch parameters, respecting backend native batching capability."""
    if not configured.enable_batching:
        return 1, 0
    caps = backend.capabilities
    if BackendCapability.NATIVE_BATCHING in caps:
        return 1, 0
    return configured.max_batch_size, configured.batch_timeout_ms


@dataclass(slots=True)
class RuntimeServices:
    """Process-local runtime dependencies shared by API handlers."""

    metrics: MetricsCollector
    scheduler: Scheduler
    admission: AdmissionController
    manager: WorkerManager
    request_logger: RequestLogger
    request_timeout_s: float

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        backend: Backend | None = None,
    ) -> "RuntimeServices":
        configured = settings or get_settings()
        selected_backend = backend or _build_backend(configured)
        metrics = MetricsCollector()
        scheduler = _build_scheduler(configured)
        admission = AdmissionController(
            max_queue_size=configured.max_queue_size,
            request_rate_limit_per_s=configured.request_rate_limit_per_s,
            request_rate_limit_burst=configured.request_rate_limit_burst,
        )
        request_logger = RequestLogger()
        batch_size, batch_timeout = _dispatch_batch_params(configured, selected_backend)
        manager = WorkerManager(
            scheduler=scheduler,
            backend=selected_backend,
            metrics=metrics,
            request_logger=request_logger,
            max_batch_size=batch_size,
            batch_timeout_ms=batch_timeout,
        )
        return cls(
            metrics=metrics,
            scheduler=scheduler,
            admission=admission,
            manager=manager,
            request_logger=request_logger,
            request_timeout_s=configured.request_timeout_s,
        )

    async def start(self) -> None:
        """Start backend-dependent services, then the worker loop."""
        backend = self.manager.backend
        if hasattr(backend, "start"):
            await backend.start()  # type: ignore[union-attr]
        await self.manager.start()

    async def stop(self) -> None:
        """Stop the worker loop, then backend-dependent services."""
        await self.manager.stop()
        backend = self.manager.backend
        if hasattr(backend, "stop"):
            await backend.stop()  # type: ignore[union-attr]
