# Roadmap

The repository is at an initial GitHub-ready baseline. The default mock backend
is designed to run on any development machine, while llama.cpp and vLLM are
optional real-backend adapters that require local model/runtime setup.

- A full async LLM serving pipeline (FastAPI + dynamic micro-batching + streaming)
- Three backend adapters (mock, llama.cpp, vLLM)
- Metrics with Prometheus histograms, optional GPU telemetry, and structured JSON logging
- Benchmark scripts and saved result artifacts
- Pinned development requirements and CI quality gates
- 60 pytest test cases covering API, scheduling, batching, streaming, metrics, and backend wiring

## Phase 0: MVP Skeleton

Completed:

- FastAPI service with health and chat completion endpoints
- OpenAI-like request and response schemas
- mock backend with simulated prefill and decode latency
- FIFO scheduler interface and implementation
- basic metrics collection
- pytest coverage for API, scheduler, streaming, batching, and backend

## Phase 1A: Async Worker Runtime

Completed:

- background async worker loop with per-request completion/stream channels
- concurrency-safe request/response association
- TTFT and generated token metrics

## Phase 1B: Dynamic Micro-Batching

Completed:

- dynamic batching queue with max batch size and timeout
- worker loop that forms and executes mock batches
- token routing for mixed completion and streaming handles
- request timeout and streaming disconnect cancellation paths
- batch size, queue wait, token count, and TTFT measurements
- FIFO baseline and dynamic batching benchmark entrypoint
- first mock-backend benchmark report across concurrency levels 1, 8, 16, 32, and 64
- high-concurrency batch size/timeout sweep and candidate configuration validation
- priority scheduler implementation (lower priority value = higher, FIFO within level)
- priority aging policy to reduce starvation under mixed-priority load
- mixed-priority scheduler benchmark with TTFT, queue wait, starvation, and fairness metrics
- llama.cpp backend adapter with llama-server subprocess lifecycle

## Phase 2: Metrics and Observability

Completed:

- Prometheus-style metrics export (JSON and text exposition format)
- latency histograms with predefined buckets for queue wait, TTFT, and total latency
- Prometheus histogram type output with cumulative buckets, sum, and count
- optional `nvidia-smi` GPU memory/utilization sampler with unavailable fallback
- structured JSON request lifecycle logging with request_id correlation
- structured logs integrated into API routes, worker manager, and lifecycle

## Phase 3: Real Backend Adapters

Completed:

- llama.cpp adapter with subprocess lifecycle
- backend capability model (STREAMING, NATIVE_BATCHING, GPU_METRICS flags)
- vLLM adapter with `vllm serve` subprocess management
- vLLM configuration through `LLM_RUNTIME_VLLM_*` environment variables
- backend capability-aware worker dispatch (bypasses micro-batching when backend has NATIVE_BATCHING)
- mock vs llama.cpp benchmark comparison script (`benchmarks/compare_backends.py`)
- CPU-only llama.cpp benchmark notes for Qwen3-4B-Instruct-Q4_K_M

## Phase 4: Reproducible Experiments

Completed:

- Dockerfile and docker-compose.yml with benchmark profiles
- reusable benchmark configs (`benchmarks/configs/`): quick, sweep, scheduler
- automated benchmark report generator (`benchmarks/generate_report.py`)
- documented performance tradeoffs (`docs/performance_tradeoffs.md`)
- curated benchmark summary separated from raw JSON and generated scratch reports
- `.env.example` and `.env.benchmark` for all configuration options
- pinned `requirements-dev.txt` used by local development and GitHub Actions
- CI checks for `ruff`, `mypy`, and `pytest`

## Open Follow-Ups

- Add stricter type-checking rules once backend adapters stabilize
- Add NVML-backed GPU sampling as a lower-overhead alternative to `nvidia-smi`
