# Requirements

## Functional Requirements

- Serve a FastAPI application locally.
- Expose `GET /health`.
- Expose `POST /v1/chat/completions`.
- Accept an OpenAI-like chat completion request shape.
- Support streaming and non-streaming responses.
- Generate mock tokens asynchronously.
- Track request lifecycle timestamps.
- Route requests through a scheduler abstraction.
- Execute queued requests through a background async worker.
- Keep each HTTP response associated with its own completion or streaming channel.
- Route generation through a backend abstraction.
- Collect basic request, failure, queue wait, TTFT, token count, and total latency metrics.
- Provide tests for API behavior, scheduling, batching, streaming, and mock generation.

## Baseline Boundaries

- Use the mock backend as the default runnable path.
- Simulate prefill and decode latency for reproducible local tests.
- Support FIFO and priority scheduling.
- Form bounded dynamic micro-batches using a configurable batch timeout and maximum size.
- Keep metrics in memory.
- Keep benchmark scripts lightweight and reproducible.
- Prefer clear interfaces over feature completeness.
- Keep llama.cpp and vLLM as optional local backend adapters.

## Non-Goals

- No agent workflows.
- No tool calling.
- No RAG pipeline.
- No fine-tuning.
- No CUDA kernel implementation.
- No distributed serving.
- No production authentication or rate limiting.
- No guaranteed GPU inference path in the default setup.

## Future Extensions

- Redis-backed queue state.
- Priority scheduling based on deadlines or aging.
- GPU memory and utilization sampling.
