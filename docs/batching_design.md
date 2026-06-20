# Batching Design

The runtime implements dynamic micro-batching on top of the background worker.
The scheduler blocks for the first waiting request, then collects additional
requests until either:

- `LLM_RUNTIME_MAX_BATCH_SIZE` is reached
- `LLM_RUNTIME_BATCH_TIMEOUT_MS` expires

The batch-aware mock backend performs one simulated prefill wait for the batch
and one decode wait per token step. Each emitted token carries its request ID,
so the worker routes it to the correct completion future or streaming queue.

Current metrics include batch count, average and maximum observed batch size,
batch size distribution, queue wait, TTFT, and generated token count.

Backends that declare `NATIVE_BATCHING`, such as vLLM, bypass runtime-level
micro-batching (`max_batch_size=1`, `batch_timeout_ms=0`). They can use
`LLM_RUNTIME_NATIVE_BACKEND_CONCURRENCY` worker fan-out to receive concurrent
requests and perform continuous batching internally.

Future extensions:

- max token budget during batch formation
- cancellation-aware removal from active batches
- continuous batching
- real llama.cpp and vLLM backend behavior
