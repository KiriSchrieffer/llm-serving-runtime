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

Future extensions:

- max token budget during batch formation
- cancellation-aware removal from active batches
- continuous batching
- real llama.cpp and vLLM backend behavior
