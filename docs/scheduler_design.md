# Scheduler Design

Schedulers implement `submit`, `next_request`, `next_batch`, and `size`.
Submitted entries pair a runtime request with its response handle.

## FIFO Scheduler

The FIFO scheduler uses an `asyncio.Queue` and is consumed by a background worker.
It waits asynchronously when empty. With batching enabled, it opens a bounded
collection window after receiving the first request. With batching disabled, it
forms only size-1 batches for a FIFO baseline.

## Priority Scheduler

The priority scheduler uses an `asyncio.PriorityQueue` internally. Each submitted
item is stored as a `(priority, sequence_number, ScheduledRequest)` tuple.

- **Lower priority values are higher priority**: `0` is the highest priority,
  consistent with the `ChatCompletionRequest.priority` default.
- **FIFO ordering within priority levels** is preserved by a monotonic sequence counter
  assigned at submit() time.
- `next_batch()` dequeues from the priority queue, so the highest-priority items are always
  consumed first during batch formation.
- The interface remains identical to FIFOScheduler, so WorkerManager can use either
  implementation without changes.

### Configuration

Set the scheduler type via the LLM_RUNTIME_SCHEDULER environment variable:

- `LLM_RUNTIME_SCHEDULER=fifo` (default) uses `FIFOScheduler`
- `LLM_RUNTIME_SCHEDULER=priority` uses `PriorityScheduler`

### Priority Scheduler Benchmark

The repository includes a mixed-priority benchmark that compares FIFO against
PriorityScheduler without requiring an HTTP server:

```bash
python benchmarks/run_priority_scheduler_benchmark.py --output benchmarks/results/priority_scheduler_mixed.json
```

The default workload enqueues 24 low-priority requests at time zero, then injects
8 high-priority requests after 40 ms. It uses the mock backend with 25 ms prefill,
10 ms decode, and software batches of 4 requests.

The benchmark records:

- high-priority TTFT (`created_at` to `first_token_at`)
- low-priority queue wait (`enqueued_at` to `started_at`)
- low-priority starvation count over a configurable queue-wait threshold
- Jain fairness index over inverse average queue wait, where `1.0` means equal delay
- FIFO vs priority deltas for high-priority TTFT and low-priority queue wait

The saved artifact is intentionally self-describing: workload parameters, raw
per-request observations, grouped summaries, comparison metrics, and metric notes
are all stored together.
