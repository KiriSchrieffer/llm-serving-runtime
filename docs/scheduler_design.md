# Scheduler Design

Schedulers implement `submit`, `next_request`, `next_batch`, and `size`.
Submitted entries pair a runtime request with its response handle.

## FIFO Scheduler

The FIFO scheduler uses an `asyncio.Queue` and is consumed by a background worker.
It waits asynchronously when empty. With batching enabled, it opens a bounded
collection window after receiving the first request. With batching disabled, it
forms only size-1 batches for a FIFO baseline.

## Priority Scheduler

The priority scheduler keeps queued requests in memory and selects the best
candidate at dequeue time. This allows optional aging because effective priority
depends on how long each request has waited.

- **Lower priority values are higher priority**: `0` is the highest priority,
  consistent with the `ChatCompletionRequest.priority` default.
- **FIFO ordering within effective priority levels** is preserved by a monotonic
  sequence counter assigned at submit() time.
- `next_batch()` repeatedly selects the best effective priority, so strict mode
  consumes high-priority items first while aging mode can promote old low-priority
  items.
- **Optional aging**: when `aging_boost_interval_s` is greater than zero, each
  queued request gains one effective priority level for every interval spent
  waiting. Effective priority is clamped at 0.
- The interface remains identical to FIFOScheduler, so WorkerManager can use either
  implementation without changes.

### Configuration

Set the scheduler type via the LLM_RUNTIME_SCHEDULER environment variable:

- `LLM_RUNTIME_SCHEDULER=fifo` (default) uses `FIFOScheduler`
- `LLM_RUNTIME_SCHEDULER=priority` uses `PriorityScheduler`
- `LLM_RUNTIME_PRIORITY_AGING_BOOST_INTERVAL_S=0` disables aging
- Set `LLM_RUNTIME_PRIORITY_AGING_BOOST_INTERVAL_S` above 0 to enable aging

### Priority Scheduler Benchmark

The repository includes a mixed-priority benchmark that compares FIFO, strict
priority, and priority aging without requiring an HTTP server:

```bash
python benchmarks/run_priority_scheduler_benchmark.py --output benchmarks/results/priority_scheduler_mixed.json
```

The default workload enqueues 24 low-priority requests at time zero, then injects
8 high-priority requests after 40 ms. It uses the mock backend with 25 ms prefill,
10 ms decode, software batches of 4 requests, and a 20 ms aging boost interval.

The benchmark records:

- high-priority TTFT (`created_at` to `first_token_at`)
- low-priority queue wait (`enqueued_at` to `started_at`)
- low-priority starvation count over a configurable queue-wait threshold
- Jain fairness index over inverse average queue wait, where `1.0` means equal delay
- FIFO vs strict priority deltas for high-priority TTFT and low-priority queue wait
- aging vs strict priority deltas for fairness and starvation tradeoffs

The saved artifact is intentionally self-describing: workload parameters, raw
per-request observations, grouped summaries, comparison metrics, and metric notes
are all stored together.
