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

### Benchmarking Plan

Planned priority scheduler benchmarks will compare:

- FIFO baseline vs priority scheduling under mixed-priority workloads
- Whether high-priority requests improve TTFT at the cost of low-priority queue wait time
- Batch size distribution under priority-aware batch formation
