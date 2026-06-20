# Performance Tradeoffs

Documented tradeoffs observed from benchmark runs using the mock backend (simulated
25ms prefill / 10ms decode) and the llama.cpp backend (Qwen3-4B-Instruct-Q4_K_M, CPU-only,
Alder Lake backend).

## Mock Backend vs LlamaCpp Backend

### Throughput (tokens/s)

| Concurrency | Mock FIFO | Mock Dynamic | LlamaCpp FIFO | LlamaCpp Dynamic |
|---|---|---|---|---|
| 1 | 14.9 | 14.8 | 14.8 | 13.9 |
| 4 | 15.1 | 15.3 | 14.5 | 15.0 |
| 8 | 14.6 | 15.2 | 14.3 | 14.7 |

### Key Observations

1. **Latency gap**: The real backend is approximately 2.3x slower in total latency and
   3.6x slower in TTFT compared to mock. This gap is expected for CPU-only inference on
   a 4B-parameter model. With GPU offload (llama.cpp `n_gpu_layers` > 0 or vLLM), the
   gap narrows significantly.

2. **Token count mismatch**: The mock backend always generates exactly `max_tokens`
   (16 per request, 512 total for 32 requests). The real model stops early when it
   detects the response is complete, producing far fewer tokens (68 total). Benchmarks
   comparing mock vs real backends should account for this asymmetry by comparing
   per-token metrics rather than per-request totals.

3. **Batching effectiveness**: Both backends achieved an average batch size of 4.0
   under 8-way concurrency with batch_timeout_ms=50. The mock backend benefits from
   micro-batching by amortizing the simulated 25ms prefill across batch members. The
   llama.cpp backend's batching depends on its internal parallel-processing capacity,
   which is limited on CPU-only hardware.

4. **Dynamic batching helps throughput on CPU too**: Even on CPU-only inference,
   dynamic batching with batch_size=4 and timeout=50ms yields a 3-4% throughput
   improvement at concurrency=4 (14.5 -> 15.0 tok/s). However, at high concurrency
   (8), individual request latency degrades significantly (up to 7-8 seconds for
   the last request in a batch), confirming that micro-batching trades per-request
   latency for throughput.

## FIFO vs Priority Scheduling

Mixed-priority benchmark command:

```bash
python benchmarks/run_priority_scheduler_benchmark.py --output benchmarks/results/priority_scheduler_mixed.json
```

Default workload: 24 low-priority requests at time zero, then 8 high-priority
requests after 40 ms. Mock latency is 25 ms prefill / 10 ms decode, with
`max_batch_size=4`, `batch_timeout_ms=0`, and a 20 ms aging boost interval.

| Metric | FIFO | Strict priority | Priority aging |
|---|---:|---:|---:|
| High-priority avg TTFT | 0.978s | 0.190s | 0.583s |
| High-priority P95 TTFT | 1.013s | 0.225s | 1.012s |
| Low-priority avg queue wait | 0.393s | 0.511s | 0.453s |
| Low-priority P95 queue wait | 0.788s | 0.930s | 0.858s |
| Queue-wait Jain fairness | 0.856 | 0.776 | 0.992 |
| Low-priority starved fraction | 16.7% | 33.3% | 16.7% |

Strict priority improved high-priority average TTFT by 80.5% versus FIFO, while
increasing low-priority average queue wait by 30.0%. Priority aging recovered
most fairness (`0.776 -> 0.992`) and reduced low-priority average queue wait by
11.4% relative to strict priority, while still keeping high-priority average
TTFT 40.4% better than FIFO.

### Key Observations

1. **Fairness vs responsiveness**: FIFO ensures every request is served in order, which
   is ideal when all requests have equal importance. Priority scheduling introduces
   starvation risk for low-priority requests but can dramatically reduce TTFT for
   high-priority interactive traffic. Aging is a middle ground: old low-priority
   requests gradually become competitive without removing priority differentiation.

2. **Batching with priority**: When priority scheduling is combined with dynamic
   batching, strict priority selects high-priority requests first during batch
   formation. Aging changes later batch rounds as old low-priority requests gain
   effective priority.

## Batching vs No Batching

| Metric | No Batching | Dynamic Batching (batch=4, timeout=50ms) |
|---|---|---|
| Per-request latency | Lower variance | Higher variance |
| Throughput | Lower | Higher (amortized prefill) |
| Queue wait | Lower | Higher (waiting for batch formation) |
| TTFT | Faster (single requests) | Potentially slower (batch prefill may delay) |

### Key Observations

1. **Throughput-latency tradeoff**: Dynamic batching trades per-request latency
   for higher throughput. With the mock backend, batching amortizes the simulated
   25ms prefill across batch members. With real backends, the tradeoff depends on
   the backend's ability to parallelize token generation.

2. **Batch timeout tuning**: A `batch_timeout_ms` of 0 forms batches only from
   requests already waiting in the queue (no delay). Larger timeouts increase
   average batch size but also increase queue wait time. The optimal timeout
   depends on arrival rate and the backend's parallel processing capacity.

3. **Native batching bypass**: When a backend declares `NATIVE_BATCHING` (e.g.,
   vLLM), the worker bypasses software-layer micro-batching entirely
   (`max_batch_size=1, batch_timeout_ms=0`). This avoids double-batching and lets
   the backend manage batch formation internally.

## LlamaCpp CPU-Only Performance (Qwen3-4B, Alder Lake)

Measured on an Intel Alder Lake CPU from the `run_local_comparison.py` benchmark
suite (16 requests per level, 16 max_tokens, batch_size=4, timeout=50ms):

Raw result artifacts:

- `benchmarks/results/llama_fifo_vs_dynamic.json/fifo_baseline.json`
- `benchmarks/results/llama_fifo_vs_dynamic.json/dynamic_batching.json`

| Concurrency | Mode | tok/s | Single-req latency | Batch avg |
|---|---|---|---|---|
| 1 | FIFO | 14.8 | ~1.1s | 1.0 |
| 1 | Dynamic | 13.9 | ~1.2s | 1.0 |
| 4 | FIFO | 14.5 | ~1.1s | 1.0 |
| 4 | Dynamic | 15.0 | ~1.1-2.2s | 4.0 |
| 8 | FIFO | 14.3 | ~1.1s | 1.0 |
| 8 | Dynamic | 14.7 | ~1.1-8.8s | 4.0 |

### Takeaways

- Throughput is stable at ~14.5 tok/s: the Alder Lake CPU is the bottleneck
  regardless of batching mode.
- Dynamic batching provides a modest 3-4% throughput gain at concurrency=4 by
  grouping requests into micro-batches that llama-server can process more efficiently.
- At concurrency=8, tail latency becomes severe (8.8s for the last request in batch)
  — dynamic batching should be paired with an aggressive timeout or batch_size cap
  for latency-sensitive workloads.
- For CPU-only deployments with this model, 2-4 concurrent requests is the sweet
  spot: good throughput without excessive queue wait.
