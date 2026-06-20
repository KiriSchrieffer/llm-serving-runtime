# Benchmark Report

This is the curated, human-written benchmark summary. Raw benchmark artifacts are
kept as JSON under `benchmarks/results/`. The helper script
`benchmarks/generate_report.py` can generate a temporary Markdown view from those
JSON files, but generated Markdown is ignored by git so this document remains the
single narrative benchmark report.

## RTX 4090 vLLM GPU Smoke/Load Test

Recorded on June 20, 2026 on a single NVIDIA GeForce RTX 4090 using the vLLM
backend and `Qwen/Qwen2.5-0.5B-Instruct`. The model was downloaded to a local
`models/` directory, then served through this runtime's FastAPI API, scheduler,
worker, and vLLM backend adapter. Each request used `max_tokens=32`.

Raw result artifacts:

- `benchmarks/results/vllm_gpu_smoke_0_5b.json`
- `benchmarks/results/vllm_gpu_load_0_5b_c8.json`
- `benchmarks/results/vllm_gpu_metrics_after_0_5b.json`

| Run | Requests | Concurrency | Tokens/s | P50 latency ms | P95 latency ms | Avg TTFT ms | Avg total latency ms | Failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Smoke | 16 | 4 | 417.6 | 206.3 | 290.7 | 127.2 | 182.8 | 0 |
| Load | 32 | 8 | 426.2 | 326.6 | 539.6 | 250.5 | 302.3 | 0 |

The post-run metrics snapshot reported `gpu.status=available` with one RTX
4090, 24,564 MB total GPU memory, and 21,074 MB used by the vLLM-backed runtime.
`nvidia-smi` utilization in the saved snapshot was 0% because it was captured
after the load test had completed.

This is a real GPU backend integration result, not a vLLM maximum-throughput
tuning run. The runtime marks vLLM as `NATIVE_BATCHING`, so software-layer
micro-batching is bypassed and the recorded batch size remains 1. The result is
therefore best interpreted as evidence that the project can launch vLLM, route
real model requests, stream completions, collect GPU metrics, and save
reproducible latency/throughput artifacts on cloud GPU hardware.

## First Mock Backend Comparison

Recorded on May 26, 2026 using a local Uvicorn service and the mock backend.
Each concurrency level sends 64 requests with 32 completion tokens per request.
Mock latency settings are 25 ms prefill and 10 ms decode. Dynamic batching
uses `max_batch_size=8` and `batch_timeout_ms=10`.

Raw result artifacts:

- `benchmarks/results/fifo_baseline.json`
- `benchmarks/results/dynamic_batching.json`

| Concurrency | FIFO tok/s | Batch tok/s | Throughput delta | FIFO P95 ms | Batch P95 ms | FIFO TTFT ms | Batch TTFT ms | Avg batch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 60.7 | 58.9 | -2.9% | 531.2 | 546.7 | 44.3 | 60.3 | 1.00 |
| 8 | 60.8 | 240.7 | +295.7% | 4204.8 | 1075.2 | 1889.7 | 510.3 | 4.00 |
| 16 | 62.8 | 342.8 | +445.6% | 7714.7 | 1594.8 | 3876.2 | 680.1 | 5.33 |
| 32 | 62.2 | 443.5 | +612.5% | 15437.0 | 2385.0 | 7958.1 | 974.6 | 6.40 |
| 64 | 62.4 | 485.1 | +676.9% | 30763.9 | 4214.4 | 16017.8 | 1895.8 | 7.11 |

## Interpretation

- At concurrency 1, dynamic batching cannot form multi-request batches and
  pays the collection timeout, reducing throughput by 2.9%.
- Under concurrent load, batching shares simulated prefill/decode steps and
  substantially improves throughput while reducing queue buildup and tail
  latency relative to the single-request worker.
- At concurrency 64, average observed batch size is 7.11 out of 8,
  yielding 485.1 tokens/s compared with 62.4 tokens/s for FIFO.
- This is a mock-backend systems-path benchmark, not evidence of real GPU
  inference performance. Later backend adapters must repeat the experiment.

## Reproduce

```bash
python benchmarks/run_local_comparison.py --mode both --levels 1 8 16 32 64 --requests 64 --max-tokens 32 --max-batch-size 8 --batch-timeout-ms 10 --prefill-latency-ms 25 --decode-latency-ms 10 --output-dir benchmarks/results
python benchmarks/analyze_results.py --baseline benchmarks/results/fifo_baseline.json --dynamic benchmarks/results/dynamic_batching.json
```

## High-Concurrency Batch Parameter Sweep

Recorded on May 27, 2026. This experiment fixes the workload at concurrency
64, 64 requests, and 32 completion tokens per request. It scans:

- max_batch_size: 2, 4, 8, 16
- batch_timeout_ms: 0, 5, 10, 20

Raw result artifact: `benchmarks/results/batch_sweep_c64.json`.

| Batch size | Timeout ms | Tokens/s | vs FIFO | TTFT ms | P95 ms | Avg batch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0 | 121.3 | +94.2% | 8034.2 | 15871.1 | 1.94 |
| 2 | 5 | 122.9 | +96.8% | 7950.5 | 15622.2 | 1.94 |
| 2 | 10 | 122.1 | +95.5% | 7926.2 | 15749.6 | 1.94 |
| 2 | 20 | 120.6 | +93.1% | 8079.5 | 15901.2 | 1.94 |
| 4 | 0 | 243.6 | +290.1% | 3858.9 | 7876.1 | 3.76 |
| 4 | 5 | 245.0 | +292.4% | 3867.4 | 7871.4 | 3.76 |
| 4 | 10 | 242.4 | +288.2% | 3841.6 | 7915.6 | 3.76 |
| 4 | 20 | 256.2 | +310.3% | 3586.6 | 7462.9 | 4.00 |
| 8 | 0 | 478.6 | +666.5% | 1922.7 | 4273.5 | 7.11 |
| 8 | 5 | 479.5 | +668.0% | 1848.6 | 4265.6 | 7.11 |
| 8 | 10 | 484.6 | +676.1% | 1862.7 | 4219.8 | 7.11 |
| 8 | 20 | 517.1 | +728.2% | 1643.4 | 3954.9 | 8.00 |
| 16 | 0 | 951.0 | +1423.1% | 888.8 | 2147.8 | 12.80 |
| 16 | 5 | 879.1 | +1307.9% | 983.4 | 2324.5 | 12.80 |
| 16 | 10 | 907.9 | +1354.0% | 925.7 | 2250.4 | 12.80 |
| 16 | 20 | 1096.5 | +1656.1% | 610.5 | 1862.1 | 16.00 |

For this saturated mock workload, `batch_size=16` and `timeout_ms=20` is the
best observed configuration by throughput, TTFT, and P95. A longer collection
window helps here because it fills the batch before execution and reduces later
queueing rounds.

## Candidate Validation Across Concurrency

The 16/20 ms candidate was rerun across the original concurrency matrix.
Raw result artifact:
`benchmarks/results/candidate_b16_t20/dynamic_batching.json`.

| Concurrency | FIFO tok/s | Candidate tok/s | Throughput delta | FIFO P95 ms | Candidate P95 ms | FIFO TTFT ms | Candidate TTFT ms | Avg batch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 60.7 | 57.4 | -5.5% | 531.2 | 561.1 | 44.3 | 74.5 | 1.00 |
| 8 | 60.8 | 458.6 | +654.1% | 4204.8 | 562.3 | 1889.7 | 70.1 | 8.00 |
| 16 | 62.8 | 552.8 | +780.1% | 7714.7 | 1102.9 | 3876.2 | 304.7 | 9.14 |
| 32 | 62.2 | 777.4 | +1148.9% | 15437.0 | 1588.0 | 7958.1 | 457.0 | 12.80 |
| 64 | 62.4 | 1124.8 | +1701.4% | 30763.9 | 1814.7 | 16017.8 | 584.4 | 16.00 |

The candidate is appropriate for high-concurrency mock workloads, but it adds
collection latency when only one request is active. It is therefore recorded as
an experimental high-load configuration, not adopted as the default runtime
setting before validation with real backends and a workload-aware policy.

## Reproduce Sweep

```bash
python benchmarks/run_batch_sweep.py --concurrency 64 --requests 64 --max-tokens 32 --batch-sizes 2 4 8 16 --timeouts 0 5 10 20 --prefill-latency-ms 25 --decode-latency-ms 10 --output benchmarks/results/batch_sweep_c64.json
python benchmarks/analyze_batch_sweep.py --sweep benchmarks/results/batch_sweep_c64.json --baseline benchmarks/results/fifo_baseline.json --ttft-budget-ms 1000
```

## Reproducible Benchmark Configuration

Benchmark scripts now support selecting the backend and scheduler at the
command line, producing artifacts that record the full configuration:

| Flag | Default | Choices |
|---|---|---|
| --backend | mock | mock, llama.cpp |
| --scheduler | fifo | fifo, priority |

### Compare real model backends

```bash
python benchmarks/run_local_comparison.py --backend llama.cpp --mode both --levels 1 --requests 8 --max-tokens 16
```

This starts a temporary `llama-server` subprocess, runs the load test, and
shuts it down; no separate server management needed. Each `--mode` spawns
a fresh runtime so metrics do not leak between runs.

### Compare schedulers

```bash
python benchmarks/run_local_comparison.py --scheduler priority --mode dynamic --levels 1 8 16 --requests 16 --max-tokens 16
```

The `runtime_settings` key in every output JSON records the actual backend
and scheduler used, making each result self-describing and independently
reproducible.

### Batch sweep with any backend

```bash
python benchmarks/run_batch_sweep.py --backend mock --scheduler fifo --concurrency 64 --batch-sizes 2 4 8 16 --timeouts 0 5 10 20
```

## Mixed-Priority Scheduler Benchmark

Recorded on June 14, 2026 using the mock backend. The workload enqueues
24 low-priority requests at time zero, then injects 8 high-priority requests
after 40 ms. Mock latency settings are 25 ms prefill and 10 ms decode.
The scheduler forms batches of 4 requests with no collection timeout. The aging
run uses a 20 ms boost interval.

Raw result artifact: `benchmarks/results/priority_scheduler_mixed.json`.

| Metric | FIFO | Strict priority | Priority aging |
| --- | ---: | ---: | ---: |
| High-priority avg TTFT | 0.978s | 0.190s | 0.583s |
| High-priority P95 TTFT | 1.013s | 0.225s | 1.012s |
| Low-priority avg queue wait | 0.393s | 0.511s | 0.453s |
| Low-priority P95 queue wait | 0.788s | 0.930s | 0.858s |
| Queue-wait Jain fairness | 0.856 | 0.776 | 0.992 |
| Low-priority starved fraction | 16.7% | 33.3% | 16.7% |

Strict priority improves high-priority average TTFT by 80.5%, but it increases
low-priority average queue wait by 30.0% and doubles the fraction of
low-priority requests above the 750 ms starvation threshold. Priority aging
recovers most fairness (`0.776 -> 0.992`) and lowers low-priority average queue
wait by 11.4% relative to strict priority, while still keeping high-priority
average TTFT 40.4% better than FIFO.

Reproduce:

```bash
python benchmarks/run_priority_scheduler_benchmark.py --output benchmarks/results/priority_scheduler_mixed.json
```

## Next Experiments

- **Real model at higher concurrency**: repeat the concurrency matrix with the
  llama.cpp backend to observe real inference latency, throughput, and TTFT.
- **Backend comparison**: mock vs llama.cpp latency profiles under identical
  request patterns.
- **llama.cpp batch configuration**: sweep `n_ctx`, `n_batch`, and `n_gpu_layers`
  to find the optimal serving configuration for a given model and hardware.
