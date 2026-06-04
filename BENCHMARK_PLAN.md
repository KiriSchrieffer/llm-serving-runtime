# Benchmark Plan

## Concurrency Levels

Benchmarks should run at:

- 1 concurrent request
- 8 concurrent requests
- 16 concurrent requests
- 32 concurrent requests
- 64 concurrent requests

## Metrics

Each run should capture:

- tokens/s
- P50 latency
- P95 latency
- TTFT
- queue wait time
- batch size distribution
- GPU memory
- GPU utilization

For the mock backend, GPU metrics should be reported as unavailable or zero. Real backend phases will sample GPU metrics.

## Scheduler Comparison

Compare:

- FIFO single-request baseline (`LLM_RUNTIME_ENABLE_BATCHING=false`)
- dynamic batching (`LLM_RUNTIME_ENABLE_BATCHING=true`)
- priority scheduling in a later phase

The first benchmark should establish FIFO mock backend baseline behavior, then
run dynamic batching under matching load. Results should show whether batching
improves throughput and how the batch window affects TTFT and queue wait time.

## Backend Comparison

Compare:

- mock backend first
- llama.cpp later
- vLLM later

The mock backend validates the serving path and benchmarking harness before introducing real inference complexity.
