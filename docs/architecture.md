# Architecture Notes

The runtime is organized around a narrow serving path:

API request -> admission control -> internal request + response handle -> dynamic batch scheduler -> background worker -> batch backend events -> response channel -> API response.

The current runtime uses one background async worker started through the FastAPI
application lifespan. Non-streaming calls wait on their own completion future,
and streaming calls consume their own event queue. This prevents concurrent
responses from being mixed while leaving the API schema and backend interface
stable as requests are dynamically micro-batched and routed back independently.

Admission control runs before a request is counted active or submitted to the
scheduler. The default configuration leaves it disabled, but operators can set
a maximum scheduler queue size and a token-bucket request rate limit. Queue
overload returns `503`, rate limiting returns `429`, and both paths record
rejection metrics without consuming worker capacity.

The metrics path combines request lifecycle data with optional GPU telemetry.
`MetricsCollector` records queue wait, TTFT, total latency, batch sizes, and
generated tokens, plus rejected requests grouped by reason and priority. When
`/metrics` is requested, it also samples `nvidia-smi` for GPU memory and
utilization. Hosts without `nvidia-smi` or without accessible NVIDIA GPUs return
a structured `unavailable` GPU snapshot instead of failing the metrics endpoint.
