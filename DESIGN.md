# Design

## API Layer

The API layer is implemented with FastAPI. It exposes a health endpoint and an OpenAI-like chat completion endpoint. Request validation is handled with Pydantic schemas, while internal scheduling and execution use runtime-specific request objects.

## Request Lifecycle

1. Client submits a chat completion request.
2. API validates the payload.
3. API converts the payload into an internal `RuntimeRequest`.
4. The API pairs the request with a completion future or streaming event queue.
5. The paired request is submitted to the scheduler.
6. A background worker consumes the scheduler queue and executes the backend.
7. The backend emits mock tokens asynchronously.
8. The worker resolves the request-specific future or pushes events to its queue.
9. The API returns either a complete response or a token stream.
10. Metrics record queue wait, TTFT, total latency, tokens, completion, or failure.

## Scheduler Abstraction

Schedulers implement a small async interface:

- `submit(request)`
- `next_request()`
- `next_batch(max_batch_size, batch_timeout_ms)`
- `size()`

The runtime includes FIFO and priority ordering with dynamic micro-batching.
Each scheduler blocks asynchronously for the first request, then uses a bounded
collection window to form a batch up to `max_batch_size`. The queue is consumed
only by the background worker, keeping concurrent responses associated with
their original request.

## Backend Abstraction

Backends implement async single-request generation and a batch event interface.
The default mock backend simulates:

- prefill latency before the first token
- decode latency for each generated token
- shared prefill and decode steps across a micro-batch
- deterministic fake token output

The llama.cpp and vLLM adapters map the same interface to external OpenAI-compatible
servers managed as subprocesses.

## Worker Model

The runtime starts a background worker loop during FastAPI lifespan startup. The
worker consumes a dynamically formed batch, owns backend execution, and routes
batch token/completion events through per-request response channels.

## Streaming Design

Streaming responses use Server-Sent Events. A streaming request owns an async
event queue; the worker places token, completion, or error events in that queue.
Each token is encoded as a chat completion chunk, and the stream ends with a
final `data: [DONE]` event.

## Metrics Design

Metrics are stored in an in-memory collector. The MVP tracks:

- request count
- completed count
- failed count
- active requests
- generated tokens
- batch count
- batch size distribution
- queue wait time observations
- TTFT observations
- total latency observations

Metrics are exposed as JSON snapshots and Prometheus-style text output.

## Failure Handling

Route handlers wrap execution in failure accounting. Invalid requests are rejected by Pydantic/FastAPI validation. Backend or scheduler failures increment failure metrics and return standard FastAPI errors.
