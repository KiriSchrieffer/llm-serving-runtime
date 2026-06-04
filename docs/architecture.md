# Architecture Notes

The runtime is organized around a narrow serving path:

API request -> internal request + response handle -> dynamic batch scheduler -> background worker -> batch backend events -> response channel -> API response.

The current runtime uses one background async worker started through the FastAPI
application lifespan. Non-streaming calls wait on their own completion future,
and streaming calls consume their own event queue. This prevents concurrent
responses from being mixed while leaving the API schema and backend interface
stable as requests are dynamically micro-batched and routed back independently.
