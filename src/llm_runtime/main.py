import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI

from llm_runtime.api.routes import router
from llm_runtime.core.lifecycle import RuntimeServices
from llm_runtime.utils.logging import JSONFormatter

_log = logging.getLogger("llm_runtime")


def _setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger("llm_runtime")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def create_app(services: RuntimeServices | None = None) -> FastAPI:
    _setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = services or RuntimeServices.create()
        app.state.runtime = runtime
        await runtime.start()
        _log.info(
            "runtime started (scheduler=%s, backend=%s)",
            type(runtime.scheduler).__name__,
            type(runtime.manager.backend).__name__,
        )
        try:
            yield
        finally:
            await runtime.stop()
            _log.info("runtime stopped")

    app = FastAPI(
        title="GPU-Aware LLM Serving Runtime",
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request, call_next):
        start = perf_counter()
        response = await call_next(request)
        elapsed_ms = (perf_counter() - start) * 1000
        _log.info(
            "%s %s -> %s in %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    app.include_router(router)
    return app


app = create_app()