import asyncio
from contextlib import suppress
from time import perf_counter

from llm_runtime.backends.base import Backend, BatchCompleted, BatchToken
from llm_runtime.core.response import (
    CompletedEvent,
    CompletionHandle,
    ErrorEvent,
    GenerationResult,
    ScheduledRequest,
    StreamingHandle,
    TokenEvent,
)
from llm_runtime.metrics.collector import MetricsCollector
from llm_runtime.scheduler.base import Scheduler
from llm_runtime.scheduler.batching import Batch
from llm_runtime.utils.logging import RequestLogger


class WorkerManager:
    """Runs a background worker that consumes scheduled requests."""

    def __init__(
        self,
        scheduler: Scheduler,
        backend: Backend,
        metrics: MetricsCollector,
        request_logger: RequestLogger,
        max_batch_size: int = 1,
        batch_timeout_ms: int = 0,
        worker_count: int = 1,
    ) -> None:
        self.scheduler = scheduler
        self.backend = backend
        self.metrics = metrics
        self.request_logger = request_logger
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.worker_count = max(1, worker_count)
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        if self._tasks:
            return
        self._tasks = [
            asyncio.create_task(self._worker_loop()) for _ in range(self.worker_count)
        ]

    async def stop(self) -> None:
        if not self._tasks:
            return
        for task in self._tasks:
            task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.gather(*self._tasks)
        self._tasks = []

    async def _worker_loop(self) -> None:
        while True:
            batch = await self.scheduler.next_batch(
                max_batch_size=self.max_batch_size,
                batch_timeout_ms=self.batch_timeout_ms,
            )
            self.metrics.record_batch(batch.size)
            self.request_logger.batch_formed(
                batch_size=batch.size,
                request_ids=[item.request.request_id for item in batch.items],
            )
            await self._execute_batch(batch)

    async def _execute_batch(self, batch: Batch) -> None:
        active_items = [item for item in batch.items if not item.cancelled]
        if not active_items:
            return

        items = {item.request.request_id: item for item in active_items}
        tokens: dict[str, list[str]] = {request_id: [] for request_id in items}
        pending = set(items)
        batch_start = perf_counter()
        try:
            async for event in self.backend.generate_batch(
                [item.request for item in active_items]
            ):
                if event.request_id not in pending:
                    continue
                item = items[event.request_id]
                if item.cancelled:
                    pending.discard(event.request_id)
                    continue
                if isinstance(event, BatchToken):
                    item.request.mark_first_token()
                    tokens[event.request_id].append(event.token)
                    if isinstance(item.handle, StreamingHandle):
                        await item.handle.queue.put(TokenEvent(token=event.token))
                elif isinstance(event, BatchCompleted):
                    self._complete_item(item, tokens[event.request_id], batch_start)
                    pending.discard(event.request_id)

            for request_id in pending:
                item = items[request_id]
                if not item.cancelled:
                    self._complete_item(item, tokens[request_id], batch_start)
        except Exception as exc:
            for request_id in pending:
                item = items[request_id]
                if not item.cancelled:
                    await self._fail_item(item, exc)

    def _complete_item(
        self, item: ScheduledRequest, tokens: list[str], batch_start: float
    ) -> None:
        if item.cancelled:
            return
        item.request.mark_completed()
        elapsed_ms = (perf_counter() - batch_start) * 1000
        if isinstance(item.handle, CompletionHandle):
            if not item.handle.future.cancelled():
                item.handle.future.set_result(
                    GenerationResult(request_id=item.request.request_id, tokens=tokens)
                )
        elif isinstance(item.handle, StreamingHandle):
            if item.handle.cancelled:
                return
            item.handle.queue.put_nowait(CompletedEvent())
        self.metrics.record_success(item.request, generated_tokens=len(tokens))
        self.request_logger.request_completed(
            request_id=item.request.request_id,
            tokens=len(tokens),
            elapsed_ms=elapsed_ms,
        )

    async def _fail_item(self, item: ScheduledRequest, exc: Exception) -> None:
        if item.cancelled:
            return
        self.metrics.record_failure(item.request)
        self.request_logger.request_failed(
            request_id=item.request.request_id, error=str(exc)
        )
        if isinstance(item.handle, CompletionHandle):
            if not item.handle.future.cancelled():
                item.handle.future.set_exception(exc)
        elif isinstance(item.handle, StreamingHandle):
            await item.handle.queue.put(ErrorEvent(message=str(exc)))
