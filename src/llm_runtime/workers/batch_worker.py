from llm_runtime.scheduler.batching import Batch
from llm_runtime.workers.worker import Worker


class BatchWorker:
    """Sequential batch helper retained for isolated worker experiments.

    The serving path uses WorkerManager to route backend batch events through
    per-request response handles.
    """

    def __init__(self, worker: Worker) -> None:
        self.worker = worker

    async def execute_batch(self, batch: Batch) -> list[list[str]]:
        results: list[list[str]] = []
        for item in batch.items:
            results.append(await self.worker.collect(item.request))
        return results
