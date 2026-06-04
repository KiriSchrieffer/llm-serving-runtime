import asyncio

from llm_runtime.api.schemas import ChatCompletionRequest
from llm_runtime.core.request import RuntimeRequest
from llm_runtime.core.response import CompletionHandle, ScheduledRequest
from llm_runtime.scheduler.fifo import FIFOScheduler
from llm_runtime.scheduler.batching import build_batch


def make_request(index: int) -> RuntimeRequest:
    return RuntimeRequest.from_chat_request(
        ChatCompletionRequest(
            messages=[{"role": "user", "content": f"request {index}"}],
            max_tokens=1,
        )
    )


async def scheduled(index: int) -> ScheduledRequest:
    return ScheduledRequest(
        request=make_request(index),
        handle=CompletionHandle.create(),
    )


def test_build_batch_respects_max_batch_size() -> None:
    asyncio.run(_assert_build_batch_size())


async def _assert_build_batch_size() -> None:
    items = [await scheduled(index) for index in range(5)]

    batch = build_batch(items, max_batch_size=3)

    assert batch.size == 3
    assert batch.items == items[:3]


def test_next_batch_stops_at_max_size() -> None:
    asyncio.run(_assert_next_batch_max_size())


async def _assert_next_batch_max_size() -> None:
    scheduler = FIFOScheduler()
    items = [await scheduled(index) for index in range(3)]
    for item in items:
        await scheduler.submit(item)

    batch = await scheduler.next_batch(max_batch_size=2, batch_timeout_ms=0)

    assert batch.items == items[:2]
    assert scheduler.size() == 1


def test_next_batch_runs_single_request_after_timeout() -> None:
    asyncio.run(_assert_single_request_timeout())


async def _assert_single_request_timeout() -> None:
    scheduler = FIFOScheduler()
    item = await scheduled(0)
    await scheduler.submit(item)

    batch = await scheduler.next_batch(max_batch_size=4, batch_timeout_ms=1)

    assert batch.items == [item]
    assert item.request.started_at is not None
