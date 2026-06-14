import asyncio

from llm_runtime.api.schemas import ChatCompletionRequest
from llm_runtime.core.request import RuntimeRequest
from llm_runtime.core.response import CompletionHandle, ScheduledRequest
from llm_runtime.scheduler.fifo import FIFOScheduler
from llm_runtime.scheduler.priority import PriorityScheduler


def make_request(content: str, priority: int = 0) -> RuntimeRequest:
    return RuntimeRequest.from_chat_request(
        ChatCompletionRequest(
            messages=[{"role": "user", "content": content}],
            max_tokens=1,
            priority=priority,
        )
    )


def test_fifo_ordering() -> None:
    asyncio.run(_assert_fifo_ordering())


async def _assert_fifo_ordering() -> None:
    scheduler = FIFOScheduler()
    first = make_request("first")
    second = make_request("second")
    first_item = ScheduledRequest(request=first, handle=CompletionHandle.create())
    second_item = ScheduledRequest(request=second, handle=CompletionHandle.create())

    await scheduler.submit(first_item)
    await scheduler.submit(second_item)

    assert scheduler.size() == 2
    assert await scheduler.next_request() is first_item
    assert await scheduler.next_request() is second_item
    assert scheduler.size() == 0


def test_next_request_waits_until_item_is_submitted() -> None:
    asyncio.run(_assert_waits_for_submission())


async def _assert_waits_for_submission() -> None:
    scheduler = FIFOScheduler()
    pending = asyncio.create_task(scheduler.next_request())
    await asyncio.sleep(0)

    assert not pending.done()

    item = ScheduledRequest(
        request=make_request("late arrival"),
        handle=CompletionHandle.create(),
    )
    await scheduler.submit(item)

    assert await pending is item


# --- PriorityScheduler tests ---


def test_priority_dequeues_high_priority_first() -> None:
    asyncio.run(_assert_priority_order())


async def _assert_priority_order() -> None:
    scheduler = PriorityScheduler()
    low = ScheduledRequest(
        request=make_request("low", priority=1),
        handle=CompletionHandle.create(),
    )
    high = ScheduledRequest(
        request=make_request("high", priority=0),
        handle=CompletionHandle.create(),
    )

    await scheduler.submit(low)
    await scheduler.submit(high)

    assert scheduler.size() == 2
    assert await scheduler.next_request() is high
    assert await scheduler.next_request() is low
    assert scheduler.size() == 0


def test_priority_fifo_within_same_level() -> None:
    asyncio.run(_assert_fifo_within_priority())


async def _assert_fifo_within_priority() -> None:
    scheduler = PriorityScheduler()
    first = ScheduledRequest(
        request=make_request("first", priority=5),
        handle=CompletionHandle.create(),
    )
    second = ScheduledRequest(
        request=make_request("second", priority=5),
        handle=CompletionHandle.create(),
    )

    await scheduler.submit(first)
    await scheduler.submit(second)

    assert await scheduler.next_request() is first
    assert await scheduler.next_request() is second


def test_priority_mixed_groups() -> None:
    asyncio.run(_assert_mixed_groups())


async def _assert_mixed_groups() -> None:
    scheduler = PriorityScheduler()
    items = {
        "p0_a": ScheduledRequest(
            request=make_request("p0_a", priority=0),
            handle=CompletionHandle.create(),
        ),
        "p0_b": ScheduledRequest(
            request=make_request("p0_b", priority=0),
            handle=CompletionHandle.create(),
        ),
        "p5": ScheduledRequest(
            request=make_request("p5", priority=5),
            handle=CompletionHandle.create(),
        ),
    }

    await scheduler.submit(items["p5"])
    await scheduler.submit(items["p0_a"])
    await scheduler.submit(items["p0_b"])

    assert await scheduler.next_request() is items["p0_a"]
    assert await scheduler.next_request() is items["p0_b"]
    assert await scheduler.next_request() is items["p5"]
    assert scheduler.size() == 0


def test_priority_next_batch_selects_highest_priority() -> None:
    asyncio.run(_assert_priority_next_batch())


async def _assert_priority_next_batch() -> None:
    scheduler = PriorityScheduler()
    low = ScheduledRequest(
        request=make_request("low", priority=3),
        handle=CompletionHandle.create(),
    )
    high = ScheduledRequest(
        request=make_request("high", priority=0),
        handle=CompletionHandle.create(),
    )

    await scheduler.submit(low)
    await scheduler.submit(high)

    batch = await scheduler.next_batch(max_batch_size=2, batch_timeout_ms=0)
    assert batch.items[0] is high
    assert batch.items[1] is low
    assert batch.size == 2


def test_priority_waits_when_empty() -> None:
    asyncio.run(_assert_priority_waits())


async def _assert_priority_waits() -> None:
    scheduler = PriorityScheduler()
    pending = asyncio.create_task(scheduler.next_request())
    await asyncio.sleep(0)

    assert not pending.done()

    item = ScheduledRequest(
        request=make_request("late arrival", priority=0),
        handle=CompletionHandle.create(),
    )
    await scheduler.submit(item)

    assert await pending is item


def test_priority_aging_disabled_preserves_strict_priority() -> None:
    asyncio.run(_assert_priority_aging_disabled())


async def _assert_priority_aging_disabled() -> None:
    now = 0.0
    scheduler = PriorityScheduler(time_fn=lambda: now)
    low = ScheduledRequest(
        request=make_request("old low", priority=2),
        handle=CompletionHandle.create(),
    )
    high = ScheduledRequest(
        request=make_request("new high", priority=0),
        handle=CompletionHandle.create(),
    )

    await scheduler.submit(low)
    now = 10.0
    await scheduler.submit(high)

    assert await scheduler.next_request() is high
    assert await scheduler.next_request() is low


def test_priority_aging_boosts_old_low_priority_request() -> None:
    asyncio.run(_assert_priority_aging_boosts_old_low())


async def _assert_priority_aging_boosts_old_low() -> None:
    now = 0.0
    scheduler = PriorityScheduler(
        aging_boost_interval_s=1.0,
        time_fn=lambda: now,
    )
    low = ScheduledRequest(
        request=make_request("old low", priority=2),
        handle=CompletionHandle.create(),
    )
    high = ScheduledRequest(
        request=make_request("new high", priority=0),
        handle=CompletionHandle.create(),
    )

    await scheduler.submit(low)
    now = 2.1
    await scheduler.submit(high)

    assert await scheduler.next_request() is low
    assert await scheduler.next_request() is high
