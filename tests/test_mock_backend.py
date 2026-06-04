import asyncio

from llm_runtime.api.schemas import ChatCompletionRequest
from llm_runtime.backends.base import BatchCompleted, BatchToken
from llm_runtime.backends.mock_backend import MockBackend
from llm_runtime.core.request import RuntimeRequest


def test_mock_backend_token_generation() -> None:
    tokens = asyncio.run(_generate_tokens())

    assert tokens == ["tok0 ", "tok1 ", "tok2 "]


def test_mock_backend_generates_batched_decode_events() -> None:
    events = asyncio.run(_generate_batch_events())

    assert isinstance(events[0], BatchToken)
    assert isinstance(events[-1], BatchCompleted)
    assert sum(isinstance(event, BatchToken) for event in events) == 5


async def _generate_tokens() -> list[str]:
    backend = MockBackend(prefill_latency_ms=0, decode_latency_ms=0)
    request = RuntimeRequest.from_chat_request(
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=3,
        )
    )

    return [token async for token in backend.generate(request)]


async def _generate_batch_events():
    backend = MockBackend(prefill_latency_ms=0, decode_latency_ms=0)
    requests = [
        RuntimeRequest.from_chat_request(
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "first"}],
                max_tokens=2,
            )
        ),
        RuntimeRequest.from_chat_request(
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "second"}],
                max_tokens=3,
            )
        ),
    ]
    return [event async for event in backend.generate_batch(requests)]
