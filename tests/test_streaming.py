import asyncio

from fastapi.testclient import TestClient

from llm_runtime.api.streaming import chat_completion_stream
from llm_runtime.core.response import StreamingHandle, TokenEvent
from llm_runtime.main import create_app


def test_streaming_chat_completion() -> None:
    with TestClient(create_app()) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "mock-llm",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
                "stream": True,
            },
        ) as response:
            body = "".join(response.iter_text())
        metrics = client.get("/metrics").json()

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"content": "tok0 "' in body
    assert '"content": "tok1 "' in body
    assert "data: [DONE]" in body
    assert metrics["completed_count"] == 1
    assert metrics["generated_tokens_total"] == 2


def test_streaming_generator_cancels_handle_when_closed() -> None:
    asyncio.run(_assert_stream_generator_cancels_handle_when_closed())


async def _assert_stream_generator_cancels_handle_when_closed() -> None:
    handle = StreamingHandle.create()
    cancel_count = 0

    async def on_cancel() -> None:
        nonlocal cancel_count
        cancel_count += 1

    await handle.queue.put(TokenEvent(token="tok0 "))
    stream = chat_completion_stream(
        request_id="chatcmpl-test",
        model="mock-llm",
        handle=handle,
        on_cancel=on_cancel,
    )

    first_chunk = await stream.__anext__()
    await stream.aclose()

    assert '"content": "tok0 "' in first_chunk
    assert handle.cancelled is True
    assert cancel_count == 1
