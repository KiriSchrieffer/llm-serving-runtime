import json
from collections.abc import AsyncIterator, Awaitable, Callable

from llm_runtime.core.response import (
    CompletedEvent,
    ErrorEvent,
    StreamingHandle,
    TokenEvent,
)


def sse_event(payload: dict[str, object]) -> str:
    """Encode one Server-Sent Event data frame."""

    return f"data: {json.dumps(payload)}\n\n"


async def chat_completion_stream(
    request_id: str,
    model: str,
    handle: StreamingHandle,
    on_cancel: Callable[[], Awaitable[None]] | None = None,
) -> AsyncIterator[str]:
    """Convert per-request worker events into OpenAI-style streaming chunks."""

    try:
        while True:
            event = await handle.queue.get()
            if isinstance(event, TokenEvent):
                yield sse_event(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": event.token},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            elif isinstance(event, CompletedEvent):
                handle.mark_completed()
                yield "data: [DONE]\n\n"
                return
            elif isinstance(event, ErrorEvent):
                handle.mark_completed()
                yield sse_event({"error": {"message": event.message}})
                return
    finally:
        if handle.cancel():
            if on_cancel is not None:
                await on_cancel()
