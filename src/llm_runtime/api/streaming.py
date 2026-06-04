import asyncio
import json
from collections.abc import AsyncIterator

from llm_runtime.core.response import CompletedEvent, ErrorEvent, StreamEvent, TokenEvent


def sse_event(payload: dict[str, object]) -> str:
    """Encode one Server-Sent Event data frame."""

    return f"data: {json.dumps(payload)}\n\n"


async def chat_completion_stream(
    request_id: str,
    model: str,
    events: asyncio.Queue[StreamEvent],
) -> AsyncIterator[str]:
    """Convert per-request worker events into OpenAI-style streaming chunks."""

    while True:
        event = await events.get()
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
            yield "data: [DONE]\n\n"
            return
        elif isinstance(event, ErrorEvent):
            yield sse_event({"error": {"message": event.message}})
            return
