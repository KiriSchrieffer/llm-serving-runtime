import asyncio
from time import perf_counter

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from starlette.responses import Response

from llm_runtime.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)
from llm_runtime.api.streaming import chat_completion_stream
from llm_runtime.core.lifecycle import RuntimeServices
from llm_runtime.core.request import RuntimeRequest
from llm_runtime.core.response import CompletionHandle, ScheduledRequest, StreamingHandle

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics", response_model=None)
async def get_metrics(http_request: Request) -> Response | dict[str, object]:
    services = _services(http_request)
    accept = http_request.headers.get("accept", "")

    if "text/plain" in accept:
        return PlainTextResponse(
            services.metrics.snapshot_prometheus(
                queue_size=services.scheduler.size()
            )
        )

    snapshot = services.metrics.snapshot()
    snapshot["queue_size"] = services.scheduler.size()
    return snapshot  # FastAPI will convert this to JSON


@router.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    body: ChatCompletionRequest,
    http_request: Request,
) -> StreamingResponse | ChatCompletionResponse:
    services = _services(http_request)
    runtime_request = RuntimeRequest.from_chat_request(body)
    services.request_logger.request_received(
        request_id=runtime_request.request_id,
        model=body.model,
        stream=body.stream,
    )
    admission = services.admission.admit(queue_size=services.scheduler.size())
    if not admission.accepted:
        services.metrics.record_rejection(
            priority=runtime_request.priority,
            reason=admission.reason,
        )
        services.request_logger.request_rejected(
            request_id=runtime_request.request_id,
            reason=admission.reason,
            status_code=admission.status_code,
        )
        raise HTTPException(
            status_code=admission.status_code,
            detail=admission.detail,
        )

    services.metrics.record_request()

    if body.stream:
        stream_handle = StreamingHandle.create()
        await services.scheduler.submit(
            ScheduledRequest(request=runtime_request, handle=stream_handle)
        )
        services.request_logger.request_enqueued(runtime_request.request_id)

        async def cancel_stream() -> None:
            services.metrics.record_failure(runtime_request)
            services.request_logger.request_failed(
                request_id=runtime_request.request_id,
                error="stream disconnected",
            )

        return StreamingResponse(
            chat_completion_stream(
                runtime_request.request_id,
                body.model,
                stream_handle,
                on_cancel=cancel_stream,
            ),
            media_type="text/event-stream",
        )

    completion_handle = CompletionHandle.create()
    await services.scheduler.submit(
        ScheduledRequest(request=runtime_request, handle=completion_handle)
    )
    services.request_logger.request_enqueued(runtime_request.request_id)
    t0 = perf_counter()
    try:
        result = await asyncio.wait_for(
            asyncio.shield(completion_handle.future),
            timeout=services.request_timeout_s,
        )
    except TimeoutError as exc:
        completion_handle.cancel()
        services.metrics.record_failure(runtime_request)
        services.request_logger.request_failed(
            request_id=runtime_request.request_id,
            error=f"request timed out after {services.request_timeout_s:.3f}s",
        )
        raise HTTPException(status_code=504, detail="request timed out") from exc
    except asyncio.CancelledError:
        completion_handle.cancel()
        services.metrics.record_failure(runtime_request)
        services.request_logger.request_failed(
            request_id=runtime_request.request_id,
            error="client disconnected",
        )
        raise
    except Exception as exc:
        services.request_logger.request_failed(
            request_id=runtime_request.request_id, error=str(exc)
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed_ms = (perf_counter() - t0) * 1000
    content = "".join(result.tokens)
    completion_tokens = len(result.tokens)
    prompt_tokens = runtime_request.prompt_token_estimate
    services.request_logger.request_completed(
        request_id=runtime_request.request_id,
        tokens=completion_tokens,
        elapsed_ms=elapsed_ms,
    )
    return ChatCompletionResponse(
        id=result.request_id,
        model=body.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _services(http_request: Request) -> RuntimeServices:
    return http_request.app.state.runtime
