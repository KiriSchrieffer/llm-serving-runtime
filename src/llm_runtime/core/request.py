from dataclasses import dataclass, field
from time import perf_counter

from llm_runtime.api.schemas import ChatCompletionRequest, ChatMessage
from llm_runtime.utils.ids import new_request_id


@dataclass(slots=True)
class SamplingParams:
    max_tokens: int
    temperature: float
    top_p: float


@dataclass(slots=True)
class RuntimeRequest:
    """Internal request object used by schedulers, workers, and backends."""

    request_id: str
    messages: list[ChatMessage]
    priority: int
    stream: bool
    sampling: SamplingParams
    created_at: float = field(default_factory=perf_counter)
    enqueued_at: float | None = None
    started_at: float | None = None
    first_token_at: float | None = None
    completed_at: float | None = None
    failed_at: float | None = None

    @classmethod
    def from_chat_request(cls, request: ChatCompletionRequest) -> "RuntimeRequest":
        return cls(
            request_id=new_request_id(),
            messages=request.messages,
            priority=request.priority,
            stream=request.stream,
            sampling=SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ),
        )

    @property
    def prompt(self) -> str:
        return "\n".join(f"{message.role}: {message.content}" for message in self.messages)

    @property
    def prompt_token_estimate(self) -> int:
        return max(1, len(self.prompt.split()))

    def mark_enqueued(self) -> None:
        self.enqueued_at = perf_counter()

    def mark_started(self) -> None:
        self.started_at = perf_counter()

    def mark_first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = perf_counter()

    def mark_completed(self) -> None:
        self.completed_at = perf_counter()

    def mark_failed(self) -> None:
        self.failed_at = perf_counter()

