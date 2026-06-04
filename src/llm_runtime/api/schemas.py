from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mock-llm"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=16, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    priority: int = Field(default=0, ge=0)


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage

