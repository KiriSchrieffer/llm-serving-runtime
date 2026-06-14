from fastapi.testclient import TestClient

from llm_runtime.config import Settings
from llm_runtime.core.lifecycle import RuntimeServices
from llm_runtime.main import create_app


def test_health_endpoint() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_non_streaming_chat_completion() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-llm",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 3,
            },
        )
        metrics = client.get("/metrics").json()

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "tok0 tok1 tok2 "
    assert body["usage"]["completion_tokens"] == 3
    assert metrics["request_count"] == 1
    assert metrics["completed_count"] == 1
    assert metrics["active_requests"] == 0
    assert metrics["generated_tokens_total"] == 3
    assert metrics["ttft_avg_s"] >= 0


def test_rate_limited_request_returns_429_without_entering_runtime() -> None:
    services = RuntimeServices.create(
        settings=Settings(
            request_rate_limit_per_s=1,
            request_rate_limit_burst=1,
        )
    )

    with TestClient(create_app(services)) as client:
        accepted = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-llm",
                "messages": [{"role": "user", "content": "first"}],
                "max_tokens": 1,
                "priority": 5,
            },
        )
        rejected = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-llm",
                "messages": [{"role": "user", "content": "second"}],
                "max_tokens": 1,
                "priority": 5,
            },
        )
        metrics = client.get("/metrics").json()

    assert accepted.status_code == 200
    assert rejected.status_code == 429
    assert rejected.json()["detail"] == "request rate limit exceeded"
    assert metrics["request_count"] == 1
    assert metrics["completed_count"] == 1
    assert metrics["rejected_count"] == 1
    assert metrics["active_requests"] == 0
    assert metrics["rejections_by_reason"] == {"rate_limited": 1}
    assert metrics["rejections_by_priority"] == {"5": 1}
