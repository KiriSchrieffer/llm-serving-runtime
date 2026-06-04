from fastapi.testclient import TestClient

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
