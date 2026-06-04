from fastapi.testclient import TestClient

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
