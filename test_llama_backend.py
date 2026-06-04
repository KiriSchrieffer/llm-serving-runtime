"""Quick integration test for the llama.cpp backend via subprocess."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

os.environ.setdefault("LLM_RUNTIME_BACKEND", "llama.cpp")
os.environ.setdefault("LLM_RUNTIME_N_CTX", "512")
os.environ.setdefault(
    "LLM_RUNTIME_MODEL_PATH",
    r"E:\llama.cpp\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
)

from fastapi.testclient import TestClient
from llm_runtime.backends.llama_cpp_backend import LlamaCppBackend
from llm_runtime.config import Settings
from llm_runtime.core.lifecycle import RuntimeServices
from llm_runtime.main import create_app

settings = Settings(
    backend="llama.cpp",
    model_path=r"E:\llama.cpp\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    n_ctx=512,
    n_gpu_layers=0,
)

backend = LlamaCppBackend(
    model_path=settings.model_path,
    n_ctx=settings.n_ctx,
    n_gpu_layers=settings.n_gpu_layers,
    port=18081,
    verbose=False,
)

services = RuntimeServices.create(settings=settings, backend=backend)

print("Starting llama.cpp backend (model loads, may take 30-60s) ...")

with TestClient(create_app(services)) as client:
    health = client.get("/health")
    print(f"Health: {health.status_code} {health.json()}")

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 10,
            "temperature": 0.0,
        },
    )
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"Response: {content!r}")
        print(f"Usage: {data['usage']}")

    metrics = client.get("/metrics").json()
    print(f"Completed: {metrics['completed_count']}")
    print(f"Tokens: {metrics['generated_tokens_total']}")

print("Done.")
