"""Integration test suite for the llama.cpp backend with a real 4B model.

Runs all API-level paths against the real model and saves every
request/response pair plus metrics snapshots to a JSON result file.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from llm_runtime.backends.llama_cpp_backend import LlamaCppBackend
from llm_runtime.config import Settings
from llm_runtime.core.lifecycle import RuntimeServices
from llm_runtime.main import create_app

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = Path(r"E:\llama.cpp\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
LLAMA_SERVER = Path(r"E:\llama.cpp\llama-server.exe")
PORT = 18083
RESULTS_DIR = Path(__file__).resolve().parent / "benchmarks" / "results"
RESULT_FILE = RESULTS_DIR / "integration_result.json"

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def _build_client():
    settings = Settings(
        backend="llama.cpp",
        model_path=str(MODEL_PATH),
        n_ctx=512,
        n_gpu_layers=0,
    )
    backend = LlamaCppBackend(
        model_path=str(MODEL_PATH),
        llama_server_path=str(LLAMA_SERVER),
        n_ctx=512,
        n_gpu_layers=0,
        port=PORT,
        verbose=False,
    )
    services = RuntimeServices.create(settings=settings, backend=backend)
    return TestClient(create_app(services))


def _log(case: dict[str, Any]) -> dict[str, Any]:
    """Print a single-line summary for a test case."""
    status = "PASS" if case.get("passed") else "FAIL"
    ms = case.get("duration_ms", "?")
    print(f"  [{status}] {case['name']:40s} {ms}ms")
    return case


def run() -> dict[str, Any]:
    results: dict[str, Any] = {
        "suite": "real_model_integration",
        "model": str(MODEL_PATH),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases": [],
        "metrics_before": None,
        "metrics_after": None,
        "errors": [],
    }

    print(f"Starting llama.cpp backend (model={MODEL_PATH.name}) ...")

    client = _build_client()

    with client:
        # -------------------------------------------------------------------
        # 1. Health
        # -------------------------------------------------------------------
        t0 = time.perf_counter()
        r = client.get("/health")
        t1 = time.perf_counter()
        _log(case := {
            "name": "health",
            "method": "GET", "path": "/health",
            "status": r.status_code,
            "response": r.json(),
            "duration_ms": round((t1 - t0) * 1000, 2),
            "passed": r.status_code == 200,
        })
        results["cases"].append(case)

        # -------------------------------------------------------------------
        # 2. Metrics snapshot before workload
        # -------------------------------------------------------------------
        results["metrics_before"] = client.get("/metrics").json()

        # -------------------------------------------------------------------
        # 3. Simple chat completion (1 token expected)
        # -------------------------------------------------------------------
        body = {
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 5,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        r = client.post("/v1/chat/completions", json=body)
        t1 = time.perf_counter()
        data = r.json() if r.status_code == 200 else None
        _log(case := {
            "name": "chat_completion_simple",
            "method": "POST", "path": "/v1/chat/completions",
            "request_body": body,
            "status": r.status_code,
            "response": data,
            "duration_ms": round((t1 - t0) * 1000, 2),
            "passed": r.status_code == 200 and len(
                data["choices"][0]["message"]["content"]) > 0,
        })
        results["cases"].append(case)

        # -------------------------------------------------------------------
        # 4. Longer generation (20 tokens)
        # -------------------------------------------------------------------
        body = {
            "messages": [{"role": "user", "content": "List three colors."}],
            "max_tokens": 20,
            "temperature": 0.0,
        }
        t0 = time.perf_counter()
        r = client.post("/v1/chat/completions", json=body)
        t1 = time.perf_counter()
        data = r.json() if r.status_code == 200 else None
        _log(case := {
            "name": "chat_completion_longer",
            "method": "POST", "path": "/v1/chat/completions",
            "request_body": body,
            "status": r.status_code,
            "response": data,
            "duration_ms": round((t1 - t0) * 1000, 2),
            "passed": r.status_code == 200 and (data["usage"]["completion_tokens"] or 0) >= 1,
        })
        results["cases"].append(case)

        # -------------------------------------------------------------------
        # 5. Streaming chat completion
        # -------------------------------------------------------------------
        body = {
            "messages": [{"role": "user", "content": "Count from one to three."}],
            "max_tokens": 15,
            "temperature": 0.0,
            "stream": True,
        }
        t0 = time.perf_counter()
        with client.stream("POST", "/v1/chat/completions", json=body) as r:
            chunks = []
            for chunk in r.iter_lines():
                line = chunk.strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks.append(json.loads(line[6:]))
            status = r.status_code
        t1 = time.perf_counter()
        collected = "".join(
            c["choices"][0]["delta"].get("content", "")
            for c in chunks if c.get("choices")
        )
        _log(case := {
            "name": "streaming_chat",
            "method": "POST", "path": "/v1/chat/completions",
            "request_body": body,
            "status": status,
            "chunks_count": len(chunks),
            "collected_text": collected,
            "duration_ms": round((t1 - t0) * 1000, 2),
            "passed": status == 200 and len(collected) > 0,
        })
        results["cases"].append(case)

        # -------------------------------------------------------------------
        # 6. Request with priority
        # -------------------------------------------------------------------
        body = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 5,
            "temperature": 0.0,
            "priority": 0,
        }
        t0 = time.perf_counter()
        r = client.post("/v1/chat/completions", json=body)
        t1 = time.perf_counter()
        data = r.json() if r.status_code == 200 else None
        _log(case := {
            "name": "chat_completion_priority",
            "method": "POST", "path": "/v1/chat/completions",
            "request_body": body,
            "status": r.status_code,
            "response": data,
            "duration_ms": round((t1 - t0) * 1000, 2),
            "passed": r.status_code == 200,
        })
        results["cases"].append(case)

        # -------------------------------------------------------------------
        # 7. Metrics after workload
        # -------------------------------------------------------------------
        results["metrics_after"] = client.get("/metrics").json()

        # -------------------------------------------------------------------
        # 8. Prometheus-format metrics
        # -------------------------------------------------------------------
        r = client.get("/metrics", headers={"accept": "text/plain"})
        _log(case := {
            "name": "prometheus_format",
            "method": "GET", "path": "/metrics",
            "headers": {"accept": "text/plain"},
            "status": r.status_code,
            "has_help": "# HELP" in r.text,
            "has_type": "# TYPE" in r.text,
            "lines": len(r.text.strip().split("\n")),
            "passed": r.status_code == 200 and "# HELP" in r.text,
        })
        results["cases"].append(case)

        # -------------------------------------------------------------------
        # 9. Metrics consistency check
        # -------------------------------------------------------------------
        mb = results["metrics_before"]
        ma = results["metrics_after"]
        delta_completed = ma["completed_count"] - mb["completed_count"]
        delta_tokens = ma["generated_tokens_total"] - mb["generated_tokens_total"]
        delta_batches = ma["batch_count"] - mb["batch_count"]
        _log(case := {
            "name": "metrics_consistency",
            "delta_completed": delta_completed,
            "delta_tokens": delta_tokens,
            "delta_batches": delta_batches,
            "passed": delta_completed >= 1 and delta_tokens >= 1,
        })
        results["cases"].append(case)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total = len(results["cases"])
    passed = sum(1 for c in results["cases"] if c.get("passed"))
    results["summary"] = {"total": total, "passed": passed, "failed": total - passed}
    print(f"\nResults: {passed}/{total} passed\n")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"Result file: {RESULT_FILE}")
