import asyncio, json, time, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from llm_runtime.main import create_app
from llm_runtime.core.lifecycle import RuntimeServices
from llm_runtime.config import Settings
from llm_runtime.backends.mock_backend import MockBackend
from llm_runtime.backends.llama_cpp_backend import LlamaCppBackend

MESSAGES = [{"role": "user", "content": "Hello, respond in one sentence."}]
MAX_TOKENS = 16
CONCURRENCY = 8
REQUESTS = 32

def run_backend(label, services):
    app = create_app(services)
    with TestClient(app) as client:
        def send_one(idx):
            resp = client.post("/v1/chat/completions", json={
                "messages": MESSAGES, "max_tokens": MAX_TOKENS
            })
            return resp.status_code
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            results = list(ex.map(send_one, range(REQUESTS)))
        snap = client.get("/metrics").json()
    ok = sum(1 for r in results if r == 200)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Requests: {ok}/{REQUESTS} OK")
    print(f"  Total tokens: {snap['generated_tokens_total']}")
    total_s = (snap.get('total_latency_p50_s', 0) or 0) * REQUESTS
    print(f"  TTFT p50: {snap.get('ttft_p50_s', 'N/A')}s, p95: {snap.get('ttft_p95_s', 'N/A')}s")
    print(f"  Total latency p50: {snap.get('total_latency_p50_s', 'N/A')}s, p95: {snap.get('total_latency_p95_s', 'N/A')}s")
    print(f"  Queue wait p50: {snap.get('queue_wait_time_p50_s', 'N/A')}s")
    if snap.get('batch_count', 0) > 0:
        print(f"  Batch count: {snap['batch_count']}, avg size: {snap.get('batch_size_avg', 'N/A')}")
    return snap

print("Testing mock backend...")
mock_svc = RuntimeServices.create(
    backend=MockBackend(prefill_latency_ms=25, decode_latency_ms=10),
    settings=Settings(enable_batching=True, max_batch_size=8, batch_timeout_ms=10),
)
mock_snap = run_backend("Mock Backend (simulated 25ms/10ms)", mock_svc)

print("\nTesting llama.cpp backend (Qwen3-4B Q4_K_M)...")
print("Starting server (may take ~10s)...")
llama_svc = RuntimeServices.create(
    backend=LlamaCppBackend(
        model_path=r"E:\llama.cpp\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        n_ctx=2048, n_gpu_layers=0
    ),
    settings=Settings(backend="llama.cpp", enable_batching=True,
                      max_batch_size=4, batch_timeout_ms=50),
)
t0 = time.perf_counter()
llama_snap = run_backend(
    f"LlamaCpp (Qwen3-4B Q4_K_M, CPU, {CONCURRENCY}-concurrent)",
    llama_svc
)
t1 = time.perf_counter()
print(f"\n  Wall-clock: {t1-t0:.1f}s")

print(f"\n{'='*60}")
print("  COMPARISON")
print(f"{'='*60}")
for metric in ["ttft_p50_s", "ttft_p95_s", "total_latency_p50_s",
               "total_latency_p95_s", "queue_wait_time_p50_s",
               "generated_tokens_total"]:
    m = mock_snap.get(metric, "N/A")
    l = llama_snap.get(metric, "N/A")
    print(f"  {metric:30s}  mock={str(m):>10s}  llama={str(l):>10s}")