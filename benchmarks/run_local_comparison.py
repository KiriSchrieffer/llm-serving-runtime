import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_runtime.config import Settings
from llm_runtime.core.lifecycle import RuntimeServices
from llm_runtime.main import create_app
from run_suite import run_suite


async def wait_until_ready(base_url: str) -> None:
    async with httpx.AsyncClient(timeout=5) as client:
        for _ in range(120):  # longer poll window for real model loading
            try:
                response = await client.get(f"{base_url}/health")
                response.raise_for_status()
                return
            except httpx.HTTPError:
                await asyncio.sleep(1)
    raise RuntimeError(f"benchmark server failed to start at {base_url}")


async def measure_mode(
    mode: str,
    enable_batching: bool,
    base_url: str,
    host: str,
    port: int,
    levels: list[int],
    requests: int,
    max_tokens: int,
    max_batch_size: int,
    batch_timeout_ms: int,
    prefill_latency_ms: int,
    decode_latency_ms: int,
    backend: str = "mock",
    scheduler: str = "fifo",
) -> dict[str, object]:
    settings = Settings(
        enable_batching=enable_batching,
        prefill_latency_ms=prefill_latency_ms,
        decode_latency_ms=decode_latency_ms,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        backend=backend,
        scheduler=scheduler,
    )
    services = RuntimeServices.create(settings=settings)
    config = uvicorn.Config(
        create_app(services),
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    try:
        await wait_until_ready(base_url)
        report = await run_suite(base_url, mode, levels, requests, max_tokens)
        report["runtime_settings"] = {
            "backend": settings.backend,
            "scheduler": settings.scheduler,
            "enable_batching": settings.enable_batching,
            "prefill_latency_ms": settings.prefill_latency_ms,
            "decode_latency_ms": settings.decode_latency_ms,
            "max_batch_size": (
                settings.max_batch_size if settings.enable_batching else 1
            ),
            "batch_timeout_ms": (
                settings.batch_timeout_ms if settings.enable_batching else 0
            ),
        }
        return report
    finally:
        server.should_exit = True
        await server_task


async def run_comparison(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{args.host}:{args.port}"
    kwargs = dict(
        host=args.host,
        port=args.port,
        levels=args.levels,
        requests=args.requests,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        batch_timeout_ms=args.batch_timeout_ms,
        prefill_latency_ms=args.prefill_latency_ms,
        decode_latency_ms=args.decode_latency_ms,
        backend=args.backend,
        scheduler=args.scheduler,
    )
    if args.mode in {"fifo", "both"}:
        fifo = await measure_mode("fifo_baseline", False, base_url, **kwargs)
        (args.output_dir / "fifo_baseline.json").write_text(
            json.dumps(fifo, indent=2), encoding="utf-8",
        )
    if args.mode in {"dynamic", "both"}:
        dynamic = await measure_mode("dynamic_batching", True, base_url, **kwargs)
        (args.output_dir / "dynamic_batching.json").write_text(
            json.dumps(dynamic, indent=2), encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FIFO baseline vs dynamic batching."
    )
    parser.add_argument("--mode", choices=["fifo", "dynamic", "both"], default="both")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-timeout-ms", type=int, default=10)
    parser.add_argument("--prefill-latency-ms", type=int, default=25)
    parser.add_argument("--decode-latency-ms", type=int, default=10)
    parser.add_argument(
        "--backend", choices=["mock", "llama.cpp"], default="mock",
    )
    parser.add_argument(
        "--scheduler", choices=["fifo", "priority"], default="fifo",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "results",
    )
    args = parser.parse_args()
    asyncio.run(run_comparison(args))


if __name__ == "__main__":
    main()
