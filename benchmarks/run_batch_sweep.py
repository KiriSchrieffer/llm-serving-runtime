import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from run_local_comparison import PROJECT_ROOT, measure_mode


async def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    base_url = f"http://{args.host}:{args.port}"
    configurations: list[dict[str, object]] = []
    for max_batch_size in args.batch_sizes:
        for batch_timeout_ms in args.timeouts:
            label = f"batch_size_{max_batch_size}_timeout_{batch_timeout_ms}ms"
            report = await measure_mode(
                label,
                True,
                base_url,
                args.host,
                args.port,
                [args.concurrency],
                args.requests,
                args.max_tokens,
                max_batch_size,
                batch_timeout_ms,
                args.prefill_latency_ms,
                args.decode_latency_ms,
                backend=args.backend,
                scheduler=args.scheduler,
            )
            run = report["runs"][0]
            configurations.append(
                {
                    "max_batch_size": max_batch_size,
                    "batch_timeout_ms": batch_timeout_ms,
                    "run": run,
                }
            )
            metrics = run["runtime_metrics"]
            print(
                f"{label}: tokens/s={run['tokens_per_second']:.2f} "
                f"ttft_ms={metrics['ttft_avg_s'] * 1000:.1f} "
                f"avg_batch={metrics['batch_size_avg']:.2f}"
            )
    return {
        "experiment": "dynamic_batch_parameter_sweep",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "scheduler": args.scheduler,
        "workload": {
            "concurrency": args.concurrency,
            "requests": args.requests,
            "max_tokens": args.max_tokens,
            "prefill_latency_ms": args.prefill_latency_ms,
            "decode_latency_ms": args.decode_latency_ms,
        },
        "search_space": {
            "max_batch_sizes": args.batch_sizes,
            "batch_timeout_ms": args.timeouts,
        },
        "configurations": configurations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep max_batch_size x batch_timeout for dynamic batching."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--timeouts", type=int, nargs="+", default=[0, 5, 10, 20])
    parser.add_argument("--prefill-latency-ms", type=int, default=25)
    parser.add_argument("--decode-latency-ms", type=int, default=10)
    parser.add_argument(
        "--backend", choices=["mock", "llama.cpp"], default="mock",
    )
    parser.add_argument(
        "--scheduler", choices=["fifo", "priority"], default="fifo",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "results" / "batch_sweep_c64.json",
    )
    args = parser.parse_args()
    report = asyncio.run(run_sweep(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
