import argparse
import json
from pathlib import Path


def summarize(baseline_path: Path, dynamic_path: Path) -> str:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    dynamic = json.loads(dynamic_path.read_text(encoding="utf-8"))
    dynamic_by_concurrency = {
        run["concurrency"]: run for run in dynamic["runs"]
    }
    lines = [
        "| Concurrency | FIFO tok/s | Batch tok/s | Throughput delta | "
        "FIFO P95 ms | Batch P95 ms | FIFO TTFT ms | Batch TTFT ms | Avg batch |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fifo_run in baseline["runs"]:
        dynamic_run = dynamic_by_concurrency[fifo_run["concurrency"]]
        fifo_tps = fifo_run["tokens_per_second"]
        dynamic_tps = dynamic_run["tokens_per_second"]
        throughput_delta = (dynamic_tps / fifo_tps - 1) * 100
        fifo_metrics = fifo_run["runtime_metrics"]
        batch_metrics = dynamic_run["runtime_metrics"]
        lines.append(
            f"| {fifo_run['concurrency']} | {fifo_tps:.1f} | {dynamic_tps:.1f} | "
            f"{throughput_delta:+.1f}% | {fifo_run['p95_latency_s'] * 1000:.1f} | "
            f"{dynamic_run['p95_latency_s'] * 1000:.1f} | "
            f"{fifo_metrics['ttft_avg_s'] * 1000:.1f} | "
            f"{batch_metrics['ttft_avg_s'] * 1000:.1f} | "
            f"{batch_metrics['batch_size_avg']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path, required=True)
    args = parser.parse_args()
    print(summarize(args.baseline, args.dynamic))


if __name__ == "__main__":
    main()
