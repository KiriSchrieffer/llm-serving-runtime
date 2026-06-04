import argparse
import json
from pathlib import Path


def summarize(
    sweep_path: Path,
    baseline_path: Path | None = None,
    ttft_budget_ms: float | None = None,
) -> str:
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    workload = sweep["workload"]
    baseline_run = _find_baseline_run(baseline_path, workload["concurrency"])
    configurations = sweep["configurations"]
    lines = [
        "| Batch size | Timeout ms | Tokens/s | vs FIFO | TTFT ms | P95 ms | Avg batch |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in configurations:
        run = result["run"]
        metrics = run["runtime_metrics"]
        delta = _throughput_delta(run, baseline_run)
        lines.append(
            f"| {result['max_batch_size']} | {result['batch_timeout_ms']} | "
            f"{run['tokens_per_second']:.1f} | {delta} | "
            f"{metrics['ttft_avg_s'] * 1000:.1f} | "
            f"{run['p95_latency_s'] * 1000:.1f} | "
            f"{metrics['batch_size_avg']:.2f} |"
        )

    highest_throughput = max(configurations, key=lambda result: result["run"]["tokens_per_second"])
    lowest_ttft = min(
        configurations,
        key=lambda result: result["run"]["runtime_metrics"]["ttft_avg_s"],
    )
    lowest_p95 = min(configurations, key=lambda result: result["run"]["p95_latency_s"])
    lines.extend(
        [
            "",
            _selection_line("Highest throughput", highest_throughput),
            _selection_line("Lowest TTFT", lowest_ttft),
            _selection_line("Lowest P95", lowest_p95),
        ]
    )
    if ttft_budget_ms is not None:
        eligible = [
            result
            for result in configurations
            if result["run"]["runtime_metrics"]["ttft_avg_s"] * 1000 <= ttft_budget_ms
        ]
        if eligible:
            chosen = max(eligible, key=lambda result: result["run"]["tokens_per_second"])
            lines.append(
                _selection_line(f"Highest throughput under {ttft_budget_ms:.0f} ms TTFT", chosen)
            )
        else:
            lines.append(f"No configuration met the {ttft_budget_ms:.0f} ms TTFT budget.")
    return "\n".join(lines)


def _find_baseline_run(path: Path | None, concurrency: int) -> dict[str, object] | None:
    if path is None:
        return None
    baseline = json.loads(path.read_text(encoding="utf-8"))
    return next(run for run in baseline["runs"] if run["concurrency"] == concurrency)


def _throughput_delta(
    run: dict[str, object],
    baseline: dict[str, object] | None,
) -> str:
    if baseline is None:
        return "-"
    delta = (run["tokens_per_second"] / baseline["tokens_per_second"] - 1) * 100
    return f"{delta:+.1f}%"


def _selection_line(label: str, result: dict[str, object]) -> str:
    run = result["run"]
    metrics = run["runtime_metrics"]
    return (
        f"- {label}: batch_size={result['max_batch_size']}, "
        f"timeout_ms={result['batch_timeout_ms']}, "
        f"tokens/s={run['tokens_per_second']:.1f}, "
        f"TTFT={metrics['ttft_avg_s'] * 1000:.1f} ms, "
        f"P95={run['p95_latency_s'] * 1000:.1f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--ttft-budget-ms", type=float)
    args = parser.parse_args()
    print(summarize(args.sweep, args.baseline, args.ttft_budget_ms))


if __name__ == "__main__":
    main()
