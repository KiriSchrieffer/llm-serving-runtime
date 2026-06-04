"""Generate a formatted benchmark report from JSON result files."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _fmt(value, decimals=3):
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)


def _row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _is_backend_entry(entry):
    """Check if a dict value looks like a backend metric snapshot."""
    if not isinstance(entry, dict):
        return False
    metric_keys = {"request_count", "completed_count", "ttft_p50_s", "total_latency_p50_s",
                   "queue_wait_time_p50_s", "generated_tokens_total", "batch_count"}
    return bool(set(entry.keys()) & metric_keys)


def generate_report(result_paths: list[str], output_path: str = "benchmark_report.md") -> None:
    lines = []
    a = lines.append

    a(f"# Benchmark Report")
    a(f"")
    a(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    a(f"")

    for path in result_paths:
        data = json.loads(Path(path).read_text())
        a(f"## {Path(path).stem}")
        a(f"")

        if isinstance(data, dict):
            # Check if it's a multi-backend comparison
            backend_entries = {k: v for k, v in data.items() if _is_backend_entry(v)}
            if len(backend_entries) >= 2:
                a(_row("Metric", *backend_entries.keys()))
                a(_row(*(("---",) * (len(backend_entries) + 1))))
                metrics_set = set()
                for v in backend_entries.values():
                    metrics_set.update(k for k in v if isinstance(v[k], (int, float)))
                for metric in sorted(metrics_set):
                    a(_row(metric, *[_fmt(backend_entries[b].get(metric, "N/A"), 4) for b in backend_entries]))
                a(f"")
                continue

            # Single backend snapshot
            a(_row("Metric", "Value"))
            a(_row("---", "---"))
            for key, value in sorted(data.items()):
                if isinstance(value, (int, float)):
                    a(_row(key, _fmt(value, 4)))
                elif isinstance(value, dict):
                    a(_row(key, json.dumps(value)))
                else:
                    a(_row(key, str(value)))
        elif isinstance(data, list) and len(data) > 0:
            keys = [k for k in data[0].keys() if isinstance(data[0][k], (int, float, str))]
            a(_row(*(k for k in keys if k != "label")))
            a(_row(*(("---",) * (len(keys)))))
            for entry in data:
                a(_row(*[_fmt(entry.get(k, ""), 3) for k in keys]))
        a(f"")

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        results_dir = Path(__file__).parent / "results"
        paths = sorted(str(p) for p in results_dir.glob("*.json") if p.name != ".gitkeep")
    generate_report(paths)