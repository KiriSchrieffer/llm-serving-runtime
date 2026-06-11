import argparse
import asyncio
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_runtime.api.schemas import ChatCompletionRequest
from llm_runtime.backends.mock_backend import MockBackend
from llm_runtime.core.request import RuntimeRequest
from llm_runtime.core.response import CompletionHandle, ScheduledRequest
from llm_runtime.metrics.collector import MetricsCollector
from llm_runtime.scheduler.fifo import FIFOScheduler
from llm_runtime.scheduler.priority import PriorityScheduler
from llm_runtime.utils.logging import RequestLogger
from llm_runtime.workers.manager import WorkerManager


HIGH_PRIORITY = 0
LOW_PRIORITY = 10
EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    label: str
    priority_class: str
    priority: int
    arrival_offset_ms: int
    max_tokens: int


@dataclass(frozen=True, slots=True)
class RequestObservation:
    label: str
    priority_class: str
    priority: int
    arrival_offset_ms: int
    queue_wait_s: float
    ttft_s: float
    total_latency_s: float
    completion_tokens: int


def build_mixed_workload(
    low_requests: int,
    high_requests: int,
    high_arrival_delay_ms: int,
    low_max_tokens: int,
    high_max_tokens: int,
) -> list[WorkloadSpec]:
    """Build a low-priority backlog followed by urgent high-priority work."""

    workload: list[WorkloadSpec] = []
    for index in range(low_requests):
        workload.append(
            WorkloadSpec(
                label=f"low-{index:03d}",
                priority_class="low",
                priority=LOW_PRIORITY,
                arrival_offset_ms=0,
                max_tokens=low_max_tokens,
            )
        )
    for index in range(high_requests):
        workload.append(
            WorkloadSpec(
                label=f"high-{index:03d}",
                priority_class="high",
                priority=HIGH_PRIORITY,
                arrival_offset_ms=high_arrival_delay_ms,
                max_tokens=high_max_tokens,
            )
        )
    return workload


async def run_scheduler_benchmark(
    scheduler_name: str,
    workload: list[WorkloadSpec],
    prefill_latency_ms: int,
    decode_latency_ms: int,
    max_batch_size: int,
    batch_timeout_ms: int,
    benchmark_timeout_s: float,
) -> dict[str, object]:
    scheduler = PriorityScheduler() if scheduler_name == "priority" else FIFOScheduler()
    manager = WorkerManager(
        scheduler=scheduler,
        backend=MockBackend(
            prefill_latency_ms=prefill_latency_ms,
            decode_latency_ms=decode_latency_ms,
        ),
        metrics=MetricsCollector(),
        request_logger=RequestLogger(),
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
    )
    await manager.start()
    try:
        tasks = [
            asyncio.create_task(_submit_and_observe(scheduler, spec))
            for spec in workload
        ]
        observations = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=benchmark_timeout_s,
        )
    finally:
        await manager.stop()

    return {
        "scheduler": scheduler_name,
        "observations": [asdict(observation) for observation in observations],
        "summary": summarize_observations(observations),
    }


def summarize_observations(
    observations: list[RequestObservation],
    starvation_threshold_s: float = 0.75,
) -> dict[str, object]:
    high = [item for item in observations if item.priority_class == "high"]
    low = [item for item in observations if item.priority_class == "low"]
    high_summary = _summarize_group(high)
    low_summary = _summarize_group(low)
    high_wait = float(high_summary["queue_wait_avg_s"])
    low_wait = float(low_summary["queue_wait_avg_s"])
    low_starved_count = sum(
        1 for item in low if item.queue_wait_s >= starvation_threshold_s
    )

    return {
        "request_count": len(observations),
        "high_priority": high_summary,
        "low_priority": low_summary,
        "fairness": {
            "queue_wait_jain_index": _jain_index(
                [1.0 / (high_wait + EPSILON), 1.0 / (low_wait + EPSILON)]
            ),
            "low_to_high_queue_wait_ratio": _safe_ratio(low_wait, high_wait),
        },
        "starvation": {
            "threshold_s": starvation_threshold_s,
            "low_starved_count": low_starved_count,
            "low_starved_fraction": _safe_ratio(low_starved_count, len(low)),
            "low_queue_wait_max_s": max((item.queue_wait_s for item in low), default=0.0),
        },
    }


def compare_summaries(
    fifo_summary: dict[str, object],
    priority_summary: dict[str, object],
) -> dict[str, object]:
    fifo_high = _section(fifo_summary, "high_priority")
    priority_high = _section(priority_summary, "high_priority")
    fifo_low = _section(fifo_summary, "low_priority")
    priority_low = _section(priority_summary, "low_priority")
    fifo_fairness = _section(fifo_summary, "fairness")
    priority_fairness = _section(priority_summary, "fairness")

    fifo_high_ttft = float(fifo_high["ttft_avg_s"])
    priority_high_ttft = float(priority_high["ttft_avg_s"])
    fifo_low_wait = float(fifo_low["queue_wait_avg_s"])
    priority_low_wait = float(priority_low["queue_wait_avg_s"])
    fifo_jain = float(fifo_fairness["queue_wait_jain_index"])
    priority_jain = float(priority_fairness["queue_wait_jain_index"])

    return {
        "high_priority_ttft_delta_s": priority_high_ttft - fifo_high_ttft,
        "high_priority_ttft_improvement_pct": _percent_change(
            fifo_high_ttft,
            fifo_high_ttft - priority_high_ttft,
        ),
        "low_priority_queue_wait_delta_s": priority_low_wait - fifo_low_wait,
        "low_priority_queue_wait_increase_pct": _percent_change(
            fifo_low_wait,
            priority_low_wait - fifo_low_wait,
        ),
        "fairness_jain_delta": priority_jain - fifo_jain,
    }


async def run_priority_suite(args: argparse.Namespace) -> dict[str, object]:
    workload = build_mixed_workload(
        low_requests=args.low_requests,
        high_requests=args.high_requests,
        high_arrival_delay_ms=args.high_arrival_delay_ms,
        low_max_tokens=args.low_max_tokens,
        high_max_tokens=args.high_max_tokens,
    )
    fifo = await run_scheduler_benchmark(
        scheduler_name="fifo",
        workload=workload,
        prefill_latency_ms=args.prefill_latency_ms,
        decode_latency_ms=args.decode_latency_ms,
        max_batch_size=args.max_batch_size,
        batch_timeout_ms=args.batch_timeout_ms,
        benchmark_timeout_s=args.benchmark_timeout_s,
    )
    priority = await run_scheduler_benchmark(
        scheduler_name="priority",
        workload=workload,
        prefill_latency_ms=args.prefill_latency_ms,
        decode_latency_ms=args.decode_latency_ms,
        max_batch_size=args.max_batch_size,
        batch_timeout_ms=args.batch_timeout_ms,
        benchmark_timeout_s=args.benchmark_timeout_s,
    )
    return _round_nested(
        {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "workload": {
                "low_requests": args.low_requests,
                "high_requests": args.high_requests,
                "high_arrival_delay_ms": args.high_arrival_delay_ms,
                "low_priority": LOW_PRIORITY,
                "high_priority": HIGH_PRIORITY,
                "low_max_tokens": args.low_max_tokens,
                "high_max_tokens": args.high_max_tokens,
                "prefill_latency_ms": args.prefill_latency_ms,
                "decode_latency_ms": args.decode_latency_ms,
                "max_batch_size": args.max_batch_size,
                "batch_timeout_ms": args.batch_timeout_ms,
            },
            "runs": {
                "fifo": fifo,
                "priority": priority,
            },
            "comparison": compare_summaries(
                fifo["summary"],  # type: ignore[arg-type]
                priority["summary"],  # type: ignore[arg-type]
            ),
            "metric_notes": {
                "ttft_s": "created_at to first_token_at",
                "queue_wait_s": "enqueued_at to started_at",
                "queue_wait_jain_index": (
                    "Jain index over inverse average queue wait; 1.0 is equal delay"
                ),
                "starvation": (
                    "Low-priority requests whose queue wait exceeds threshold_s"
                ),
            },
        }
    )


async def _submit_and_observe(
    scheduler: FIFOScheduler | PriorityScheduler,
    spec: WorkloadSpec,
) -> RequestObservation:
    await asyncio.sleep(spec.arrival_offset_ms / 1000)
    request = _make_request(spec)
    handle = CompletionHandle.create()
    await scheduler.submit(ScheduledRequest(request=request, handle=handle))
    result = await handle.future

    if (
        request.enqueued_at is None
        or request.started_at is None
        or request.first_token_at is None
        or request.completed_at is None
    ):
        raise RuntimeError(f"incomplete timing data for {spec.label}")

    return RequestObservation(
        label=spec.label,
        priority_class=spec.priority_class,
        priority=spec.priority,
        arrival_offset_ms=spec.arrival_offset_ms,
        queue_wait_s=request.started_at - request.enqueued_at,
        ttft_s=request.first_token_at - request.created_at,
        total_latency_s=request.completed_at - request.created_at,
        completion_tokens=len(result.tokens),
    )


def _make_request(spec: WorkloadSpec) -> RuntimeRequest:
    return RuntimeRequest.from_chat_request(
        ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": f"{spec.priority_class} priority request {spec.label}",
                }
            ],
            max_tokens=spec.max_tokens,
            priority=spec.priority,
        )
    )


def _summarize_group(items: list[RequestObservation]) -> dict[str, object]:
    queue_waits = [item.queue_wait_s for item in items]
    ttfts = [item.ttft_s for item in items]
    total_latencies = [item.total_latency_s for item in items]
    return {
        "count": len(items),
        "queue_wait_avg_s": _mean(queue_waits),
        "queue_wait_p50_s": _percentile(queue_waits, 0.50),
        "queue_wait_p95_s": _percentile(queue_waits, 0.95),
        "queue_wait_max_s": max(queue_waits, default=0.0),
        "ttft_avg_s": _mean(ttfts),
        "ttft_p50_s": _percentile(ttfts, 0.50),
        "ttft_p95_s": _percentile(ttfts, 0.95),
        "total_latency_avg_s": _mean(total_latencies),
        "total_latency_p95_s": _percentile(total_latencies, 0.95),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    lower_weight = upper - index
    upper_weight = index - lower
    return ordered[lower] * lower_weight + ordered[upper] * upper_weight


def _jain_index(values: list[float]) -> float:
    if not values:
        return 0.0
    denominator = len(values) * sum(value * value for value in values)
    if denominator == 0:
        return 0.0
    numerator = sum(values) ** 2
    return numerator / denominator


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _percent_change(base: float, delta: float) -> float:
    if base == 0:
        return 0.0
    return delta / base * 100


def _section(summary: dict[str, object], key: str) -> dict[str, object]:
    section = summary[key]
    assert isinstance(section, dict)
    return section


def _round_nested(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FIFO vs priority scheduling on mixed-priority load."
    )
    parser.add_argument("--low-requests", type=int, default=24)
    parser.add_argument("--high-requests", type=int, default=8)
    parser.add_argument("--high-arrival-delay-ms", type=int, default=40)
    parser.add_argument("--low-max-tokens", type=int, default=12)
    parser.add_argument("--high-max-tokens", type=int, default=4)
    parser.add_argument("--prefill-latency-ms", type=int, default=25)
    parser.add_argument("--decode-latency-ms", type=int, default=10)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--batch-timeout-ms", type=int, default=0)
    parser.add_argument("--benchmark-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "results" / "priority_scheduler_mixed.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_priority_suite(args))
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
