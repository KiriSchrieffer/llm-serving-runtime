import pytest

from benchmarks.run_priority_scheduler_benchmark import (
    RequestObservation,
    build_mixed_workload,
    compare_summaries,
    summarize_observations,
)


def observation(
    label: str,
    priority_class: str,
    queue_wait_s: float,
    ttft_s: float,
) -> RequestObservation:
    return RequestObservation(
        label=label,
        priority_class=priority_class,
        priority=0 if priority_class == "high" else 10,
        arrival_offset_ms=0,
        queue_wait_s=queue_wait_s,
        ttft_s=ttft_s,
        total_latency_s=ttft_s + 0.1,
        completion_tokens=4,
    )


def test_build_mixed_workload_creates_delayed_high_priority_requests() -> None:
    workload = build_mixed_workload(
        low_requests=2,
        high_requests=1,
        high_arrival_delay_ms=50,
        low_max_tokens=12,
        high_max_tokens=4,
    )

    assert [item.priority_class for item in workload] == ["low", "low", "high"]
    assert [item.priority for item in workload] == [10, 10, 0]
    assert workload[-1].arrival_offset_ms == 50
    assert workload[-1].max_tokens == 4


def test_summarize_observations_reports_fairness_and_starvation() -> None:
    summary = summarize_observations(
        [
            observation("high-0", "high", queue_wait_s=0.1, ttft_s=0.2),
            observation("high-1", "high", queue_wait_s=0.2, ttft_s=0.3),
            observation("low-0", "low", queue_wait_s=0.8, ttft_s=0.9),
            observation("low-1", "low", queue_wait_s=1.0, ttft_s=1.1),
        ],
        starvation_threshold_s=0.75,
    )

    assert summary["request_count"] == 4
    assert summary["high_priority"]["ttft_avg_s"] == 0.25
    assert summary["low_priority"]["queue_wait_avg_s"] == 0.9
    assert summary["starvation"]["low_starved_count"] == 2
    assert summary["fairness"]["low_to_high_queue_wait_ratio"] == pytest.approx(6.0)
    assert summary["fairness"]["queue_wait_jain_index"] < 1.0


def test_compare_summaries_captures_priority_tradeoff() -> None:
    fifo_summary = summarize_observations(
        [
            observation("high-0", "high", queue_wait_s=1.0, ttft_s=1.1),
            observation("low-0", "low", queue_wait_s=0.3, ttft_s=0.4),
        ]
    )
    priority_summary = summarize_observations(
        [
            observation("high-0", "high", queue_wait_s=0.2, ttft_s=0.3),
            observation("low-0", "low", queue_wait_s=0.6, ttft_s=0.7),
        ]
    )

    comparison = compare_summaries(fifo_summary, priority_summary)

    assert comparison["high_priority_ttft_delta_s"] < 0
    assert comparison["high_priority_ttft_improvement_pct"] > 0
    assert comparison["low_priority_queue_wait_delta_s"] > 0
    assert comparison["low_priority_queue_wait_increase_pct"] > 0
