from collections import Counter
from collections.abc import Callable, Sequence

from llm_runtime.core.request import RuntimeRequest
from llm_runtime.metrics.gpu import GPUSnapshot, gpu_snapshot
from llm_runtime.metrics.latency import percentile


_LATENCY_BUCKETS: list[float] = [
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
]


class Histogram:
    """A simple latency histogram with predefined bucket boundaries.

    Each observation is counted into the first bucket whose upper bound is
    greater than or equal to the observed value. Observations larger than
    the largest bucket are counted in a trailing ``+Inf`` bucket. Prometheus
    export emits cumulative bucket counts, as required by its histogram format.
    """

    def __init__(self, buckets: list[float] | None = None) -> None:
        self._boundaries = buckets or _LATENCY_BUCKETS
        self._counts: list[int] = [0] * (len(self._boundaries) + 1)
        self._sum = 0.0
        self._total = 0

    def observe(self, value: float) -> None:
        self._sum += value
        self._total += 1
        for idx, boundary in enumerate(self._boundaries):
            if value <= boundary:
                self._counts[idx] += 1
                return
        self._counts[-1] += 1  # +Inf bucket

    @property
    def sum_samples(self) -> float:
        return self._sum

    @property
    def count(self) -> int:
        return self._total

    def prometheus_lines(self, name: str, help_text: str) -> str:
        if self._total == 0:
            return ""
        lines: list[str] = []
        _add = lines.append
        _add(f"# HELP {name} {help_text}")
        _add(f"# TYPE {name} histogram")
        cumulative = 0
        for idx, boundary in enumerate(self._boundaries):
            cumulative += self._counts[idx]
            _add(f'{name}_bucket{{le="{boundary}"}} {cumulative}')
        _add(f'{name}_bucket{{le="+Inf"}} {self._total}')
        _add(f"{name}_sum {self._sum}")
        _add(f"{name}_count {self._total}")
        return "\n".join(lines) + "\n"

    def snapshot(self) -> dict[str, object]:
        return {
            "count": self._total,
            "sum": round(self._sum, 6),
            "buckets": {
                str(b): self._counts[i] for i, b in enumerate(self._boundaries)
            },
            "+Inf": self._counts[-1],
        }


class MetricsCollector:
    """In-memory metrics collector with latency histograms and Prometheus export."""

    def __init__(self, gpu_sampler: Callable[[], GPUSnapshot] | None = None) -> None:
        self.request_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.rejected_count = 0
        self.rejections_by_reason: Counter[str] = Counter()
        self.rejections_by_priority: Counter[str] = Counter()
        self.rejections_by_reason_priority: Counter[tuple[str, str]] = Counter()
        self.active_requests = 0
        self.generated_tokens_total = 0
        self.batch_count = 0
        self.batch_sizes: list[int] = []
        self.queue_wait_times: list[float] = []
        self.ttft_times: list[float] = []
        self.total_latencies: list[float] = []
        self.queue_wait_histogram = Histogram()
        self.ttft_histogram = Histogram()
        self.total_latency_histogram = Histogram()
        self._gpu_sampler = gpu_sampler or gpu_snapshot

    def record_request(self) -> None:
        self.request_count += 1
        self.active_requests += 1

    def record_success(self, request: RuntimeRequest, generated_tokens: int = 0) -> None:
        self.completed_count += 1
        self.active_requests = max(0, self.active_requests - 1)
        self.generated_tokens_total += generated_tokens
        self._record_timing(request)

    def record_failure(self, request: RuntimeRequest) -> None:
        request.mark_failed()
        self.failed_count += 1
        self.active_requests = max(0, self.active_requests - 1)
        self._record_timing(request)

    def record_rejection(self, priority: int, reason: str) -> None:
        normalized_reason = reason or "unknown"
        priority_label = str(priority)
        self.rejected_count += 1
        self.rejections_by_reason[normalized_reason] += 1
        self.rejections_by_priority[priority_label] += 1
        self.rejections_by_reason_priority[(normalized_reason, priority_label)] += 1

    def record_batch(self, batch_size: int) -> None:
        self.batch_count += 1
        self.batch_sizes.append(batch_size)

    def snapshot(self) -> dict[str, object]:
        batch_distribution = Counter(self.batch_sizes)
        return {
            "request_count": self.request_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "rejected_count": self.rejected_count,
            "rejections_by_reason": dict(sorted(self.rejections_by_reason.items())),
            "rejections_by_priority": dict(
                sorted(self.rejections_by_priority.items())
            ),
            "active_requests": self.active_requests,
            "generated_tokens_total": self.generated_tokens_total,
            "batch_count": self.batch_count,
            "batch_size_avg": _avg(self.batch_sizes),
            "batch_size_max": max(self.batch_sizes, default=0),
            "batch_size_distribution": {
                str(size): count for size, count in sorted(batch_distribution.items())
            },
            "queue_wait_time_avg_s": _avg(self.queue_wait_times),
            "queue_wait_time_p50_s": percentile(self.queue_wait_times, 50),
            "queue_wait_time_p95_s": percentile(self.queue_wait_times, 95),
            "queue_wait_time_p99_s": percentile(self.queue_wait_times, 99),
            "queue_wait_histogram": self.queue_wait_histogram.snapshot(),
            "ttft_avg_s": _avg(self.ttft_times),
            "ttft_p50_s": percentile(self.ttft_times, 50),
            "ttft_p95_s": percentile(self.ttft_times, 95),
            "ttft_p99_s": percentile(self.ttft_times, 99),
            "ttft_histogram": self.ttft_histogram.snapshot(),
            "total_latency_avg_s": _avg(self.total_latencies),
            "total_latency_p50_s": percentile(self.total_latencies, 50),
            "total_latency_p95_s": percentile(self.total_latencies, 95),
            "total_latency_p99_s": percentile(self.total_latencies, 99),
            "total_latency_histogram": self.total_latency_histogram.snapshot(),
            "gpu": self._gpu_sampler(),
        }

    def snapshot_prometheus(self, queue_size: int = 0) -> str:
        """Return runtime metrics as Prometheus exposition-format text."""
        lines: list[str] = []
        _add = lines.append

        _add("# HELP llm_requests_total Total requests received.")
        _add("# TYPE llm_requests_total counter")
        received_count = self.request_count + self.rejected_count
        _add(f'llm_requests_total{{status="received"}} {received_count}')
        _add(f'llm_requests_total{{status="accepted"}} {self.request_count}')
        _add(f'llm_requests_total{{status="completed"}} {self.completed_count}')
        _add(f'llm_requests_total{{status="failed"}} {self.failed_count}')
        _add(f'llm_requests_total{{status="rejected"}} {self.rejected_count}')

        _add("")
        _add("# HELP llm_rejected_requests_total Total rejected requests.")
        _add("# TYPE llm_rejected_requests_total counter")
        for (
            reason,
            priority,
        ), count in sorted(self.rejections_by_reason_priority.items()):
            _add(
                'llm_rejected_requests_total'
                f'{{reason="{_escape_label(reason)}",priority="{_escape_label(priority)}"}} '
                f"{count}"
            )

        _add("")
        _add("# HELP llm_generated_tokens_total Total generated tokens.")
        _add("# TYPE llm_generated_tokens_total counter")
        _add(f"llm_generated_tokens_total {self.generated_tokens_total}")

        _add("")
        _add("# HELP llm_batches_total Total batch executions.")
        _add("# TYPE llm_batches_total counter")
        _add(f"llm_batches_total {self.batch_count}")

        _add("")
        _add("# HELP llm_active_requests Currently active requests.")
        _add("# TYPE llm_active_requests gauge")
        _add(f"llm_active_requests {self.active_requests}")

        _add("")
        _add("# HELP llm_queue_size Current scheduler queue depth.")
        _add("# TYPE llm_queue_size gauge")
        _add(f"llm_queue_size {queue_size}")

        _add("")
        _add("# HELP llm_batch_size_avg Average observed batch size.")
        _add("# TYPE llm_batch_size_avg gauge")
        _add(f"llm_batch_size_avg {_avg(self.batch_sizes)}")

        _add("")
        _add("# HELP llm_batch_size_max Maximum observed batch size.")
        _add("# TYPE llm_batch_size_max gauge")
        _add(f"llm_batch_size_max {max(self.batch_sizes, default=0)}")

        _add("")
        lines.extend(_gpu_prometheus_lines(self._gpu_sampler()))

        # Per-latency quantile gauges. Histogram metrics keep the base names.
        for name, values in [
            ("llm_queue_wait_quantile_seconds", self.queue_wait_times),
            ("llm_ttft_quantile_seconds", self.ttft_times),
            ("llm_total_latency_quantile_seconds", self.total_latencies),
        ]:
            _add("")
            _add(f"# HELP {name} Latency quantiles in seconds.")
            _add(f"# TYPE {name} gauge")
            _add(f'{name}{{quantile="0.5"}} {percentile(values, 50)}')
            _add(f'{name}{{quantile="0.95"}} {percentile(values, 95)}')
            _add(f'{name}{{quantile="0.99"}} {percentile(values, 99)}')

        # Histogram metrics
        for name, histogram, help_text in [
            (
                "llm_queue_wait_seconds",
                self.queue_wait_histogram,
                "Queue wait time histogram.",
            ),
            (
                "llm_ttft_seconds",
                self.ttft_histogram,
                "Time-to-first-token histogram.",
            ),
            (
                "llm_total_latency_seconds",
                self.total_latency_histogram,
                "Total request latency histogram.",
            ),
        ]:
            text = histogram.prometheus_lines(name, help_text)
            if text:
                _add("")
                lines.append(text.rstrip())

        return "\n".join(lines) + "\n"

    def _record_timing(self, request: RuntimeRequest) -> None:
        if request.enqueued_at is not None and request.started_at is not None:
            wait = request.started_at - request.enqueued_at
            self.queue_wait_times.append(wait)
            self.queue_wait_histogram.observe(wait)
        if request.first_token_at is not None:
            ttft = request.first_token_at - request.created_at
            self.ttft_times.append(ttft)
            self.ttft_histogram.observe(ttft)
        end_time = request.completed_at or request.failed_at
        if end_time is not None:
            total = end_time - request.created_at
            self.total_latencies.append(total)
            self.total_latency_histogram.observe(total)


def _avg(values: Sequence[float | int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _gpu_prometheus_lines(snapshot: GPUSnapshot) -> list[str]:
    source = _escape_label(snapshot["source"])
    available = 1 if snapshot["status"] == "available" else 0
    gpu_count = snapshot["gpu_count"]
    memory_used_mb = snapshot["memory_used_mb"]
    memory_total_mb = snapshot["memory_total_mb"]
    utilization_pct = snapshot["utilization_pct"]

    lines = [
        "# HELP llm_gpu_available Whether GPU metrics are available from the sampler.",
        "# TYPE llm_gpu_available gauge",
        f'llm_gpu_available{{source="{source}"}} {available}',
        "",
        "# HELP llm_gpu_count Number of GPUs reported by the sampler.",
        "# TYPE llm_gpu_count gauge",
        f'llm_gpu_count{{source="{source}"}} {gpu_count}',
        "",
        "# HELP llm_gpu_memory_used_bytes Total GPU memory used.",
        "# TYPE llm_gpu_memory_used_bytes gauge",
        (
            f'llm_gpu_memory_used_bytes{{source="{source}"}} '
            f"{memory_used_mb * 1024 * 1024}"
        ),
        "",
        "# HELP llm_gpu_memory_total_bytes Total GPU memory capacity.",
        "# TYPE llm_gpu_memory_total_bytes gauge",
        (
            f'llm_gpu_memory_total_bytes{{source="{source}"}} '
            f"{memory_total_mb * 1024 * 1024}"
        ),
        "",
        "# HELP llm_gpu_utilization_percent Average GPU utilization percent.",
        "# TYPE llm_gpu_utilization_percent gauge",
        f'llm_gpu_utilization_percent{{source="{source}"}} {utilization_pct}',
    ]

    if snapshot["gpus"]:
        lines.extend(
            [
                "",
                "# HELP llm_gpu_device_memory_used_bytes Per-GPU memory used.",
                "# TYPE llm_gpu_device_memory_used_bytes gauge",
                "# HELP llm_gpu_device_memory_total_bytes Per-GPU memory capacity.",
                "# TYPE llm_gpu_device_memory_total_bytes gauge",
                "# HELP llm_gpu_device_utilization_percent Per-GPU utilization percent.",
                "# TYPE llm_gpu_device_utilization_percent gauge",
            ]
        )
        for gpu in snapshot["gpus"]:
            index = _escape_label(str(gpu["index"]))
            name = _escape_label(gpu["name"])
            labels = f'gpu="{index}",name="{name}",source="{source}"'
            lines.extend(
                [
                    f'llm_gpu_device_memory_used_bytes{{{labels}}} '
                    f'{gpu["memory_used_mb"] * 1024 * 1024}',
                    f'llm_gpu_device_memory_total_bytes{{{labels}}} '
                    f'{gpu["memory_total_mb"] * 1024 * 1024}',
                    f'llm_gpu_device_utilization_percent{{{labels}}} '
                    f'{gpu["utilization_pct"]}',
                ]
            )
    return lines


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
