from llm_runtime.metrics.collector import MetricsCollector


def export_metrics(collector: MetricsCollector) -> dict[str, object]:
    """Export metrics as a JSON-serializable dictionary."""

    return collector.snapshot()

