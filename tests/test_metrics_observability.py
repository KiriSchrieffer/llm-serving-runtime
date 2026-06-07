import json
import logging

from llm_runtime.api.schemas import ChatCompletionRequest
from llm_runtime.core.request import RuntimeRequest
from llm_runtime.metrics.collector import Histogram, MetricsCollector, _LATENCY_BUCKETS
from llm_runtime.utils.logging import JSONFormatter, RequestLogger


def test_histogram_observations_fall_into_correct_buckets():
    hist = Histogram()
    hist.observe(0.003)
    hist.observe(0.008)
    hist.observe(0.05)
    hist.observe(2.0)
    hist.observe(5.5)
    hist.observe(20.0)
    assert hist.count == 6
    assert hist._counts[1] == 1
    assert hist._counts[2] == 1
    assert hist._counts[4] == 1
    assert hist._counts[-1] == 1


def test_histogram_empty_produces_empty_prometheus():
    hist = Histogram()
    assert hist.prometheus_lines("test", "help") == ""
    snap = hist.snapshot()
    assert snap["count"] == 0
    assert snap["sum"] == 0


def test_histogram_prometheus_output_is_valid():
    hist = Histogram()
    hist.observe(0.3)
    hist.observe(1.2)
    output = hist.prometheus_lines("llm_queue_wait_seconds", "Queue wait time histogram.")
    assert "HELP" in output
    assert "TYPE" in output
    assert "histogram" in output
    assert "llm_queue_wait_seconds_bucket" in output
    assert "llm_queue_wait_seconds_sum" in output
    assert "llm_queue_wait_seconds_count 2" in output


def test_histogram_prometheus_buckets_are_cumulative():
    hist = Histogram(buckets=[0.1, 0.5, 1.0])
    hist.observe(0.05)
    hist.observe(0.4)
    hist.observe(2.0)

    output = hist.prometheus_lines("llm_test_seconds", "Test histogram.")

    assert 'llm_test_seconds_bucket{le="0.1"} 1' in output
    assert 'llm_test_seconds_bucket{le="0.5"} 2' in output
    assert 'llm_test_seconds_bucket{le="1.0"} 2' in output
    assert 'llm_test_seconds_bucket{le="+Inf"} 3' in output
    assert "llm_test_seconds_count 3" in output


def test_histogram_snapshot_format():
    hist = Histogram()
    hist.observe(0.1)
    snap = hist.snapshot()
    assert snap["count"] == 1
    assert "buckets" in snap
    assert "+Inf" in snap
    assert snap["buckets"]["0.1"] == 1


def test_collector_records_histogram_timing():
    collector = MetricsCollector()
    request = RuntimeRequest.from_chat_request(
        ChatCompletionRequest(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=2,
        )
    )
    request.mark_enqueued()
    request.mark_started()
    request.mark_first_token()
    request.mark_completed()
    collector.record_request()
    collector.record_success(request, generated_tokens=2)
    assert collector.queue_wait_histogram.count == 1
    assert collector.ttft_histogram.count == 1
    assert collector.total_latency_histogram.count == 1


def test_collector_snapshot_includes_histograms():
    collector = MetricsCollector()
    snap = collector.snapshot()
    assert "queue_wait_histogram" in snap
    assert "ttft_histogram" in snap
    assert "total_latency_histogram" in snap
    assert snap["queue_wait_histogram"]["count"] == 0


def test_json_formatter_produces_valid_json():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=1,
        msg="hello world", args=(), exc_info=None,
    )
    record.request_id = "chatcmpl-abc123"
    record.event = "request_received"
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["message"] == "hello world"
    assert parsed["request_id"] == "chatcmpl-abc123"
    assert parsed["event"] == "request_received"
    assert parsed["level"] == "INFO"


def test_request_logger_emits_structured_events(caplog):
    caplog.set_level(logging.INFO, logger="llm_runtime.request")
    logger = RequestLogger()
    logger.request_received("req-1", "mock-llm", stream=False)
    logger.request_enqueued("req-1")
    logger.request_completed("req-1", tokens=3, elapsed_ms=45.2)
    logger.request_failed("req-2", error="timeout")
    assert len(caplog.records) == 4
    assert caplog.records[2].tokens == 3
    assert caplog.records[2].elapsed_ms == 45.2


def test_request_logger_batch_formed_event(caplog):
    caplog.set_level(logging.INFO, logger="llm_runtime.request")
    logger = RequestLogger()
    logger.batch_formed(batch_size=3, request_ids=["a", "b", "c"])
    record = caplog.records[0]
    assert record.event == "batch_formed"
    assert record.batch_size == 3


def test_metrics_endpoint_includes_histogram_data():
    from llm_runtime.main import create_app
    from llm_runtime.core.lifecycle import RuntimeServices
    from fastapi.testclient import TestClient
    services = RuntimeServices.create()
    app = create_app(services)
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1},
        )
        snap = client.get("/metrics").json()
    hist = snap["total_latency_histogram"]
    assert hist["count"] == 1
    assert hist["sum"] > 0
    assert len(hist["buckets"]) == len(_LATENCY_BUCKETS)


def test_metrics_prometheus_output_includes_histogram():
    from llm_runtime.main import create_app
    from llm_runtime.core.lifecycle import RuntimeServices
    from fastapi.testclient import TestClient
    services = RuntimeServices.create()
    app = create_app(services)
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1},
        )
        prom = client.get("/metrics", headers={"accept": "text/plain"}).text
    assert "TYPE llm_total_latency_seconds histogram" in prom
    assert "llm_total_latency_seconds_bucket" in prom
    assert "llm_total_latency_seconds_sum" in prom
    assert "llm_total_latency_seconds_count" in prom


def test_prometheus_output_retains_percentile_gauges():
    from llm_runtime.main import create_app
    from llm_runtime.core.lifecycle import RuntimeServices
    from fastapi.testclient import TestClient
    services = RuntimeServices.create()
    app = create_app(services)
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1},
        )
        prom = client.get("/metrics", headers={"accept": "text/plain"}).text
    assert "llm_queue_wait_quantile_seconds" in prom
    assert "llm_ttft_quantile_seconds" in prom


def test_prometheus_quantile_gauges_use_seconds_and_distinct_names():
    collector = MetricsCollector()
    collector.queue_wait_times.extend([0.25, 0.5, 1.0])
    collector.queue_wait_histogram.observe(0.5)

    prom = collector.snapshot_prometheus()

    assert "# TYPE llm_queue_wait_quantile_seconds gauge" in prom
    assert 'llm_queue_wait_quantile_seconds{quantile="0.5"} 0.5' in prom
    assert "# TYPE llm_queue_wait_seconds histogram" in prom
    assert "# TYPE llm_queue_wait_seconds gauge" not in prom
