import subprocess
from collections.abc import Sequence

from llm_runtime.metrics.collector import MetricsCollector
from llm_runtime.metrics.gpu import NvidiaSmiSampler


def test_nvidia_smi_sampler_parses_gpu_snapshot() -> None:
    def runner(args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        assert args[0] == "nvidia-smi"
        assert timeout_s == 1.0
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=(
                "0, NVIDIA A100-SXM4-40GB, 1024, 40960, 73\n"
                "1, NVIDIA L4, 2048, 24576, 45\n"
            ),
            stderr="",
        )

    snapshot = NvidiaSmiSampler(runner=runner).snapshot()

    assert snapshot["status"] == "available"
    assert snapshot["gpu_count"] == 2
    assert snapshot["memory_used_mb"] == 3072
    assert snapshot["memory_total_mb"] == 65536
    assert snapshot["utilization_pct"] == 59
    assert snapshot["gpus"][0]["name"] == "NVIDIA A100-SXM4-40GB"


def test_nvidia_smi_sampler_returns_unavailable_on_command_failure() -> None:
    def runner(args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args,
            returncode=9,
            stdout="",
            stderr="NVIDIA-SMI has failed",
        )

    snapshot = NvidiaSmiSampler(runner=runner).snapshot()

    assert snapshot["status"] == "unavailable"
    assert snapshot["reason"] == "NVIDIA-SMI has failed"
    assert snapshot["gpu_count"] == 0
    assert snapshot["gpus"] == []


def test_metrics_snapshot_includes_gpu_sampler_output() -> None:
    collector = MetricsCollector(gpu_sampler=_fake_gpu_snapshot)

    snapshot = collector.snapshot()

    assert snapshot["gpu"]["status"] == "available"
    assert snapshot["gpu"]["memory_used_mb"] == 1024
    assert snapshot["gpu"]["utilization_pct"] == 75


def test_prometheus_output_includes_gpu_gauges() -> None:
    collector = MetricsCollector(gpu_sampler=_fake_gpu_snapshot)

    output = collector.snapshot_prometheus()

    assert "# TYPE llm_gpu_available gauge" in output
    assert 'llm_gpu_available{source="nvidia-smi"} 1' in output
    assert 'llm_gpu_count{source="nvidia-smi"} 1' in output
    assert 'llm_gpu_memory_used_bytes{source="nvidia-smi"} 1073741824' in output
    assert 'llm_gpu_utilization_percent{source="nvidia-smi"} 75' in output
    assert (
        'llm_gpu_device_utilization_percent{gpu="0",name="NVIDIA L4",'
        'source="nvidia-smi"} 75'
    ) in output


def _fake_gpu_snapshot() -> dict[str, object]:
    return {
        "status": "available",
        "source": "nvidia-smi",
        "gpu_count": 1,
        "memory_used_mb": 1024,
        "memory_total_mb": 24576,
        "utilization_pct": 75,
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA L4",
                "memory_used_mb": 1024,
                "memory_total_mb": 24576,
                "utilization_pct": 75,
            }
        ],
    }
