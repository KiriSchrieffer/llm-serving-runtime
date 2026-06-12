import csv
import shutil
import subprocess
from collections.abc import Callable, Sequence
from io import StringIO


GPUSnapshot = dict[str, object]
Runner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]

_QUERY_FIELDS = [
    "index",
    "name",
    "memory.used",
    "memory.total",
    "utilization.gpu",
]


class NvidiaSmiSampler:
    """Collect GPU utilization and memory from nvidia-smi when available."""

    def __init__(
        self,
        command: str = "nvidia-smi",
        timeout_s: float = 1.0,
        runner: Runner | None = None,
    ) -> None:
        self.command = command
        self.timeout_s = timeout_s
        self._runner = runner

    def snapshot(self) -> GPUSnapshot:
        if self._runner is None and shutil.which(self.command) is None:
            return _unavailable("nvidia-smi not found")

        args = [
            self.command,
            f"--query-gpu={','.join(_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = (self._runner or _run)(args, self.timeout_s)
        except subprocess.TimeoutExpired:
            return _unavailable("nvidia-smi timed out")
        except OSError as exc:
            return _unavailable(str(exc))

        if result.returncode != 0:
            reason = result.stderr.strip() or f"nvidia-smi exited {result.returncode}"
            return _unavailable(reason)

        gpus = _parse_nvidia_smi(result.stdout)
        if not gpus:
            return _unavailable("no GPU data")

        memory_used_mb = sum(int(gpu["memory_used_mb"]) for gpu in gpus)
        memory_total_mb = sum(int(gpu["memory_total_mb"]) for gpu in gpus)
        utilization_pct = round(
            sum(int(gpu["utilization_pct"]) for gpu in gpus) / len(gpus)
        )
        return {
            "status": "available",
            "source": "nvidia-smi",
            "gpu_count": len(gpus),
            "memory_used_mb": memory_used_mb,
            "memory_total_mb": memory_total_mb,
            "utilization_pct": utilization_pct,
            "gpus": gpus,
        }


def gpu_snapshot() -> GPUSnapshot:
    """Return a best-effort GPU metrics snapshot."""

    return NvidiaSmiSampler().snapshot()


def _run(
    args: Sequence[str],
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        args,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout_s,
    )


def _parse_nvidia_smi(output: str) -> list[dict[str, object]]:
    gpus: list[dict[str, object]] = []
    reader = csv.reader(StringIO(output))
    for row in reader:
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) < len(_QUERY_FIELDS):
            continue
        index, name, memory_used, memory_total, utilization = [
            value.strip() for value in row[: len(_QUERY_FIELDS)]
        ]
        gpus.append(
            {
                "index": _parse_int(index),
                "name": name,
                "memory_used_mb": _parse_int(memory_used),
                "memory_total_mb": _parse_int(memory_total),
                "utilization_pct": _parse_int(utilization),
            }
        )
    return gpus


def _parse_int(value: str) -> int:
    digits = "".join(char for char in value if char.isdigit())
    if not digits:
        return 0
    return int(digits)


def _unavailable(reason: str) -> GPUSnapshot:
    return {
        "status": "unavailable",
        "source": "nvidia-smi",
        "reason": reason,
        "gpu_count": 0,
        "memory_used_mb": 0,
        "memory_total_mb": 0,
        "utilization_pct": 0,
        "gpus": [],
    }
