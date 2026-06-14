from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from threading import Lock
from time import perf_counter


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    accepted: bool
    reason: str = ""
    detail: str = ""
    status_code: int = 200


class AdmissionController:
    """Fail-fast admission control for overload and request-rate protection."""

    def __init__(
        self,
        max_queue_size: int = 0,
        request_rate_limit_per_s: float = 0.0,
        request_rate_limit_burst: int = 0,
        time_fn: Callable[[], float] = perf_counter,
    ) -> None:
        self.max_queue_size = max_queue_size
        self.request_rate_limit_per_s = request_rate_limit_per_s
        self.request_rate_limit_burst = request_rate_limit_burst
        self._time_fn = time_fn
        self._capacity = _bucket_capacity(
            request_rate_limit_per_s=request_rate_limit_per_s,
            request_rate_limit_burst=request_rate_limit_burst,
        )
        self._tokens = float(self._capacity)
        self._last_refill_at = time_fn()
        self._lock = Lock()

    def admit(self, queue_size: int) -> AdmissionDecision:
        if self.max_queue_size > 0 and queue_size >= self.max_queue_size:
            return AdmissionDecision(
                accepted=False,
                reason="queue_full",
                detail="scheduler queue is full",
                status_code=503,
            )

        if self.request_rate_limit_per_s <= 0:
            return AdmissionDecision(accepted=True)

        with self._lock:
            self._refill_locked()
            if self._tokens < 1:
                return AdmissionDecision(
                    accepted=False,
                    reason="rate_limited",
                    detail="request rate limit exceeded",
                    status_code=429,
                )
            self._tokens -= 1
        return AdmissionDecision(accepted=True)

    def _refill_locked(self) -> None:
        now = self._time_fn()
        elapsed_s = max(0.0, now - self._last_refill_at)
        self._last_refill_at = now
        self._tokens = min(
            float(self._capacity),
            self._tokens + elapsed_s * self.request_rate_limit_per_s,
        )


def _bucket_capacity(
    request_rate_limit_per_s: float,
    request_rate_limit_burst: int,
) -> int:
    if request_rate_limit_per_s <= 0:
        return 0
    if request_rate_limit_burst > 0:
        return request_rate_limit_burst
    return max(1, ceil(request_rate_limit_per_s))
