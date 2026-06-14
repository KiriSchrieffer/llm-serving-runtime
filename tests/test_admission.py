from llm_runtime.core.admission import AdmissionController


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_disabled_admission_controller_accepts_requests() -> None:
    controller = AdmissionController()

    decision = controller.admit(queue_size=10_000)

    assert decision.accepted


def test_max_queue_size_rejects_full_scheduler_queue() -> None:
    controller = AdmissionController(max_queue_size=2)

    accepted = controller.admit(queue_size=1)
    rejected = controller.admit(queue_size=2)

    assert accepted.accepted
    assert not rejected.accepted
    assert rejected.reason == "queue_full"
    assert rejected.status_code == 503


def test_token_bucket_rate_limit_rejects_until_refilled() -> None:
    clock = FakeClock()
    controller = AdmissionController(
        request_rate_limit_per_s=2,
        request_rate_limit_burst=2,
        time_fn=clock,
    )

    assert controller.admit(queue_size=0).accepted
    assert controller.admit(queue_size=0).accepted

    limited = controller.admit(queue_size=0)
    assert not limited.accepted
    assert limited.reason == "rate_limited"
    assert limited.status_code == 429

    clock.advance(0.5)

    assert controller.admit(queue_size=0).accepted
