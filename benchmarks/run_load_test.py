import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(slots=True)
class Result:
    latency_s: float
    tokens: int


async def run_one(client: httpx.AsyncClient, max_tokens: int) -> Result:
    payload = {
        "model": "mock-llm",
        "messages": [{"role": "user", "content": "benchmark request"}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    start = time.perf_counter()
    response = await client.post("/v1/chat/completions", json=payload)
    response.raise_for_status()
    elapsed = time.perf_counter() - start
    usage = response.json()["usage"]
    return Result(latency_s=elapsed, tokens=usage["completion_tokens"])


async def run_load(
    base_url: str,
    concurrency: int,
    requests: int,
    max_tokens: int,
) -> dict[str, object]:
    async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
        before_metrics = (await client.get("/metrics")).json()
        pending = [run_one(client, max_tokens) for _ in range(requests)]
        start = time.perf_counter()
        results: list[Result] = []
        for offset in range(0, len(pending), concurrency):
            results.extend(await asyncio.gather(*pending[offset : offset + concurrency]))
        total_s = time.perf_counter() - start
        after_metrics = (await client.get("/metrics")).json()

    latencies = sorted(result.latency_s for result in results)
    total_tokens = sum(result.tokens for result in results)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    return {
        "requests": requests,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "tokens_per_second": total_tokens / total_s,
        "p50_latency_s": p50,
        "p95_latency_s": p95,
        "runtime_metrics": _workload_metrics(before_metrics, after_metrics),
    }


def _workload_metrics(
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, object]:
    completed = int(after["completed_count"]) - int(before["completed_count"])
    batch_count = int(after["batch_count"]) - int(before["batch_count"])
    return {
        "completed_count": completed,
        "failed_count": int(after["failed_count"]) - int(before["failed_count"]),
        "generated_tokens_total": int(after["generated_tokens_total"])
        - int(before["generated_tokens_total"]),
        "queue_wait_time_avg_s": _delta_average(
            before,
            after,
            "queue_wait_time_avg_s",
            "completed_count",
        ),
        "ttft_avg_s": _delta_average(
            before,
            after,
            "ttft_avg_s",
            "completed_count",
        ),
        "total_latency_avg_s": _delta_average(
            before,
            after,
            "total_latency_avg_s",
            "completed_count",
        ),
        "batch_count": batch_count,
        "batch_size_avg": _delta_average(
            before,
            after,
            "batch_size_avg",
            "batch_count",
        ),
        "batch_size_max": _max_new_batch_size(before, after),
        "batch_size_distribution": _delta_distribution(before, after),
    }


def _delta_average(
    before: dict[str, object],
    after: dict[str, object],
    average_key: str,
    count_key: str,
) -> float:
    before_count = int(before[count_key])
    after_count = int(after[count_key])
    delta_count = after_count - before_count
    if delta_count == 0:
        return 0.0
    before_total = float(before[average_key]) * before_count
    after_total = float(after[average_key]) * after_count
    return (after_total - before_total) / delta_count


def _delta_distribution(
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, int]:
    before_distribution = before["batch_size_distribution"]
    after_distribution = after["batch_size_distribution"]
    assert isinstance(before_distribution, dict)
    assert isinstance(after_distribution, dict)
    return {
        str(size): int(count) - int(before_distribution.get(str(size), 0))
        for size, count in after_distribution.items()
        if int(count) - int(before_distribution.get(str(size), 0)) > 0
    }


def _max_new_batch_size(
    before: dict[str, object],
    after: dict[str, object],
) -> int:
    distribution = _delta_distribution(before, after)
    return max((int(size) for size in distribution), default=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(
        run_load(args.base_url, args.concurrency, args.requests, args.max_tokens)
    )
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
