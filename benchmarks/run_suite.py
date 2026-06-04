import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from run_load_test import run_load


async def run_suite(
    base_url: str,
    mode: str,
    concurrency_levels: list[int],
    requests: int,
    max_tokens: int,
) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    for concurrency in concurrency_levels:
        result = await run_load(base_url, concurrency, requests, max_tokens)
        runs.append(result)
        print(
            f"{mode}: concurrency={concurrency} "
            f"tokens/s={result['tokens_per_second']:.2f}"
        )
    return {
        "mode": mode,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "requests_per_level": requests,
        "max_tokens": max_tokens,
        "concurrency_levels": concurrency_levels,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(
        run_suite(
            args.base_url,
            args.mode,
            args.levels,
            args.requests,
            args.max_tokens,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
