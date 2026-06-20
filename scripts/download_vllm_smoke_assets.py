"""Download the smallest model assets needed for the vLLM GPU smoke test.

The script is intentionally retryable. If a cloud GPU connection drops while
weights are downloading, run the same command again and Hugging Face Hub will
reuse the partial local cache where possible.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from collections.abc import Sequence
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_DIR = Path("models")
DEFAULT_HF_HOME = Path(".hf-cache")

DEFAULT_INCLUDE_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "*.py",
    "*.tiktoken",
)
DEFAULT_EXCLUDE_PATTERNS = (
    "*.bin",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.ot",
    "*.pt",
    "flax_model*",
    "tf_model*",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download model assets for the cloud GPU vLLM smoke test."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model repo to download. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the local model folder will be created.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        help="Exact destination directory. Overrides --output-dir.",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help="Hugging Face cache directory used for resumable downloads.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=os.getenv("HF_ENDPOINT", ""),
        help="Optional mirror endpoint, for example https://hf-mirror.com.",
    )
    parser.add_argument("--revision", help="Optional model revision or commit SHA.")
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--retry-sleep-s", type=float, default=20.0)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Lower values are slower but friendlier to unstable connections.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=list(DEFAULT_INCLUDE_PATTERNS),
        help="Allowed file patterns.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=list(DEFAULT_EXCLUDE_PATTERNS),
        help="Ignored file patterns.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Verify an existing cache/local dir without network access.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved settings without downloading.",
    )
    return parser.parse_args(argv)


def default_local_dir(model: str, output_dir: Path) -> Path:
    return output_dir / model.rstrip("/").split("/")[-1]


def configure_huggingface_env(hf_home: Path, hf_endpoint: str) -> None:
    os.environ["HF_HOME"] = str(hf_home)
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint


def runtime_export_lines(
    model_dir: Path,
    hf_home: Path,
    hf_endpoint: str = "",
) -> list[str]:
    lines = [
        f"export HF_HOME={shlex.quote(str(hf_home))}",
        f"export LLM_RUNTIME_MODEL_PATH={shlex.quote(str(model_dir))}",
    ]
    if hf_endpoint:
        lines.insert(1, f"export HF_ENDPOINT={shlex.quote(hf_endpoint)}")
    return lines


def missing_required_assets(model_dir: Path) -> list[str]:
    missing: list[str] = []
    if not (model_dir / "config.json").is_file():
        missing.append("config.json")
    if not any(model_dir.glob("*.safetensors")):
        missing.append("*.safetensors")
    tokenizer_files = ("tokenizer.json", "tokenizer.model", "vocab.json")
    if not any((model_dir / name).is_file() for name in tokenizer_files):
        missing.append("tokenizer.json/tokenizer.model/vocab.json")
    return missing


def download_with_retries(args: argparse.Namespace, local_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with:\n"
            "  python -m pip install 'huggingface_hub>=0.24'"
        ) from exc

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            print(
                f"[download] attempt {attempt}/{args.retries}: "
                f"{args.model} -> {local_dir}",
                flush=True,
            )
            downloaded = snapshot_download(
                repo_id=args.model,
                revision=args.revision,
                local_dir=str(local_dir),
                allow_patterns=args.include,
                ignore_patterns=args.exclude,
                max_workers=args.max_workers,
                local_files_only=args.local_files_only,
            )
            return Path(downloaded)
        except Exception as exc:  # noqa: BLE001 - surface retryable network errors
            last_error = exc
            print(f"[download] failed: {exc}", file=sys.stderr, flush=True)
            if attempt < args.retries:
                print(
                    f"[download] retrying in {args.retry_sleep_s:.1f}s...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(args.retry_sleep_s)

    raise SystemExit(
        f"Download failed after {args.retries} attempts: {last_error}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    local_dir = (args.local_dir or default_local_dir(args.model, args.output_dir)).resolve()
    hf_home = args.hf_home.resolve()
    configure_huggingface_env(hf_home, args.hf_endpoint)

    print(f"model:      {args.model}")
    print(f"local dir:  {local_dir}")
    print(f"HF_HOME:    {hf_home}")
    if args.hf_endpoint:
        print(f"HF_ENDPOINT:{args.hf_endpoint}")
    print(f"include:    {', '.join(args.include)}")
    print(f"exclude:    {', '.join(args.exclude)}")

    if args.dry_run:
        return 0

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded_dir = download_with_retries(args, local_dir).resolve()
    missing = missing_required_assets(downloaded_dir)
    if missing:
        raise SystemExit(
            "Download finished, but required model assets are missing: "
            + ", ".join(missing)
        )

    print("\nDownloaded model assets successfully.")
    print("\nUse these exports before starting the vLLM runtime:")
    for line in runtime_export_lines(downloaded_dir, hf_home, args.hf_endpoint):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
