import importlib.util
from pathlib import Path
from types import ModuleType


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "download_vllm_smoke_assets.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_vllm_smoke_assets", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_local_dir_uses_model_repo_suffix(tmp_path: Path) -> None:
    script = _load_script()

    assert script.default_local_dir("Qwen/Qwen2.5-1.5B-Instruct", tmp_path) == (
        tmp_path / "Qwen2.5-1.5B-Instruct"
    )


def test_default_model_matches_saved_vllm_artifact_family() -> None:
    script = _load_script()

    assert script.DEFAULT_MODEL == "Qwen/Qwen2.5-0.5B-Instruct"


def test_missing_required_assets_detects_complete_model_dir(tmp_path: Path) -> None:
    script = _load_script()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_text("", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    assert script.missing_required_assets(tmp_path) == []


def test_runtime_export_lines_include_optional_endpoint(tmp_path: Path) -> None:
    script = _load_script()
    model_dir = tmp_path / "models" / "qwen"
    hf_home = tmp_path / ".hf-cache"

    lines = script.runtime_export_lines(
        model_dir,
        hf_home,
        "https://hf-mirror.com",
    )

    assert lines == [
        f"export HF_HOME={hf_home}",
        "export HF_ENDPOINT=https://hf-mirror.com",
        f"export LLM_RUNTIME_MODEL_PATH={model_dir}",
    ]
