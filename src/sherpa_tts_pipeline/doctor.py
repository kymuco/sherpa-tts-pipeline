from __future__ import annotations

import importlib
import importlib.metadata
import logging
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sherpa_tts_pipeline.config import load_yaml_config

LOGGER = logging.getLogger(__name__)


@dataclass
class DoctorCheck:
    name: str
    status: str
    details: str


def _module_version(module_name: str, fallback: str = "unknown") -> str:
    try:
        return importlib.metadata.version(module_name)
    except Exception:
        return fallback


def _import_check(module_name: str, package_name: str | None = None) -> DoctorCheck:
    package = package_name or module_name
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return DoctorCheck(package, "FAIL", str(exc))
    return DoctorCheck(package, "OK", f"version={_module_version(package)}")


def _binary_check(binary_name: str, required: bool = True) -> DoctorCheck:
    location = shutil.which(binary_name)
    if location:
        return DoctorCheck(binary_name, "OK", location)
    return DoctorCheck(binary_name, "FAIL" if required else "WARN", "not found in PATH")


def _python_check() -> DoctorCheck:
    return DoctorCheck(
        "python",
        "OK",
        f"{platform.python_version()} | {platform.platform()}",
    )


def _torch_cuda_check() -> DoctorCheck:
    try:
        import torch
    except Exception as exc:
        return DoctorCheck("torch.cuda", "WARN", f"torch unavailable: {exc}")

    return DoctorCheck(
        "torch.cuda",
        "OK" if torch.cuda.is_available() else "WARN",
        "CUDA available" if torch.cuda.is_available() else "CUDA not available",
    )


def _ctranslate2_cuda_check() -> DoctorCheck:
    try:
        import ctranslate2
    except Exception as exc:
        return DoctorCheck("ctranslate2.cuda", "WARN", f"ctranslate2 unavailable: {exc}")

    try:
        device_count = ctranslate2.get_cuda_device_count()
    except Exception as exc:
        return DoctorCheck("ctranslate2.cuda", "WARN", f"could not query CUDA: {exc}")

    return DoctorCheck(
        "ctranslate2.cuda",
        "OK" if device_count > 0 else "WARN",
        f"cuda_devices={device_count}",
    )


def _config_check(config_path: str | None) -> DoctorCheck | None:
    if not config_path:
        return None
    path = Path(config_path).expanduser().resolve()
    try:
        load_yaml_config(path)
    except Exception as exc:
        return DoctorCheck("config", "FAIL", f"{path} | {exc}")
    return DoctorCheck("config", "OK", str(path))


def _dataset_dir_check(dataset_dir: str | None) -> DoctorCheck | None:
    if not dataset_dir:
        return None
    path = Path(dataset_dir).expanduser().resolve()
    if not path.is_dir():
        return DoctorCheck("dataset_dir", "FAIL", f"not found: {path}")

    expected = ["metadata.csv", "clips.csv", "rejected.csv", "sources.csv", "wavs"]
    missing = [name for name in expected if not (path / name).exists()]
    if missing:
        return DoctorCheck("dataset_dir", "WARN", f"{path} | missing: {', '.join(missing)}")
    return DoctorCheck("dataset_dir", "OK", str(path))


def _model_dir_check(model_dir: str | None) -> DoctorCheck | None:
    if not model_dir:
        return None
    path = Path(model_dir).expanduser().resolve()
    if not path.is_dir():
        return DoctorCheck("model_dir", "FAIL", f"not found: {path}")

    expected = ["model.onnx", "tokens.txt", "espeak-ng-data"]
    missing = [name for name in expected if not (path / name).exists()]
    if missing:
        return DoctorCheck("model_dir", "WARN", f"{path} | missing: {', '.join(missing)}")
    return DoctorCheck("model_dir", "OK", str(path))


def _piper_src_check(piper_src: str | None) -> DoctorCheck | None:
    if not piper_src:
        return None
    path = Path(piper_src).expanduser().resolve()
    expected_file = path / "piper" / "train" / "vits" / "lightning.py"
    if not path.is_dir():
        return DoctorCheck("piper_src", "FAIL", f"not found: {path}")
    if not expected_file.is_file():
        return DoctorCheck(
            "piper_src",
            "WARN",
            f"{path} | expected training module not found: {expected_file}",
        )
    return DoctorCheck("piper_src", "OK", str(path))


def _log_check(check: DoctorCheck) -> None:
    LOGGER.info("[%s] %s | %s", check.status, check.name, check.details)


def run_doctor_stage(args: Any) -> int:
    checks: list[DoctorCheck] = [
        _python_check(),
        _binary_check("ffmpeg"),
        _binary_check("ffprobe", required=False),
        _import_check("yaml", "PyYAML"),
        _import_check("ctranslate2"),
        _import_check("faster_whisper", "faster-whisper"),
        _import_check("soundfile"),
        _import_check("torch"),
        _import_check("sherpa_onnx", "sherpa-onnx-bin"),
        _torch_cuda_check(),
        _ctranslate2_cuda_check(),
    ]

    optional_checks = [
        _config_check(args.config),
        _dataset_dir_check(args.dataset_dir),
        _model_dir_check(args.model_dir),
        _piper_src_check(args.piper_src),
    ]
    checks.extend(check for check in optional_checks if check is not None)

    LOGGER.info("Command: doctor")
    for check in checks:
        _log_check(check)

    fail_count = sum(1 for check in checks if check.status == "FAIL")
    warn_count = sum(1 for check in checks if check.status == "WARN")
    LOGGER.info("Doctor summary: fail=%s | warn=%s | ok=%s", fail_count, warn_count, len(checks) - fail_count - warn_count)
    return 1 if fail_count else 0
