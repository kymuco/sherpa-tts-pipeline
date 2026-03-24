from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from sherpa_tts_pipeline.config import get_nested, load_optional_yaml_config
from sherpa_tts_pipeline.dataset.audio import SUPPORTED_AUDIO_SUFFIXES

LOGGER = logging.getLogger(__name__)

PREPARE_MODES = {"normalize-only", "training-ready"}


@dataclass
class PrepareJob:
    source_path: Path
    relative_output_path: Path

    def resolve_output(self, output_dir: Path) -> Path:
        return output_dir / self.relative_output_path


@dataclass
class PrepareOptions:
    inputs: list[str]
    output_dir: Path
    config_path: Path | None = None
    mode: str = "normalize-only"
    target_lufs: float = -18.0
    loudness_range: float = 7.0
    true_peak: float = -1.5
    sample_rate: int | None = None
    mono: bool | None = None
    output_codec: str | None = None
    overwrite: bool = False
    dry_run: bool = False

    @property
    def resolved_sample_rate(self) -> int | None:
        if self.sample_rate is not None:
            return self.sample_rate
        if self.mode == "training-ready":
            return 22050
        return None

    @property
    def resolved_mono(self) -> bool | None:
        if self.mono is not None:
            return self.mono
        if self.mode == "training-ready":
            return True
        return None

    @property
    def resolved_output_codec(self) -> str:
        if self.output_codec:
            return self.output_codec
        return "pcm_s16le" if self.mode == "training-ready" else "pcm_s24le"


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _collect_audio_files(directory: Path) -> list[Path]:
    files = [
        candidate.resolve()
        for candidate in directory.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    ]
    return sorted(files, key=lambda candidate: str(candidate).lower())


def _unique_relative_path(relative_path: Path, used_paths: set[str]) -> Path:
    candidate = relative_path
    suffix = candidate.suffix
    stem = candidate.stem
    counter = 2

    while str(candidate).lower() in used_paths:
        candidate = candidate.with_name(f"{stem}-{counter}{suffix}")
        counter += 1

    used_paths.add(str(candidate).lower())
    return candidate


def resolve_prepare_jobs(input_values: Sequence[str]) -> list[PrepareJob]:
    jobs: list[PrepareJob] = []
    used_paths: set[str] = set()
    missing_paths: list[str] = []
    empty_directories: list[str] = []
    unsupported_files: list[str] = []

    for value in input_values:
        path = Path(value).expanduser().resolve()

        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
                unsupported_files.append(str(path))
                continue

            relative_path = _unique_relative_path(Path(path.stem).with_suffix(".wav"), used_paths)
            jobs.append(PrepareJob(source_path=path, relative_output_path=relative_path))
            continue

        if path.is_dir():
            audio_files = _collect_audio_files(path)
            if not audio_files:
                empty_directories.append(str(path))
                continue

            for audio_file in audio_files:
                relative_path = audio_file.relative_to(path).with_suffix(".wav")
                relative_path = _unique_relative_path(relative_path, used_paths)
                jobs.append(
                    PrepareJob(
                        source_path=audio_file,
                        relative_output_path=relative_path,
                    )
                )
            continue

        missing_paths.append(str(path))

    errors: list[str] = []
    if missing_paths:
        errors.append("Input paths were not found:\n" + "\n".join(missing_paths))
    if unsupported_files:
        errors.append(
            "Unsupported input files were provided:\n" + "\n".join(unsupported_files)
        )
    if empty_directories:
        errors.append(
            "No supported audio files were found in:\n" + "\n".join(empty_directories)
        )
    if errors:
        raise FileNotFoundError("\n\n".join(errors))
    if not jobs:
        raise FileNotFoundError("No input audio files were found.")

    return jobs


def validate_options(options: PrepareOptions, require_ffmpeg: bool) -> None:
    if options.mode not in PREPARE_MODES:
        raise ValueError(
            f"Unsupported prepare mode: {options.mode}. Use one of: {', '.join(sorted(PREPARE_MODES))}."
        )
    if options.resolved_sample_rate is not None and options.resolved_sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero.")
    if options.loudness_range <= 0:
        raise ValueError("lra must be greater than zero.")
    if require_ffmpeg and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found in PATH.")


def build_ffmpeg_command(
    source_path: Path,
    output_path: Path,
    options: PrepareOptions,
) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-af",
        (
            "loudnorm="
            f"I={options.target_lufs}:"
            f"LRA={options.loudness_range}:"
            f"TP={options.true_peak}"
        ),
    ]

    if options.resolved_sample_rate is not None:
        command.extend(["-ar", str(options.resolved_sample_rate)])

    if options.resolved_mono:
        command.extend(["-ac", "1"])

    command.extend(
        [
            "-c:a",
            options.resolved_output_codec,
            str(output_path),
        ]
    )
    return command


def prepare_audio(options: PrepareOptions) -> tuple[int, int]:
    validate_options(options, require_ffmpeg=True)
    jobs = resolve_prepare_jobs(options.inputs)

    converted_count = 0
    skipped_count = 0

    for job in jobs:
        output_path = job.resolve_output(options.output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not options.overwrite:
            LOGGER.info("[SKIP] %s -> %s (already exists)", job.source_path.name, output_path)
            skipped_count += 1
            continue

        command = build_ffmpeg_command(job.source_path, output_path, options)
        subprocess.run(command, check=True)
        LOGGER.info("[OK] %s -> %s", job.source_path.name, output_path)
        converted_count += 1

    return converted_count, skipped_count


def _build_options(args: Any, config: dict[str, Any], config_path: Path | None) -> PrepareOptions:
    mode = str(
        args.mode
        if args.mode is not None
        else get_nested(config, "prepare", "mode", default="normalize-only")
    ).strip()

    mono = (
        args.mono
        if args.mono is not None
        else _optional_bool(get_nested(config, "prepare", "audio", "mono", default=None))
    )
    overwrite = (
        args.overwrite
        if args.overwrite is not None
        else bool(get_nested(config, "prepare", "output", "overwrite", default=False))
    )

    return PrepareOptions(
        inputs=list(args.inputs),
        output_dir=Path(args.out).expanduser().resolve(),
        config_path=config_path,
        mode=mode,
        target_lufs=float(
            args.target_lufs
            if args.target_lufs is not None
            else get_nested(config, "prepare", "loudness", "target_lufs", default=-18.0)
        ),
        loudness_range=float(
            args.lra
            if args.lra is not None
            else get_nested(config, "prepare", "loudness", "loudness_range", default=7.0)
        ),
        true_peak=float(
            args.true_peak
            if args.true_peak is not None
            else get_nested(config, "prepare", "loudness", "true_peak", default=-1.5)
        ),
        sample_rate=_optional_int(
            args.sample_rate
            if args.sample_rate is not None
            else get_nested(config, "prepare", "audio", "sample_rate", default=None)
        ),
        mono=mono,
        output_codec=(
            str(args.codec)
            if args.codec is not None
            else get_nested(config, "prepare", "audio", "codec", default=None)
        ),
        overwrite=bool(overwrite),
        dry_run=bool(args.dry_run),
    )


def _format_audio_setting(value: Any, fallback_label: str = "keep-original") -> str:
    if value is None:
        return fallback_label
    return str(value)


def _log_plan(options: PrepareOptions, jobs: Sequence[PrepareJob]) -> None:
    preview_count = min(5, len(jobs))

    LOGGER.info("Command: prepare")
    LOGGER.info("Output dir: %s", options.output_dir)
    LOGGER.info("Mode: %s", options.mode)
    LOGGER.info(
        "Loudness: target_lufs=%.1f | lra=%.1f | true_peak=%.1f",
        options.target_lufs,
        options.loudness_range,
        options.true_peak,
    )
    LOGGER.info(
        "Audio: sample_rate=%s | mono=%s | codec=%s | overwrite=%s",
        _format_audio_setting(options.resolved_sample_rate),
        _format_audio_setting(options.resolved_mono),
        options.resolved_output_codec,
        options.overwrite,
    )
    LOGGER.info("Resolved inputs: %s", len(jobs))
    for index, job in enumerate(jobs[:preview_count], start=1):
        LOGGER.info(
            "  %s. %s -> %s",
            index,
            job.source_path,
            job.resolve_output(options.output_dir),
        )
    if len(jobs) > preview_count:
        LOGGER.info("  ... and %s more", len(jobs) - preview_count)
    if options.config_path is not None:
        LOGGER.info("Config: %s", options.config_path)


def run_prepare_stage(args: Any) -> int:
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    config = load_optional_yaml_config(config_path)
    options = _build_options(args, config, config_path)
    jobs = resolve_prepare_jobs(options.inputs)

    _log_plan(options, jobs)

    if options.dry_run:
        validate_options(options, require_ffmpeg=False)
        LOGGER.info("Dry run complete. No files were written and ffmpeg was not started.")
        return 0

    converted_count, skipped_count = prepare_audio(options)
    LOGGER.info(
        "Prepare finished. converted=%s | skipped=%s | output_dir=%s",
        converted_count,
        skipped_count,
        options.output_dir,
    )
    return 0
