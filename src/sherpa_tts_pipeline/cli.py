from __future__ import annotations

import argparse
from typing import Sequence

from sherpa_tts_pipeline.dataset.build import run_dataset_stage
from sherpa_tts_pipeline.dataset.report import run_report_stage
from sherpa_tts_pipeline.dataset.review import run_review_stage
from sherpa_tts_pipeline.doctor import run_doctor_stage
from sherpa_tts_pipeline.export.piper_onnx import run_export_stage
from sherpa_tts_pipeline.infer.sherpa import run_speak_stage
from sherpa_tts_pipeline.prepare.normalize import run_prepare_stage
from sherpa_tts_pipeline.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sherpa-tts",
        description="Simple CLI for building, exporting, and testing sherpa-onnx TTS voices.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Normalize source audio before dataset generation.",
    )
    prepare_parser.add_argument(
        "inputs",
        nargs="+",
        help="Source audio files or directories.",
    )
    prepare_parser.add_argument(
        "--out",
        required=True,
        help="Output directory for normalized audio.",
    )
    prepare_parser.add_argument(
        "--config",
        default=None,
        help="Optional config file with advanced knobs.",
    )
    prepare_parser.add_argument(
        "--mode",
        choices=["normalize-only", "training-ready"],
        default=None,
        help="Preparation mode. normalize-only preserves format by default, training-ready makes mono 22050 Hz WAVs.",
    )
    prepare_parser.add_argument(
        "--target-lufs",
        type=float,
        default=None,
        help="Target integrated loudness in LUFS.",
    )
    prepare_parser.add_argument(
        "--lra",
        type=float,
        default=None,
        help="Target loudness range.",
    )
    prepare_parser.add_argument(
        "--true-peak",
        type=float,
        default=None,
        help="Maximum true peak in dBTP.",
    )
    prepare_parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Optional output sample rate override. Leave unset to keep the original rate in normalize-only mode.",
    )
    prepare_parser.add_argument(
        "--mono",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write mono output. Use --no-mono to keep the original channel count.",
    )
    prepare_parser.add_argument(
        "--codec",
        default=None,
        help="Optional ffmpeg audio codec override, for example pcm_s16le or pcm_s24le.",
    )
    prepare_parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overwrite existing prepared files.",
    )
    prepare_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and config without running ffmpeg.",
    )
    prepare_parser.set_defaults(handler=run_prepare_stage)

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Build a TTS dataset from one or more source audio files or directories.",
    )
    dataset_parser.add_argument(
        "inputs",
        nargs="+",
        help="Source audio files or directories.",
    )
    dataset_parser.add_argument(
        "--out",
        required=True,
        help="Output dataset directory.",
    )
    dataset_parser.add_argument(
        "--config",
        default=None,
        help="Optional config file with advanced knobs.",
    )
    dataset_parser.add_argument(
        "--language",
        default=None,
        help="Optional language code. Leave unset for auto-detection.",
    )
    dataset_parser.add_argument(
        "--whisper-model",
        default=None,
        help="Optional faster-whisper model name or local path override.",
    )
    dataset_parser.add_argument(
        "--append",
        action="store_true",
        help="Append clips to an existing dataset directory.",
    )
    dataset_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild the dataset directory from scratch.",
    )
    dataset_parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate clips when appending to an existing dataset.",
    )
    dataset_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and config without running Whisper or writing files.",
    )
    dataset_parser.set_defaults(handler=run_dataset_stage)

    review_parser = subparsers.add_parser(
        "review",
        help="Prepare a manual review queue and optional rescue previews from an existing dataset.",
    )
    review_parser.add_argument(
        "dataset_dir",
        help="Existing dataset directory that contains clips.csv and rejected.csv.",
    )
    review_parser.add_argument(
        "--out",
        default=None,
        help="Output directory for review manifests. Defaults to <dataset_dir>/review.",
    )
    review_parser.add_argument(
        "--subset",
        choices=["all", "kept", "rejected", "rescue"],
        default="all",
        help="Choose which entries to include in the review queue.",
    )
    review_parser.add_argument(
        "--extract-rejected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract preview WAVs for rejected rows so they can be reviewed directly.",
    )
    review_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing review outputs.",
    )
    review_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and show the planned review queue without writing files.",
    )
    review_parser.set_defaults(handler=run_review_stage)

    report_parser = subparsers.add_parser(
        "report",
        help="Generate a human-readable summary report for an existing dataset.",
    )
    report_parser.add_argument(
        "dataset_dir",
        help="Existing dataset directory.",
    )
    report_parser.add_argument(
        "--out",
        default=None,
        help="Optional output directory for report files. Defaults to the dataset directory.",
    )
    report_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show summary counts without writing report files.",
    )
    report_parser.set_defaults(handler=run_report_stage)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check the local environment and optional project paths.",
    )
    doctor_parser.add_argument(
        "--config",
        default=None,
        help="Optional config file to validate.",
    )
    doctor_parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Optional dataset directory to validate.",
    )
    doctor_parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional exported model directory to validate.",
    )
    doctor_parser.add_argument(
        "--piper-src",
        default=None,
        help="Optional piper1-gpl/src directory to validate for export.",
    )
    doctor_parser.set_defaults(handler=run_doctor_stage)

    export_parser = subparsers.add_parser(
        "export",
        help="Export a Piper checkpoint to ONNX for sherpa-onnx workflows.",
    )
    export_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a Piper .ckpt file.",
    )
    export_parser.add_argument(
        "--out",
        required=True,
        help="Output directory for the exported model bundle.",
    )
    export_parser.add_argument(
        "--config",
        default=None,
        help="Optional config file with advanced knobs.",
    )
    export_parser.add_argument(
        "--piper-src",
        default=None,
        help="Optional path to piper1-gpl/src.",
    )
    export_parser.add_argument(
        "--tokens",
        default=None,
        help="Optional tokens.txt to copy into the output bundle.",
    )
    export_parser.add_argument(
        "--voice-config-json",
        default=None,
        help="Optional Piper voice config JSON to turn into tokens.txt automatically.",
    )
    export_parser.add_argument(
        "--espeak-data-dir",
        default=None,
        help="Optional espeak-ng-data directory to copy into the output bundle.",
    )
    export_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and config without exporting the model.",
    )
    export_parser.set_defaults(handler=run_export_stage)

    speak_parser = subparsers.add_parser(
        "speak",
        help="Run local TTS inference from an exported model directory.",
    )
    speak_parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory that contains model.onnx, tokens.txt, and espeak-ng-data.",
    )
    speak_parser.add_argument(
        "--text",
        required=True,
        help="Text to synthesize.",
    )
    speak_parser.add_argument(
        "--config",
        default=None,
        help="Optional config file with advanced knobs.",
    )
    speak_parser.add_argument(
        "--output",
        default=None,
        help="Optional output wav path override.",
    )
    speak_parser.add_argument(
        "--provider",
        default=None,
        help="Optional runtime provider override, for example cpu or cuda.",
    )
    speak_parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Optional CPU thread count override.",
    )
    speak_parser.add_argument(
        "--sid",
        type=int,
        default=None,
        help="Optional speaker id override.",
    )
    speak_parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Optional speech speed override.",
    )
    speak_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and config without generating audio.",
    )
    speak_parser.set_defaults(handler=run_speak_stage)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
