from __future__ import annotations

import csv
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sherpa_tts_pipeline.dataset.report import load_csv_rows, summarize_dataset

LOGGER = logging.getLogger(__name__)

REVIEW_FIELDNAMES = [
    "entry_type",
    "clip_id",
    "audio_path",
    "dataset_clip_path",
    "source_file",
    "speech_start_sec",
    "speech_end_sec",
    "export_start_sec",
    "export_end_sec",
    "duration_sec",
    "text",
    "reason",
    "suggested_action",
    "review_status",
    "review_notes",
]
SALVAGEABLE_REASONS = {
    "too_short",
    "too_long",
    "high_no_speech_prob",
    "low_avg_logprob",
    "low_avg_word_probability",
    "too_few_words",
}


@dataclass
class ReviewOptions:
    dataset_dir: Path
    output_dir: Path
    subset: str = "all"
    extract_rejected: bool = True
    overwrite: bool = False
    dry_run: bool = False


def _suggested_action(entry_type: str, reason: str) -> str:
    if entry_type == "kept":
        return "keep"
    if reason in SALVAGEABLE_REASONS:
        return "rescue"
    return "drop"


def _extract_preview_clip(
    source_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-vn",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _load_review_rows(dataset_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    clips_rows = load_csv_rows(dataset_dir / "clips.csv")
    rejected_rows = load_csv_rows(dataset_dir / "rejected.csv")
    return clips_rows, rejected_rows


def _build_queue_rows(
    dataset_dir: Path,
    output_dir: Path,
    subset: str,
    extract_rejected: bool,
    dry_run: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    clips_rows, rejected_rows = _load_review_rows(dataset_dir)
    queue_rows: list[dict[str, str]] = []
    rescue_rows: list[dict[str, str]] = []
    rejected_preview_dir = output_dir / "rejected_wavs"

    include_kept = subset in {"all", "kept"}
    include_rejected = subset in {"all", "rejected", "rescue"}

    if include_kept:
        for row in clips_rows:
            dataset_clip_path = row.get("clip_path", "")
            audio_path = (dataset_dir / dataset_clip_path).resolve() if dataset_clip_path else None
            queue_rows.append(
                {
                    "entry_type": "kept",
                    "clip_id": row.get("clip_id", ""),
                    "audio_path": str(audio_path) if audio_path else "",
                    "dataset_clip_path": dataset_clip_path,
                    "source_file": row.get("source_file", ""),
                    "speech_start_sec": row.get("speech_start_sec", ""),
                    "speech_end_sec": row.get("speech_end_sec", ""),
                    "export_start_sec": row.get("export_start_sec", ""),
                    "export_end_sec": row.get("export_end_sec", ""),
                    "duration_sec": row.get("duration_sec", ""),
                    "text": row.get("text", ""),
                    "reason": "",
                    "suggested_action": "keep",
                    "review_status": "",
                    "review_notes": "",
                }
            )

    if include_rejected:
        rejected_index = 1
        for row in rejected_rows:
            reason = (row.get("reason") or "").strip()
            suggested_action = _suggested_action("rejected", reason)
            if subset == "rescue" and suggested_action != "rescue":
                continue

            audio_path = ""
            if extract_rejected:
                source_path = Path(row.get("source_file", "")).expanduser().resolve()
                output_path = rejected_preview_dir / f"rejected_{rejected_index:04d}.wav"
                if not dry_run:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    _extract_preview_clip(
                        source_path=source_path,
                        output_path=output_path,
                        start_sec=float(row.get("export_start_sec") or 0.0),
                        end_sec=float(row.get("export_end_sec") or 0.0),
                    )
                audio_path = str(output_path.resolve())
            rejected_index += 1

            queue_row = {
                "entry_type": "rejected",
                "clip_id": "",
                "audio_path": audio_path,
                "dataset_clip_path": "",
                "source_file": row.get("source_file", ""),
                "speech_start_sec": row.get("speech_start_sec", ""),
                "speech_end_sec": row.get("speech_end_sec", ""),
                "export_start_sec": row.get("export_start_sec", ""),
                "export_end_sec": row.get("export_end_sec", ""),
                "duration_sec": row.get("duration_sec", ""),
                "text": row.get("text", ""),
                "reason": reason,
                "suggested_action": suggested_action,
                "review_status": "",
                "review_notes": "",
            }
            queue_rows.append(queue_row)
            if suggested_action == "rescue":
                rescue_rows.append(queue_row.copy())

    return queue_rows, rescue_rows


def _validate_review_options(options: ReviewOptions) -> None:
    if options.subset not in {"all", "kept", "rejected", "rescue"}:
        raise ValueError("subset must be one of: all, kept, rejected, rescue")
    if not options.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {options.dataset_dir}")
    if not (options.dataset_dir / "clips.csv").exists() and not (
        options.dataset_dir / "rejected.csv"
    ).exists():
        raise FileNotFoundError(
            f"{options.dataset_dir} does not look like a dataset directory. Expected clips.csv or rejected.csv."
        )
    if options.extract_rejected and shutil.which("ffmpeg") is None and not options.dry_run:
        raise RuntimeError("ffmpeg was not found in PATH.")
    queue_path = options.output_dir / "review_queue.csv"
    if queue_path.exists() and not options.overwrite:
        raise FileExistsError(
            f"{queue_path} already exists. Use --overwrite to replace review outputs."
        )


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_review_stage(args: Any) -> int:
    options = ReviewOptions(
        dataset_dir=Path(args.dataset_dir).expanduser().resolve(),
        output_dir=(
            Path(args.out).expanduser().resolve()
            if args.out
            else Path(args.dataset_dir).expanduser().resolve() / "review"
        ),
        subset=str(args.subset),
        extract_rejected=bool(args.extract_rejected),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
    _validate_review_options(options)
    summary = summarize_dataset(options.dataset_dir)

    LOGGER.info("Command: review")
    LOGGER.info("Dataset dir: %s", options.dataset_dir)
    LOGGER.info("Output dir: %s", options.output_dir)
    LOGGER.info(
        "Subset: %s | extract_rejected=%s | overwrite=%s",
        options.subset,
        options.extract_rejected,
        options.overwrite,
    )
    LOGGER.info(
        "Dataset snapshot: kept=%s | rejected=%s | sources=%s",
        summary["counts"]["kept_clips"],
        summary["counts"]["rejected_clips"],
        summary["counts"]["source_files"],
    )

    queue_rows, rescue_rows = _build_queue_rows(
        dataset_dir=options.dataset_dir,
        output_dir=options.output_dir,
        subset=options.subset,
        extract_rejected=options.extract_rejected,
        dry_run=options.dry_run,
    )
    LOGGER.info(
        "Review queue: rows=%s | rescue_candidates=%s",
        len(queue_rows),
        len(rescue_rows),
    )

    if options.dry_run:
        LOGGER.info("Dry run complete. No review files were written.")
        return 0

    queue_path = options.output_dir / "review_queue.csv"
    rescue_path = options.output_dir / "rescue_candidates.csv"
    _write_review_csv(queue_path, queue_rows)
    _write_review_csv(rescue_path, rescue_rows)

    LOGGER.info("Review files written:")
    LOGGER.info("  queue=%s", queue_path)
    LOGGER.info("  rescue=%s", rescue_path)
    return 0
