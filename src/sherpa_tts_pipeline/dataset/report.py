from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

LOGGER = logging.getLogger(__name__)

REPORT_JSON_NAME = "dataset_report.json"
REPORT_MARKDOWN_NAME = "dataset_report.md"


@dataclass
class DatasetReportResult:
    dataset_dir: Path
    json_path: Path
    markdown_path: Path
    summary: dict[str, Any]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_float(value: str | None) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    return float(text)


def _duration_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "total_sec": 0.0,
            "avg_sec": 0.0,
            "min_sec": 0.0,
            "max_sec": 0.0,
        }

    return {
        "total_sec": round(sum(values), 3),
        "avg_sec": round(mean(values), 3),
        "min_sec": round(min(values), 3),
        "max_sec": round(max(values), 3),
    }


def _count_metadata_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def summarize_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_dir).expanduser().resolve()

    clips_rows = load_csv_rows(dataset_path / "clips.csv")
    rejected_rows = load_csv_rows(dataset_path / "rejected.csv")
    sources_rows = load_csv_rows(dataset_path / "sources.csv")
    metadata_path = dataset_path / "metadata.csv"

    kept_durations = [_parse_float(row.get("duration_sec")) for row in clips_rows]
    rejected_durations = [_parse_float(row.get("duration_sec")) for row in rejected_rows]
    kept_word_counts = [int(row.get("word_count") or 0) for row in clips_rows]
    rejection_counts = Counter(
        (row.get("reason") or "unknown").strip() or "unknown" for row in rejected_rows
    )

    top_problem_sources = sorted(
        [
            {
                "source_file": row.get("source_file", ""),
                "kept_chunks": int(row.get("kept_chunks") or 0),
                "rejected_chunks": int(row.get("rejected_chunks") or 0),
                "candidate_chunks": int(row.get("candidate_chunks") or 0),
                "duration_sec": _parse_float(row.get("duration_sec")),
                "duration_after_vad_sec": _parse_float(row.get("duration_after_vad_sec")),
            }
            for row in sources_rows
        ],
        key=lambda row: (row["rejected_chunks"], -row["kept_chunks"], row["source_file"]),
        reverse=True,
    )[:5]

    summary = {
        "dataset_dir": str(dataset_path),
        "paths": {
            "metadata": str(metadata_path),
            "clips": str(dataset_path / "clips.csv"),
            "rejected": str(dataset_path / "rejected.csv"),
            "sources": str(dataset_path / "sources.csv"),
            "wavs_dir": str(dataset_path / "wavs"),
        },
        "counts": {
            "metadata_rows": _count_metadata_rows(metadata_path),
            "kept_clips": len(clips_rows),
            "rejected_clips": len(rejected_rows),
            "source_files": len(sources_rows),
        },
        "durations": {
            "kept": _duration_stats(kept_durations),
            "rejected": _duration_stats(rejected_durations),
        },
        "quality": {
            "avg_words_per_kept_clip": round(mean(kept_word_counts), 3) if kept_word_counts else 0.0,
        },
        "rejections": {
            "by_reason": dict(rejection_counts),
        },
        "sources": {
            "top_problem_sources": top_problem_sources,
        },
    }
    return summary


def render_dataset_report_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    kept = summary["durations"]["kept"]
    rejected = summary["durations"]["rejected"]
    by_reason = summary["rejections"]["by_reason"]
    top_sources = summary["sources"]["top_problem_sources"]

    lines = [
        "# Dataset Report",
        "",
        f"Dataset: `{summary['dataset_dir']}`",
        "",
        "## Summary",
        "",
        f"- Kept clips: {counts['kept_clips']}",
        f"- Rejected clips: {counts['rejected_clips']}",
        f"- Source files: {counts['source_files']}",
        f"- Kept duration: {kept['total_sec']:.3f}s",
        f"- Rejected duration: {rejected['total_sec']:.3f}s",
        f"- Average kept clip length: {kept['avg_sec']:.3f}s",
        f"- Average words per kept clip: {summary['quality']['avg_words_per_kept_clip']:.3f}",
        "",
        "## Rejections",
        "",
    ]

    if by_reason:
        for reason, count in sorted(by_reason.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- No rejected clips.")

    lines.extend(["", "## Top Problem Sources", ""])

    if top_sources:
        for row in top_sources:
            lines.append(
                "- "
                f"{row['source_file']} | rejected={row['rejected_chunks']} | "
                f"kept={row['kept_chunks']} | candidates={row['candidate_chunks']}"
            )
    else:
        lines.append("- No source rows were found.")

    lines.append("")
    return "\n".join(lines)


def write_dataset_report(
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
) -> DatasetReportResult:
    dataset_path = Path(dataset_dir).expanduser().resolve()
    report_dir = Path(output_dir).expanduser().resolve() if output_dir else dataset_path
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_dataset(dataset_path)
    json_path = report_dir / REPORT_JSON_NAME
    markdown_path = report_dir / REPORT_MARKDOWN_NAME

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_dataset_report_markdown(summary) + "\n",
        encoding="utf-8",
    )

    LOGGER.info("Report written:")
    LOGGER.info("  json=%s", json_path)
    LOGGER.info("  markdown=%s", markdown_path)

    return DatasetReportResult(
        dataset_dir=dataset_path,
        json_path=json_path,
        markdown_path=markdown_path,
        summary=summary,
    )


def run_report_stage(args: Any) -> int:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve() if args.out else dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    summary = summarize_dataset(dataset_dir)

    LOGGER.info("Command: report")
    LOGGER.info("Dataset dir: %s", dataset_dir)
    LOGGER.info("Output dir: %s", output_dir)
    LOGGER.info(
        "Counts: kept=%s | rejected=%s | sources=%s",
        summary["counts"]["kept_clips"],
        summary["counts"]["rejected_clips"],
        summary["counts"]["source_files"],
    )
    LOGGER.info(
        "Durations: kept=%.3fs | rejected=%.3fs",
        summary["durations"]["kept"]["total_sec"],
        summary["durations"]["rejected"]["total_sec"],
    )

    if args.dry_run:
        LOGGER.info("Dry run complete. No report files were written.")
        return 0

    write_dataset_report(dataset_dir=dataset_dir, output_dir=output_dir)
    return 0
