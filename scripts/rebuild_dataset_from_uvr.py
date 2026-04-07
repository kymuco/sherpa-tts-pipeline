from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import unicodedata
import wave
from collections import defaultdict
from pathlib import Path


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def source_key_from_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"^_+", "", stem)
    return normalize_text(stem)


def cleaned_key_from_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"^\d+_+", "", stem)
    stem = re.sub(r"[_\s-]*\((?:Vocals|vocals)\)$", "", stem)
    return normalize_text(stem)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_clean_file_map(clean_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for file_path in sorted(clean_dir.iterdir()):
        if not file_path.is_file():
            continue
        key = cleaned_key_from_name(file_path.name)
        if key in mapping:
            raise RuntimeError(
                f"Duplicate cleaned source match for key {key!r}: "
                f"{mapping[key].name} and {file_path.name}"
            )
        mapping[key] = file_path
    return mapping


def find_ffmpeg(ffmpeg_arg: str | None) -> str:
    if ffmpeg_arg:
        return ffmpeg_arg
    return shutil.which("ffmpeg") or "ffmpeg"


def normalize_clean_sources(
    source_rows: list[dict[str, str]],
    clean_map: dict[str, Path],
    cache_dir: Path,
    ffmpeg_bin: str,
) -> tuple[dict[str, Path], dict[str, Path]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_leaf_to_cleaned: dict[str, Path] = {}
    source_leaf_to_cache: dict[str, Path] = {}

    for index, row in enumerate(source_rows, start=1):
        source_leaf = Path(row["source_file"]).name
        source_key = source_key_from_name(source_leaf)
        cleaned_path = clean_map.get(source_key)
        if cleaned_path is None:
            raise FileNotFoundError(f"No cleaned UVR file found for source: {source_leaf}")

        cache_path = cache_dir / f"{index:02d}__{source_key}.wav"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(cleaned_path),
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(cache_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        source_leaf_to_cleaned[source_leaf] = cleaned_path
        source_leaf_to_cache[source_leaf] = cache_path

    return source_leaf_to_cleaned, source_leaf_to_cache


def rebuild_dataset(
    source_dataset_dir: Path,
    clean_dir: Path,
    output_dataset_dir: Path,
    cache_dir: Path,
    ffmpeg_bin: str,
) -> None:
    metadata_path = source_dataset_dir / "metadata.csv"
    clips_path = source_dataset_dir / "clips.csv"
    rejected_path = source_dataset_dir / "rejected.csv"
    sources_path = source_dataset_dir / "sources.csv"

    if output_dataset_dir.exists():
        raise FileExistsError(f"Output dataset directory already exists: {output_dataset_dir}")

    output_wavs_dir = output_dataset_dir / "wavs"
    output_wavs_dir.mkdir(parents=True, exist_ok=False)

    metadata_text = metadata_path.read_text(encoding="utf-8")
    (output_dataset_dir / "metadata.csv").write_text(metadata_text, encoding="utf-8")
    shutil.copy2(rejected_path, output_dataset_dir / "rejected.csv")

    clips_rows = load_csv_rows(clips_path)
    source_rows = load_csv_rows(sources_path)
    clean_map = build_clean_file_map(clean_dir)
    source_leaf_to_cleaned, source_leaf_to_cache = normalize_clean_sources(
        source_rows=source_rows,
        clean_map=clean_map,
        cache_dir=cache_dir,
        ffmpeg_bin=ffmpeg_bin,
    )

    grouped_clips: dict[str, list[dict[str, str]]] = defaultdict(list)
    updated_clips_rows: list[dict[str, str]] = []
    for row in clips_rows:
        source_leaf = Path(row["source_file"]).name
        updated_row = dict(row)
        updated_row["source_file"] = str(source_leaf_to_cleaned[source_leaf])
        updated_clips_rows.append(updated_row)
        grouped_clips[source_leaf].append(updated_row)

    updated_source_rows: list[dict[str, str]] = []
    for row in source_rows:
        updated_row = dict(row)
        source_leaf = Path(row["source_file"]).name
        updated_row["source_file"] = str(source_leaf_to_cleaned[source_leaf])
        updated_source_rows.append(updated_row)

    for source_leaf, clip_group in grouped_clips.items():
        cache_path = source_leaf_to_cache[source_leaf]
        with wave.open(str(cache_path), "rb") as source_wav:
            sample_rate = source_wav.getframerate()
            channels = source_wav.getnchannels()
            sample_width = source_wav.getsampwidth()
            total_frames = source_wav.getnframes()

            if channels != 1 or sample_width != 2:
                raise RuntimeError(
                    f"Unexpected cached WAV format for {cache_path.name}: "
                    f"channels={channels}, sample_width={sample_width}"
                )

            for row in clip_group:
                clip_id = row["clip_id"].strip()
                clip_output_path = output_dataset_dir / row["clip_path"]
                clip_output_path.parent.mkdir(parents=True, exist_ok=True)

                start_sec = float(row["export_start_sec"])
                end_sec = float(row["export_end_sec"])
                start_frame = max(0, int(round(start_sec * sample_rate)))
                end_frame = min(total_frames, int(round(end_sec * sample_rate)))
                frame_count = max(0, end_frame - start_frame)

                source_wav.setpos(start_frame)
                frames = source_wav.readframes(frame_count)

                with wave.open(str(clip_output_path), "wb") as clip_wav:
                    clip_wav.setnchannels(1)
                    clip_wav.setsampwidth(2)
                    clip_wav.setframerate(sample_rate)
                    clip_wav.writeframes(frames)

                if not clip_output_path.exists():
                    raise FileNotFoundError(f"Expected clip was not written: {clip_id}")

    write_csv_rows(
        output_dataset_dir / "clips.csv",
        fieldnames=list(updated_clips_rows[0].keys()),
        rows=updated_clips_rows,
    )
    write_csv_rows(
        output_dataset_dir / "sources.csv",
        fieldnames=list(updated_source_rows[0].keys()),
        rows=updated_source_rows,
    )
    write_csv_rows(
        output_dataset_dir / "uvr_source_map.csv",
        fieldnames=["source_leaf", "cleaned_source_path", "cache_wav_path"],
        rows=[
            {
                "source_leaf": source_leaf,
                "cleaned_source_path": str(source_leaf_to_cleaned[source_leaf]),
                "cache_wav_path": str(source_leaf_to_cache[source_leaf]),
            }
            for source_leaf in source_leaf_to_cleaned
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild an accepted dataset from UVR-cleaned long sources while "
            "keeping the existing clip ids, texts, and timings."
        )
    )
    parser.add_argument("--source-dataset-dir", required=True)
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--output-dataset-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--ffmpeg", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rebuild_dataset(
        source_dataset_dir=Path(args.source_dataset_dir).expanduser().resolve(),
        clean_dir=Path(args.clean_dir).expanduser().resolve(),
        output_dataset_dir=Path(args.output_dataset_dir).expanduser().resolve(),
        cache_dir=Path(args.cache_dir).expanduser().resolve(),
        ffmpeg_bin=find_ffmpeg(args.ffmpeg),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
