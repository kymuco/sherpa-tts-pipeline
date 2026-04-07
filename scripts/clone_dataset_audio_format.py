from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_ffmpeg(ffmpeg_arg: str | None) -> str:
    if ffmpeg_arg:
        return ffmpeg_arg
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    return "ffmpeg"


def copy_metadata_files(source_dir: Path, output_dir: Path) -> None:
    for file_path in source_dir.iterdir():
        if file_path.name == "wavs":
            continue
        if file_path.is_file():
            shutil.copy2(file_path, output_dir / file_path.name)


def convert_dataset_audio(
    source_dir: Path,
    output_dir: Path,
    ffmpeg_bin: str,
    sample_rate: int,
    channels: int,
) -> None:
    source_wavs = source_dir / "wavs"
    output_wavs = output_dir / "wavs"

    if not source_wavs.is_dir():
        raise FileNotFoundError(f"wavs directory not found: {source_wavs}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    output_wavs.mkdir(parents=True, exist_ok=False)
    copy_metadata_files(source_dir, output_dir)

    wav_files = sorted(source_wavs.glob("*.wav"), key=lambda path: int(path.stem))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {source_wavs}")

    for wav_path in wav_files:
        output_path = output_wavs / wav_path.name
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(wav_path),
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone a dataset directory and convert its wavs into a new audio format "
            "while keeping metadata files unchanged."
        )
    )
    parser.add_argument("--source-dataset-dir", required=True)
    parser.add_argument("--output-dataset-dir", required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--ffmpeg", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    convert_dataset_audio(
        source_dir=Path(args.source_dataset_dir).expanduser().resolve(),
        output_dir=Path(args.output_dataset_dir).expanduser().resolve(),
        ffmpeg_bin=find_ffmpeg(args.ffmpeg),
        sample_rate=args.sample_rate,
        channels=args.channels,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
