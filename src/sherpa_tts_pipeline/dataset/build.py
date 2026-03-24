from __future__ import annotations

import csv
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from sherpa_tts_pipeline.config import get_nested, load_optional_yaml_config
from sherpa_tts_pipeline.dataset.audio import SUPPORTED_AUDIO_SUFFIXES, looks_like_audio

LOGGER = logging.getLogger(__name__)

TERMINAL_PUNCTUATION = (".", "!", "?", "...")
BREAK_PUNCTUATION = TERMINAL_PUNCTUATION + (";", ":")
WHITESPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
CLIPS_FIELDNAMES = [
    "clip_id",
    "clip_path",
    "source_file",
    "speech_start_sec",
    "speech_end_sec",
    "export_start_sec",
    "export_end_sec",
    "duration_sec",
    "text",
    "avg_logprob",
    "no_speech_prob",
    "avg_word_probability",
    "min_word_probability",
    "word_count",
    "segment_count",
    "reason",
]
SOURCES_FIELDNAMES = [
    "source_file",
    "language",
    "language_probability",
    "duration_sec",
    "duration_after_vad_sec",
    "raw_segments",
    "candidate_chunks",
    "kept_chunks",
    "rejected_chunks",
]


@dataclass
class ChunkCandidate:
    source_path: Path
    speech_start: float
    speech_end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    avg_word_probability: float
    min_word_probability: float
    word_count: int
    segment_count: int = 1
    export_start: float = 0.0
    export_end: float = 0.0

    @property
    def speech_duration(self) -> float:
        return max(0.0, self.speech_end - self.speech_start)

    @property
    def export_duration(self) -> float:
        return max(0.0, self.export_end - self.export_start)

    def merged_with(self, other: "ChunkCandidate") -> "ChunkCandidate":
        total_speech = self.speech_duration + other.speech_duration
        total_words = self.word_count + other.word_count
        avg_logprob = weighted_average(
            (self.avg_logprob, self.speech_duration),
            (other.avg_logprob, other.speech_duration),
        )
        avg_word_probability = weighted_average(
            (self.avg_word_probability, self.word_count),
            (other.avg_word_probability, other.word_count),
        )

        return ChunkCandidate(
            source_path=self.source_path,
            speech_start=self.speech_start,
            speech_end=other.speech_end,
            text=normalize_text(f"{self.text} {other.text}"),
            avg_logprob=(
                avg_logprob
                if total_speech > 0
                else min(self.avg_logprob, other.avg_logprob)
            ),
            no_speech_prob=max(self.no_speech_prob, other.no_speech_prob),
            avg_word_probability=(
                avg_word_probability
                if total_words > 0
                else min(self.avg_word_probability, other.avg_word_probability)
            ),
            min_word_probability=min(self.min_word_probability, other.min_word_probability),
            word_count=total_words,
            segment_count=self.segment_count + other.segment_count,
        )


@dataclass
class DatasetOptions:
    inputs: list[Path]
    output_dir: Path
    config_path: Path | None = None
    append: bool = False
    overwrite: bool = False
    dry_run: bool = False
    language: str | None = None
    whisper_model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 8
    best_of: int = 8
    condition_on_previous_text: bool = False
    sample_rate: int = 22050
    min_duration: float = 2.0
    max_duration: float = 12.0
    start_pad: float = 0.18
    end_pad: float = 0.45
    merge_gap: float = 0.35
    split_gap: float = 0.60
    min_words: int = 1
    min_avg_logprob: float = -0.90
    max_no_speech_prob: float = 0.60
    min_word_prob: float = 0.55
    no_vad: bool = False
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 250


@dataclass
class DatasetBuildResult:
    output_dir: Path
    metadata_path: Path
    clips_path: Path
    rejected_path: Path
    sources_path: Path
    kept_count: int
    rejected_count: int


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_language(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "auto":
        return None
    return normalized


def _collect_directory_inputs(path: Path) -> list[Path]:
    candidates = [
        child.resolve()
        for child in path.rglob("*")
        if child.is_file() and child.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    ]
    return sorted(candidates, key=lambda candidate: str(candidate).lower())


def resolve_inputs(input_values: Sequence[str]) -> list[Path]:
    seen: set[Path] = set()
    resolved_paths: list[Path] = []
    missing_paths: list[str] = []
    empty_directories: list[str] = []

    for value in input_values:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            if path not in seen:
                seen.add(path)
                resolved_paths.append(path)
            continue

        if path.is_dir():
            audio_files = _collect_directory_inputs(path)
            if not audio_files:
                empty_directories.append(str(path))
                continue
            for audio_path in audio_files:
                if audio_path not in seen:
                    seen.add(audio_path)
                    resolved_paths.append(audio_path)
            continue

        missing_paths.append(str(path))

    errors: list[str] = []
    if missing_paths:
        errors.append("Input paths were not found:\n" + "\n".join(missing_paths))
    if empty_directories:
        errors.append(
            "No supported audio files were found in:\n" + "\n".join(empty_directories)
        )
    if errors:
        raise FileNotFoundError("\n\n".join(errors))
    if not resolved_paths:
        raise FileNotFoundError("No input audio files were found.")

    return resolved_paths


def validate_output_dir_state(output_dir: Path, append: bool, overwrite: bool) -> None:
    wavs_dir = output_dir / "wavs"
    metadata_path = output_dir / "metadata.csv"

    if output_dir.exists() and not append and not overwrite:
        has_existing_data = metadata_path.exists() or any(wavs_dir.glob("*.wav"))
        if has_existing_data:
            raise FileExistsError(
                f"{output_dir} already contains dataset files. Use --append or --overwrite."
            )


def validate_options(options: DatasetOptions, require_ffmpeg: bool) -> None:
    if options.append and options.overwrite:
        raise ValueError("--append and --overwrite cannot be used together.")
    if options.min_duration <= 0 or options.max_duration <= 0:
        raise ValueError("Durations must be greater than zero.")
    if options.min_duration >= options.max_duration:
        raise ValueError("--min-duration must be smaller than --max-duration.")
    if options.start_pad < 0 or options.end_pad < 0:
        raise ValueError("Padding values cannot be negative.")
    if options.merge_gap < 0 or options.split_gap < 0:
        raise ValueError("Gap values cannot be negative.")
    if options.sample_rate <= 0:
        raise ValueError("--sample-rate must be greater than zero.")
    if options.beam_size <= 0 or options.best_of <= 0:
        raise ValueError("Whisper beam_size and best_of must be greater than zero.")
    if options.min_words <= 0:
        raise ValueError("--min-words must be greater than zero.")
    validate_output_dir_state(
        output_dir=options.output_dir,
        append=options.append,
        overwrite=options.overwrite,
    )
    if require_ffmpeg and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found in PATH.")


def load_whisper_runtime() -> tuple[Any, Any, Any]:
    try:
        import ctranslate2
    except ImportError as exc:
        raise RuntimeError(
            "ctranslate2 is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    try:
        from faster_whisper import WhisperModel
        from faster_whisper.vad import VadOptions
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    return ctranslate2, WhisperModel, VadOptions


def resolve_device_and_compute_type(
    device: str,
    compute_type: str,
    ctranslate2_module: Any | None = None,
) -> tuple[str, str]:
    resolved_device = device
    resolved_compute_type = compute_type

    if resolved_device == "auto":
        try:
            if ctranslate2_module is not None and ctranslate2_module.get_cuda_device_count() > 0:
                resolved_device = "cuda"
            else:
                resolved_device = "cpu"
        except Exception:
            resolved_device = "cpu"

    if resolved_compute_type == "auto":
        resolved_compute_type = "float16" if resolved_device == "cuda" else "int8"

    return resolved_device, resolved_compute_type


def prepare_output_dir(
    output_dir: Path,
    append: bool,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path, Path]:
    wavs_dir = output_dir / "wavs"
    metadata_path = output_dir / "metadata.csv"
    clips_path = output_dir / "clips.csv"
    rejected_path = output_dir / "rejected.csv"
    sources_path = output_dir / "sources.csv"

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    validate_output_dir_state(output_dir=output_dir, append=append, overwrite=overwrite)

    wavs_dir.mkdir(parents=True, exist_ok=True)
    return wavs_dir, metadata_path, clips_path, rejected_path, sources_path


def next_clip_index(metadata_path: Path, wavs_dir: Path) -> int:
    numeric_ids: list[int] = []

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            for line in metadata_file:
                clip_id = line.split("|", 1)[0].strip()
                if clip_id.isdigit():
                    numeric_ids.append(int(clip_id))

    for wav_file in wavs_dir.glob("*.wav"):
        if wav_file.stem.isdigit():
            numeric_ids.append(int(wav_file.stem))

    if numeric_ids:
        return max(numeric_ids) + 1

    existing_wavs = list(wavs_dir.glob("*.wav"))
    return len(existing_wavs) + 1


def open_csv_writer(path: Path, fieldnames: Sequence[str]) -> tuple[csv.DictWriter, Any]:
    file_exists = path.exists()
    file_handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    return writer, file_handle


def normalize_text(text: str) -> str:
    cleaned = text.replace("|", " ").replace("\n", " ").replace("\r", " ")
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    cleaned = SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned


def weighted_average(*pairs: tuple[float, float]) -> float:
    total_weight = sum(weight for _, weight in pairs if weight > 0)
    if total_weight <= 0:
        return pairs[0][0] if pairs else 0.0
    return sum(value * weight for value, weight in pairs if weight > 0) / total_weight


def word_text(words: Sequence[Any], fallback_text: str) -> str:
    joined = normalize_text("".join(word.word for word in words))
    return joined or normalize_text(fallback_text)


def make_chunk_from_words(source_path: Path, words: Sequence[Any], segment: Any) -> ChunkCandidate:
    text = word_text(words, segment.text)
    probabilities = [word.probability for word in words]
    avg_word_probability = sum(probabilities) / len(probabilities) if probabilities else 1.0
    min_word_probability = min(probabilities) if probabilities else 1.0

    return ChunkCandidate(
        source_path=source_path,
        speech_start=words[0].start if words else segment.start,
        speech_end=words[-1].end if words else segment.end,
        text=text,
        avg_logprob=segment.avg_logprob,
        no_speech_prob=segment.no_speech_prob,
        avg_word_probability=avg_word_probability,
        min_word_probability=min_word_probability,
        word_count=len(words) if words else max(1, len(text.split())),
    )


def should_mark_break(word_text_value: str, gap_after: float, split_gap: float) -> bool:
    stripped = word_text_value.strip()
    return stripped.endswith(BREAK_PUNCTUATION) or gap_after >= split_gap


def split_long_segment(
    source_path: Path,
    segment: Any,
    max_speech_duration: float,
    split_gap: float,
) -> list[ChunkCandidate]:
    words = list(segment.words or [])
    if not words:
        return [make_chunk_from_words(source_path, [], segment)]

    if (words[-1].end - words[0].start) <= max_speech_duration:
        return [make_chunk_from_words(source_path, words, segment)]

    ranges: list[tuple[int, int]] = []
    start_index = 0
    index = 0
    last_break_index: int | None = None

    while index < len(words):
        current_duration = words[index].end - words[start_index].start
        gap_after = (
            words[index + 1].start - words[index].end
            if index + 1 < len(words)
            else float("inf")
        )
        if should_mark_break(words[index].word, gap_after, split_gap):
            last_break_index = index + 1

        if current_duration >= max_speech_duration:
            split_at = (
                last_break_index
                if last_break_index and last_break_index > start_index
                else index + 1
            )
            ranges.append((start_index, split_at))
            start_index = split_at
            index = start_index
            last_break_index = None
            continue

        index += 1

    if start_index < len(words):
        ranges.append((start_index, len(words)))

    return [
        make_chunk_from_words(source_path, words[start:end], segment)
        for start, end in ranges
        if end > start
    ]


def text_ends_cleanly(text: str) -> bool:
    stripped = text.rstrip()
    return stripped.endswith(TERMINAL_PUNCTUATION)


def text_starts_like_continuation(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    first_char = stripped[0]
    return first_char.isalpha() and first_char.islower()


def merge_adjacent_chunks(
    chunks: Sequence[ChunkCandidate],
    min_speech_duration: float,
    max_speech_duration: float,
    merge_gap: float,
) -> list[ChunkCandidate]:
    if not chunks:
        return []

    merged: list[ChunkCandidate] = [chunks[0]]

    for next_chunk in chunks[1:]:
        current = merged[-1]
        gap = max(0.0, next_chunk.speech_start - current.speech_end)
        combined_duration = next_chunk.speech_end - current.speech_start
        current_needs_merge = current.speech_duration < min_speech_duration
        next_needs_merge = next_chunk.speech_duration < min_speech_duration
        likely_same_sentence = (
            not text_ends_cleanly(current.text)
            or text_starts_like_continuation(next_chunk.text)
        )

        if gap <= merge_gap and combined_duration <= max_speech_duration and (
            current_needs_merge or next_needs_merge or likely_same_sentence
        ):
            merged[-1] = current.merged_with(next_chunk)
        else:
            merged.append(next_chunk)

    while len(merged) >= 2:
        tail = merged[-1]
        previous = merged[-2]
        gap = max(0.0, tail.speech_start - previous.speech_end)
        combined_duration = tail.speech_end - previous.speech_start
        if (
            tail.speech_duration < min_speech_duration
            and gap <= merge_gap
            and combined_duration <= max_speech_duration
        ):
            merged[-2] = previous.merged_with(tail)
            merged.pop()
        else:
            break

    return merged


def apply_padding(
    chunks: Sequence[ChunkCandidate],
    audio_duration: float,
    start_pad: float,
    end_pad: float,
) -> list[ChunkCandidate]:
    padded_chunks: list[ChunkCandidate] = []

    for index, chunk in enumerate(chunks):
        previous_end = chunks[index - 1].speech_end if index > 0 else 0.0
        next_start = (
            chunks[index + 1].speech_start if index + 1 < len(chunks) else audio_duration
        )

        start_room = max(0.0, chunk.speech_start - previous_end)
        end_room = max(0.0, next_start - chunk.speech_end)

        usable_start_pad = min(start_pad, start_room * 0.90 if index > 0 else start_pad)
        usable_end_pad = min(end_pad, end_room * 0.90 if index + 1 < len(chunks) else end_pad)

        chunk.export_start = max(0.0, chunk.speech_start - usable_start_pad)
        chunk.export_end = min(audio_duration, chunk.speech_end + usable_end_pad)
        padded_chunks.append(chunk)

    return padded_chunks


def rejection_reason(
    chunk: ChunkCandidate,
    min_duration: float,
    max_duration: float,
    min_words: int,
    min_avg_logprob: float,
    max_no_speech_prob: float,
    min_word_prob: float,
) -> str | None:
    if not chunk.text:
        return "empty_text"
    if chunk.export_duration < min_duration:
        return "too_short"
    if chunk.export_duration > max_duration:
        return "too_long"
    if chunk.word_count < min_words:
        return "too_few_words"
    if chunk.avg_logprob < min_avg_logprob:
        return "low_avg_logprob"
    if chunk.no_speech_prob > max_no_speech_prob:
        return "high_no_speech_prob"
    if chunk.avg_word_probability < min_word_prob:
        return "low_avg_word_probability"
    return None


def transcribe_file(
    model: Any,
    source_path: Path,
    options: DatasetOptions,
    vad_options_cls: Any,
) -> tuple[list[ChunkCandidate], Any, int]:
    vad_options = None
    if not options.no_vad:
        vad_options = vad_options_cls(
            min_silence_duration_ms=options.vad_min_silence_ms,
            speech_pad_ms=options.vad_speech_pad_ms,
        )

    segments_iter, info = model.transcribe(
        str(source_path),
        language=options.language,
        beam_size=options.beam_size,
        best_of=options.best_of,
        word_timestamps=True,
        vad_filter=not options.no_vad,
        vad_parameters=vad_options,
        condition_on_previous_text=options.condition_on_previous_text,
    )
    segments = list(segments_iter)

    max_speech_duration = max(0.5, options.max_duration - options.start_pad - options.end_pad)
    min_speech_duration = max(0.5, options.min_duration - options.start_pad - options.end_pad)

    split_chunks: list[ChunkCandidate] = []
    for segment in segments:
        split_chunks.extend(
            split_long_segment(
                source_path=source_path,
                segment=segment,
                max_speech_duration=max_speech_duration,
                split_gap=options.split_gap,
            )
        )

    merged_chunks = merge_adjacent_chunks(
        chunks=split_chunks,
        min_speech_duration=min_speech_duration,
        max_speech_duration=max_speech_duration,
        merge_gap=options.merge_gap,
    )
    padded_chunks = apply_padding(
        chunks=merged_chunks,
        audio_duration=float(info.duration),
        start_pad=options.start_pad,
        end_pad=options.end_pad,
    )

    return padded_chunks, info, len(segments)


def export_clip(
    source_path: Path,
    clip_path: Path,
    start: float,
    end: float,
    sample_rate: int,
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
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(clip_path),
    ]
    subprocess.run(command, check=True)


def row_from_chunk(
    clip_id: str,
    chunk: ChunkCandidate,
    reason: str,
) -> dict[str, object]:
    relative_clip_path = Path("wavs") / f"{clip_id}.wav" if clip_id else None
    return {
        "clip_id": clip_id,
        "clip_path": str(relative_clip_path).replace("\\", "/") if relative_clip_path else "",
        "source_file": str(chunk.source_path),
        "speech_start_sec": f"{chunk.speech_start:.3f}",
        "speech_end_sec": f"{chunk.speech_end:.3f}",
        "export_start_sec": f"{chunk.export_start:.3f}",
        "export_end_sec": f"{chunk.export_end:.3f}",
        "duration_sec": f"{chunk.export_duration:.3f}",
        "text": chunk.text,
        "avg_logprob": f"{chunk.avg_logprob:.4f}",
        "no_speech_prob": f"{chunk.no_speech_prob:.4f}",
        "avg_word_probability": f"{chunk.avg_word_probability:.4f}",
        "min_word_probability": f"{chunk.min_word_probability:.4f}",
        "word_count": chunk.word_count,
        "segment_count": chunk.segment_count,
        "reason": reason,
    }


def log_chunk(prefix: str, clip_name: str, chunk: ChunkCandidate) -> None:
    LOGGER.info(
        "%s %s | %.2fs | logprob=%.2f | word_prob=%.2f | %s",
        prefix,
        clip_name,
        chunk.export_duration,
        chunk.avg_logprob,
        chunk.avg_word_probability,
        chunk.text,
    )


def build_dataset(options: DatasetOptions) -> DatasetBuildResult:
    validate_options(options, require_ffmpeg=True)

    (
        wavs_dir,
        metadata_path,
        clips_path,
        rejected_path,
        sources_path,
    ) = prepare_output_dir(
        output_dir=options.output_dir,
        append=options.append,
        overwrite=options.overwrite,
    )

    clip_index = next_clip_index(metadata_path, wavs_dir) if options.append else 1
    ctranslate2_module, whisper_model_cls, vad_options_cls = load_whisper_runtime()
    device, compute_type = resolve_device_and_compute_type(
        options.device,
        options.compute_type,
        ctranslate2_module=ctranslate2_module,
    )

    LOGGER.info("Whisper model: %s", options.whisper_model)
    LOGGER.info("Device: %s | compute_type: %s", device, compute_type)

    model = whisper_model_cls(
        options.whisper_model,
        device=device,
        compute_type=compute_type,
    )

    metadata_mode = "a" if metadata_path.exists() and options.append else "w"
    metadata_file = metadata_path.open(metadata_mode, encoding="utf-8", newline="")
    clips_writer, clips_file = open_csv_writer(clips_path, CLIPS_FIELDNAMES)
    rejected_writer, rejected_file = open_csv_writer(rejected_path, CLIPS_FIELDNAMES)
    sources_writer, sources_file = open_csv_writer(sources_path, SOURCES_FIELDNAMES)

    total_kept = 0
    total_rejected = 0

    try:
        for source_number, source_path in enumerate(options.inputs, start=1):
            LOGGER.info(
                "[%s/%s] Transcribing %s",
                source_number,
                len(options.inputs),
                source_path.name,
            )
            chunks, info, raw_segment_count = transcribe_file(
                model=model,
                source_path=source_path,
                options=options,
                vad_options_cls=vad_options_cls,
            )
            LOGGER.info(
                "  language=%s (%.2f) | duration=%.2fs | raw_segments=%s | candidate_chunks=%s",
                info.language,
                info.language_probability,
                float(info.duration),
                raw_segment_count,
                len(chunks),
            )

            kept_for_source = 0
            rejected_for_source = 0

            for chunk in chunks:
                reason = rejection_reason(
                    chunk=chunk,
                    min_duration=options.min_duration,
                    max_duration=options.max_duration,
                    min_words=options.min_words,
                    min_avg_logprob=options.min_avg_logprob,
                    max_no_speech_prob=options.max_no_speech_prob,
                    min_word_prob=options.min_word_prob,
                )

                if reason is not None:
                    rejected_writer.writerow(row_from_chunk("", chunk, reason))
                    log_chunk("[SKIP]", reason, chunk)
                    rejected_for_source += 1
                    continue

                clip_id = str(clip_index)
                clip_index += 1
                clip_path = wavs_dir / f"{clip_id}.wav"

                export_clip(
                    source_path=source_path,
                    clip_path=clip_path,
                    start=chunk.export_start,
                    end=chunk.export_end,
                    sample_rate=options.sample_rate,
                )
                metadata_file.write(f"{clip_id}|{chunk.text}\n")
                clips_writer.writerow(row_from_chunk(clip_id, chunk, ""))
                log_chunk("[KEEP]", f"{clip_id}.wav", chunk)
                kept_for_source += 1

            sources_writer.writerow(
                {
                    "source_file": str(source_path),
                    "language": info.language,
                    "language_probability": f"{info.language_probability:.4f}",
                    "duration_sec": f"{float(info.duration):.3f}",
                    "duration_after_vad_sec": (
                        f"{float(getattr(info, 'duration_after_vad', info.duration)):.3f}"
                    ),
                    "raw_segments": raw_segment_count,
                    "candidate_chunks": len(chunks),
                    "kept_chunks": kept_for_source,
                    "rejected_chunks": rejected_for_source,
                }
            )

            total_kept += kept_for_source
            total_rejected += rejected_for_source
            LOGGER.info(
                "  done: kept=%s | rejected=%s",
                kept_for_source,
                rejected_for_source,
            )
    finally:
        metadata_file.close()
        clips_file.close()
        rejected_file.close()
        sources_file.close()

    LOGGER.info("Dataset build finished.")
    LOGGER.info("  kept=%s | rejected=%s", total_kept, total_rejected)
    LOGGER.info("  metadata=%s", metadata_path)
    LOGGER.info("  clips=%s", clips_path)
    LOGGER.info("  rejected=%s", rejected_path)
    LOGGER.info("  sources=%s", sources_path)

    return DatasetBuildResult(
        output_dir=options.output_dir,
        metadata_path=metadata_path,
        clips_path=clips_path,
        rejected_path=rejected_path,
        sources_path=sources_path,
        kept_count=total_kept,
        rejected_count=total_rejected,
    )


def _build_dataset_options(
    args: Any,
    config: dict[str, Any],
    config_path: Path | None,
) -> DatasetOptions:
    return DatasetOptions(
        inputs=resolve_inputs(args.inputs),
        output_dir=Path(args.out).expanduser().resolve(),
        config_path=config_path,
        append=bool(args.append),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        language=_normalize_language(
            _first_not_none(args.language, get_nested(config, "dataset", "language"))
        ),
        whisper_model=str(
            _first_not_none(
                args.whisper_model,
                get_nested(config, "dataset", "whisper", "model"),
                "large-v3",
            )
        ),
        device=str(get_nested(config, "dataset", "whisper", "device", default="auto")),
        compute_type=str(
            get_nested(config, "dataset", "whisper", "compute_type", default="auto")
        ),
        beam_size=int(get_nested(config, "dataset", "whisper", "beam_size", default=8)),
        best_of=int(get_nested(config, "dataset", "whisper", "best_of", default=8)),
        condition_on_previous_text=bool(
            get_nested(
                config,
                "dataset",
                "whisper",
                "condition_on_previous_text",
                default=False,
            )
        ),
        sample_rate=int(get_nested(config, "dataset", "audio", "sample_rate", default=22050)),
        min_duration=float(get_nested(config, "dataset", "audio", "min_duration", default=2.0)),
        max_duration=float(get_nested(config, "dataset", "audio", "max_duration", default=12.0)),
        start_pad=float(get_nested(config, "dataset", "audio", "start_pad", default=0.18)),
        end_pad=float(get_nested(config, "dataset", "audio", "end_pad", default=0.45)),
        merge_gap=float(get_nested(config, "dataset", "audio", "merge_gap", default=0.35)),
        split_gap=float(get_nested(config, "dataset", "audio", "split_gap", default=0.60)),
        min_words=int(get_nested(config, "dataset", "quality", "min_words", default=1)),
        min_avg_logprob=float(
            get_nested(config, "dataset", "quality", "min_avg_logprob", default=-0.90)
        ),
        max_no_speech_prob=float(
            get_nested(config, "dataset", "quality", "max_no_speech_prob", default=0.60)
        ),
        min_word_prob=float(
            get_nested(config, "dataset", "quality", "min_word_prob", default=0.55)
        ),
        no_vad=bool(get_nested(config, "dataset", "whisper", "no_vad", default=False)),
        vad_min_silence_ms=int(
            get_nested(config, "dataset", "whisper", "vad_min_silence_ms", default=500)
        ),
        vad_speech_pad_ms=int(
            get_nested(config, "dataset", "whisper", "vad_speech_pad_ms", default=250)
        ),
    )


def _log_dataset_plan(options: DatasetOptions) -> None:
    preview_count = min(5, len(options.inputs))

    LOGGER.info("Command: dataset")
    LOGGER.info("Output dataset dir: %s", options.output_dir)
    LOGGER.info("Resolved inputs: %s", len(options.inputs))
    for index, path in enumerate(options.inputs[:preview_count], start=1):
        kind = "audio" if looks_like_audio(path) else "media"
        LOGGER.info("  %s. %s [%s]", index, path, kind)
    if len(options.inputs) > preview_count:
        LOGGER.info("  ... and %s more", len(options.inputs) - preview_count)

    if options.config_path is not None:
        LOGGER.info("Config: %s", options.config_path)

    LOGGER.info(
        "Whisper: model=%s | language=%s | device=%s | compute_type=%s",
        options.whisper_model,
        options.language or "auto",
        options.device,
        options.compute_type,
    )
    LOGGER.info(
        "Audio: sample_rate=%s | min=%.2fs | max=%.2fs",
        options.sample_rate,
        options.min_duration,
        options.max_duration,
    )
    LOGGER.info(
        "Quality: min_words=%s | min_avg_logprob=%.2f | min_word_prob=%.2f",
        options.min_words,
        options.min_avg_logprob,
        options.min_word_prob,
    )


def run_dataset_stage(args: Any) -> int:
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    config = load_optional_yaml_config(config_path)
    options = _build_dataset_options(args, config, config_path)

    _log_dataset_plan(options)

    if options.dry_run:
        validate_options(options, require_ffmpeg=False)
        LOGGER.info("Dry run complete. No files were written and Whisper was not started.")
        return 0

    build_dataset(options)
    return 0
