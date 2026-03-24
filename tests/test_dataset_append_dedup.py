import csv
import wave
from pathlib import Path
from types import SimpleNamespace

from sherpa_tts_pipeline.dataset.build import (
    ChunkCandidate,
    DatasetOptions,
    build_dataset,
)


def _write_dummy_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_append_skips_existing_duplicate_chunks(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "demo_voice"
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True)

    source_wav = tmp_path / "raw_audio" / "voice.wav"
    _write_dummy_wav(source_wav)
    _write_dummy_wav(wavs_dir / "1.wav")
    (dataset_dir / "metadata.csv").write_text("1|Hello there\n", encoding="utf-8")

    fieldnames = [
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
    _write_csv(
        dataset_dir / "clips.csv",
        fieldnames,
        [
            {
                "clip_id": "1",
                "clip_path": "wavs/1.wav",
                "source_file": str(source_wav),
                "speech_start_sec": "0.000",
                "speech_end_sec": "1.000",
                "export_start_sec": "0.000",
                "export_end_sec": "1.000",
                "duration_sec": "1.000",
                "text": "Hello there",
                "avg_logprob": "-0.10",
                "no_speech_prob": "0.10",
                "avg_word_probability": "0.90",
                "min_word_probability": "0.80",
                "word_count": "2",
                "segment_count": "1",
                "reason": "",
            }
        ],
    )
    _write_csv(
        dataset_dir / "rejected.csv",
        fieldnames,
        [],
    )
    _write_csv(
        dataset_dir / "sources.csv",
        [
            "source_file",
            "language",
            "language_probability",
            "duration_sec",
            "duration_after_vad_sec",
            "raw_segments",
            "candidate_chunks",
            "kept_chunks",
            "rejected_chunks",
        ],
        [],
    )

    duplicate_chunk = ChunkCandidate(
        source_path=source_wav,
        speech_start=0.0,
        speech_end=1.0,
        text="Hello there",
        avg_logprob=-0.10,
        no_speech_prob=0.10,
        avg_word_probability=0.90,
        min_word_probability=0.80,
        word_count=2,
        export_start=0.0,
        export_end=1.0,
    )
    new_chunk = ChunkCandidate(
        source_path=source_wav,
        speech_start=1.2,
        speech_end=3.8,
        text="Brand new clip",
        avg_logprob=-0.10,
        no_speech_prob=0.10,
        avg_word_probability=0.95,
        min_word_probability=0.90,
        word_count=3,
        export_start=1.2,
        export_end=3.8,
    )

    class DummyWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    class DummyCTranslate2:
        @staticmethod
        def get_cuda_device_count() -> int:
            return 0

    monkeypatch.setattr(
        "sherpa_tts_pipeline.dataset.build.load_whisper_runtime",
        lambda: (DummyCTranslate2, DummyWhisperModel, object),
    )
    monkeypatch.setattr(
        "sherpa_tts_pipeline.dataset.build.transcribe_file",
        lambda **kwargs: (
            [duplicate_chunk, new_chunk],
            SimpleNamespace(duration=3.0, language="en", language_probability=1.0),
            2,
        ),
    )
    monkeypatch.setattr(
        "sherpa_tts_pipeline.dataset.build.export_clip",
        lambda source_path, clip_path, start, end, sample_rate: _write_dummy_wav(clip_path),
    )
    monkeypatch.setattr(
        "sherpa_tts_pipeline.dataset.build.shutil.which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )

    result = build_dataset(
        DatasetOptions(
            inputs=[source_wav],
            output_dir=dataset_dir,
            append=True,
            sample_rate=22050,
        )
    )

    assert result.kept_count == 1
    assert result.duplicate_count == 1
    metadata_lines = (dataset_dir / "metadata.csv").read_text(encoding="utf-8").splitlines()
    assert metadata_lines == ["1|Hello there", "2|Brand new clip"]
