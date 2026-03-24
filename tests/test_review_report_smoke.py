import csv
import wave
from pathlib import Path

from sherpa_tts_pipeline.cli import main


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


def test_review_and_report_commands_smoke(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "demo_voice"
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True)

    kept_wav = wavs_dir / "1.wav"
    source_wav = tmp_path / "raw_audio" / "voice.wav"
    _write_dummy_wav(kept_wav)
    _write_dummy_wav(source_wav)

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
        [
            {
                "clip_id": "",
                "clip_path": "",
                "source_file": str(source_wav),
                "speech_start_sec": "1.000",
                "speech_end_sec": "1.400",
                "export_start_sec": "0.950",
                "export_end_sec": "1.450",
                "duration_sec": "0.500",
                "text": "Hi",
                "avg_logprob": "-0.20",
                "no_speech_prob": "0.20",
                "avg_word_probability": "0.95",
                "min_word_probability": "0.95",
                "word_count": "1",
                "segment_count": "1",
                "reason": "too_short",
            }
        ],
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
        [
            {
                "source_file": str(source_wav),
                "language": "en",
                "language_probability": "1.0000",
                "duration_sec": "2.000",
                "duration_after_vad_sec": "1.500",
                "raw_segments": "2",
                "candidate_chunks": "2",
                "kept_chunks": "1",
                "rejected_chunks": "1",
            }
        ],
    )

    report_exit_code = main(["report", str(dataset_dir)])
    assert report_exit_code == 0
    assert (dataset_dir / "dataset_report.json").exists()
    assert (dataset_dir / "dataset_report.md").exists()

    review_out = tmp_path / "reviews" / "demo_voice"
    review_exit_code = main(
        [
            "review",
            str(dataset_dir),
            "--out",
            str(review_out),
            "--subset",
            "rescue",
            "--no-extract-rejected",
        ]
    )
    assert review_exit_code == 0
    assert (review_out / "review_queue.csv").exists()
    assert (review_out / "rescue_candidates.csv").exists()
