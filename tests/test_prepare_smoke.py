import wave
from pathlib import Path

from sherpa_tts_pipeline.cli import main


def _write_dummy_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00" * 44100 * 2)


def test_prepare_command_smoke(tmp_path: Path) -> None:
    raw_audio_dir = tmp_path / "raw_audio"
    nested_dir = raw_audio_dir / "session_a"
    nested_dir.mkdir(parents=True)
    _write_dummy_wav(nested_dir / "voice.wav")

    exit_code = main(
        [
            "prepare",
            str(raw_audio_dir),
            "--out",
            str(tmp_path / "prepared_audio" / "demo_voice"),
            "--dry-run",
        ]
    )
    assert exit_code == 0
