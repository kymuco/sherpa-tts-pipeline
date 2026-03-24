import wave
from pathlib import Path

from sherpa_tts_pipeline.cli import main
from sherpa_tts_pipeline.config import load_yaml_config


def _write_dummy_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)


def test_dataset_command_smoke(tmp_path: Path) -> None:
    raw_audio_dir = tmp_path / "raw_audio"
    raw_audio_dir.mkdir()
    _write_dummy_wav(raw_audio_dir / "voice.wav")

    exit_code = main(
        [
            "dataset",
            str(raw_audio_dir),
            "--out",
            str(tmp_path / "data" / "demo_voice"),
            "--dry-run",
        ]
    )
    assert exit_code == 0


def test_example_config_loads() -> None:
    config = load_yaml_config(Path("examples/voice.yaml"))
    assert "dataset" in config
    assert "speak" in config
