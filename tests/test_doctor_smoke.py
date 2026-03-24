from pathlib import Path

from sherpa_tts_pipeline.cli import main


def test_doctor_command_smoke(monkeypatch) -> None:
    monkeypatch.setattr(
        "sherpa_tts_pipeline.doctor.shutil.which",
        lambda name: f"/usr/bin/{name}" if name in {"ffmpeg", "ffprobe"} else None,
    )
    exit_code = main(["doctor", "--config", str(Path("examples/voice.yaml"))])
    assert exit_code == 0
