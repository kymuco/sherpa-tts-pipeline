from pathlib import Path

from sherpa_tts_pipeline.cli import main


def test_doctor_command_smoke() -> None:
    exit_code = main(["doctor", "--config", str(Path("examples/voice.yaml"))])
    assert exit_code == 0
