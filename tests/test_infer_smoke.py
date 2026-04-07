from pathlib import Path

from sherpa_tts_pipeline.cli import main


def _prepare_bundle_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.onnx").write_bytes(b"")
    (path / "tokens.txt").write_text("a 1\n", encoding="utf-8")
    (path / "espeak-ng-data").mkdir(exist_ok=True)


def test_speak_command_smoke(tmp_path: Path) -> None:
    model_dir = tmp_path / "release" / "demo_voice"
    _prepare_bundle_dir(model_dir)

    exit_code = main(
        [
            "speak",
            "--model-dir",
            str(model_dir),
            "--text",
            "Smoke test sentence.",
            "--dry-run",
        ]
    )
    assert exit_code == 0


def test_export_command_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "demo.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"")

    piper_src = tmp_path / "external" / "piper1-gpl" / "src"
    piper_src.mkdir(parents=True, exist_ok=True)
    voice_config_json = tmp_path / "release" / "demo_voice.json"
    voice_config_json.parent.mkdir(parents=True, exist_ok=True)
    voice_config_json.write_text(
        '{"audio":{"sample_rate":22050},"espeak":{"voice":"ru"},"phoneme_type":"espeak","num_speakers":1,"phoneme_id_map":{"a":[1]}}',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "export",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(tmp_path / "release" / "demo_voice"),
            "--piper-src",
            str(piper_src),
            "--voice-config-json",
            str(voice_config_json),
            "--dry-run",
        ]
    )
    assert exit_code == 0
