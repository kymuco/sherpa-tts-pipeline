from pathlib import Path

from sherpa_tts_pipeline.prepare.normalize import PrepareOptions, build_ffmpeg_command


def test_normalize_only_does_not_force_training_format() -> None:
    options = PrepareOptions(
        inputs=[],
        output_dir=Path("."),
        mode="normalize-only",
    )

    command = build_ffmpeg_command(Path("input.wav"), Path("output.wav"), options)

    assert "-ar" not in command
    assert "-ac" not in command
    assert "pcm_s24le" in command


def test_training_ready_forces_training_format_by_default() -> None:
    options = PrepareOptions(
        inputs=[],
        output_dir=Path("."),
        mode="training-ready",
    )

    command = build_ffmpeg_command(Path("input.wav"), Path("output.wav"), options)

    assert "-ar" in command
    assert "22050" in command
    assert "-ac" in command
    assert "1" in command
    assert "pcm_s16le" in command
