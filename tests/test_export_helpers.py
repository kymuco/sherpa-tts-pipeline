import json
from pathlib import Path

from sherpa_tts_pipeline.export.piper_onnx import (
    _build_sherpa_metadata,
    _write_tokens_from_voice_config_json,
)


def _write_voice_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "audio": {"sample_rate": 22050},
                "espeak": {"voice": "ru"},
                "phoneme_type": "espeak",
                "num_speakers": 1,
                "phoneme_id_map": {
                    "_": [0],
                    "^": [1],
                    "$": [2],
                    "a": [14],
                    "b": [15],
                },
            }
        ),
        encoding="utf-8",
    )


def test_write_tokens_from_voice_config_json(tmp_path: Path) -> None:
    config_path = tmp_path / "voice.json"
    tokens_path = tmp_path / "tokens.txt"
    _write_voice_config(config_path)

    _write_tokens_from_voice_config_json(config_path, tokens_path)

    assert tokens_path.read_text(encoding="utf-8").splitlines() == [
        "_ 0",
        "^ 1",
        "$ 2",
        "a 14",
        "b 15",
    ]


def test_build_sherpa_metadata_from_voice_config() -> None:
    metadata = _build_sherpa_metadata(
        {
            "audio": {"sample_rate": 22050},
            "espeak": {"voice": "ru"},
            "phoneme_type": "espeak",
            "num_speakers": 1,
        }
    )

    assert metadata == {
        "model_type": "vits",
        "comment": "piper",
        "language": "ru",
        "voice": "ru",
        "has_espeak": 1,
        "n_speakers": 1,
        "sample_rate": 22050,
    }
