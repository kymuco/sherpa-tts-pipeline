from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sherpa_tts_pipeline.config import get_nested, load_optional_yaml_config

LOGGER = logging.getLogger(__name__)


@dataclass
class ExportOptions:
    checkpoint_path: Path
    output_dir: Path
    piper_src: Path
    config_path: Path | None = None
    voice_config_json: Path | None = None
    tokens_path: Path | None = None
    espeak_data_dir: Path | None = None
    opset_version: int = 15
    dry_run: bool = False

    @property
    def output_model_path(self) -> Path:
        return self.output_dir / "model.onnx"


def _resolve_optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _load_torch_runtime() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is not installed. Run `pip install -r requirements-dev.txt`.") from exc

    return torch


def _load_onnx_runtime() -> Any:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx is not installed. Run `pip install onnx`.") from exc

    return onnx


def _add_piper_src_to_path(piper_src: Path) -> None:
    piper_src_value = str(piper_src)
    if piper_src_value not in sys.path:
        sys.path.insert(0, piper_src_value)


def _validate_options(options: ExportOptions) -> None:
    if not options.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {options.checkpoint_path}")
    if options.checkpoint_path.suffix.lower() != ".ckpt":
        raise ValueError(f"Expected a Piper .ckpt checkpoint: {options.checkpoint_path.name}")
    if not options.piper_src.is_dir():
        raise FileNotFoundError(f"piper1-gpl src directory not found: {options.piper_src}")
    if options.voice_config_json is not None and not options.voice_config_json.is_file():
        raise FileNotFoundError(f"Voice config JSON not found: {options.voice_config_json}")
    if options.tokens_path is not None and not options.tokens_path.is_file():
        raise FileNotFoundError(f"tokens.txt not found: {options.tokens_path}")
    if options.espeak_data_dir is not None and not options.espeak_data_dir.is_dir():
        raise FileNotFoundError(f"espeak-ng-data directory not found: {options.espeak_data_dir}")
    if options.opset_version <= 0:
        raise ValueError("opset_version must be greater than zero.")


def _export_onnx(options: ExportOptions) -> None:
    torch = _load_torch_runtime()
    _add_piper_src_to_path(options.piper_src)

    try:
        from piper.train.vits.lightning import VitsModel
    except ImportError as exc:
        raise RuntimeError(
            "Could not import Piper training code from piper1-gpl/src. "
            "Check --piper-src or export.piper_src in the config."
        ) from exc

    model = VitsModel.load_from_checkpoint(options.checkpoint_path, map_location="cpu")
    model_g = model.model_g
    model_g.eval()

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    def infer_forward(text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )[0].unsqueeze(1)
        return audio

    model_g.forward = infer_forward  # type: ignore[method-assign,assignment]

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers
    dummy_input_length = 50

    sequences = torch.randint(
        low=0,
        high=num_symbols,
        size=(1, dummy_input_length),
        dtype=torch.long,
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    sid = torch.LongTensor([0]) if num_speakers > 1 else None
    dummy_input = (sequences, sequence_lengths, scales, sid)

    options.output_dir.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model=model_g,
        args=dummy_input,
        f=options.output_model_path,
        verbose=False,
        opset_version=options.opset_version,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 2: "time"},
        },
        dynamo=False,
    )


def _copy_optional_assets(options: ExportOptions) -> None:
    if options.tokens_path is not None:
        shutil.copy2(options.tokens_path, options.output_dir / "tokens.txt")
    elif options.voice_config_json is not None:
        _write_tokens_from_voice_config_json(
            voice_config_json=options.voice_config_json,
            output_path=options.output_dir / "tokens.txt",
        )

    if options.espeak_data_dir is not None:
        shutil.copytree(
            options.espeak_data_dir,
            options.output_dir / "espeak-ng-data",
            dirs_exist_ok=True,
        )


def _load_voice_config_json(voice_config_json: Path) -> dict[str, Any]:
    data = json.loads(voice_config_json.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Voice config JSON must contain an object: {voice_config_json}")
    return data


def _build_sherpa_metadata(voice_config: dict[str, Any]) -> dict[str, Any]:
    audio = voice_config.get("audio")
    if not isinstance(audio, dict):
        raise ValueError("Voice config JSON must contain an 'audio' object.")

    sample_rate = audio.get("sample_rate")
    if sample_rate is None:
        raise ValueError("Voice config JSON is missing audio.sample_rate.")

    espeak = voice_config.get("espeak")
    espeak_voice = ""
    if isinstance(espeak, dict):
        espeak_voice = str(espeak.get("voice") or "")

    language = espeak_voice
    language_config = voice_config.get("language")
    if isinstance(language_config, dict):
        language = str(
            language_config.get("name_english")
            or language_config.get("name_native")
            or espeak_voice
        )

    return {
        "model_type": "vits",
        "comment": "piper",
        "language": language,
        "voice": espeak_voice,
        "has_espeak": int(voice_config.get("phoneme_type") == "espeak"),
        "n_speakers": int(voice_config.get("num_speakers", 1)),
        "sample_rate": int(sample_rate),
    }


def _add_sherpa_metadata_to_onnx(model_path: Path, voice_config_json: Path) -> None:
    onnx = _load_onnx_runtime()
    voice_config = _load_voice_config_json(voice_config_json)
    metadata = _build_sherpa_metadata(voice_config)

    model = onnx.load(str(model_path))
    existing = {item.key: item for item in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = str(value)
        else:
            meta = model.metadata_props.add()
            meta.key = key
            meta.value = str(value)

    onnx.save(model, str(model_path))


def _write_tokens_from_voice_config_json(voice_config_json: Path, output_path: Path) -> None:
    voice_config = _load_voice_config_json(voice_config_json)
    phoneme_id_map = voice_config.get("phoneme_id_map")
    if not isinstance(phoneme_id_map, dict) or not phoneme_id_map:
        raise ValueError(
            f"Voice config JSON does not contain a valid phoneme_id_map: {voice_config_json}"
        )

    id_to_symbol: dict[int, str] = {}
    for symbol, value in phoneme_id_map.items():
        if isinstance(value, int):
            phoneme_ids = [value]
        elif isinstance(value, list) and value and all(isinstance(item, int) for item in value):
            phoneme_ids = value
        else:
            raise ValueError(
                f"Unsupported phoneme_id_map entry for {symbol!r} in {voice_config_json}"
            )

        for phoneme_id in phoneme_ids:
            existing = id_to_symbol.get(phoneme_id)
            if existing is not None and existing != symbol:
                raise ValueError(
                    "Conflicting phoneme_id_map entries for "
                    f"id {phoneme_id}: {existing!r} vs {symbol!r}"
                )
            id_to_symbol[phoneme_id] = symbol

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for phoneme_id in sorted(id_to_symbol):
            output_file.write(f"{id_to_symbol[phoneme_id]} {phoneme_id}\n")


def _build_options(args: Any, config: dict[str, Any], config_path: Path | None) -> ExportOptions:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()

    piper_src_value = (
        args.piper_src
        or get_nested(config, "export", "piper_src", default=None)
        or "external/piper1-gpl/src"
    )

    tokens_value = args.tokens or get_nested(config, "export", "tokens", default=None)
    voice_config_value = args.voice_config_json or get_nested(
        config, "export", "voice_config_json", default=None
    )
    espeak_value = args.espeak_data_dir or get_nested(
        config,
        "export",
        "espeak_data_dir",
        default=None,
    )

    return ExportOptions(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        piper_src=Path(piper_src_value).expanduser().resolve(),
        config_path=config_path,
        voice_config_json=_resolve_optional_path(voice_config_value),
        tokens_path=_resolve_optional_path(tokens_value),
        espeak_data_dir=_resolve_optional_path(espeak_value),
        opset_version=int(get_nested(config, "export", "opset_version", default=15)),
        dry_run=bool(args.dry_run),
    )


def _log_plan(options: ExportOptions) -> None:
    LOGGER.info("Command: export")
    LOGGER.info("Checkpoint: %s", options.checkpoint_path)
    LOGGER.info("Output dir: %s", options.output_dir)
    LOGGER.info("Output model: %s", options.output_model_path)
    LOGGER.info("Piper source: %s", options.piper_src)
    LOGGER.info("Opset version: %s", options.opset_version)
    if options.voice_config_json is not None:
        LOGGER.info("Voice config JSON: %s", options.voice_config_json)
    if options.tokens_path is not None:
        LOGGER.info("Tokens: %s", options.tokens_path)
    if options.espeak_data_dir is not None:
        LOGGER.info("espeak-ng-data: %s", options.espeak_data_dir)
    if options.config_path is not None:
        LOGGER.info("Config: %s", options.config_path)


def run_export_stage(args: Any) -> int:
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    config = load_optional_yaml_config(config_path)
    options = _build_options(args, config, config_path)

    _log_plan(options)
    _validate_options(options)

    if options.dry_run:
        LOGGER.info("Dry run complete. No model files were exported.")
        return 0

    _export_onnx(options)
    if options.voice_config_json is not None:
        _add_sherpa_metadata_to_onnx(options.output_model_path, options.voice_config_json)
    _copy_optional_assets(options)

    LOGGER.info("Export finished.")
    LOGGER.info("Model ONNX: %s", options.output_model_path)
    if options.tokens_path is None and options.voice_config_json is None:
        LOGGER.warning(
            "tokens.txt was not copied. Add --tokens or --voice-config-json if you want "
            "a speak-ready bundle."
        )
    if options.voice_config_json is None:
        LOGGER.warning(
            "Sherpa metadata was not added to model.onnx. Add --voice-config-json if you want "
            "a runtime-ready bundle."
        )
    if options.espeak_data_dir is None:
        LOGGER.warning(
            "espeak-ng-data was not copied. Add --espeak-data-dir if you want a speak-ready bundle."
        )
    return 0
