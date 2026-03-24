from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sherpa_tts_pipeline.config import get_nested, load_optional_yaml_config

LOGGER = logging.getLogger(__name__)


@dataclass
class SpeakOptions:
    model_dir: Path
    text: str
    output_path: Path
    config_path: Path | None = None
    provider: str = "cpu"
    num_threads: int = 4
    sid: int = 0
    speed: float = 1.0
    dry_run: bool = False

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.onnx"

    @property
    def tokens_path(self) -> Path:
        return self.model_dir / "tokens.txt"

    @property
    def data_dir(self) -> Path:
        return self.model_dir / "espeak-ng-data"


def _load_runtime() -> tuple[Any, Any]:
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "sherpa-onnx is not installed. Run `pip install -r requirements-dev.txt`."
        ) from exc

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is not installed. Run `pip install -r requirements-dev.txt`."
        ) from exc

    return sherpa_onnx, sf


def _validate_options(options: SpeakOptions) -> None:
    if not options.model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {options.model_dir}")
    if not options.model_path.is_file():
        raise FileNotFoundError(f"model.onnx not found: {options.model_path}")
    if not options.tokens_path.is_file():
        raise FileNotFoundError(f"tokens.txt not found: {options.tokens_path}")
    if not options.data_dir.is_dir():
        raise FileNotFoundError(f"espeak-ng-data not found: {options.data_dir}")
    if not options.text.strip():
        raise ValueError("Text for synthesis cannot be empty.")
    if options.num_threads <= 0:
        raise ValueError("num_threads must be greater than zero.")
    if options.speed <= 0:
        raise ValueError("speed must be greater than zero.")
    if options.sid < 0:
        raise ValueError("sid cannot be negative.")


def _build_tts(sherpa_onnx: Any, options: SpeakOptions) -> Any:
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(options.model_path),
                tokens=str(options.tokens_path),
                data_dir=str(options.data_dir),
            ),
            provider=options.provider,
            num_threads=options.num_threads,
        )
    )

    if not config.validate():
        raise RuntimeError("Invalid sherpa-onnx TTS config.")

    return sherpa_onnx.OfflineTts(config)


def _build_options(args: Any, config: dict[str, Any], config_path: Path | None) -> SpeakOptions:
    num_threads = (
        args.num_threads
        if args.num_threads is not None
        else get_nested(config, "speak", "num_threads", default=4)
    )
    speed = (
        args.speed
        if args.speed is not None
        else get_nested(config, "speak", "speed", default=1.0)
    )

    return SpeakOptions(
        model_dir=Path(args.model_dir).expanduser().resolve(),
        text=args.text,
        output_path=Path(
            args.output or get_nested(config, "speak", "output", default="outputs/tts.wav")
        ).expanduser().resolve(),
        config_path=config_path,
        provider=str(args.provider or get_nested(config, "speak", "provider", default="cpu")),
        num_threads=int(num_threads),
        sid=int(args.sid if args.sid is not None else get_nested(config, "speak", "sid", default=0)),
        speed=float(speed),
        dry_run=bool(args.dry_run),
    )


def _log_plan(options: SpeakOptions) -> None:
    LOGGER.info("Command: speak")
    LOGGER.info("Model dir: %s", options.model_dir)
    LOGGER.info("Model: %s", options.model_path)
    LOGGER.info("Tokens: %s", options.tokens_path)
    LOGGER.info("Data dir: %s", options.data_dir)
    LOGGER.info("Output wav: %s", options.output_path)
    LOGGER.info("Provider: %s | num_threads: %s", options.provider, options.num_threads)
    LOGGER.info("sid: %s | speed: %.2f", options.sid, options.speed)
    LOGGER.info("Text: %s", options.text)
    if options.config_path is not None:
        LOGGER.info("Config: %s", options.config_path)


def run_speak_stage(args: Any) -> int:
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    config = load_optional_yaml_config(config_path)
    options = _build_options(args, config, config_path)

    _log_plan(options)
    _validate_options(options)

    if options.dry_run:
        LOGGER.info("Dry run complete. No audio was generated.")
        return 0

    sherpa_onnx, soundfile = _load_runtime()
    tts = _build_tts(sherpa_onnx, options)

    start_time = time.time()
    audio = tts.generate(options.text, sid=options.sid, speed=options.speed)
    generation_time = time.time() - start_time

    if audio is None:
        raise RuntimeError("TTS generation returned None.")

    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(options.output_path), audio.samples, audio.sample_rate)

    duration = len(audio.samples) / float(audio.sample_rate)
    rtf = generation_time / duration if duration > 0 else 0.0

    LOGGER.info("Saved WAV: %s", options.output_path)
    LOGGER.info(
        "sample_rate=%s | duration=%.2fs | generation_time=%.3fs | RTF=%.3f",
        audio.sample_rate,
        duration,
        generation_time,
        rtf,
    )
    return 0
