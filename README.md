# sherpa-tts-pipeline

Simple CLI and library for building custom TTS voices with:

- `faster-whisper` dataset preparation
- Piper training in Google Colab
- ONNX export for local runtime use

This repo is meant to feel simple from the outside and clean from the inside.

## Fast Path

1. Install the package.
2. Build a dataset from your source audio.
3. Train in Colab.
4. Export ONNX.
5. Test the voice locally.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

`dataset` also expects `ffmpeg` to be available in `PATH`.

## Commands

Build a dataset:

```bash
sherpa-tts dataset raw_audio --out data/my_voice
```

Build a dataset with advanced knobs:

```bash
sherpa-tts dataset raw_audio --out data/my_voice --config examples/voice.yaml
```

Preview the resolved files and settings without starting Whisper:

```bash
sherpa-tts dataset raw_audio --out data/my_voice --dry-run
```

Export a checkpoint:

```bash
sherpa-tts export --checkpoint path/to/model.ckpt --out release/my_voice --piper-src path/to/piper1-gpl/src
```

Test a voice locally:

```bash
sherpa-tts speak --model-dir release/my_voice --text "Hello"
```

## Train In Colab

Open:

- `notebooks/train_piper_colab.ipynb`

The notebook is the current training entry point. It handles:

1. Colab GPU check
2. Piper install
3. Drive mount
4. Dataset validation
5. Training
6. ONNX export

If you want a local export bundle that is ready for `speak`, you can also copy assets during export:

```bash
sherpa-tts export ^
  --checkpoint path/to/model.ckpt ^
  --out release/my_voice ^
  --piper-src path/to/piper1-gpl/src ^
  --tokens path/to/tokens.txt ^
  --espeak-data-dir path/to/espeak-ng-data
```

## Optional Config

You do not need a config file to start.

If you want more control, use:

- `examples/voice.yaml`

That keeps the public UX simple:

- no required project bootstrap
- no required multi-file config system
- no need to understand the package layout before first use

## Package Layout

```text
sherpa-tts-pipeline/
  README.md
  pyproject.toml
  requirements.txt
  requirements-colab.txt
  requirements-dev.txt
  notebooks/
    train_piper_colab.ipynb
  examples/
    voice.yaml
    text_samples.txt
  src/
    sherpa_tts_pipeline/
  tests/
```

## Current Status

The public CLI is intentionally simple:

- `dataset`
- `export`
- `speak`

All three commands have a `--dry-run` mode so you can validate paths and config before launching the heavy step.
