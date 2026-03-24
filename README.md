# sherpa-tts-pipeline

CLI pipeline for building Sherpa-ONNX TTS voices from raw audio.

This repository is for the practical workflow:

`raw audio -> prepare -> dataset -> review -> train -> export -> speak`

It is meant to stay simple for users and still clean for developers:

- one public CLI: `sherpa-tts`
- one main config example: `examples/voice.yaml`
- one softer rescue config: `examples/voice_rescue.yaml`
- one Colab notebook for dataset building
- one Colab notebook for training

## What It Does

- normalizes source audio safely before dataset generation
- builds TTS datasets with `faster-whisper`
- produces inspectable outputs: `metadata.csv`, `clips.csv`, `rejected.csv`, `sources.csv`
- creates dataset reports automatically
- prepares manual review and rescue queues
- exports Piper checkpoints to ONNX for `sherpa-onnx`
- runs local CLI inference from exported bundles

## What It Does Not Do

- it does not train locally out of the box
- it does not include Piper itself
- it does not magically clean bad source material
- it does not replace manual review of the dataset

## Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
```

Requirements:

- Python `3.10+`
- `ffmpeg` in `PATH` for `prepare`, `dataset`, and review preview extraction
- local `piper1-gpl/src` checkout for `export`

## Quick Start

Check that the local environment is sane:

```bash
sherpa-tts doctor --config examples\voice.yaml
```

Normalize source audio without silently forcing mono or `22050 Hz`:

```bash
sherpa-tts prepare raw_audio\my_voice --out prepared_audio\my_voice
```

Build a dataset:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice
```

Generate a summary report:

```bash
sherpa-tts report data\my_voice
```

Create a rescue-focused review queue:

```bash
sherpa-tts review data\my_voice --subset rescue
```

Train in Colab:

- open `notebooks/build_dataset_colab.ipynb` if you want dataset generation in Colab
- open `notebooks/train_piper_colab.ipynb` for Piper training

Export a checkpoint:

```bash
sherpa-tts export --checkpoint path/to/model.ckpt --out release/my_voice --piper-src path/to/piper1-gpl/src
```

Run local inference:

```bash
sherpa-tts speak --model-dir release/my_voice --text "Hello"
```

## Detailed Guide

For the full walkthrough, command usage, config knobs, rescue workflow, append behavior,
dataset file meanings, Colab notes, and common gotchas, see [USAGE.md](USAGE.md).

## Command Summary

- `sherpa-tts doctor`
- `sherpa-tts prepare`
- `sherpa-tts dataset`
- `sherpa-tts report`
- `sherpa-tts review`
- `sherpa-tts export`
- `sherpa-tts speak`

All commands except `doctor` support `--dry-run`.

## Repository Layout

```text
sherpa-tts-pipeline/
  README.md
  USAGE.md
  pyproject.toml
  requirements.txt
  requirements-colab.txt
  requirements-dev.txt
  notebooks/
    build_dataset_colab.ipynb
    train_piper_colab.ipynb
  examples/
    voice.yaml
    voice_rescue.yaml
    text_samples.txt
  src/
    sherpa_tts_pipeline/
  tests/
```
