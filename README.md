# sherpa-tts-pipeline

Tool-first pipeline for building Sherpa-ONNX TTS voices from raw audio with:

- safer source audio preparation and loudness normalization
- `faster-whisper` dataset generation
- Piper training in Google Colab
- ONNX export for `sherpa-onnx`
- local CLI inference

The repo is meant to stay simple for users and clean for developers: one CLI, one optional config file, and focused Colab notebooks for dataset building and training.

## Workflow

1. Prepare raw source audio.
2. Normalize audio with `sherpa-tts prepare`.
3. Build a TTS dataset with `sherpa-tts dataset` or `notebooks/build_dataset_colab.ipynb`.
4. Review the result with `sherpa-tts report` and `sherpa-tts review`.
5. Sanity-check the environment with `sherpa-tts doctor` when needed.
6. Train the voice in `notebooks/train_piper_colab.ipynb`.
7. Export the Piper checkpoint with `sherpa-tts export`.
8. Test the exported bundle with `sherpa-tts speak`.

## Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
```

Requirements:

- `ffmpeg` in `PATH` for `prepare` and dataset generation
- a local `piper1-gpl/src` checkout for export

## Quick Start

Normalize raw audio into a clean working directory:

```bash
sherpa-tts prepare raw_audio\my_voice --out prepared_audio\my_voice
```

Preview normalization without writing files:

```bash
sherpa-tts prepare raw_audio\my_voice --out prepared_audio\my_voice --dry-run
```

`prepare` writes normalized `.wav` files to the output directory and keeps the original files untouched.

By default it now uses the safer `normalize-only` mode:

- keeps the original sample rate unless you override it
- keeps the original channel count unless you ask for mono
- applies loudness normalization without silently forcing training format

If you want training-ready files right away:

```bash
sherpa-tts prepare raw_audio\my_voice --out prepared_audio\my_voice --mode training-ready
```

Build a dataset from a directory with audio files:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice
```

Preview what will be used without starting Whisper:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice --dry-run
```

`append` now protects you from exact duplicate clips by default. If you really want duplicates, use:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice --append --allow-duplicates
```

Generate a quick summary report after or between runs:

```bash
sherpa-tts report data\my_voice
```

Create a manual review queue and rescue previews:

```bash
sherpa-tts review data\my_voice --subset rescue
```

Check the local environment and optional paths:

```bash
sherpa-tts doctor --config examples\voice.yaml --dataset-dir data\my_voice
```

Train in Colab:

- open `notebooks/build_dataset_colab.ipynb` to build a dataset in Colab
- open `notebooks/train_piper_colab.ipynb`

Export a checkpoint:

```bash
sherpa-tts export --checkpoint path/to/model.ckpt --out release/my_voice --piper-src path/to/piper1-gpl/src
```

Export a bundle that is ready for `speak`:

```bash
sherpa-tts export ^
  --checkpoint path/to/model.ckpt ^
  --out release/my_voice ^
  --piper-src path/to/piper1-gpl/src ^
  --tokens path/to/tokens.txt ^
  --espeak-data-dir path/to/espeak-ng-data
```

Run local inference:

```bash
sherpa-tts speak --model-dir release/my_voice --text "Hello"
```

Preview inference settings without generating audio:

```bash
sherpa-tts speak --model-dir release/my_voice --text "Hello" --dry-run
```

## Config

You do not need a config file to start.

If you want extra control, use:

- `examples/voice.yaml`
- `examples/voice_rescue.yaml` for a second, softer pass over rejected clips

It can hold knobs for:

- loudness normalization targets
- safe prepare mode, output sample rate, mono conversion, and codec
- dataset language and Whisper settings
- clip duration and quality thresholds
- export paths and ONNX opset
- inference provider, speed, speaker id, and output path

`voice_rescue.yaml` is meant for "save what can be saved" runs. It is more permissive about short clips, pause merging, padding, and `no_speech_prob`, so review rescued clips manually before training.

## What `dataset` Creates

Inside the output directory:

- `metadata.csv`
- `clips.csv`
- `rejected.csv`
- `sources.csv`
- `dataset_report.json`
- `dataset_report.md`
- `wavs/`

This matches the workflow used by the Colab notebook and keeps the generated dataset inspectable before training.

## Commands

The public CLI is intentionally small:

- `sherpa-tts prepare`
- `sherpa-tts dataset`
- `sherpa-tts review`
- `sherpa-tts report`
- `sherpa-tts doctor`
- `sherpa-tts export`
- `sherpa-tts speak`

All commands except `doctor` support `--dry-run` for path and config validation before the heavy step.

## Repository Layout

```text
sherpa-tts-pipeline/
  README.md
  pyproject.toml
  requirements.txt
  requirements-colab.txt
  requirements-dev.txt
  notebooks/
    build_dataset_colab.ipynb
    train_piper_colab.ipynb
  examples/
    voice.yaml
    text_samples.txt
  src/
    sherpa_tts_pipeline/
  tests/
```
