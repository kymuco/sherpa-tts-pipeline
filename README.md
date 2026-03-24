# sherpa-tts-pipeline

Tool-first pipeline for building Sherpa-ONNX TTS voices from raw audio with:

- source audio preparation and loudness normalization
- `faster-whisper` dataset generation
- Piper training in Google Colab
- ONNX export for `sherpa-onnx`
- local CLI inference

The repo is meant to stay simple for users and clean for developers: one CLI, one optional config file, one Colab notebook for training.

## Workflow

1. Prepare raw source audio.
2. Normalize audio with `sherpa-tts prepare`.
3. Build a TTS dataset with `sherpa-tts dataset`.
4. Train the voice in `notebooks/train_piper_colab.ipynb`.
5. Export the Piper checkpoint with `sherpa-tts export`.
6. Test the exported bundle with `sherpa-tts speak`.

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

Build a dataset from a directory with audio files:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice
```

Preview what will be used without starting Whisper:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice --dry-run
```

Train in Colab:

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

It can hold knobs for:

- loudness normalization targets
- output sample rate and mono conversion for prepared audio
- dataset language and Whisper settings
- clip duration and quality thresholds
- export paths and ONNX opset
- inference provider, speed, speaker id, and output path

## What `dataset` Creates

Inside the output directory:

- `metadata.csv`
- `clips.csv`
- `rejected.csv`
- `sources.csv`
- `wavs/`

This matches the workflow used by the Colab notebook and keeps the generated dataset inspectable before training.

## Commands

The public CLI is intentionally small:

- `sherpa-tts prepare`
- `sherpa-tts dataset`
- `sherpa-tts export`
- `sherpa-tts speak`

All four commands support `--dry-run` for path and config validation before the heavy step.

## Repository Layout

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
