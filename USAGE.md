# `sherpa-tts-pipeline` Usage Guide

Detailed usage guide for the CLI pipeline that prepares audio, builds a TTS dataset with `faster-whisper`, reviews the result, exports Piper checkpoints to ONNX, and runs local `sherpa-onnx` inference.

This document describes the workflow that exists in the repository today.

## Contents

- [What This Project Is](#what-this-project-is)
- [Requirements](#requirements)
- [Install](#install)
- [Recommended Folder Layout](#recommended-folder-layout)
- [Workflow At A Glance](#workflow-at-a-glance)
- [1. Check The Environment With `doctor`](#1-check-the-environment-with-doctor)
- [2. Prepare Source Audio With `prepare`](#2-prepare-source-audio-with-prepare)
- [3. Build A Dataset With `dataset`](#3-build-a-dataset-with-dataset)
- [4. Read The Dataset Outputs](#4-read-the-dataset-outputs)
- [5. Generate A Report With `report`](#5-generate-a-report-with-report)
- [6. Create A Review Queue With `review`](#6-create-a-review-queue-with-review)
- [7. Build The Dataset In Colab](#7-build-the-dataset-in-colab)
- [8. Train In Colab](#8-train-in-colab)
- [9. Export A Piper Checkpoint With `export`](#9-export-a-piper-checkpoint-with-export)
- [10. Run Local Inference With `speak`](#10-run-local-inference-with-speak)
- [Config Files](#config-files)
- [Rescue Workflow](#rescue-workflow)
- [Append, Overwrite, And Duplicates](#append-overwrite-and-duplicates)
- [Common Gotchas](#common-gotchas)

## What This Project Is

`sherpa-tts-pipeline` is a tool-first repository for building a custom TTS voice from existing audio material.

The intended workflow is:

1. collect source audio
2. normalize it if needed
3. build a dataset with `faster-whisper`
4. review and rescue borderline clips
5. train with Piper in Colab
6. export the checkpoint to ONNX
7. test the exported voice locally with `sherpa-onnx`

It is not a general audio editor, not a fully automatic voice-cloning product, and not a local training framework.

## Requirements

- Python `3.10+`
- system `ffmpeg` in `PATH`
- local `piper1-gpl/src` checkout for export
- enough disk space for raw audio, prepared audio, datasets, and checkpoints

Optional but useful:

- GPU for faster dataset building
- Google Colab for training

## Install

For local development:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
```

Quick sanity check:

```bash
sherpa-tts doctor --config examples\voice.yaml
```

## Recommended Folder Layout

```text
raw_audio/
  my_voice/
prepared_audio/
  my_voice/
data/
  my_voice/
release/
  my_voice/
```

Suggested meaning:

- `raw_audio/my_voice/`: untouched source files
- `prepared_audio/my_voice/`: normalized WAV copies
- `data/my_voice/`: generated dataset
- `release/my_voice/`: exported ONNX bundle

## Workflow At A Glance

```bash
sherpa-tts doctor --config examples\voice.yaml
sherpa-tts prepare raw_audio\my_voice --out prepared_audio\my_voice
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice
sherpa-tts report data\my_voice
sherpa-tts review data\my_voice --subset rescue
```

Then:

- open `notebooks/train_piper_colab.ipynb`
- train the model
- export with `sherpa-tts export`
- test with `sherpa-tts speak`

## 1. Check The Environment With `doctor`

Use `doctor` first if setup is new or something feels off.

```bash
sherpa-tts doctor --config examples\voice.yaml
```

It checks:

- Python
- `ffmpeg`
- `ffprobe`
- `PyYAML`
- `ctranslate2`
- `faster-whisper`
- `soundfile`
- `torch`
- `sherpa-onnx-bin`
- CUDA visibility for `torch` and `ctranslate2`

Optional checks:

```bash
sherpa-tts doctor ^
  --config examples\voice.yaml ^
  --dataset-dir data\my_voice ^
  --model-dir release\my_voice ^
  --piper-src path\to\piper1-gpl\src
```

`doctor` returns non-zero only on real failures. Missing CUDA is currently a warning, not a hard failure.

## 2. Prepare Source Audio With `prepare`

Basic usage:

```bash
sherpa-tts prepare raw_audio\my_voice --out prepared_audio\my_voice
```

Preview only:

```bash
sherpa-tts prepare raw_audio\my_voice --out prepared_audio\my_voice --dry-run
```

### Safe Default: `normalize-only`

`prepare` now defaults to `normalize-only`.

That means:

- it normalizes loudness
- it keeps the original sample rate unless you override it
- it keeps the original channel count unless you ask for mono
- it writes normalized WAV files without silently forcing training format

### Training-Ready Mode

If you explicitly want mono `22050 Hz` WAVs during preparation:

```bash
sherpa-tts prepare raw_audio\my_voice --out prepared_audio\my_voice --mode training-ready
```

### Useful Flags

```bash
sherpa-tts prepare raw_audio\my_voice ^
  --out prepared_audio\my_voice ^
  --target-lufs -18 ^
  --lra 7 ^
  --true-peak -1.5
```

Optional format overrides:

```bash
sherpa-tts prepare raw_audio\my_voice ^
  --out prepared_audio\my_voice ^
  --sample-rate 22050 ^
  --mono ^
  --codec pcm_s16le
```

When in doubt, keep the default safe mode.

## 3. Build A Dataset With `dataset`

Basic usage:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice
```

You can also point it directly at raw audio:

```bash
sherpa-tts dataset raw_audio\my_voice --out data\my_voice
```

Preview without running Whisper:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice --dry-run
```

Use a config:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice --config examples\voice.yaml
```

Override the Whisper model:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice --whisper-model medium
```

### What `dataset` Does

For each source file it:

- runs `faster-whisper`
- applies optional VAD
- splits long segments
- merges short adjacent segments when they likely belong together
- pads clip boundaries
- filters clips by quality thresholds
- exports final WAV clips into `wavs/`
- writes CSV metadata files
- writes a dataset report automatically

## 4. Read The Dataset Outputs

Inside the dataset directory:

- `metadata.csv`: final training list in `clip_id|text` format
- `clips.csv`: all kept clips with timings and quality fields
- `rejected.csv`: rejected candidates with reasons
- `sources.csv`: per-source summary stats
- `dataset_report.json`: machine-readable summary
- `dataset_report.md`: human-readable summary
- `wavs/`: exported clip audio

## 5. Generate A Report With `report`

If you want a fresh summary without rebuilding:

```bash
sherpa-tts report data\my_voice
```

Preview counts only:

```bash
sherpa-tts report data\my_voice --dry-run
```

This is useful after manual edits or merges.

## 6. Create A Review Queue With `review`

Basic usage:

```bash
sherpa-tts review data\my_voice
```

This creates a review folder with:

- `review_queue.csv`
- `rescue_candidates.csv`
- optional preview WAVs for rejected clips

Rescue-only review:

```bash
sherpa-tts review data\my_voice --subset rescue
```

Skip preview WAV extraction:

```bash
sherpa-tts review data\my_voice --subset rescue --no-extract-rejected
```

Why this exists:

- `rejected.csv` is useful for diagnostics
- `review_queue.csv` is better for actual human review
- rescue candidates are often worth keeping after a second pass

## 7. Build The Dataset In Colab

If your local machine is slow or you want everything in Google Drive, use:

- `notebooks/build_dataset_colab.ipynb`

The notebook lets you choose:

- `raw`
- `prepared`
- `prepare-first`

from one edit cell.

## 8. Train In Colab

Use:

- `notebooks/train_piper_colab.ipynb`

Expected input:

- dataset directory with `metadata.csv`
- `wavs/`

The notebook is the recommended path for training rather than local training scripts.

## 9. Export A Piper Checkpoint With `export`

Basic export:

```bash
sherpa-tts export ^
  --checkpoint path\to\model.ckpt ^
  --out release\my_voice ^
  --piper-src path\to\piper1-gpl\src
```

Bundle export for `speak`:

```bash
sherpa-tts export ^
  --checkpoint path\to\model.ckpt ^
  --out release\my_voice ^
  --piper-src path\to\piper1-gpl\src ^
  --tokens path\to\tokens.txt ^
  --espeak-data-dir path\to\espeak-ng-data
```

If `tokens.txt` or `espeak-ng-data` are missing, export still works, but the bundle is not ready for `speak`.

## 10. Run Local Inference With `speak`

Basic usage:

```bash
sherpa-tts speak --model-dir release\my_voice --text "Hello"
```

Preview only:

```bash
sherpa-tts speak --model-dir release\my_voice --text "Hello" --dry-run
```

Override runtime settings:

```bash
sherpa-tts speak ^
  --model-dir release\my_voice ^
  --text "Hello" ^
  --provider cpu ^
  --num-threads 4 ^
  --sid 0 ^
  --speed 1.0
```

## Config Files

Main config:

- `examples/voice.yaml`

Softer second-pass config:

- `examples/voice_rescue.yaml`

High-level sections:

- `prepare`
- `dataset`
- `export`
- `speak`

Typical reasons to touch the config:

- change Whisper model or language
- relax duration thresholds
- adjust VAD pause behavior
- switch prepare mode
- set export paths
- set inference defaults

## Rescue Workflow

Recommended pattern:

1. Build the strict dataset with `examples/voice.yaml`.
2. Inspect `rejected.csv` and `dataset_report.md`.
3. Run `review --subset rescue`.
4. Run a second pass with `examples/voice_rescue.yaml`.
5. Manually listen to rescued clips before merging them into training data.

Example:

```bash
sherpa-tts dataset raw_audio\my_voice --out data\my_voice_rescue --config examples\voice_rescue.yaml
```

## Append, Overwrite, And Duplicates

### `--overwrite`

Rebuild the dataset directory from scratch:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice --overwrite
```

### `--append`

Add new clips into an existing dataset:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice --append
```

### Duplicate Protection

When appending, the tool now skips exact duplicates by default.

Current duplicate signature:

- `source_file`
- `export_start_sec`
- `export_end_sec`
- `text`

If you really want duplicates:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice --append --allow-duplicates
```

## Common Gotchas

- `prepare` is not a magic cleaner. If the source audio is noisy, the normalized result can still be noisy.
- `faster-whisper` auto-segmentation still needs manual review for best training quality.
- `high_no_speech_prob` does not always mean the clip is useless. Some of those are salvageable.
- `append` is for adding new material, not for casually rerunning the same source set over and over.
- `voice_rescue.yaml` is intentionally softer. Do not trust rescued clips blindly.
- `export` needs a real Piper checkpoint and a local `piper1-gpl/src`.
- `speak` needs `model.onnx`, `tokens.txt`, and `espeak-ng-data` together in the model directory.
