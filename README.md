# sherpa-tts-pipeline

[![CI](https://github.com/kymuco/sherpa-tts-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/kymuco/sherpa-tts-pipeline/actions/workflows/ci.yml)

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

## Proof

What a new user can trust right now:

- the public CLI is covered by local tests and CI
- `prepare`, `dataset`, `report`, `review`, `export`, and `speak` all have real implementations
- dataset generation writes inspectable CSVs instead of hiding decisions
- duplicate clips are skipped by default when appending
- Colab notebooks are included for dataset building and training

What still remains your responsibility:

- source audio quality
- manual review of the generated dataset
- training quality checks
- rights to the source audio, dataset, checkpoint, and released model

## Demo

What a successful dataset build looks like on disk:

```text
data/
  my_voice/
    metadata.csv
    clips.csv
    rejected.csv
    sources.csv
    dataset_report.json
    dataset_report.md
    wavs/
      1.wav
      2.wav
      3.wav
    review/
      review_queue.csv
      rescue_candidates.csv
```

What a user should notice:

- the tool does not hide decisions inside one opaque file
- kept clips, rejected clips, and source summaries stay inspectable
- review artifacts are separate from the training metadata

Example from one local test run:

```text
kept clips: 223
rejected clips: 38
kept audio: about 19.5 minutes
top rejected reasons:
- too_short: 30
- high_no_speech_prob: 7
- too_long: 1
```

Those numbers are only an example. Your counts will depend on source quality, speaking style, language, and config thresholds.

## What It Does Not Do

- it does not train locally out of the box
- it does not include Piper itself
- it does not magically clean bad source material
- it does not replace manual review of the dataset

## Known Limitations

- the workflow is currently optimized for single-speaker voice building first
- dataset quality still depends heavily on source quality and manual review
- `export` depends on a separate local `piper1-gpl/src` checkout
- training is Colab-first rather than a one-command local training experience
- the command examples are PowerShell-first, even though the core code is cross-platform

## Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
```

The command examples below use PowerShell-style paths. On Linux or macOS, use `source venv/bin/activate` and normal `/` path separators.

Requirements:

- Python `3.10+`
- `ffmpeg` in `PATH` for `prepare`, `dataset`, and review preview extraction
- local `piper1-gpl/src` checkout for `export`

## Tested On

- Windows with PowerShell during local development
- Ubuntu in GitHub Actions CI
- Python `3.11` in CI
- Python `3.10+` as the supported package target
- CPU-only environments by default, with optional CUDA acceleration when available

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

## First Successful Run

If you want one concrete "I just want to see it work" path, use this:

1. Put a few source files into `raw_audio/my_voice/`.
2. Check the environment:

```bash
sherpa-tts doctor --config examples\voice.yaml
```

Expected result:

- no `FAIL` lines
- `ffmpeg` is found
- `examples/voice.yaml` is accepted

3. Normalize audio:

```bash
sherpa-tts prepare raw_audio\my_voice --out prepared_audio\my_voice
```

Expected result:

- `prepared_audio/my_voice/` contains normalized `.wav` files

4. Build the dataset:

```bash
sherpa-tts dataset prepared_audio\my_voice --out data\my_voice
```

Expected result:

- `data/my_voice/metadata.csv`
- `data/my_voice/clips.csv`
- `data/my_voice/rejected.csv`
- `data/my_voice/sources.csv`
- `data/my_voice/dataset_report.md`
- `data/my_voice/wavs/`

5. Inspect the report:

```bash
sherpa-tts report data\my_voice
```

Expected result:

- `dataset_report.json` and `dataset_report.md` are refreshed
- you can immediately see kept vs rejected counts

6. Build a rescue queue:

```bash
sherpa-tts review data\my_voice --subset rescue
```

Expected result:

- `data/my_voice/review/review_queue.csv`
- `data/my_voice/review/rescue_candidates.csv`
- optional `rejected_wavs/` previews if extraction is enabled

At that point the project has already proven the most important part: it can turn raw audio into a reviewable TTS dataset.

## Detailed Guide

For the full walkthrough, command usage, config knobs, rescue workflow, append behavior,
dataset file meanings, Colab notes, FAQ, and common gotchas, see [USAGE.md](USAGE.md).

Supplemental reference notes live in [docs/README.md](docs/README.md). `README.md` and `USAGE.md` are the canonical user-facing docs.

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
