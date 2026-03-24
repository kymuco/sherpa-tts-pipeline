# Pipeline Overview

The repository is organized around a small number of explicit stages.

## 1. Prepare

Optional audio cleanup happens outside the core repository logic or in a future preprocessing stage.

Examples:

- vocal isolation
- denoise
- loudness normalization

## 2. Build Dataset

This stage turns long-form source audio into a TTS dataset.

Expected outputs:

- `metadata.csv`
- `clips.csv`
- `rejected.csv`
- `sources.csv`
- `wavs/`

## 3. Review Dataset

No automatic dataset pipeline should skip review. The review stage exists to catch:

- wrong transcriptions
- broken boundaries
- noisy clips
- repeated clips
- speaker drift

## 4. Train

Training is intended to run in Google Colab first. The notebook should become the canonical training path.

## 5. Export

This stage takes a trained checkpoint, exports ONNX, and prepares a `sherpa-onnx` bundle.

## 6. Infer

This stage runs local TTS inference for smoke tests, benchmarking, and demos.

