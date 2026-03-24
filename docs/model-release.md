# Model Release Notes

Canonical user-facing docs live in `README.md` and `USAGE.md`. This file is a small supplemental reference.

Code, model weights, and training data should be treated as separate release concerns.

## Recommended Export Bundle

```text
artifacts/export/<voice_name>/
  model.onnx
  tokens.txt
  espeak-ng-data/
  model-card.md
  samples/
```

## Release Checklist

- verify the voice data can be shared
- verify the trained weights can be shared
- write a short model card
- include inference instructions
- include known limitations

## Important

The repository code license does not automatically grant rights to redistribute:

- source audio
- cleaned vocals
- datasets
- checkpoints
- released models
