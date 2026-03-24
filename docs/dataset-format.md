# Dataset Format

The repository standardizes the dataset output contract so the training notebook and future tools can rely on it.

## Required Files

- `metadata.csv`
- `clips.csv`
- `rejected.csv`
- `sources.csv`
- `wavs/`

## `metadata.csv`

Pipe-delimited training manifest:

```text
clip_id|normalized text
```

## `clips.csv`

Accepted clips with quality metadata.

Suggested columns:

- `clip_id`
- `clip_path`
- `source_file`
- `speech_start_sec`
- `speech_end_sec`
- `export_start_sec`
- `export_end_sec`
- `duration_sec`
- `text`
- `avg_logprob`
- `no_speech_prob`
- `avg_word_probability`
- `min_word_probability`
- `word_count`
- `segment_count`
- `reason`

## `rejected.csv`

Rejected clips with the same columns as `clips.csv`, plus the rejection reason.

## `sources.csv`

One summary row per source file.

Suggested columns:

- `source_file`
- `language`
- `language_probability`
- `duration_sec`
- `duration_after_vad_sec`
- `raw_segments`
- `candidate_chunks`
- `kept_chunks`
- `rejected_chunks`

