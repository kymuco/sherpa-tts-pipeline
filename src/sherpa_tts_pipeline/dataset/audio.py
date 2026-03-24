from __future__ import annotations

from pathlib import Path


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def looks_like_audio(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_AUDIO_SUFFIXES

