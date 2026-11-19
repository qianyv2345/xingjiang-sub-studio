"""Abstract ASR/alignment interfaces and shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from drama_studio.models import CharTimestamp, SubtitleCue


@dataclass
class ASRSegment:
    """Coarse speech recognition segment grounded on a VAD timeline window."""

    start_ms: int
    end_ms: int
    text: str
    language: str = "zh"  # "zh" | "en"
    confidence: float = 0.85
    model_used: str = ""
    raw_char_timings: list[CharTimestamp] = field(default_factory=list)
    src_win_start_ms: int = -1
    src_win_end_ms: int = -1


@dataclass
class UtteranceWindow:
    """Contiguous group of cues aligned together within an audio window."""

    cues: list[SubtitleCue]
    window_start_ms: int
    window_end_ms: int
    clean_text: str


@dataclass
class AlignerOutput:
    """Character-level or word-level timestamps produced for a single utterance window."""

    char_timings: list[CharTimestamp]
    backend_name: str
    raw_asr_text: str = ""
    warnings: list[str] = field(default_factory=list)


class AlignmentBackend(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def align_utterance(
        self,
        audio_16k: np.ndarray,
        window_start_ms: int,
        window_end_ms: int,
        spoken_text: str,
        *,
        language: str = "zh",
        device: str = "auto",
        cache_dir: Path | None = None,
    ) -> AlignerOutput:
        ...
