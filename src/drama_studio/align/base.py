"""Abstract alignment interfaces and shared confidence/validation primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from drama_studio.models import CharTimestamp, SubtitleCue


@dataclass
class UtteranceWindow:
    """Contiguous group of SRT cues aligned together within an audio window."""

    cues: list[SubtitleCue]
    window_start_ms: int
    window_end_ms: int
    clean_text: str


@dataclass
class AlignerOutput:
    """Character-level timestamps produced for a single utterance window."""

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
        device: str = "auto",
        cache_dir: Path | None = None,
    ) -> AlignerOutput:
        ...
