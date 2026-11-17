"""Undo/Redo commands for subtitle editing and local manual audio gain adjustments."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

from PySide6.QtGui import QUndoCommand

from drama_studio.models import ManualGainPoint, SubtitleCue


class ReplaceCuesCommand(QUndoCommand):
    """Undoable snapshot replacement for subtitle cue edits (timing, text wrap, merge, split)."""

    def __init__(
        self,
        description: str,
        before_cues: list[SubtitleCue],
        after_cues: list[SubtitleCue],
        apply_fn: Callable[[list[SubtitleCue]], None],
    ) -> None:
        super().__init__(description)
        self.before_cues = deepcopy(before_cues)
        self.after_cues = deepcopy(after_cues)
        self.apply_fn = apply_fn
        self._first = True

    def redo(self) -> None:
        if self._first:
            self._first = False
        self.apply_fn(deepcopy(self.after_cues))

    def undo(self) -> None:
        self.apply_fn(deepcopy(self.before_cues))


class ReplaceManualGainCommand(QUndoCommand):
    """Undoable replacement for local manual volume adjustment points."""

    def __init__(
        self,
        description: str,
        before_points: list[ManualGainPoint],
        after_points: list[ManualGainPoint],
        apply_fn: Callable[[list[ManualGainPoint]], None],
    ) -> None:
        super().__init__(description)
        self.before_points = deepcopy(before_points)
        self.after_points = deepcopy(after_points)
        self.apply_fn = apply_fn

    def redo(self) -> None:
        self.apply_fn(deepcopy(self.after_points))

    def undo(self) -> None:
        self.apply_fn(deepcopy(self.before_points))
