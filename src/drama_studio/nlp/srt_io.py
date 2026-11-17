"""SRT parsing and DaVinci Resolve Studio 20-compatible SRT serialization."""

from __future__ import annotations

from pathlib import Path
import re
import pysubs2

from drama_studio.models import SubtitleCue

_BOM = "\ufeff"


def _read_text_with_fallback(path: Path) -> str:
    raw_bytes = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "utf-16"):
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def format_srt_time(ms: int) -> str:
    """Format milliseconds into strict HH:MM:SS,mmm SRT timecode."""
    ms = max(0, int(round(ms)))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def load_srt(path: str | Path) -> list[SubtitleCue]:
    """Load an SRT file (including Jianying UTF-8-BOM/CRLF exports) preserving raw text and timestamps."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"字幕文件不存在: {p}")

    content = _read_text_with_fallback(p)
    content = content.lstrip(_BOM).replace("\r\n", "\n").replace("\r", "\n")

    subs = pysubs2.SSAFile.from_string(content, format_="srt")
    cues: list[SubtitleCue] = []
    for idx, ev in enumerate(subs.events, start=1):
        # Convert ASS line-breaks (\N) or whitespace
        text = ev.plaintext if hasattr(ev, "plaintext") else ev.text
        text = text.replace("\\N", "\n").replace("\\n", "\n")
        # Strip inline ASS override tags if any leaked into Jianying SRT
        text = re.sub(r"\{\\[^}]*\}", "", text).strip()
        if not text:
            continue
        start_ms = max(0, int(ev.start))
        end_ms = max(start_ms + 1, int(ev.end))
        cues.append(
            SubtitleCue(
                index=idx,
                start_ms=start_ms,
                end_ms=end_ms,
                raw_text=text,
                display_text=text,
                orig_start_ms=start_ms,
                orig_end_ms=end_ms,
                source_cue_indices=[idx],
            )
        )
    return cues


def normalize_cues_for_resolve(
    cues: list[SubtitleCue],
    *,
    min_gap_ms: int = 1,
    min_duration_ms: int = 80,
) -> list[SubtitleCue]:
    """Ensure strictly monotonic, non-overlapping cues safe for DaVinci Resolve Studio 20."""
    ordered = sorted(cues, key=lambda c: (c.start_ms, c.end_ms))
    result: list[SubtitleCue] = []

    for cue in ordered:
        text = (cue.display_text or cue.raw_text).strip()
        if not text:
            continue
        start = max(0, int(cue.start_ms))
        end = max(start + min_duration_ms, int(cue.end_ms))

        if result:
            prev = result[-1]
            if start < prev.end_ms + min_gap_ms:
                # Trim previous cue end if there is sufficient room; otherwise push current start
                needed_prev_end = start - min_gap_ms
                if needed_prev_end - prev.start_ms >= min_duration_ms:
                    prev.end_ms = needed_prev_end
                else:
                    start = prev.end_ms + min_gap_ms
                    if end <= start + min_duration_ms:
                        end = start + min_duration_ms

        cue.start_ms = start
        cue.end_ms = end
        cue.index = len(result) + 1
        result.append(cue)

    return result


def dumps_srt(cues: list[SubtitleCue], *, enforce_resolve_safe: bool = True) -> str:
    """Serialize cues to clean UTF-8 (no BOM) Unix-LF SRT string compatible with DaVinci Resolve 20."""
    items = normalize_cues_for_resolve(cues) if enforce_resolve_safe else cues
    blocks: list[str] = []
    for idx, cue in enumerate(items, start=1):
        text = (cue.display_text or cue.raw_text).replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            continue
        blocks.append(f"{idx}\n{format_srt_time(cue.start_ms)} --> {format_srt_time(cue.end_ms)}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def save_srt(cues: list[SubtitleCue], output_path: str | Path, *, enforce_resolve_safe: bool = True) -> Path:
    """Write SRT without BOM using LF line endings (best practice for DaVinci Resolve Studio 20)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    content = dumps_srt(cues, enforce_resolve_safe=enforce_resolve_safe)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return out
