"""Deterministic, episode-isolated stage cache with atomic writes and LRU disk quota pruning."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

from drama_studio.align.asr_engine import ASRRunResult
from drama_studio.align.base import ASRSegment
from drama_studio.align.engine import (
    AlignedUtteranceData,
    AlignmentEngineResult,
    ForcedAlignmentStageResult,
)
from drama_studio.audio.leveler import LevelingOutput
from drama_studio.models import (
    AnomalyItem,
    AnomalySeverity,
    AnomalyType,
    AudioAnalysisReport,
    CharTimestamp,
    SubtitleCue,
)
from drama_studio.nlp.script_matcher import ScriptMatchResult


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def stable_param_hash(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return _sha256_text(serialized)


def compute_media_fingerprint(video_path: str | Path) -> str:
    """Fast, content-grounded fingerprint combining size, mtime, and head/tail SHA-256."""
    p = Path(video_path)
    st = p.stat()
    h = hashlib.sha256()
    h.update(p.name.encode("utf-8", errors="ignore"))
    h.update(f"|{st.st_size}|{st.st_mtime_ns}|".encode("ascii"))
    try:
        with open(p, "rb") as f:
            h.update(f.read(65536))
            if st.st_size > 131072:
                f.seek(max(0, st.st_size - 65536))
                h.update(f.read(65536))
    except OSError:
        pass
    return h.hexdigest()[:16]


def compute_file_hash(path: str | Path | None) -> str:
    if not path:
        return "none"
    p = Path(path)
    if not p.is_file():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def _safe_slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff._-]+", "_", name).strip("._-")
    return cleaned[:40] or "episode"


def get_episode_cache_dir(cache_root: Path, video_path: str | Path, media_fp: str) -> Path:
    stem = _safe_slug(Path(video_path).stem)
    ep_dir = cache_root / f"{stem}_{media_fp[:12]}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    # Touch mtime for LRU tracking
    try:
        os.utime(ep_dir, None)
    except OSError:
        pass
    return ep_dir


def atomic_replace_file(
    src: Path,
    dst: Path,
    *,
    retries: int = 10,
    initial_delay_sec: float = 0.04,
) -> None:
    """Atomically replace `dst` with `src`, retrying brief Windows file-handle teardown locks."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_exc: PermissionError | None = None
    for attempt in range(max(1, retries)):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(initial_delay_sec * (attempt + 1))
    src.unlink(missing_ok=True)
    raise PermissionError(
        f"无法替换目标文件（可能正被音频播放器、剪辑软件或系统预览占用）：{dst} ({last_exc})"
    ) from last_exc


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        atomic_replace_file(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def read_json_safe(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_stage_cache_size_mb(cache_root: Path) -> float:
    if not cache_root.is_dir():
        return 0.0
    total = sum(f.stat().st_size for f in cache_root.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)


def clear_stage_cache(cache_root: Path) -> float:
    freed = get_stage_cache_size_mb(cache_root)
    if cache_root.is_dir():
        for child in list(cache_root.iterdir()):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
    return freed


def prune_stage_cache(cache_root: Path, max_size_mb: int = 4096, *, protect_dir: Path | None = None) -> int:
    """Enforce disk quota by evicting least-recently-used episode cache directories."""
    if not cache_root.is_dir() or max_size_mb <= 0:
        return 0
    max_bytes = int(max_size_mb) * 1024 * 1024
    target_bytes = int(max_bytes * 0.85)

    entries: list[tuple[float, int, Path]] = []
    total_bytes = 0
    for d in cache_root.iterdir():
        if not d.is_dir():
            continue
        sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        total_bytes += sz
        if protect_dir is not None and d.resolve() == protect_dir.resolve():
            continue
        try:
            mtime = d.stat().st_mtime
        except OSError:
            mtime = time.time()
        entries.append((mtime, sz, d))

    if total_bytes <= max_bytes:
        return 0

    entries.sort(key=lambda item: item[0])  # oldest first
    removed = 0
    for _mt, sz, d in entries:
        if total_bytes <= target_bytes:
            break
        shutil.rmtree(d, ignore_errors=True)
        total_bytes -= sz
        removed += 1
    return removed


# ---- Serialization Helpers for Stage Artifacts ----

def _cue_to_dict(c: SubtitleCue) -> dict[str, Any]:
    return {
        "index": c.index,
        "start_ms": c.start_ms,
        "end_ms": c.end_ms,
        "raw_text": c.raw_text,
        "display_text": c.display_text,
        "orig_start_ms": c.orig_start_ms,
        "orig_end_ms": c.orig_end_ms,
        "confidence": c.confidence,
        "char_timings": [asdict(ct) for ct in c.char_timings],
        "anomaly_flags": list(c.anomaly_flags),
        "source_cue_indices": list(c.source_cue_indices),
        "locked": c.locked,
        "turn_id": c.turn_id,
        "src_win_start_ms": c.src_win_start_ms,
        "src_win_end_ms": c.src_win_end_ms,
    }


def _cue_from_dict(d: dict[str, Any]) -> SubtitleCue:
    return SubtitleCue(
        index=int(d["index"]),
        start_ms=int(d["start_ms"]),
        end_ms=int(d["end_ms"]),
        raw_text=str(d["raw_text"]),
        display_text=str(d.get("display_text", "")),
        orig_start_ms=d.get("orig_start_ms"),
        orig_end_ms=d.get("orig_end_ms"),
        confidence=float(d.get("confidence", 1.0)),
        char_timings=[CharTimestamp(**ct) for ct in d.get("char_timings", [])],
        anomaly_flags=list(d.get("anomaly_flags", [])),
        source_cue_indices=list(d.get("source_cue_indices", [])),
        locked=bool(d.get("locked", False)),
        turn_id=int(d.get("turn_id", 0)),
        src_win_start_ms=int(d.get("src_win_start_ms", -1)),
        src_win_end_ms=int(d.get("src_win_end_ms", -1)),
    )


def _anomaly_to_dict(a: AnomalyItem) -> dict[str, Any]:
    return a.to_dict()


def _anomaly_from_dict(d: dict[str, Any]) -> AnomalyItem:
    return AnomalyItem(
        anomaly_type=AnomalyType(d["anomaly_type"]),
        severity=AnomalySeverity(d["severity"]),
        start_ms=int(d["start_ms"]),
        end_ms=int(d["end_ms"]),
        cue_index=d.get("cue_index"),
        message=str(d["message"]),
        suggestion=str(d.get("suggestion", "")),
    )


def save_asr_cache(path: Path, res: ASRRunResult) -> None:
    payload = {
        "detected_language": res.detected_language,
        "model_used": res.model_used,
        "warnings": list(res.warnings),
        "segments": [
            {
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "text": s.text,
                "language": s.language,
                "confidence": s.confidence,
                "model_used": s.model_used,
                "raw_char_timings": [asdict(ct) for ct in s.raw_char_timings],
                "src_win_start_ms": s.src_win_start_ms,
                "src_win_end_ms": s.src_win_end_ms,
            }
            for s in res.segments
        ],
    }
    atomic_write_json(path, payload)


def load_asr_cache(path: Path) -> ASRRunResult | None:
    d = read_json_safe(path)
    if not isinstance(d, dict) or "segments" not in d:
        return None
    try:
        segs = [
            ASRSegment(
                start_ms=int(s["start_ms"]),
                end_ms=int(s["end_ms"]),
                text=str(s["text"]),
                language=str(s.get("language", "zh")),
                confidence=float(s.get("confidence", 0.85)),
                model_used=str(s.get("model_used", "")),
                raw_char_timings=[CharTimestamp(**ct) for ct in s.get("raw_char_timings", [])],
                src_win_start_ms=int(s.get("src_win_start_ms", -1)),
                src_win_end_ms=int(s.get("src_win_end_ms", -1)),
            )
            for s in d["segments"]
        ]
        return ASRRunResult(
            segments=segs,
            detected_language=str(d.get("detected_language", "zh")),
            model_used=str(d.get("model_used", "")),
            warnings=list(d.get("warnings", [])),
        )
    except Exception:
        return None


def save_script_match_cache(path: Path, res: ScriptMatchResult) -> None:
    payload = {
        "cues": [_cue_to_dict(c) for c in res.cues],
        "anomalies": [_anomaly_to_dict(a) for a in res.anomalies],
        "overall_confidence": res.overall_confidence,
        "matched_count": res.matched_count,
        "total_asr_segments": res.total_asr_segments,
        "matched_script_unit_indices": res.matched_script_unit_indices,
        "skipped_script_turns": res.skipped_script_turns,
        "extra_audio_segments": res.extra_audio_segments,
        "duplicate_lines": res.duplicate_lines,
        "details": res.details,
    }
    atomic_write_json(path, payload)


def load_script_match_cache(path: Path) -> ScriptMatchResult | None:
    d = read_json_safe(path)
    if not isinstance(d, dict) or "cues" not in d:
        return None
    try:
        return ScriptMatchResult(
            cues=[_cue_from_dict(c) for c in d["cues"]],
            anomalies=[_anomaly_from_dict(a) for a in d.get("anomalies", [])],
            overall_confidence=float(d["overall_confidence"]),
            matched_count=int(d["matched_count"]),
            total_asr_segments=int(d["total_asr_segments"]),
            matched_script_unit_indices=list(d.get("matched_script_unit_indices", [])),
            skipped_script_turns=list(d.get("skipped_script_turns", [])),
            extra_audio_segments=list(d.get("extra_audio_segments", [])),
            duplicate_lines=list(d.get("duplicate_lines", [])),
            details=list(d.get("details", [])),
        )
    except Exception:
        return None


def save_forced_align_cache(path: Path, stage: ForcedAlignmentStageResult) -> None:
    payload = {
        "global_offset_ms": stage.global_offset_ms,
        "aligner_used": stage.aligner_used,
        "effective_language": stage.effective_language,
        "extra_phrases": stage.extra_phrases,
        "pre_warnings": [_anomaly_to_dict(a) for a in stage.pre_warnings],
        "script_mismatch_ratio": stage.script_mismatch_ratio,
        "aligned_windows": [
            {
                "clean_text": w.clean_text,
                "window_start_ms": w.window_start_ms,
                "window_end_ms": w.window_end_ms,
                "char_timings": [asdict(ct) for ct in w.char_timings],
                "win_char_turns": w.win_char_turns,
                "win_char_hints": w.win_char_hints,
                "per_cue_turn": w.per_cue_turn,
                "cues_snapshot": [_cue_to_dict(c) for c in w.cues_snapshot],
                "max_local_turn": w.max_local_turn,
            }
            for w in stage.aligned_windows
        ],
    }
    atomic_write_json(path, payload)


def load_forced_align_cache(path: Path) -> ForcedAlignmentStageResult | None:
    d = read_json_safe(path)
    if not isinstance(d, dict) or "aligned_windows" not in d:
        return None
    try:
        windows = [
            AlignedUtteranceData(
                clean_text=str(w["clean_text"]),
                window_start_ms=int(w["window_start_ms"]),
                window_end_ms=int(w["window_end_ms"]),
                char_timings=[CharTimestamp(**ct) for ct in w["char_timings"]],
                win_char_turns=list(w["win_char_turns"]),
                win_char_hints=list(w["win_char_hints"]),
                per_cue_turn=list(w["per_cue_turn"]),
                cues_snapshot=[_cue_from_dict(c) for c in w["cues_snapshot"]],
                max_local_turn=int(w["max_local_turn"]),
            )
            for w in d["aligned_windows"]
        ]
        return ForcedAlignmentStageResult(
            aligned_windows=windows,
            global_offset_ms=int(d.get("global_offset_ms", 0)),
            aligner_used=str(d.get("aligner_used", "")),
            effective_language=str(d.get("effective_language", "zh")),
            extra_phrases=list(d.get("extra_phrases", [])),
            pre_warnings=[_anomaly_from_dict(a) for a in d.get("pre_warnings", [])],
            script_mismatch_ratio=d.get("script_mismatch_ratio"),
        )
    except Exception:
        return None


def save_layout_cache(path: Path, res: AlignmentEngineResult) -> None:
    payload = {
        "cues": [_cue_to_dict(c) for c in res.cues],
        "anomalies": [_anomaly_to_dict(a) for a in res.anomalies],
        "global_offset_ms": res.global_offset_ms,
        "aligner_used": res.aligner_used,
        "nlp_used": res.nlp_used,
        "text_integrity_ok": res.text_integrity_ok,
    }
    atomic_write_json(path, payload)


def load_layout_cache(path: Path) -> AlignmentEngineResult | None:
    d = read_json_safe(path)
    if not isinstance(d, dict) or "cues" not in d:
        return None
    try:
        return AlignmentEngineResult(
            cues=[_cue_from_dict(c) for c in d["cues"]],
            anomalies=[_anomaly_from_dict(a) for a in d.get("anomalies", [])],
            global_offset_ms=int(d.get("global_offset_ms", 0)),
            aligner_used=str(d.get("aligner_used", "")),
            nlp_used=str(d.get("nlp_used", "")),
            text_integrity_ok=bool(d.get("text_integrity_ok", True)),
        )
    except Exception:
        return None


def save_leveling_meta_cache(path: Path, out: LevelingOutput) -> None:
    payload = {
        "report": asdict(out.report),
        "vad_spans_ms": [list(sp) for sp in out.vad_spans_ms],
        "gain_curve_times_ms": out.gain_curve_times_ms,
        "gain_curve_db": out.gain_curve_db,
        "waveform_times_ms": out.waveform_times_ms,
        "waveform_before_peaks": out.waveform_before_peaks,
        "waveform_after_peaks": out.waveform_after_peaks,
    }
    atomic_write_json(path, payload)


def load_leveling_meta_cache(path: Path) -> dict[str, Any] | None:
    d = read_json_safe(path)
    if not isinstance(d, dict) or "report" not in d:
        return None
    try:
        d["report_obj"] = AudioAnalysisReport(**d["report"])
        d["vad_spans_ms"] = [(int(s), int(e)) for s, e in d.get("vad_spans_ms", [])]
        return d
    except Exception:
        return None
