"""Batch import and automatic video + SRT pairing utilities."""

from __future__ import annotations

from pathlib import Path
import re
import uuid

from drama_studio.media.ffmpeg_wrapper import SUPPORTED_SUBTITLE_EXTS, SUPPORTED_VIDEO_EXTS
from drama_studio.models import EpisodeTask

SUPPORTED_SCRIPT_EXTS = {".md", ".txt"}
_IGNORED_SCRIPT_STEMS = {"readme", "index", "summary", "notes", "changelog", "license"}

_EPISODE_NUM_PATTERNS = [
    re.compile(r"第\s*(\d+)\s*[集话回期]"),
    re.compile(r"(?:^|[^A-Za-z0-9])(?:episode|ep)[_\-\s]*(\d+)(?!\w)", re.IGNORECASE),
    re.compile(r"(?:^|[^A-Za-z0-9])(?:s\d+[_\-\s]*)?e[_\-\s]*(\d+)(?=[^A-Za-z0-9]|$)", re.IGNORECASE),
    re.compile(r"(?:^|[_\-\s])(\d{1,4})(?:[_\-\s]|$)"),
]

_NOISE_SUFFIXES = [
    "粗剪", "精剪", "成片", "原片", "视频", "字幕", "剪映", "导出版",
    "台词提取", "台词稿", "台词", "文稿", "剧本", "final", "rough", "sub", "srt",
]


def extract_episode_number(stem: str) -> int | None:
    """Extract numeric episode index if present in filename stem."""
    for pat in _EPISODE_NUM_PATTERNS:
        m = pat.search(stem)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


def format_episode_label(stem: str, fallback_idx: int) -> str:
    """Build clean display label such as '第01集'."""
    ep_num = extract_episode_number(stem)
    if ep_num is not None:
        return f"第{ep_num:02d}集"
    cleaned = stem.strip()
    return cleaned if cleaned else f"第{fallback_idx:02d}集"


def normalize_match_key(stem: str) -> str:
    """Normalize a filename stem for fuzzy video <-> SRT pairing."""
    key = stem.lower()
    for word in _NOISE_SUFFIXES:
        key = re.sub(re.escape(word), "", key, flags=re.IGNORECASE)
    key = re.sub(r"[\s_\-\.\(\)\[\]【】]+", "", key)
    return key


def _is_valid_script_file(path: Path) -> bool:
    """Return True if path looks like an episode dialogue script (.md/.txt), skipping repo docs/readme."""
    if path.suffix.lower() not in SUPPORTED_SCRIPT_EXTS:
        return False
    stem_lower = path.stem.lower().strip()
    if not stem_lower or stem_lower.startswith((".", "~$")):
        return False
    if stem_lower in _IGNORED_SCRIPT_STEMS:
        return False
    return True


def _dedup_paths(items: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    uniq: list[Path] = []
    for item in items:
        rp = item.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(item)
    return uniq


def _scan_dropped_paths(paths: list[str | Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """Collect deduplicated video, SRT, and dialogue script paths from files/folders."""
    videos: list[Path] = []
    srts: list[Path] = []
    scripts: list[Path] = []

    def _classify_file(f: Path) -> None:
        ext = f.suffix.lower()
        if ext in SUPPORTED_VIDEO_EXTS:
            videos.append(f)
        elif ext in SUPPORTED_SUBTITLE_EXTS:
            srts.append(f)
        elif _is_valid_script_file(f):
            scripts.append(f)

    for p_raw in paths:
        p = Path(p_raw)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if not child.is_file():
                    continue
                if any(part.startswith(".") or part == "智能修正输出" for part in child.relative_to(p).parts[:-1]):
                    continue
                _classify_file(child)
        elif p.is_file():
            _classify_file(p)

    return _dedup_paths(videos), _dedup_paths(srts), _dedup_paths(scripts)


def _match_candidate_for_stems(
    anchor_paths: list[Path],
    candidates: list[Path],
    allow_single_fallback: bool = False,
) -> Path | None:
    """Match a candidate SRT/script file against one or more anchor paths (video/srt stems)."""
    if not candidates:
        return None
    # Pass 1: exact stem (prefer same dir first)
    for anchor in anchor_paths:
        for cand in candidates:
            if cand.stem == anchor.stem and cand.parent == anchor.parent:
                return cand
    for anchor in anchor_paths:
        for cand in candidates:
            if cand.stem == anchor.stem:
                return cand
    # Pass 2: normalized fuzzy key
    for anchor in anchor_paths:
        ak = normalize_match_key(anchor.stem)
        if ak:
            for cand in candidates:
                if normalize_match_key(cand.stem) == ak:
                    return cand
    # Pass 3: same episode number
    for anchor in anchor_paths:
        ep = extract_episode_number(anchor.stem)
        if ep is not None:
            for cand in candidates:
                if extract_episode_number(cand.stem) == ep:
                    return cand
    # Pass 4: single candidate fallback
    if allow_single_fallback and len(candidates) == 1:
        return candidates[0]
    return None


def assign_script_files_to_tasks(tasks: list[EpisodeTask], paths: list[str | Path]) -> int:
    """Match dropped script files/folders onto existing EpisodeTasks; return number of matched tasks."""
    _, _, scripts = _scan_dropped_paths(paths)
    if not scripts or not tasks:
        return 0
    remaining = list(scripts)
    matched_count = 0
    for idx, task in enumerate(tasks):
        anchors: list[Path] = []
        if task.video_path:
            anchors.append(Path(task.video_path))
        if task.srt_path:
            anchors.append(Path(task.srt_path))
        if task.episode_label:
            anchors.append(Path(task.episode_label))
        matched = _match_candidate_for_stems(
            anchors,
            remaining,
            allow_single_fallback=(len(tasks) == 1 and idx == 0),
        )
        if matched is not None:
            remaining.remove(matched)
            task.script_path = str(matched)
            matched_count += 1
    return matched_count


def pair_media_files(paths: list[str | Path]) -> tuple[list[EpisodeTask], list[str]]:
    """Pair dropped video files (.mp4/.mov/.mkv), subtitle files (.srt), and optional scripts (.md/.txt).

    Returns (tasks, unmatched_srt_paths).
    """
    uniq_videos, remaining_srts, remaining_scripts = _scan_dropped_paths(paths)
    tasks: list[EpisodeTask] = []

    single_video = len(uniq_videos) == 1
    for idx, v in enumerate(uniq_videos, start=1):
        matched_srt = _match_candidate_for_stems([v], remaining_srts, allow_single_fallback=single_video)
        if matched_srt is not None:
            remaining_srts.remove(matched_srt)

        script_anchors = [v] + ([matched_srt] if matched_srt is not None else [])
        matched_script = _match_candidate_for_stems(
            script_anchors,
            remaining_scripts,
            allow_single_fallback=single_video,
        )
        if matched_script is not None:
            remaining_scripts.remove(matched_script)

        tasks.append(
            EpisodeTask(
                task_id=uuid.uuid4().hex[:10],
                episode_label=format_episode_label(v.stem, idx),
                video_path=str(v),
                srt_path=str(matched_srt) if matched_srt else "",
                script_path=str(matched_script) if matched_script else "",
            )
        )

    # Sort tasks naturally by episode number if available
    tasks.sort(
        key=lambda t: (
            0 if extract_episode_number(Path(t.video_path).stem) is not None else 1,
            extract_episode_number(Path(t.video_path).stem) or 0,
            t.episode_label,
        )
    )

    return tasks, [str(s) for s in remaining_srts]
