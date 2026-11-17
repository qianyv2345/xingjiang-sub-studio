"""Persistent application configuration, template library, and project settings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import uuid

from drama_studio.models import ProjectConfig, SubtitleTemplate


def default_app_data_dir() -> Path:
    """Return a stable Windows AppData (or user-home fallback) config directory.

    Preserves existing legacy ``DramaSubStudio`` directories so previously
    downloaded AI models and saved projects remain immediately available.
    """
    override = os.environ.get("DRAMA_STUDIO_HOME")
    if override:
        path = Path(override)
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        legacy = base / "DramaSubStudio"
        target = base / "XingJiangSubStudio"
        path = legacy if legacy.exists() and not target.exists() else target
    else:
        base = Path.home() / ".config"
        legacy = base / "DramaSubStudio"
        target = base / "XingJiangSubStudio"
        path = legacy if legacy.exists() and not target.exists() else target
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_model_cache_dir(base_dir: Path | None = None) -> Path:
    root = base_dir or default_app_data_dir()
    cache = root / "models"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


class ConfigStore:
    """Loads and saves templates, per-project settings, and global preferences."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or default_app_data_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.templates_path = self.base_dir / "templates.json"
        self.projects_path = self.base_dir / "projects.json"
        self.prefs_path = self.base_dir / "preferences.json"

        self.templates: list[SubtitleTemplate] = []
        self.projects: dict[str, ProjectConfig] = {}
        self.active_project_id: str = "default"
        self.preferences: dict[str, Any] = {
            "model_cache_dir": str(default_model_cache_dir(self.base_dir)),
            "default_output_dir": "",
            "last_import_dir": "",
            "window_geometry": None,
            "use_modelscope_mirror": True,
            "theme_mode": "light",
        }
        self.load_all()

    def load_all(self) -> None:
        self._load_templates()
        self._load_projects()
        self._load_prefs()

    def _load_templates(self) -> None:
        builtins = [
            SubtitleTemplate.preset_portrait_9_16(),
            SubtitleTemplate.preset_landscape_16_9(),
        ]
        custom: list[SubtitleTemplate] = []
        if self.templates_path.exists():
            try:
                raw = json.loads(self.templates_path.read_text(encoding="utf-8"))
                for item in raw:
                    tpl = SubtitleTemplate.from_dict(item)
                    if not tpl.is_builtin and tpl.template_id not in {b.template_id for b in builtins}:
                        custom.append(tpl)
            except Exception:
                custom = []
        self.templates = builtins + custom

    def save_templates(self) -> None:
        custom = [t.to_dict() for t in self.templates if not t.is_builtin]
        self.templates_path.write_text(
            json.dumps(custom, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_or_update_template(self, tpl: SubtitleTemplate) -> SubtitleTemplate:
        if tpl.is_builtin:
            tpl = tpl.clone(new_name=f"{tpl.name} (自定义)")
        for idx, existing in enumerate(self.templates):
            if existing.template_id == tpl.template_id:
                self.templates[idx] = tpl
                self.save_templates()
                return tpl
        self.templates.append(tpl)
        self.save_templates()
        return tpl

    def delete_template(self, template_id: str) -> bool:
        target = next((t for t in self.templates if t.template_id == template_id), None)
        if target is None or target.is_builtin:
            return False
        self.templates = [t for t in self.templates if t.template_id != template_id]
        self.save_templates()
        return True

    def rename_template(self, template_id: str, new_name: str) -> bool:
        target = next((t for t in self.templates if t.template_id == template_id), None)
        if target is None or target.is_builtin or not new_name.strip():
            return False
        target.name = new_name.strip()
        self.save_templates()
        return True

    def duplicate_template(self, template_id: str, new_name: str | None = None) -> SubtitleTemplate | None:
        source = next((t for t in self.templates if t.template_id == template_id), None)
        if source is None:
            return None
        copy_name = new_name or f"{source.name} - 副本"
        cloned = source.clone(new_name=copy_name)
        self.templates.append(cloned)
        self.save_templates()
        return cloned

    def _load_projects(self) -> None:
        self.projects = {}
        if self.projects_path.exists():
            try:
                raw = json.loads(self.projects_path.read_text(encoding="utf-8"))
                self.active_project_id = raw.get("active_project_id", "default")
                for pid, pdata in raw.get("projects", {}).items():
                    self.projects[pid] = ProjectConfig.from_dict(pdata)
            except Exception:
                self.projects = {}
        if "default" not in self.projects:
            self.projects["default"] = ProjectConfig(
                project_id="default",
                project_name="默认短剧项目",
            )
        if self.active_project_id not in self.projects:
            self.active_project_id = "default"

    def save_projects(self) -> None:
        payload = {
            "active_project_id": self.active_project_id,
            "projects": {pid: proj.to_dict() for pid, proj in self.projects.items()},
        }
        self.projects_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_active_project(self) -> ProjectConfig:
        return self.projects[self.active_project_id]

    def create_project(self, name: str) -> ProjectConfig:
        pid = f"proj_{uuid.uuid4().hex[:8]}"
        proj = ProjectConfig(project_id=pid, project_name=name.strip() or "未命名项目")
        self.projects[pid] = proj
        self.active_project_id = pid
        self.save_projects()
        return proj

    def switch_project(self, project_id: str) -> ProjectConfig:
        if project_id in self.projects:
            self.active_project_id = project_id
            self.save_projects()
        return self.projects[self.active_project_id]

    def delete_project(self, project_id: str) -> bool:
        if project_id == "default" or project_id not in self.projects:
            return False
        del self.projects[project_id]
        if self.active_project_id == project_id:
            self.active_project_id = "default"
        self.save_projects()
        return True

    def _load_prefs(self) -> None:
        if self.prefs_path.exists():
            try:
                raw = json.loads(self.prefs_path.read_text(encoding="utf-8"))
                self.preferences.update(raw)
            except Exception:
                pass
        Path(self.preferences["model_cache_dir"]).mkdir(parents=True, exist_ok=True)

    def save_prefs(self) -> None:
        self.prefs_path.write_text(
            json.dumps(self.preferences, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
