"""Application settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class AppSettings:
    """Wrapper around the merged settings dictionary."""

    raw: dict[str, Any]

    def get(self, key: str, default: Any | None = None) -> Any:
        value: Any = self.raw
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    @property
    def root(self) -> Path:
        return Path(self.get("paths.root", ".")).resolve()


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    defaults_path = Path(__file__).with_name("defaults.yaml")
    with defaults_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle) or {}

    candidate_paths: list[Path] = []
    if config_path is not None:
        candidate_paths.append(Path(config_path))
    candidate_paths.append(Path("config.yaml"))
    env_path = os.environ.get("SEVEREWX_CONFIG")
    if env_path:
        candidate_paths.append(Path(env_path))

    for candidate in candidate_paths:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                override = yaml.safe_load(handle) or {}
            config = _deep_merge(config, override)

    return AppSettings(raw=config)
