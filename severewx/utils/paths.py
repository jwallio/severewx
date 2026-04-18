"""Filesystem path helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from severewx.config import AppSettings


@dataclass(slots=True)
class DataPaths:
    root: Path
    raw: Path
    interim: Path
    processed: Path
    feature_archive: Path
    archive_metadata: Path
    labels: Path
    models: Path
    outputs: Path
    maps: Path
    verification: Path
    archive: Path


def build_paths(settings: AppSettings) -> DataPaths:
    root = settings.root

    def resolve(key: str) -> Path:
        return (root / settings.get(key)).resolve()

    paths = DataPaths(
        root=root,
        raw=resolve("paths.raw"),
        interim=resolve("paths.interim"),
        processed=resolve("paths.processed"),
        feature_archive=resolve("paths.feature_archive"),
        archive_metadata=resolve("paths.archive_metadata"),
        labels=resolve("paths.labels"),
        models=resolve("paths.models"),
        outputs=resolve("paths.outputs"),
        maps=resolve("paths.maps"),
        verification=resolve("paths.verification"),
        archive=resolve("paths.archive"),
    )
    for path in asdict(paths).values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    return paths


def dated_filename(prefix: str, date: str, suffix: str, cycle: str | None = None) -> str:
    cycle_part = f"_{cycle}" if cycle else ""
    return f"{prefix}_{date}{cycle_part}.{suffix.lstrip('.')}"
