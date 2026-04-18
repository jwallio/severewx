"""Model persistence."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

from severewx.utils.paths import DataPaths


def model_path(paths: DataPaths, hazard: str) -> Path:
    return paths.models / f"{hazard}_model.joblib"


def save_model_artifact(artifact: dict[str, Any], paths: DataPaths, hazard: str) -> Path:
    path = model_path(paths, hazard)
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)
    return path


def load_model_artifact(paths: DataPaths, hazard: str) -> dict[str, Any]:
    with model_path(paths, hazard).open("rb") as handle:
        return pickle.load(handle)
