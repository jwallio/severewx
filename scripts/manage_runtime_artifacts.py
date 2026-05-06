"""Inventory and package local runtime artifacts.

The repository intentionally ignores ``data/``. This helper makes that local
state auditable without accidentally committing generated forecast data or model
binaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


RUNTIME_DIRS = (
    "data/models",
    "data/model_bundles",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/labels",
    "data/outputs",
    "data/archive",
)
CACHE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}
SKIP_CLEANUP_DIR_NAMES = {".git", ".venv"}
MODEL_PATTERNS = ("*_model.joblib", "analog_reference.parquet", "training_data_summary.json")


@dataclass(slots=True)
class ArtifactRow:
    path: str
    category: str
    size_bytes: int
    modified_utc: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _category(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        return "external"
    for directory in RUNTIME_DIRS:
        if relative == directory or relative.startswith(f"{directory}/"):
            return directory
    if path.name in CACHE_DIR_NAMES:
        return "cache"
    return "other"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> list[ArtifactRow]:
    rows: list[ArtifactRow] = []
    for directory in [root / value for value in RUNTIME_DIRS if (root / value).exists()]:
        for file_path in sorted(path for path in directory.rglob("*") if path.is_file()):
            stat = file_path.stat()
            rows.append(
                ArtifactRow(
                    path=file_path.relative_to(root).as_posix(),
                    category=_category(file_path, root),
                    size_bytes=int(stat.st_size),
                    modified_utc=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                )
            )
    return rows


def write_inventory(root: Path, output_dir: Path) -> tuple[Path, Path]:
    rows = inventory(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "runtime_artifact_inventory.json"
    csv_path = output_dir / "runtime_artifact_inventory.csv"
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "category", "size_bytes", "modified_utc"])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return json_path, csv_path


def clean_caches(root: Path, *, dry_run: bool) -> list[Path]:
    removed: list[Path] = []
    for current_root, dir_names, _file_names in os.walk(root):
        dir_names[:] = [name for name in dir_names if name not in SKIP_CLEANUP_DIR_NAMES]
        for name in list(dir_names):
            if name not in CACHE_DIR_NAMES:
                continue
            cache_dir = Path(current_root) / name
            removed.append(cache_dir)
            if not dry_run:
                shutil.rmtree(cache_dir)
            dir_names.remove(name)
    return removed


def bundle_models(root: Path, output_dir: Path) -> Path:
    model_dir = root / "data" / "models"
    if not model_dir.exists():
        raise FileNotFoundError(f"missing model directory: {model_dir}")
    files: list[Path] = []
    for pattern in MODEL_PATTERNS:
        files.extend(sorted(model_dir.glob(pattern)))
    files = sorted(dict.fromkeys(path for path in files if path.is_file()))
    if not files:
        raise FileNotFoundError(f"no model artifacts found in {model_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    zip_path = output_dir / f"severewx_model_bundle_{stamp}.zip"
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_root": str(root),
        "files": [
            {
                "path": file_path.relative_to(root).as_posix(),
                "size_bytes": file_path.stat().st_size,
                "sha256": _file_digest(file_path),
            }
            for file_path in files
        ],
    }
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, file_path.relative_to(root).as_posix())
        archive.writestr("model_bundle_manifest.json", json.dumps(manifest, indent=2))
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory, clean, and package local severewx runtime artifacts.")
    parser.add_argument("--root", type=Path, default=_repo_root(), help="Repository root. Defaults to this script's parent repo.")
    parser.add_argument("--inventory", action="store_true", help="Write data/model_bundles runtime inventory JSON and CSV.")
    parser.add_argument("--clean-caches", action="store_true", help="Remove transient Python and pytest cache folders.")
    parser.add_argument("--bundle-models", action="store_true", help="Create a zip bundle of local model artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Show cleanup targets without deleting them.")
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = root / "data" / "model_bundles"
    if not (args.inventory or args.clean_caches or args.bundle_models):
        parser.error("choose at least one action: --inventory, --clean-caches, or --bundle-models")
    if args.inventory:
        json_path, csv_path = write_inventory(root, output_dir)
        print(f"wrote inventory json={json_path}")
        print(f"wrote inventory csv={csv_path}")
    if args.clean_caches:
        removed = clean_caches(root, dry_run=args.dry_run)
        action = "would remove" if args.dry_run else "removed"
        for path in removed:
            print(f"{action} {path}")
        print(f"{action} {len(removed)} cache directories")
    if args.bundle_models:
        zip_path = bundle_models(root, output_dir)
        print(f"wrote model bundle {zip_path}")


if __name__ == "__main__":
    main()
