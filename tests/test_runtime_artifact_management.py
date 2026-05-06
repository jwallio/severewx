from pathlib import Path

from scripts.manage_runtime_artifacts import bundle_models, clean_caches, inventory, write_inventory


def test_runtime_artifact_inventory_and_cache_cleanup(tmp_path: Path) -> None:
    model_dir = tmp_path / "data" / "models"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "tornado_model.joblib"
    model_file.write_bytes(b"model")
    cache_dir = tmp_path / "severewx" / "__pycache__"
    cache_dir.mkdir(parents=True)
    (cache_dir / "module.pyc").write_bytes(b"cache")

    rows = inventory(tmp_path)
    assert any(row.path == "data/models/tornado_model.joblib" for row in rows)
    json_path, csv_path = write_inventory(tmp_path, tmp_path / "data" / "model_bundles")
    assert json_path.exists()
    assert csv_path.exists()

    dry_run_removed = clean_caches(tmp_path, dry_run=True)
    assert cache_dir in dry_run_removed
    assert cache_dir.exists()
    removed = clean_caches(tmp_path, dry_run=False)
    assert cache_dir in removed
    assert not cache_dir.exists()


def test_model_bundle_contains_manifest_and_artifacts(tmp_path: Path) -> None:
    model_dir = tmp_path / "data" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "tornado_model.joblib").write_bytes(b"model")
    (model_dir / "training_data_summary.json").write_text("{}", encoding="utf-8")

    bundle_path = bundle_models(tmp_path, tmp_path / "data" / "model_bundles")

    assert bundle_path.exists()
    import zipfile

    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
    assert "data/models/tornado_model.joblib" in names
    assert "data/models/training_data_summary.json" in names
    assert "model_bundle_manifest.json" in names
