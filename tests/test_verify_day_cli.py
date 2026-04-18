import os
from pathlib import Path
from types import SimpleNamespace

from severewx.cli.verify_day import (
    _latest_label_cube,
    _latest_metadata_for_date,
    _latest_outbreak_table,
    _latest_prediction_for_date,
)


def test_verify_day_artifact_selection_prefers_newest_mtime(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    outputs_dir = tmp_path / "outputs"
    labels_dir.mkdir(parents=True)
    outputs_dir.mkdir(parents=True)

    older_label = labels_dir / "labels_2024-04-26_2024-04-26.nc"
    newer_label = labels_dir / "labels_2024-03-01_2025-06-30.nc"
    older_outbreak = labels_dir / "outbreaks_2024-04-26_2024-04-26.parquet"
    newer_outbreak = labels_dir / "outbreaks_2024-03-01_2025-06-30.parquet"
    older_pred = outputs_dir / "forecast_products_2024-04-26_00.nc"
    newer_pred = outputs_dir / "forecast_products_2024-04-26_12.nc"
    older_meta = outputs_dir / "forecast_metadata_2024-04-26_00.json"
    newer_meta = outputs_dir / "forecast_metadata_2024-04-26_12.json"

    for file_path in [
        older_label,
        newer_label,
        older_outbreak,
        newer_outbreak,
        older_pred,
        newer_pred,
        older_meta,
        newer_meta,
    ]:
        file_path.write_text("x", encoding="utf-8")

    os.utime(older_label, (1, 1))
    os.utime(newer_label, (2, 2))
    os.utime(older_outbreak, (1, 1))
    os.utime(newer_outbreak, (2, 2))
    os.utime(older_pred, (1, 1))
    os.utime(newer_pred, (2, 2))
    os.utime(older_meta, (1, 1))
    os.utime(newer_meta, (2, 2))

    paths = SimpleNamespace(labels=labels_dir, outputs=outputs_dir)
    assert _latest_label_cube(paths) == newer_label
    assert _latest_outbreak_table(paths) == newer_outbreak
    assert _latest_prediction_for_date(paths, "2024-04-26") == newer_pred
    assert _latest_metadata_for_date(paths, "2024-04-26") == newer_meta
