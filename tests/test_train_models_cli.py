import os
from pathlib import Path

from severewx.cli.train_models import _latest_matching_file


def test_latest_matching_file_prefers_newest_mtime(tmp_path: Path) -> None:
    older = tmp_path / "labels_2024-04-26_2024-04-26.nc"
    newer = tmp_path / "labels_2024-03-01_2025-06-30.nc"
    older.write_text("older", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    assert _latest_matching_file(tmp_path, "labels_*.nc") == newer
