import json
import sys

import yaml

from severewx.cli.validate_staged_gfs import main as validate_staged_gfs_main
from severewx.config import load_settings
from severewx.ingest.nomads import SyntheticForecastSource, validate_local_staged_gfs_inventory
from severewx.utils.paths import build_paths


def _settings_for_staged(tmp_path):
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["ingest"]["source"] = "local_staged_gfs"
    settings.raw["ingest"]["allow_synthetic_fallback"] = False
    settings.raw["ingest"]["allow_partial_cycle"] = True
    settings.raw["ingest"]["leads"] = [0, 6]
    settings.raw["ingest"]["local_staged_gfs"]["file_patterns"] = [
        "{root}/data/raw/staged_gfs/{date}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.nc"
    ]
    return settings


def _stage_cycle(settings, date: str, cycle: str, leads: list[int]):
    paths = build_paths(settings)
    staged_root = paths.raw / "staged_gfs" / date / cycle
    staged_root.mkdir(parents=True, exist_ok=True)
    synthetic, _ = SyntheticForecastSource().fetch_cycle(date, cycle, settings, paths)
    requested = [int(value) for value in settings.get("ingest.leads", [])]
    for lead in leads:
        lead_index = requested.index(int(lead))
        output_path = staged_root / f"gfs.t{cycle}z.pgrb2.0p25.f{int(lead):03d}.nc"
        synthetic.isel(time=lead_index).to_netcdf(output_path)


def test_validate_local_staged_gfs_inventory_reports_complete_and_partial(tmp_path) -> None:
    settings = _settings_for_staged(tmp_path)
    _stage_cycle(settings, "2026-04-09", "00", [0, 6])
    _stage_cycle(settings, "2026-04-09", "12", [0])

    report = validate_local_staged_gfs_inventory("2026-04-09", "2026-04-09", ["00", "12"], settings)
    assert report["detected_dates"] == ["2026-04-09"]
    assert report["detected_cycles"] == ["00", "12"]
    assert report["detected_lead_hours"] == [0, 6]
    assert report["status_counts"]["complete"] == 1
    assert report["status_counts"]["partial"] == 1
    assert any(row["cycle"] == "00" and row["status"] == "complete" for row in report["cycle_results"])
    assert any(row["cycle"] == "12" and row["status"] == "partial" for row in report["cycle_results"])


def test_validate_staged_gfs_cli_writes_report(tmp_path, monkeypatch) -> None:
    settings = _settings_for_staged(tmp_path)
    _stage_cycle(settings, "2026-04-09", "00", [0])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(settings.raw), encoding="utf-8")
    monkeypatch.setenv("SEVEREWX_CONFIG", str(config_path))
    monkeypatch.setattr(sys, "argv", ["validate_staged_gfs", "--start", "2026-04-09", "--cycles", "00"])

    validate_staged_gfs_main()

    report_path = build_paths(settings).interim / "staged_gfs_validation_2026-04-09_2026-04-09_00.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path.exists()
    assert payload["detected_dates"] == ["2026-04-09"]
    assert payload["detected_cycles"] == ["00"]
    assert payload["status_counts"]["partial"] == 1
    assert payload["cycle_results"][0]["available_leads"] == [0]
    assert payload["cycle_results"][0]["missing_leads"] == [6]
