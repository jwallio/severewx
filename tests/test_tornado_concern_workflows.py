from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from severewx.cli import build_tornado_concern_product as product_cli
from severewx.cli import build_tornado_concern_public_candidates as public_candidates_cli
from severewx.cli import build_tornado_concern_spc_comparison as spc_comparison_cli
from severewx.cli import compare_tornado_concern_component_delta as delta_cli
from severewx.cli import prepare_tornado_concern_public_prototype as prototype_cli
from severewx.cli import prototype_tornado_concern_map_styles as style_restart_cli
from severewx.cli import prototype_tornado_concern_coherent_field as coherent_field_cli
from severewx.cli import rescore_tornado_concern_eval_csv as rescore_cli
from severewx.cli import run_tornado_concern_checkpoint as checkpoint_cli
from severewx.cli import run_tornado_concern_recovery_wave as wave_cli
from severewx.cli import select_tornado_concern_hard_negative_candidates as hard_negative_candidates_cli
from severewx.cli import tornado_concern_eval as eval_cli
from severewx.cli import tornado_concern_failure_diagnostics as diagnostics_cli
from severewx.cli import tornado_concern_failure_review as review_cli
from severewx.cli import tornado_concern_field_diagnostics as field_diagnostics_cli
from severewx.cli import tornado_concern_product_audit as product_audit_cli
from severewx.cli import tornado_concern_product_quality as product_quality_cli
from severewx.config import AppSettings
from severewx.models.tornado_concern import coherent_tornado_concern_grid, envelope_tornado_concern_grid


def _workflow_paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        outputs=tmp_path / "outputs",
        verification=tmp_path / "verification",
        labels=tmp_path / "labels",
        interim=tmp_path / "interim",
    )


def test_build_tornado_concern_product_generates_bundle(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    prediction = xr.Dataset(
        data_vars={
            "tornado_concern_prob": (
                ("time", "lat", "lon"),
                np.array(
                    [
                        [[0.05, 0.10], [0.02, 0.01]],
                        [[0.15, 0.35], [0.08, 0.03]],
                    ],
                    dtype=float,
                ),
            ),
            "outbreak_risk": (
                ("time", "lat", "lon"),
                np.array(
                    [
                        [[0.02, 0.05], [0.01, 0.01]],
                        [[0.10, 0.30], [0.06, 0.02]],
                    ],
                    dtype=float,
                ),
            ),
            "sig_tor_support": (
                ("time", "lat", "lon"),
                np.array(
                    [
                        [[0.10, 0.20], [0.05, 0.01]],
                        [[0.30, 0.60], [0.15, 0.04]],
                    ],
                    dtype=float,
                ),
            ),
        },
        coords={
            "time": pd.to_datetime(["2024-04-26T00:00:00", "2024-04-27T00:00:00"]),
            "lat": [35.0, 36.0],
            "lon": [-98.0, -97.0],
        },
    )
    prediction.to_netcdf(paths.outputs / "forecast_products_2024-04-26_00.nc")
    (paths.outputs / "forecast_metadata_2024-04-26_00.json").write_text(
        '{"run_summary":{"init_date":"2024-04-26"},"ingest_summary":{"source":"local_staged_gfs"}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(product_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(product_cli, "build_paths", lambda _settings: paths)

    outdir = tmp_path / "product"
    manifest = tmp_path / "manifest.json"
    product_cli.main(
        [
            "--date",
            "2024-04-26",
            "--cycle",
            "00",
            "--outdir",
            str(outdir),
            "--field",
            "tornado_concern_prob",
            "--map-style",
            "contours",
            "--map-domain",
            "conus",
            "--archive-manifest",
            str(manifest),
        ]
    )

    image_path = outdir / "tornado_concern_init_2024-04-26_00z_valid_2024-04-26.png"
    summary_path = outdir / "tornado_concern_init_2024-04-26_00z_valid_2024-04-26.md"
    metadata_path = outdir / "tornado_concern_init_2024-04-26_00z_valid_2024-04-26.json"
    assert image_path.exists()
    assert summary_path.exists()
    metadata = pd.read_json(metadata_path, typ="series")
    assert metadata["variant"] == "baseline"
    assert metadata["date"] == "2024-04-26"
    assert metadata["cycle"] == "00"
    assert metadata["valid_date"] == "2024-04-26"
    assert metadata["valid_date_start"] == "2024-04-26"
    assert metadata["valid_date_end"] == "2024-04-26"
    assert metadata["valid_day_count"] == 1
    assert metadata["map_style"] == "contours"
    assert metadata["color_ramp"] == "red_gradient"
    assert metadata["display_transform"] == "neighborhood_supported_public_field"
    assert metadata["max_public_display_tornado_concern_prob"] <= metadata["max_tornado_concern_prob"]
    assert bool(metadata["public_ready"])
    assert metadata["audit_status"] == "ok"
    assert metadata["publication_status"] == "public_candidate"
    assert Path(str(metadata["main_image_path"])).name == image_path.name
    manifest_payload = pd.read_json(manifest)
    assert str(manifest_payload.loc[0, "date"]).startswith("2024-04-26")


def test_build_tornado_concern_product_defaults_to_hybrid_outlook_product(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    shape = (1, 16, 16)
    sig_tor_support = np.zeros(shape, dtype=float)
    tornado_favored_overlap = np.zeros(shape, dtype=float)
    scp_proxy = np.zeros(shape, dtype=float)
    sig_tor_support[:, 5:11, 5:11] = 1.6
    tornado_favored_overlap[:, 5:11, 5:11] = 2.8
    scp_proxy[:, 5:11, 5:11] = 1.1
    xr.Dataset(
        {
            "tornado_concern_prob": (("time", "lat", "lon"), np.zeros(shape, dtype=float)),
            "sig_tor_support": (("time", "lat", "lon"), sig_tor_support),
            "tornado_favored_overlap": (("time", "lat", "lon"), tornado_favored_overlap),
            "scp_proxy": (("time", "lat", "lon"), scp_proxy),
        },
        coords={
            "time": pd.to_datetime(["2024-05-19T00:00:00"]),
            "lat": np.linspace(30.0, 45.0, 16),
            "lon": np.linspace(-105.0, -85.0, 16),
        },
    ).to_netcdf(paths.outputs / "forecast_products_2024-05-19_00.nc")

    monkeypatch.setattr(product_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(product_cli, "build_paths", lambda _settings: paths)

    outdir = tmp_path / "default_product"
    product_cli.main(["--date", "2024-05-19", "--cycle", "00", "--outdir", str(outdir)])

    metadata = pd.read_json(outdir / "tornado_concern_init_2024-05-19_00z_valid_2024-05-19.json", typ="series")
    assert metadata["field_name"] == "tornado_environment_outlook_hybrid"
    assert metadata["variant"] == "environment_outlook_hybrid"
    assert metadata["map_style"] == "outlook"
    assert metadata["map_domain"] == "regional"
    assert metadata["display_preset_requested"] == "auto"


def test_prototype_tornado_concern_map_styles_generates_comparison_board(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    prediction = xr.Dataset(
        data_vars={
            "tornado_concern_prob": (
                ("time", "lat", "lon"),
                np.array(
                    [
                        [[0.01, 0.03, 0.00], [0.02, 0.08, 0.01], [0.00, 0.02, 0.01]],
                        [[0.02, 0.05, 0.01], [0.04, 0.20, 0.03], [0.01, 0.04, 0.02]],
                    ],
                    dtype=float,
                ),
            )
        },
        coords={
            "time": pd.to_datetime(["2024-05-19T00:00:00", "2024-05-21T00:00:00"]),
            "lat": [35.0, 36.0, 37.0],
            "lon": [-99.0, -98.0, -97.0],
        },
    )
    prediction.to_netcdf(paths.outputs / "forecast_products_2024-05-19_00.nc")
    monkeypatch.setattr(style_restart_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(style_restart_cli, "build_paths", lambda _settings: paths)

    outdir = tmp_path / "style_restart"
    style_restart_cli.main(
        [
            "--date",
            "2024-05-19",
            "--cycle",
            "00",
            "--valid-date-mode",
            "best",
            "--outdir",
            str(outdir),
        ]
    )

    manifest = pd.read_csv(outdir / "map_style_restart_manifest.csv")
    assert manifest.loc[0, "valid_date"] == "2024-05-21"
    assert Path(manifest.loc[0, "board_path"]).exists()
    assert "Tornado Concern Map Style Restart" in (outdir / "README.md").read_text(encoding="utf-8")


def test_coherent_tornado_concern_grid_spreads_seed_only_with_support() -> None:
    raw = np.zeros((9, 9), dtype=float)
    raw[4, 4] = 0.50
    support = np.zeros((9, 9), dtype=float)
    support[3:6, 3:6] = 1.0

    coherent = coherent_tornado_concern_grid(raw, sig_tor_support=support, radius=3)

    assert float(coherent[4, 4]) == 0.50
    assert float(coherent[4, 5]) > 0.05
    assert float(coherent[0, 0]) == 0.0
    assert int(np.count_nonzero(coherent >= 0.02)) > int(np.count_nonzero(raw >= 0.02))


def test_envelope_tornado_concern_grid_builds_broad_low_tier_from_support() -> None:
    raw = np.zeros((9, 9), dtype=float)
    raw[4, 4] = 0.50
    support = np.zeros((9, 9), dtype=float)
    support[2:7, 2:7] = 1.00

    envelope = envelope_tornado_concern_grid(raw, sig_tor_support=support, tornado_favored_overlap=support, scp_proxy=support)

    assert int(np.count_nonzero(envelope >= 0.02)) > int(np.count_nonzero(raw >= 0.02))
    assert float(envelope[2, 2]) >= 0.02
    assert float(envelope[4, 4]) >= 0.50


def test_prototype_tornado_concern_coherent_field_generates_board(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    raw = np.zeros((1, 9, 9), dtype=float)
    raw[:, 4, 4] = 0.50
    support = np.zeros((1, 9, 9), dtype=float)
    support[:, 3:6, 3:6] = 1.0
    prediction = xr.Dataset(
        data_vars={
            "tornado_concern_prob": (("time", "lat", "lon"), raw),
            "sig_tor_support": (("time", "lat", "lon"), support),
            "tornado_favored_overlap": (("time", "lat", "lon"), support),
            "scp_proxy": (("time", "lat", "lon"), support),
            "outbreak_risk": (("time", "lat", "lon"), support),
        },
        coords={"time": pd.to_datetime(["2024-05-19T00:00:00"]), "lat": np.arange(9), "lon": np.arange(9)},
    )
    prediction.to_netcdf(paths.outputs / "forecast_products_2024-05-19_00.nc")
    monkeypatch.setattr(coherent_field_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(coherent_field_cli, "build_paths", lambda _settings: paths)

    outdir = tmp_path / "coherent"
    coherent_field_cli.main(["--date", "2024-05-19", "--cycle", "00", "--outdir", str(outdir)])

    manifest = pd.read_csv(outdir / "coherent_field_manifest.csv")
    assert int(manifest.loc[0, "coherent_cells_ge_02pct"]) > int(manifest.loc[0, "raw_cells_ge_02pct"])
    assert int(manifest.loc[0, "envelope_cells_ge_02pct"]) >= int(manifest.loc[0, "coherent_cells_ge_02pct"])
    assert Path(manifest.loc[0, "board_path"]).exists()


def test_build_tornado_concern_product_supports_explicit_24h_valid_date(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    prediction = xr.Dataset(
        data_vars={
            "tornado_concern_prob": (
                ("time", "lat", "lon"),
                np.array(
                    [
                        [[0.02, 0.03], [0.01, 0.00]],
                        [[0.30, 0.35], [0.10, 0.05]],
                    ],
                    dtype=float,
                ),
            ),
        },
        coords={
            "time": pd.to_datetime(["2024-04-26T12:00:00", "2024-04-27T12:00:00"]),
            "lat": [35.0, 36.0],
            "lon": [-98.0, -97.0],
        },
    )
    prediction.to_netcdf(paths.outputs / "forecast_products_2024-04-26_00.nc")
    monkeypatch.setattr(product_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(product_cli, "build_paths", lambda _settings: paths)

    outdir = tmp_path / "valid_day_product"
    product_cli.main(
        [
            "--date",
            "2024-04-26",
            "--cycle",
            "00",
            "--valid-date",
            "2024-04-27",
            "--field",
            "tornado_concern_prob",
            "--map-style",
            "contours",
            "--outdir",
            str(outdir),
        ]
    )

    metadata = pd.read_json(outdir / "tornado_concern_init_2024-04-26_00z_valid_2024-04-27.json", typ="series")
    assert metadata["valid_date"] == "2024-04-27"
    assert metadata["valid_date_start"] == "2024-04-27"
    assert metadata["valid_date_end"] == "2024-04-27"
    assert abs(float(metadata["max_tornado_concern_prob"]) - 0.35) < 1e-9


def test_build_tornado_concern_product_can_render_derived_environment_envelope(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    raw = np.zeros((1, 9, 9), dtype=float)
    raw[:, 4, 4] = 0.50
    support = np.zeros((1, 9, 9), dtype=float)
    support[:, 2:7, 2:7] = 1.0
    prediction = xr.Dataset(
        data_vars={
            "tornado_concern_prob": (("time", "lat", "lon"), raw),
            "sig_tor_support": (("time", "lat", "lon"), support),
            "tornado_favored_overlap": (("time", "lat", "lon"), support),
            "scp_proxy": (("time", "lat", "lon"), support),
            "outbreak_risk": (("time", "lat", "lon"), support),
        },
        coords={"time": pd.to_datetime(["2024-05-19T00:00:00"]), "lat": np.arange(9), "lon": np.arange(9)},
    )
    prediction.to_netcdf(paths.outputs / "forecast_products_2024-05-19_00.nc")
    monkeypatch.setattr(product_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(product_cli, "build_paths", lambda _settings: paths)

    outdir = tmp_path / "envelope_product"
    product_cli.main(
        [
            "--date",
            "2024-05-19",
            "--cycle",
            "00",
            "--field",
            "tornado_concern_environment_envelope",
            "--map-style",
            "outlook",
            "--outdir",
            str(outdir),
        ]
    )

    metadata = pd.read_json(outdir / "tornado_concern_init_2024-05-19_00z_valid_2024-05-19.json", typ="series")
    assert metadata["variant"] == "environment_envelope"
    assert metadata["field_name"] == "tornado_concern_environment_envelope"
    assert metadata["map_style"] == "outlook"
    assert float(metadata["max_tornado_concern_prob"]) >= 0.50
    assert (outdir / "tornado_concern_init_2024-05-19_00z_valid_2024-05-19.png").exists()


def test_build_tornado_concern_product_can_render_derived_environment_outlook(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    raw = np.zeros((1, 9, 9), dtype=float)
    raw[:, 4, 4] = 0.80
    sig_tor_support = np.zeros((1, 9, 9), dtype=float)
    tornado_favored_overlap = np.zeros((1, 9, 9), dtype=float)
    scp_proxy = np.zeros((1, 9, 9), dtype=float)
    sig_tor_support[:, 1:4, 1:4] = 1.25
    tornado_favored_overlap[:, 1:4, 1:4] = 2.50
    scp_proxy[:, 1:4, 1:4] = 0.75
    prediction = xr.Dataset(
        data_vars={
            "tornado_concern_prob": (("time", "lat", "lon"), raw),
            "sig_tor_support": (("time", "lat", "lon"), sig_tor_support),
            "tornado_favored_overlap": (("time", "lat", "lon"), tornado_favored_overlap),
            "scp_proxy": (("time", "lat", "lon"), scp_proxy),
        },
        coords={"time": pd.to_datetime(["2024-05-19T00:00:00"]), "lat": np.arange(9), "lon": np.arange(9)},
    )
    prediction.to_netcdf(paths.outputs / "forecast_products_2024-05-19_00.nc")
    monkeypatch.setattr(product_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(product_cli, "build_paths", lambda _settings: paths)

    outdir = tmp_path / "outlook_product"
    product_cli.main(
        [
            "--date",
            "2024-05-19",
            "--cycle",
            "00",
            "--field",
            "tornado_environment_outlook",
            "--map-style",
            "outlook",
            "--outdir",
            str(outdir),
        ]
    )

    metadata = pd.read_json(outdir / "tornado_concern_init_2024-05-19_00z_valid_2024-05-19.json", typ="series")
    assert metadata["variant"] == "environment_outlook"
    assert metadata["field_name"] == "tornado_environment_outlook"
    assert metadata["map_style"] == "outlook"
    assert float(metadata["max_tornado_concern_prob"]) >= 0.30
    assert "ingredient-only" in str(metadata["summary_text"])
    assert metadata["ingredient_diagnostics"]["limiting_ingredient_at_peak"] in {
        "sig_tor_support",
        "tornado_favored_overlap",
        "scp_proxy",
    }
    assert "TORP/TorNet-style object filtering" in metadata["reference_design_notes"][1]
    assert (outdir / "tornado_concern_init_2024-05-19_00z_valid_2024-05-19.png").exists()


def test_public_display_outlook_field_removes_isolated_specks() -> None:
    values = np.zeros((9, 9), dtype=float)
    values[0, 8] = 0.45
    values[2:7, 2:7] = 0.10

    display = product_cli._public_display_outlook_field(values)

    assert float(display[0, 8]) == 0.0
    assert int(np.count_nonzero(display >= 0.02)) == 25
    assert float(display[2, 2]) == 0.10


def test_display_extent_auto_zooms_to_outlook_footprint() -> None:
    lon = np.linspace(-125.0, -66.5, 20)
    lat = np.linspace(24.0, 50.0, 12)
    values = np.zeros((12, 20), dtype=float)
    values[4:7, 8:11] = 0.10

    extent = product_cli._display_extent(lon, lat, values, map_domain="regional")

    assert extent is not None
    lon_min, lon_max, lat_min, lat_max = extent
    assert lon_max - lon_min >= product_cli.REGIONAL_MIN_WIDTH_DEGREES
    assert lat_max - lat_min >= product_cli.REGIONAL_MIN_HEIGHT_DEGREES
    assert lon_min > -125.0
    assert lon_max < -66.5


def test_display_preset_auto_selects_weak_standard_and_broad() -> None:
    weak = np.zeros((12, 12), dtype=float)
    weak[3:8, 3:8] = 0.10
    standard = np.zeros((30, 30), dtype=float)
    standard[5:20, 6:24] = 0.10
    broad = np.zeros((50, 50), dtype=float)
    broad[5:45, 5:45] = 0.10

    assert product_cli._resolve_display_preset(weak, "auto") == "weak"
    assert product_cli._resolve_display_preset(standard, "auto") == "standard"
    assert product_cli._resolve_display_preset(broad, "auto") == "broad"


def test_tiered_outlook_filter_preserves_compact_high_core_in_broad_preset() -> None:
    values = np.zeros((20, 20), dtype=float)
    values[0:3, 0:3] = 0.02
    values[10:13, 10:14] = 0.10

    display = product_cli._public_display_outlook_field(values, display_preset="broad")

    assert int(np.count_nonzero(display[:3, :3] >= 0.02)) == 0
    assert int(np.count_nonzero(display >= 0.10)) == 12


def test_broad_outlook_label_levels_suppress_low_end_clutter() -> None:
    values = np.zeros((30, 30), dtype=float)
    values[4:26, 4:26] = 0.05
    values[8:22, 8:22] = 0.10
    values[12:18, 12:18] = 0.30

    levels = product_cli._visible_label_levels(values, display_preset="broad")

    assert levels[0] >= 0.10
    assert 0.30 in levels


def test_build_outlook_products_records_auto_display_presets(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)
    cases = {
        "2024-04-18": (12, (slice(3, 8), slice(3, 8)), 0.10),
        "2024-05-21": (30, (slice(5, 20), slice(6, 24)), 0.10),
        "2024-05-24": (50, (slice(5, 45), slice(5, 45)), 0.10),
    }
    for date, (size, region, value) in cases.items():
        field = np.zeros((1, size, size), dtype=float)
        field[(0, *region)] = value
        xr.Dataset(
            {"tornado_environment_outlook": (("time", "lat", "lon"), field)},
            coords={
                "time": pd.to_datetime([f"{date}T00:00:00"]),
                "lat": np.linspace(24.0, 50.0, size),
                "lon": np.linspace(-125.0, -66.5, size),
            },
        ).to_netcdf(paths.outputs / f"forecast_products_{date}_00.nc")
    monkeypatch.setattr(product_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(product_cli, "build_paths", lambda _settings: paths)
    dates_file = tmp_path / "dates.txt"
    dates_file.write_text("\n".join(cases) + "\n", encoding="utf-8")
    outdir = tmp_path / "preset_products"

    product_cli.main(
        [
            "--dates-file",
            str(dates_file),
            "--cycle",
            "00",
            "--field",
            "tornado_environment_outlook",
            "--map-style",
            "outlook",
            "--map-domain",
            "regional",
            "--outdir",
            str(outdir),
        ]
    )

    presets = {}
    for date in cases:
        metadata = pd.read_json(outdir / f"tornado_concern_init_{date}_00z_valid_{date}.json", typ="series")
        presets[date] = metadata["display_preset_effective"]
        assert metadata["map_domain"] == "regional"
        assert int(metadata["contour_label_count"]) == 0
    assert presets == {"2024-04-18": "weak", "2024-05-21": "standard", "2024-05-24": "broad"}


def test_product_quality_cli_scores_and_splits_review_folders(tmp_path: Path) -> None:
    products_dir = tmp_path / "products"
    products_dir.mkdir()
    cases = [
        ("2024-05-21", 0.45, 900, 300, 40, 900, 10, "standard"),
        ("2024-05-30", 0.05, 120, 0, 0, 90, 0, "weak"),
        ("2024-05-24", 0.45, 1800, 420, 60, 1700, 22, "broad"),
        ("2024-06-01", 0.50, 800, 120, 2, 700, 8, "standard"),
    ]
    for date, max_value, ge02, ge10, ge35, largest, labels, preset in cases:
        image_path = products_dir / f"{date}.png"
        summary_path = products_dir / f"{date}.md"
        metadata_path = products_dir / f"{date}.json"
        image_path.write_bytes(b"png")
        summary_path.write_text(f"# {date}\n", encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                {
                    "date": date,
                    "valid_date": date,
                    "main_image_path": str(image_path),
                    "summary_path": str(summary_path),
                    "map_domain": "regional",
                    "display_preset_effective": preset,
                    "max_tornado_concern_prob": max_value,
                    "public_display_grid_cells_ge_02pct": ge02,
                    "public_display_grid_cells_ge_05pct": ge02,
                    "public_display_grid_cells_ge_10pct": ge10,
                    "public_display_grid_cells_ge_20pct": ge10,
                    "public_display_grid_cells_ge_35pct": ge35,
                    "display_object_count_ge_02pct": 1,
                    "display_largest_object_cells_ge_02pct": largest,
                    "contour_label_count": labels,
                }
            ),
            encoding="utf-8",
        )

    output_csv = tmp_path / "quality.csv"
    failure_queue_csv = tmp_path / "failure_queue.csv"
    product_quality_cli.main(
        [
            "--products-dir",
            str(products_dir),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(tmp_path / "quality.md"),
            "--best-dir",
            str(tmp_path / "best"),
            "--needs-dir",
            str(tmp_path / "needs"),
            "--best-count",
            "1",
            "--needs-count",
            "1",
            "--failure-queue-csv",
            str(failure_queue_csv),
            "--failure-queue-md",
            str(tmp_path / "failure_queue.md"),
        ]
    )

    frame = pd.read_csv(output_csv)
    assert frame.iloc[0]["date"] == "2024-05-21"
    weak_row = frame.loc[frame["date"] == "2024-05-30"].iloc[0]
    assert "weak_or_sparse_display" in weak_row["quality_issues"]
    broad_row = frame.loc[frame["date"] == "2024-05-24"].iloc[0]
    assert "broad_low_end_footprint" in broad_row["quality_issues"]
    queue = pd.read_csv(failure_queue_csv)
    assert set(queue["queue"]) == {"weak_or_sparse_display", "broad_low_end_footprint", "tiny_high_end_area"}
    assert (tmp_path / "best" / "2024-05-21.png").exists()
    assert (tmp_path / "needs" / "2024-05-30.png").exists()
    assert "Queue Counts" in (tmp_path / "failure_queue.md").read_text(encoding="utf-8")


def test_field_diagnostics_cli_compares_outlook_challenger(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)
    size = 24
    sig = np.zeros((1, size, size), dtype=float)
    overlap = np.zeros((1, size, size), dtype=float)
    scp = np.zeros((1, size, size), dtype=float)
    sig[:, 3:21, 3:21] = 1.25
    overlap[:, 3:21, 3:21] = 2.50
    scp[:, 3:21, 3:21] = 0.75
    sig[:, 9:15, 9:15] = 1.9
    overlap[:, 9:15, 9:15] = 3.4
    scp[:, 9:15, 9:15] = 1.2
    xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), sig),
            "tornado_favored_overlap": (("time", "lat", "lon"), overlap),
            "scp_proxy": (("time", "lat", "lon"), scp),
        },
        coords={
            "time": pd.to_datetime(["2024-05-08T00:00:00"]),
            "lat": np.linspace(24.0, 50.0, size),
            "lon": np.linspace(-125.0, -66.5, size),
        },
    ).to_netcdf(paths.outputs / "forecast_products_2024-05-08_00.nc")
    monkeypatch.setattr(field_diagnostics_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(field_diagnostics_cli, "build_paths", lambda _settings: paths)
    dates_file = tmp_path / "dates.txt"
    dates_file.write_text("2024-05-08\n", encoding="utf-8")
    outdir = tmp_path / "diagnostics"

    field_diagnostics_cli.main(["--dates-file", str(dates_file), "--cycle", "00", "--outdir", str(outdir)])

    frame = pd.read_csv(outdir / "tornado_environment_outlook_field_diagnostics.csv")
    assert frame.loc[0, "status"] == "ok"
    assert int(frame.loc[0, "delta_ge02"]) < 0
    assert (outdir / "tornado_environment_outlook_field_diagnostics.md").exists()


def test_build_tornado_concern_product_best_valid_date_mode_uses_peak_day(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    prediction = xr.Dataset(
        data_vars={
            "tornado_concern_prob": (
                ("time", "lat", "lon"),
                np.array(
                    [
                        [[0.02, 0.03], [0.01, 0.00]],
                        [[0.30, 0.45], [0.10, 0.05]],
                    ],
                    dtype=float,
                ),
            ),
        },
        coords={
            "time": pd.to_datetime(["2024-04-26T12:00:00", "2024-04-28T12:00:00"]),
            "lat": [35.0, 36.0],
            "lon": [-98.0, -97.0],
        },
    )
    prediction.to_netcdf(paths.outputs / "forecast_products_2024-04-26_00.nc")
    monkeypatch.setattr(product_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(product_cli, "build_paths", lambda _settings: paths)

    outdir = tmp_path / "best_valid_day_product"
    product_cli.main(
        [
            "--date",
            "2024-04-26",
            "--cycle",
            "00",
            "--valid-date-mode",
            "best",
            "--field",
            "tornado_concern_prob",
            "--outdir",
            str(outdir),
        ]
    )

    metadata_path = outdir / "tornado_concern_init_2024-04-26_00z_valid_2024-04-28.json"
    assert metadata_path.exists()
    metadata = pd.read_json(metadata_path, typ="series")
    assert metadata["valid_date"] == "2024-04-28"
    assert abs(float(metadata["max_tornado_concern_prob"]) - 0.45) < 1e-9


def test_build_tornado_concern_product_can_build_dates_file_in_one_process(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    for date in ["2024-04-26", "2024-05-19"]:
        prediction = xr.Dataset(
            data_vars={
                "tornado_concern_prob": (
                    ("time", "lat", "lon"),
                    np.array([[[0.05, 0.10], [0.02, 0.01]], [[0.15, 0.35], [0.08, 0.03]]], dtype=float),
                ),
            },
            coords={
                "time": pd.to_datetime([f"{date}T00:00:00", f"{date}T12:00:00"]),
                "lat": [35.0, 36.0],
                "lon": [-98.0, -97.0],
            },
        )
        prediction.to_netcdf(paths.outputs / f"forecast_products_{date}_00.nc")

    monkeypatch.setattr(product_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(product_cli, "build_paths", lambda _settings: paths)

    dates_file = tmp_path / "dates.txt"
    dates_file.write_text("2024-04-26\n2024-05-19\n", encoding="utf-8")
    outdir = tmp_path / "product_batch"
    product_cli.main(
        [
            "--dates-file",
            str(dates_file),
            "--cycle",
            "00",
            "--field",
            "tornado_concern_prob",
            "--outdir",
            str(outdir),
            "--map-style",
            "contours",
        ]
    )

    assert (outdir / "tornado_concern_init_2024-04-26_00z_valid_2024-04-26.png").exists()
    assert (outdir / "tornado_concern_init_2024-05-19_00z_valid_2024-05-19.png").exists()
    metadata = pd.read_json(outdir / "tornado_concern_init_2024-05-19_00z_valid_2024-05-19.json", typ="series")
    assert metadata["map_style"] == "contours"


def test_build_tornado_concern_product_dates_file_tolerates_utf8_bom(tmp_path: Path) -> None:
    dates_file = tmp_path / "dates.txt"
    dates_file.write_text("\ufeff2024-05-19\n2024-05-23\n", encoding="utf-8")

    assert product_cli._read_dates_file(dates_file) == ["2024-05-19", "2024-05-23"]


def test_tornado_concern_failure_review_outputs_selected_cases(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    for init_date in ["2024-05-26", "2024-05-11", "2024-04-18"]:
        (paths.outputs / f"forecast_products_{init_date}_00.nc").write_text("placeholder", encoding="utf-8")
        (paths.verification / f"{init_date}_verification.json").write_text("{}", encoding="utf-8")

    eval_frame = pd.DataFrame(
        [
            {
                "init_date": "2024-05-26",
                "day_rank_within_init": 1,
                "ranking_tornado_concern_score": 1.2,
                "top_valid_date": "2024-05-27",
                "top_observed_category": "hail_outbreak_day",
                "best_tornado_valid_date": "2024-05-26",
                "hail_outranks_tornado_failure": True,
                "top_day_category_mismatch": True,
                "top_minus_best_tornado_score": 0.90,
                "failure_driver": "generic_core_beats_tornado_core",
                "source_root_cause": "generic_overlap_contamination",
                "core_root_cause": "diffuse_core_beats_compact_tornado_core",
            },
            {
                "init_date": "2024-05-11",
                "day_rank_within_init": 1,
                "ranking_tornado_concern_score": 0.8,
                "top_valid_date": "2024-05-14",
                "top_observed_category": "wind_mcs_outbreak_day",
                "best_tornado_valid_date": "2024-05-11",
                "hail_outranks_tornado_failure": False,
                "top_day_category_mismatch": True,
                "top_minus_best_tornado_score": 0.40,
                "failure_driver": "context_swamps_core",
                "source_root_cause": "upstream_tornado_signal_absent",
                "core_root_cause": "none",
            },
            {
                "init_date": "2024-04-18",
                "day_rank_within_init": 1,
                "ranking_tornado_concern_score": 0.3,
                "top_valid_date": "2024-04-18",
                "top_observed_category": "tornado_outbreak_day",
                "best_tornado_valid_date": "2024-04-19",
                "hail_outranks_tornado_failure": False,
                "top_day_category_mismatch": False,
                "top_minus_best_tornado_score": 0.12,
                "failure_driver": "near_tie_small_margin",
                "source_root_cause": "upstream_tornado_signal_absent",
                "core_root_cause": "upstream_core_absent",
            },
        ]
    )
    eval_csv = tmp_path / "eval.csv"
    eval_frame.to_csv(eval_csv, index=False)
    ready_dates = tmp_path / "ready_dates.txt"
    ready_dates.write_text("2024-05-26\n2024-05-11\n2024-04-18\n", encoding="utf-8")

    monkeypatch.setattr(review_cli, "load_settings", lambda: object())
    monkeypatch.setattr(review_cli, "build_paths", lambda _settings: paths)

    outdir = tmp_path / "review"
    review_cli.main(
        [
            "--eval-csv",
            str(eval_csv),
            "--ready-dates-file",
            str(ready_dates),
            "--outdir",
            str(outdir),
            "--max-cases",
            "2",
            "--include-hail-outranks",
            "--include-category-mismatches",
            "--include-missed-tornado-first",
        ]
    )

    cases = pd.read_csv(outdir / "tornado_concern_failure_review_cases.csv")
    assert list(cases["init_date"]) == ["2024-05-26", "2024-05-11"]
    assert list(cases["primary_failure_type"]) == ["hail_outranks", "category_mismatch"]
    markdown = (outdir / "tornado_concern_failure_review.md").read_text(encoding="utf-8")
    assert "## Selected Cases" in markdown


def test_rescore_tornado_concern_eval_csv_outputs_ranked_challenger(tmp_path: Path, monkeypatch) -> None:
    input_csv = tmp_path / "baseline.csv"
    output_csv = tmp_path / "rescored.csv"
    output_md = tmp_path / "rescored.md"
    pd.DataFrame(
        [
            {
                "init_date": "2024-05-19",
                "valid_date": "2024-05-22",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "is_real_ingest": True,
                "raw_core_variant_name": "baseline",
                "raw_core_mask_factor": 1.0,
                "raw_core_final_value": 0.18,
                "raw_core_to_context_ratio": 0.18,
                "core_variant_name": "baseline",
                "source_variant_name": "baseline",
                "component_variant_name": "baseline",
                "score_variant_name": "baseline",
                "support_top": 0.22,
                "normalized_synoptic_support": 0.90,
                "context_top": 1.00,
                "joint_area": 0.004,
                "core_top": 0.18,
                "effective_core_top": 0.18,
                "scp_top": 0.28,
                "penalty_top": 0.05,
                "sig_tor_support_max": 1.45,
                "outbreak_risk_max": 0.55,
                "tornado_overlap_max": 1.55,
                "learned_tornado_concern_prob": 0.30,
                "raw_base_score_before_variant": 0.18,
                "tornado_concern_score": 0.18,
            },
            {
                "init_date": "2024-05-19",
                "valid_date": "2024-05-21",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 0,
                "is_real_ingest": True,
                "raw_core_variant_name": "baseline",
                "raw_core_mask_factor": 1.0,
                "raw_core_final_value": 0.12,
                "raw_core_to_context_ratio": 0.146,
                "core_variant_name": "baseline",
                "source_variant_name": "baseline",
                "component_variant_name": "baseline",
                "score_variant_name": "baseline",
                "support_top": 0.18,
                "normalized_synoptic_support": 0.65,
                "context_top": 0.82,
                "joint_area": 0.0007,
                "core_top": 0.12,
                "effective_core_top": 0.12,
                "scp_top": 0.55,
                "penalty_top": 0.05,
                "sig_tor_support_max": 2.20,
                "outbreak_risk_max": 0.65,
                "tornado_overlap_max": 3.20,
                "learned_tornado_concern_prob": 0.30,
                "raw_base_score_before_variant": 0.12,
                "tornado_concern_score": 0.12,
            },
        ]
    ).to_csv(input_csv, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rescore_tornado_concern_eval_csv",
            "--input-csv",
            str(input_csv),
            "--source-variant",
            "failure_targeted_contamination",
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ],
    )

    rescore_cli.main()

    ranked = pd.read_csv(output_csv)
    assert set(ranked["source_variant_name"]) == {"failure_targeted_contamination"}
    assert output_md.exists()
    assert "failure_targeted_contamination" in output_md.read_text(encoding="utf-8")


def test_failure_diagnostics_cli_buckets_top_day_mismatches(tmp_path: Path) -> None:
    input_csv = tmp_path / "baseline.csv"
    output_csv = tmp_path / "diagnostics.csv"
    output_md = tmp_path / "diagnostics.md"
    pd.DataFrame(
        [
            {
                "init_date": "2024-05-19",
                "valid_date": "2024-05-22",
                "day_rank_within_init": 1,
                "observed_category": "hail_outbreak_day",
                "top_valid_date": "2024-05-22",
                "top_observed_category": "hail_outbreak_day",
                "best_tornado_valid_date": "2024-05-21",
                "top_day_category_mismatch": True,
                "hail_outranks_tornado_failure": True,
                "non_outbreak_outranks_outbreak_failure": False,
                "failure_driver": "broad_context_dominates",
                "ranking_tornado_concern_score": 0.80,
                "best_tornado_score": 0.35,
                "top_minus_best_tornado_score": 0.45,
                "tornado_overlap_max": 1.40,
                "sig_tor_support_max": 1.10,
                "hail_overlap_max": 3.20,
                "wind_overlap_max": 0.30,
                "learned_tornado_concern_prob": 0.30,
                "raw_base_score_before_variant": 0.80,
                "core_top": 0.04,
                "scp_top": 0.20,
                "context_top": 0.95,
            },
            {
                "init_date": "2024-05-19",
                "valid_date": "2024-05-21",
                "day_rank_within_init": 2,
                "observed_category": "tornado_outbreak_day",
                "top_valid_date": "2024-05-22",
                "top_observed_category": "hail_outbreak_day",
                "best_tornado_valid_date": "2024-05-21",
                "top_day_category_mismatch": True,
                "hail_outranks_tornado_failure": True,
                "non_outbreak_outranks_outbreak_failure": False,
                "failure_driver": "broad_context_dominates",
                "ranking_tornado_concern_score": 0.35,
                "tornado_overlap_max": 3.20,
                "sig_tor_support_max": 2.20,
                "hail_overlap_max": 1.10,
                "wind_overlap_max": 0.20,
                "learned_tornado_concern_prob": 0.25,
                "raw_base_score_before_variant": 0.35,
                "core_top": 0.03,
                "scp_top": 0.55,
                "context_top": 0.70,
            },
            {
                "init_date": "2025-03-04",
                "valid_date": "2025-03-07",
                "day_rank_within_init": 1,
                "observed_category": "non_outbreak_severe_day",
                "top_valid_date": "2025-03-07",
                "top_observed_category": "non_outbreak_severe_day",
                "best_tornado_valid_date": "2025-03-04",
                "top_day_category_mismatch": True,
                "hail_outranks_tornado_failure": False,
                "non_outbreak_outranks_outbreak_failure": True,
                "failure_driver": "tornado_signal_near_zero_for_both",
                "ranking_tornado_concern_score": 0.10,
                "best_tornado_score": 0.01,
                "top_minus_best_tornado_score": 0.09,
                "tornado_signal": 0.0,
            },
        ]
    ).to_csv(input_csv, index=False)

    diagnostics_cli.main(["--eval-csv", str(input_csv), "--output-csv", str(output_csv), "--output-md", str(output_md)])

    rows = pd.read_csv(output_csv)
    assert set(rows["failure_bucket"]) == {"hail_over_tornado", "non_outbreak_over_outbreak"}
    assert rows.loc[rows["failure_bucket"] == "hail_over_tornado", "best_tornado_valid_date"].iloc[0] == "2024-05-21"
    assert "mismatch_windows: 2" in output_md.read_text(encoding="utf-8")


def test_weak_signal_diagnostics_cli_writes_ranked_bucket_report(tmp_path: Path) -> None:
    input_csv = tmp_path / "baseline.csv"
    output_csv = tmp_path / "diagnostics.csv"
    output_md = tmp_path / "diagnostics.md"
    weak_csv = tmp_path / "weak.csv"
    weak_md = tmp_path / "weak.md"
    pd.DataFrame(
        [
            {
                "init_date": "2024-05-19",
                "valid_date": "2024-05-22",
                "day_rank_within_init": 1,
                "observed_category": "hail_outbreak_day",
                "top_valid_date": "2024-05-22",
                "top_observed_category": "hail_outbreak_day",
                "best_tornado_valid_date": "2024-05-21",
                "top_day_category_mismatch": True,
                "hail_outranks_tornado_failure": True,
                "non_outbreak_outranks_outbreak_failure": False,
                "failure_driver": "broad_context_dominates",
                "ranking_tornado_concern_score": 0.80,
                "top_minus_best_tornado_score": 0.45,
                "top_tornado_signal": 0.020,
                "best_tornado_tornado_signal": 0.008,
                "top_minus_best_tornado_learned_term": 0.10,
                "top_minus_best_tornado_context_top": 0.20,
                "top_minus_best_tornado_sig_tor_support_max": -0.20,
                "top_minus_best_tornado_scp_top": -0.10,
                "top_minus_best_tornado_tornado_overlap_max": -0.25,
                "best_tornado_sig_tor_support_max": 1.00,
                "best_tornado_scp_top": 0.15,
                "best_tornado_tornado_overlap_max": 1.20,
                "hail_overlap_max": 3.20,
                "best_tornado_hail_support": 1.10,
            }
        ]
    ).to_csv(input_csv, index=False)

    diagnostics_cli.main(
        [
            "--eval-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
            "--weak-signal-output-csv",
            str(weak_csv),
            "--weak-signal-output-md",
            str(weak_md),
        ]
    )

    rows = pd.read_csv(weak_csv)
    assert bool(rows.loc[0, "missing_or_weak_sig_tor_support"])
    assert bool(rows.loc[0, "weak_scp_proxy"])
    assert bool(rows.loc[0, "weak_tornado_favored_overlap"])
    assert bool(rows.loc[0, "learned_probability_overpowering_ingredients"])
    assert bool(rows.loc[0, "broad_hail_or_wind_context_dominates"])
    assert "relevant_failures: 1" in weak_md.read_text(encoding="utf-8")


def test_rescore_tornado_hail_separation_challenger_changes_hail_dominant_fixture(tmp_path: Path, monkeypatch) -> None:
    input_csv = tmp_path / "baseline.csv"
    baseline_csv = tmp_path / "baseline_rescored.csv"
    challenger_csv = tmp_path / "challenger.csv"
    pd.DataFrame(
        [
            {
                "init_date": "2024-05-19",
                "valid_date": "2024-05-22",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "is_real_ingest": True,
                "support_top": 0.20,
                "normalized_synoptic_support": 0.90,
                "context_top": 0.95,
                "joint_area": 0.020,
                "core_top": 0.030,
                "effective_core_top": 0.030,
                "scp_top": 0.20,
                "penalty_top": 0.18,
                "sig_tor_support_max": 1.10,
                "outbreak_risk_max": 0.30,
                "tornado_overlap_max": 1.35,
                "hail_overlap_max": 3.20,
                "learned_tornado_concern_prob": 0.30,
                "raw_base_score_before_variant": 0.30,
                "tornado_concern_score": 0.30,
            },
            {
                "init_date": "2024-05-19",
                "valid_date": "2024-05-21",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 0,
                "is_real_ingest": True,
                "support_top": 0.18,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.75,
                "joint_area": 0.016,
                "core_top": 0.026,
                "effective_core_top": 0.026,
                "scp_top": 0.55,
                "penalty_top": 0.02,
                "sig_tor_support_max": 2.10,
                "outbreak_risk_max": 0.24,
                "tornado_overlap_max": 3.10,
                "hail_overlap_max": 1.20,
                "learned_tornado_concern_prob": 0.30,
                "raw_base_score_before_variant": 0.26,
                "tornado_concern_score": 0.26,
            },
        ]
    ).to_csv(input_csv, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "rescore_tornado_concern_eval_csv",
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(baseline_csv),
            "--output-md",
            str(tmp_path / "baseline.md"),
        ],
    )
    rescore_cli.main()
    monkeypatch.setattr(
        "sys.argv",
        [
            "rescore_tornado_concern_eval_csv",
            "--input-csv",
            str(input_csv),
            "--component-variant",
            "tornado_hail_separation_v1",
            "--output-csv",
            str(challenger_csv),
            "--output-md",
            str(tmp_path / "challenger.md"),
        ],
    )
    rescore_cli.main()

    baseline = pd.read_csv(baseline_csv)
    challenger = pd.read_csv(challenger_csv)
    baseline_hail = baseline.loc[baseline["valid_date"] == "2024-05-22"].iloc[0]
    challenger_hail = challenger.loc[challenger["valid_date"] == "2024-05-22"].iloc[0]
    assert set(challenger["component_variant_name"]) == {"tornado_hail_separation_v1"}
    assert float(challenger_hail["tornado_concern_score"]) < float(baseline_hail["tornado_concern_score"])
    assert "component_variant: `tornado_hail_separation_v1`" in (tmp_path / "challenger.md").read_text(encoding="utf-8")


def test_lead_time_calibrated_score_variant_penalizes_later_valid_days() -> None:
    day0 = {
        "init_date": "2024-05-07",
        "valid_date": "2024-05-07",
        "raw_base_score_before_variant": 1.0,
        "tornado_signal": 0.20,
        "broad_signal": 0.40,
        "discriminator": 0.333333,
        "learned_tornado_concern_prob": 0.30,
    }
    day3 = {**day0, "valid_date": "2024-05-10"}

    day0_terms = eval_cli._variant_terms_from_components(day0, variant="lead_time_calibrated")
    day3_terms = eval_cli._variant_terms_from_components(day3, variant="lead_time_calibrated")

    assert day0_terms["variant_penalty_term"] == 1.0
    assert day3_terms["variant_penalty_term"] < 0.25
    assert day3_terms["tornado_concern_score"] < day0_terms["tornado_concern_score"]


def test_build_tornado_concern_product_marks_failed_audit_internal_only(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    values = np.zeros((1, 8, 8), dtype=float)
    values[:, 1:7, 1:7] = 0.08
    prediction = xr.Dataset(
        data_vars={"tornado_concern_prob": (("time", "lat", "lon"), values)},
        coords={"time": pd.to_datetime(["2024-05-26T00:00:00"]), "lat": np.arange(8), "lon": np.arange(8)},
    )
    prediction.to_netcdf(paths.outputs / "forecast_products_2024-05-26_00.nc")
    monkeypatch.setattr(product_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(product_cli, "build_paths", lambda _settings: paths)

    outdir = tmp_path / "product_failed_audit"
    product_cli.main(
        [
            "--date",
            "2024-05-26",
            "--cycle",
            "00",
            "--field",
            "tornado_concern_prob",
            "--map-style",
            "contours",
            "--outdir",
            str(outdir),
        ]
    )

    metadata = pd.read_json(outdir / "tornado_concern_init_2024-05-26_00z_valid_2024-05-26.json", typ="series")
    assert not bool(metadata["public_ready"])
    assert metadata["audit_status"] == "flagged"
    assert metadata["publication_status"] == "internal_review_only"
    assert "overbroad_low_risk_footprint" in str(metadata["failure_reasons"])


def test_build_tornado_concern_product_marks_dot_only_display_internal_only(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    values = np.zeros((1, 9, 9), dtype=float)
    values[:, 4, 4] = 0.50
    prediction = xr.Dataset(
        data_vars={"tornado_concern_prob": (("time", "lat", "lon"), values)},
        coords={"time": pd.to_datetime(["2024-04-26T00:00:00"]), "lat": np.arange(9), "lon": np.arange(9)},
    )
    prediction.to_netcdf(paths.outputs / "forecast_products_2024-04-26_00.nc")
    monkeypatch.setattr(product_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(product_cli, "build_paths", lambda _settings: paths)

    outdir = tmp_path / "dot_only_product"
    product_cli.main(
        [
            "--date",
            "2024-04-26",
            "--cycle",
            "00",
            "--field",
            "tornado_concern_prob",
            "--map-style",
            "contours",
            "--outdir",
            str(outdir),
        ]
    )

    metadata = pd.read_json(outdir / "tornado_concern_init_2024-04-26_00z_valid_2024-04-26.json", typ="series")
    assert not bool(metadata["public_ready"])
    assert "display_too_sparse_ge_02pct" in str(metadata["failure_reasons"])


def test_tornado_concern_product_audit_can_target_exact_valid_date(tmp_path: Path) -> None:
    values = np.zeros((2, 8, 8), dtype=float)
    values[0, 1:7, 1:7] = 0.08
    values[1, 3:5, 3:5] = 0.12
    dataset = xr.Dataset(
        {"tornado_concern_prob": (("time", "lat", "lon"), values)},
        coords={"time": pd.to_datetime(["2024-05-26T00:00:00", "2024-05-27T00:00:00"]), "lat": np.arange(8), "lon": np.arange(8)},
    )
    path = tmp_path / "forecast_products_2024-05-26_00.nc"
    dataset.to_netcdf(path)

    broad = product_audit_cli.audit_tornado_concern_product("2024-05-26", path, valid_date="2024-05-26")
    compact = product_audit_cli.audit_tornado_concern_product("2024-05-26", path, valid_date="2024-05-27")

    assert not broad["public_ready"]
    assert compact["public_ready"]


def test_public_display_concern_field_damps_isolated_bullseye() -> None:
    values = np.zeros((9, 9), dtype=float)
    values[4, 4] = 0.66

    display = product_cli._public_display_concern_field(values)

    assert float(display[4, 4]) < 0.10
    assert int(np.count_nonzero(display >= 0.01)) <= int(np.count_nonzero(values >= 0.01))
    assert float(display[4, 4]) < float(values[4, 4])


def test_neighborhood_mean_uses_circular_weighted_kernel() -> None:
    values = np.zeros((7, 7), dtype=float)
    values[3, 3] = 1.0

    smoothed = product_cli._neighborhood_mean(values, radius=2)

    assert float(smoothed[3, 3]) > float(smoothed[3, 1])
    assert float(smoothed[3, 1]) > 0.0
    assert float(smoothed[1, 1]) == 0.0
    assert abs(float(smoothed.sum()) - 1.0) < 1e-9


def test_visible_label_levels_include_highest_displayed_band() -> None:
    values = np.array([[0.0, 0.021], [0.19, 0.31]], dtype=float)

    assert product_cli._visible_label_levels(values)[-1] == 0.30


def test_public_display_concern_field_keeps_supported_cluster_visible() -> None:
    values = np.zeros((9, 9), dtype=float)
    values[3:6, 3:6] = 0.20

    display = product_cli._public_display_concern_field(values)

    assert float(display[4, 4]) >= 0.20
    assert int(np.count_nonzero(display >= 0.05)) >= int(np.count_nonzero(values >= 0.05))


def test_public_display_concern_field_keeps_compact_signal_from_becoming_specks() -> None:
    values = np.zeros((9, 9), dtype=float)
    values[4:6, 4:6] = 0.12

    display = product_cli._public_display_concern_field(values)

    assert float(display.max()) >= 0.12
    assert int(np.count_nonzero(display >= 0.02)) > int(np.count_nonzero(values >= 0.02))


def test_public_candidate_builder_ranks_ready_public_tornado_cases(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    compact = np.zeros((1, 8, 8), dtype=float)
    compact[:, 3:5, 3:5] = 0.18
    broad = np.zeros((1, 8, 8), dtype=float)
    broad[:, 1:7, 1:7] = 0.08
    for date, values in [("2024-05-19", compact), ("2024-05-26", broad)]:
        xr.Dataset(
            {"tornado_concern_prob": (("time", "lat", "lon"), values)},
            coords={"time": pd.to_datetime([f"{date}T00:00:00"]), "lat": np.arange(8), "lon": np.arange(8)},
        ).to_netcdf(paths.outputs / f"forecast_products_{date}_00.nc")
    (paths.verification / "2024-05-19_verification.json").write_text(
        json.dumps({"per_day": [{"observed_category": "tornado_outbreak_day", "observed_tornado_outbreak": 1}]}),
        encoding="utf-8",
    )
    (paths.verification / "2024-05-26_verification.json").write_text(
        json.dumps({"per_day": [{"observed_category": "hail_outbreak_day", "observed_tornado_outbreak": 0}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(public_candidates_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(public_candidates_cli, "build_paths", lambda _settings: paths)

    dates_file = tmp_path / "dates.txt"
    dates_file.write_text("2024-05-26\n2024-05-19\n", encoding="utf-8")
    output_csv = tmp_path / "candidates.csv"
    output_md = tmp_path / "candidates.md"
    public_candidates_cli.main(
        [
            "--dates-file",
            str(dates_file),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
            "--field",
            "tornado_concern_prob",
        ]
    )

    rows = pd.read_csv(output_csv)
    assert list(rows["date"]) == ["2024-05-19", "2024-05-26"]
    assert bool(rows.loc[0, "public_ready"])
    assert not bool(rows.loc[1, "public_ready"])
    assert "Ranked Candidates" in output_md.read_text(encoding="utf-8")


def test_public_candidate_builder_records_best_valid_date(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    values = np.zeros((2, 6, 6), dtype=float)
    values[0, 2:4, 2:4] = 0.08
    values[1, 2:4, 2:4] = 0.22
    xr.Dataset(
        {"tornado_concern_prob": (("time", "lat", "lon"), values)},
        coords={"time": pd.to_datetime(["2024-05-19T00:00:00", "2024-05-21T00:00:00"]), "lat": np.arange(6), "lon": np.arange(6)},
    ).to_netcdf(paths.outputs / "forecast_products_2024-05-19_00.nc")
    monkeypatch.setattr(public_candidates_cli, "load_settings", lambda: AppSettings(raw={}))
    monkeypatch.setattr(public_candidates_cli, "build_paths", lambda _settings: paths)

    dates_file = tmp_path / "dates.txt"
    dates_file.write_text("2024-05-19\n", encoding="utf-8")
    output_csv = tmp_path / "candidates.csv"
    output_md = tmp_path / "candidates.md"
    public_candidates_cli.main(
        [
            "--dates-file",
            str(dates_file),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
            "--valid-date-mode",
            "best",
            "--field",
            "tornado_concern_prob",
        ]
    )

    rows = pd.read_csv(output_csv)
    assert rows.loc[0, "selected_valid_date"] == "2024-05-21"
    assert rows.loc[0, "valid_date_mode"] == "best"


def test_tornado_concern_product_audit_flags_overbroad_low_risk_area(tmp_path: Path) -> None:
    values = np.zeros((1, 6, 6), dtype=float)
    values[:, 1:5, 1:5] = 0.08
    dataset = xr.Dataset(
        {
            "tornado_concern_prob": (("time", "lat", "lon"), values),
            "tornado_concern_guardrail_applied": (("time", "lat", "lon"), np.zeros_like(values, dtype=bool)),
        },
        coords={"time": pd.to_datetime(["2024-06-02T00:00:00"]), "lat": np.arange(6), "lon": np.arange(6)},
    )
    path = tmp_path / "forecast_products_2024-06-02_00.nc"
    dataset.to_netcdf(path)

    row = product_audit_cli.audit_tornado_concern_product(
        "2024-06-02",
        path,
        max_low_risk_cells=8,
        max_component_cells=8,
    )

    assert not row["public_ready"]
    assert "overbroad_low_risk_footprint" in row["failure_reasons"]
    assert "large_contiguous_low_risk_area" in row["failure_reasons"]


def test_tornado_concern_product_audit_allows_compact_signal(tmp_path: Path) -> None:
    values = np.zeros((1, 6, 6), dtype=float)
    values[:, 2:4, 2:4] = 0.12
    dataset = xr.Dataset(
        {"tornado_concern_prob": (("time", "lat", "lon"), values)},
        coords={"time": pd.to_datetime(["2024-05-26T00:00:00"]), "lat": np.arange(6), "lon": np.arange(6)},
    )
    path = tmp_path / "forecast_products_2024-05-26_00.nc"
    dataset.to_netcdf(path)

    row = product_audit_cli.audit_tornado_concern_product(
        "2024-05-26",
        path,
        max_low_risk_cells=8,
        max_low_risk_fraction=0.20,
        max_component_cells=8,
    )

    assert row["public_ready"]
    assert row["low_risk_cells"] == 4
    assert row["largest_low_risk_component_cells"] == 4


def test_recovery_wave_mixed_dry_run_batches_deterministically(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    candidate_rows = pd.DataFrame(
        [
            {"date": "2024-04-01", "priority_tier": "sig_tor", "priority_sort_key": 0, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
            {"date": "2024-04-02", "priority_tier": "sig_tor", "priority_sort_key": 0, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
            {"date": "2024-04-03", "priority_tier": "outbreak", "priority_sort_key": 1, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
            {"date": "2024-04-04", "priority_tier": "outbreak", "priority_sort_key": 1, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
            {"date": "2024-04-05", "priority_tier": "all_tornado", "priority_sort_key": 2, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
        ]
    )

    monkeypatch.setattr(wave_cli, "load_settings", lambda: object())
    monkeypatch.setattr(wave_cli, "build_paths", lambda _settings: paths)
    monkeypatch.setattr(wave_cli, "build_tornado_concern_candidate_rows", lambda *args, **kwargs: candidate_rows)

    outdir = tmp_path / "wave"
    wave_cli.main(
        [
            "--wave",
            "mixed",
            "--max-dates",
            "5",
            "--batch-size",
            "2",
            "--output-dir",
            str(outdir),
            "--dry-run",
        ]
    )

    summary = pd.read_csv(outdir / "recovery_wave_summary.csv")
    assert list(summary["planned_dates"]) == [2, 2, 1]
    batch_one = pd.read_csv(outdir / "batches" / "batch_01_recovery.csv")
    assert list(batch_one["date"]) == ["2024-04-01", "2024-04-03"]
    batch_two = pd.read_csv(outdir / "batches" / "batch_02_recovery.csv")
    assert list(batch_two["date"]) == ["2024-04-05", "2024-04-02"]
    assert set(batch_one["recovery_status"]) == {"dry_run"}


def test_recovery_wave_hard_negative_proxy_excludes_sig_tor(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)

    candidate_rows = pd.DataFrame(
        [
            {"date": "2024-04-01", "priority_tier": "sig_tor", "priority_sort_key": 0, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
            {"date": "2024-04-02", "priority_tier": "outbreak", "priority_sort_key": 1, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
            {"date": "2024-04-03", "priority_tier": "all_tornado", "priority_sort_key": 2, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
            {"date": "2024-04-04", "priority_tier": "outbreak", "priority_sort_key": 1, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "interim_summary_synthetic_source"},
        ]
    )

    monkeypatch.setattr(wave_cli, "load_settings", lambda: object())
    monkeypatch.setattr(wave_cli, "build_paths", lambda _settings: paths)
    monkeypatch.setattr(wave_cli, "build_tornado_concern_candidate_rows", lambda *args, **kwargs: candidate_rows)

    outdir = tmp_path / "hard_negative_wave"
    wave_cli.main(
        [
            "--wave",
            "hard_negative",
            "--max-dates",
            "10",
            "--batch-size",
            "2",
            "--output-dir",
            str(outdir),
            "--dry-run",
        ]
    )

    cases = pd.read_csv(outdir / "recovery_wave_cases.csv")
    assert list(cases["date"]) == ["2024-04-02", "2024-04-03"]
    assert "2024-04-01" not in set(cases["date"])


def test_recovery_wave_can_use_reviewed_dates_file(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)
    candidate_rows = pd.DataFrame(
        [
            {"date": "2024-04-01", "priority_tier": "sig_tor", "priority_sort_key": 0, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
            {"date": "2024-04-02", "priority_tier": "outbreak", "priority_sort_key": 1, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
            {"date": "2024-04-03", "priority_tier": "all_tornado", "priority_sort_key": 2, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
        ]
    )
    dates_file = tmp_path / "reviewed.txt"
    dates_file.write_text("2024-04-03\n2024-04-01\n", encoding="utf-8")

    monkeypatch.setattr(wave_cli, "load_settings", lambda: object())
    monkeypatch.setattr(wave_cli, "build_paths", lambda _settings: paths)
    monkeypatch.setattr(wave_cli, "build_tornado_concern_candidate_rows", lambda *args, **kwargs: candidate_rows)

    outdir = tmp_path / "reviewed_wave"
    wave_cli.main(
        [
            "--wave",
            "hard_negative",
            "--dates-file",
            str(dates_file),
            "--batch-size",
            "10",
            "--output-dir",
            str(outdir),
            "--dry-run",
        ]
    )

    cases = pd.read_csv(outdir / "recovery_wave_cases.csv")
    assert list(cases["date"]) == ["2024-04-03", "2024-04-01"]


def test_hard_negative_wave_dates_file_is_authoritative_for_zero_tornado_dates(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)
    candidate_rows = pd.DataFrame(
        [
            {"date": "2025-05-02", "priority_tier": "all_tornado", "priority_sort_key": 2, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
        ]
    )
    dates = [f"2024-08-{day:02d}" for day in range(1, 31)] + [f"2024-09-{day:02d}" for day in range(1, 21)]
    dates_file = tmp_path / "hard_negative_reviewed.txt"
    dates_file.write_text("\n".join(dates) + "\n", encoding="utf-8")

    monkeypatch.setattr(wave_cli, "load_settings", lambda: object())
    monkeypatch.setattr(wave_cli, "build_paths", lambda _settings: paths)
    monkeypatch.setattr(wave_cli, "build_tornado_concern_candidate_rows", lambda *args, **kwargs: candidate_rows)

    outdir = tmp_path / "hard_negative_dates_file_wave"
    wave_cli.main(
        [
            "--wave",
            "hard_negative",
            "--dates-file",
            str(dates_file),
            "--batch-size",
            "10",
            "--output-dir",
            str(outdir),
            "--dry-run",
        ]
    )

    summary = pd.read_csv(outdir / "recovery_wave_summary.csv")
    cases = pd.read_csv(outdir / "recovery_wave_cases.csv")
    assert list(summary["planned_dates"]) == [10, 10, 10, 10, 10]
    assert list(cases["date"]) == dates
    assert set(cases["priority_tier"]) == {"hard_negative"}


def test_hard_negative_wave_dates_file_does_not_filter_through_tornado_positive_tiers(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)
    candidate_rows = pd.DataFrame(
        [
            {"date": "2025-05-02", "priority_tier": "all_tornado", "priority_sort_key": 2, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
            {"date": "2024-04-26", "priority_tier": "sig_tor", "priority_sort_key": 0, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
        ]
    )
    dates_file = tmp_path / "reviewed_hard_negative.txt"
    dates_file.write_text("2024-08-17\n2025-05-02\n2024-08-18\n", encoding="utf-8")

    monkeypatch.setattr(wave_cli, "load_settings", lambda: object())
    monkeypatch.setattr(wave_cli, "build_paths", lambda _settings: paths)
    monkeypatch.setattr(wave_cli, "build_tornado_concern_candidate_rows", lambda *args, **kwargs: candidate_rows)

    outdir = tmp_path / "hard_negative_no_tier_filter"
    wave_cli.main(
        [
            "--wave",
            "hard_negative",
            "--dates-file",
            str(dates_file),
            "--batch-size",
            "10",
            "--output-dir",
            str(outdir),
            "--dry-run",
        ]
    )

    cases = pd.read_csv(outdir / "recovery_wave_cases.csv")
    assert list(cases["date"]) == ["2024-08-17", "2025-05-02", "2024-08-18"]
    assert set(cases["priority_tier"]) == {"hard_negative"}


def test_recovery_wave_dates_file_positive_wave_still_uses_candidate_rows(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)
    candidate_rows = pd.DataFrame(
        [
            {"date": "2024-04-01", "priority_tier": "sig_tor", "priority_sort_key": 0, "failure_reason": "not_real_ingest_confirmed", "real_ingest_failure_detail": "no_local_real_ingest_evidence"},
        ]
    )
    dates_file = tmp_path / "positive_reviewed.txt"
    dates_file.write_text("2024-08-17\n2024-04-01\n", encoding="utf-8")

    monkeypatch.setattr(wave_cli, "load_settings", lambda: object())
    monkeypatch.setattr(wave_cli, "build_paths", lambda _settings: paths)
    monkeypatch.setattr(wave_cli, "build_tornado_concern_candidate_rows", lambda *args, **kwargs: candidate_rows)

    outdir = tmp_path / "positive_dates_file_wave"
    wave_cli.main(
        [
            "--wave",
            "mixed",
            "--dates-file",
            str(dates_file),
            "--batch-size",
            "10",
            "--output-dir",
            str(outdir),
            "--dry-run",
        ]
    )

    cases = pd.read_csv(outdir / "recovery_wave_cases.csv")
    assert list(cases["date"]) == ["2024-04-01"]
    assert list(cases["priority_tier"]) == ["sig_tor"]


def test_recovery_wave_summary_markdown_includes_skipped_dates_file_reasons(tmp_path: Path, monkeypatch) -> None:
    paths = _workflow_paths(tmp_path)
    for path in [paths.outputs, paths.verification, paths.labels, paths.interim]:
        path.mkdir(parents=True, exist_ok=True)
    dates_file = tmp_path / "reviewed_with_skips.txt"
    dates_file.write_text("2024-08-17\nnot-a-date\n2024-08-17\n", encoding="utf-8")

    monkeypatch.setattr(wave_cli, "load_settings", lambda: object())
    monkeypatch.setattr(wave_cli, "build_paths", lambda _settings: paths)
    monkeypatch.setattr(wave_cli, "build_tornado_concern_candidate_rows", lambda *args, **kwargs: pd.DataFrame())

    outdir = tmp_path / "skipped_dates_wave"
    wave_cli.main(
        [
            "--wave",
            "hard_negative",
            "--dates-file",
            str(dates_file),
            "--batch-size",
            "10",
            "--output-dir",
            str(outdir),
            "--dry-run",
        ]
    )

    markdown = (outdir / "recovery_wave_summary.md").read_text(encoding="utf-8")
    assert "| not-a-date | invalid_date |" in markdown
    assert "| 2024-08-17 | duplicate_date |" in markdown


def test_hard_negative_candidate_selector_uses_eval_failures_not_random_dates(tmp_path: Path) -> None:
    eval_csv = tmp_path / "eval.csv"
    output = tmp_path / "hard_negative_recovery_candidates.txt"
    report_csv = tmp_path / "hard_negative_recovery_candidates.csv"
    report_md = tmp_path / "hard_negative_recovery_candidates.md"
    pd.DataFrame(
        [
            {
                "init_date": "2024-05-19",
                "valid_date": "2024-05-22",
                "day_rank_within_init": 1,
                "top_valid_date": "2024-05-22",
                "top_observed_category": "hail_outbreak_day",
                "top_observed_tornado_outbreak": 0,
                "top_observed_significant_tornado_support": 0,
                "top_day_category_mismatch": True,
                "hail_outranks_tornado_failure": True,
                "top_context_top": 0.95,
                "hail_overlap_max": 3.20,
                "top_minus_best_tornado_score": 0.40,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-26",
                "day_rank_within_init": 1,
                "top_valid_date": "2024-04-26",
                "top_observed_category": "tornado_outbreak_day",
                "top_observed_tornado_outbreak": 1,
                "top_observed_significant_tornado_support": 1,
                "top_day_category_mismatch": False,
                "hail_outranks_tornado_failure": False,
                "top_context_top": 0.70,
                "hail_overlap_max": 0.80,
                "top_minus_best_tornado_score": 0.0,
            },
        ]
    ).to_csv(eval_csv, index=False)

    hard_negative_candidates_cli.main(
        [
            "--eval-csv",
            str(eval_csv),
            "--use-eval-fallback",
            "--output",
            str(output),
            "--report-csv",
            str(report_csv),
            "--report-md",
            str(report_md),
        ]
    )

    assert output.read_text(encoding="utf-8").splitlines() == ["2024-05-19"]
    rows = pd.read_csv(report_csv)
    assert list(rows["date"]) == ["2024-05-19"]
    assert "Remove `--dry-run` only after reviewing" in report_md.read_text(encoding="utf-8")


def test_hard_negative_selector_prefers_zero_tornado_hail_wind_heavy_dates(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    outputs_dir = tmp_path / "outputs"
    verification_dir = outputs_dir / "verification"
    interim_dir = tmp_path / "interim"
    labels_dir.mkdir()
    outputs_dir.mkdir()
    verification_dir.mkdir()
    interim_dir.mkdir()
    pd.DataFrame(
        [
            *[{"report_id": i, "date": "2024-08-17", "hazard": "wind", "significant": 0} for i in range(30)],
            *[{"report_id": 100 + i, "date": "2024-08-17", "hazard": "hail", "significant": 0} for i in range(10)],
            *[{"report_id": 200 + i, "date": "2024-08-18", "hazard": "wind", "significant": 0} for i in range(35)],
            {"report_id": 300, "date": "2024-08-18", "hazard": "tornado", "significant": 0},
        ]
    ).to_parquet(labels_dir / "spc_reports_test.parquet", index=False)
    pd.DataFrame(
        [
            {"date": "2024-08-17", "tornado_outbreak": 0, "significant_tornado_support": 0},
            {"date": "2024-08-18", "tornado_outbreak": 0, "significant_tornado_support": 0},
        ]
    ).to_parquet(labels_dir / "outbreaks_test.parquet", index=False)
    ready = tmp_path / "ready.txt"
    ready.write_text("", encoding="utf-8")

    pool, metadata = hard_negative_candidates_cli.build_local_hard_negative_pool(
        labels_dir=labels_dir,
        outputs_dir=outputs_dir,
        verification_dir=verification_dir,
        interim_dir=interim_dir,
        ready_dates_file=ready,
    )
    selected, counts = hard_negative_candidates_cli.select_local_hard_negative_candidates(pool, target_count=2, min_hail_wind_reports=25)

    assert list(selected["date"]) == ["2024-08-17", "2024-08-18"]
    assert list(selected["tornado_reports"]) == [0, 1]
    assert counts["zero_tornado_candidates"] == 1
    assert counts["one_tornado_fallback_candidates"] == 1
    assert metadata["unavailable_sources"] == []


def test_hard_negative_selector_excludes_outbreak_positive_and_ready_by_default(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    outputs_dir = tmp_path / "outputs"
    verification_dir = outputs_dir / "verification"
    interim_dir = tmp_path / "interim"
    labels_dir.mkdir()
    outputs_dir.mkdir()
    verification_dir.mkdir()
    interim_dir.mkdir()
    rows = []
    for date in ["2024-08-17", "2024-08-18", "2024-08-19"]:
        rows.extend({"report_id": len(rows) + i, "date": date, "hazard": "wind", "significant": 0} for i in range(30))
    pd.DataFrame(rows).to_parquet(labels_dir / "spc_reports_test.parquet", index=False)
    pd.DataFrame(
        [
            {"date": "2024-08-17", "tornado_outbreak": 1, "significant_tornado_support": 0},
            {"date": "2024-08-18", "tornado_outbreak": 0, "significant_tornado_support": 1},
            {"date": "2024-08-19", "tornado_outbreak": 0, "significant_tornado_support": 0},
        ]
    ).to_parquet(labels_dir / "outbreaks_test.parquet", index=False)
    ready = tmp_path / "ready.txt"
    ready.write_text("2024-08-19\n", encoding="utf-8")

    pool, _metadata = hard_negative_candidates_cli.build_local_hard_negative_pool(
        labels_dir=labels_dir,
        outputs_dir=outputs_dir,
        verification_dir=verification_dir,
        interim_dir=interim_dir,
        ready_dates_file=ready,
    )
    selected, counts = hard_negative_candidates_cli.select_local_hard_negative_candidates(pool, target_count=5, min_hail_wind_reports=25)
    included_ready, _ = hard_negative_candidates_cli.select_local_hard_negative_candidates(pool, target_count=5, min_hail_wind_reports=25, exclude_ready=False)

    assert selected.empty
    assert counts["excluded_outbreak_positive_count"] == 2
    assert list(included_ready["date"]) == ["2024-08-19"]


def test_hard_negative_selector_writes_txt_csv_md_outputs(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    pd.DataFrame(
        [{"report_id": i, "date": "2024-08-17", "hazard": "wind", "significant": 0} for i in range(30)]
    ).to_parquet(labels_dir / "spc_reports_test.parquet", index=False)
    pd.DataFrame([{"date": "2024-08-17", "tornado_outbreak": 0, "significant_tornado_support": 0}]).to_parquet(
        labels_dir / "outbreaks_test.parquet",
        index=False,
    )
    output = tmp_path / "candidates.txt"
    report_csv = tmp_path / "candidates.csv"
    report_md = tmp_path / "candidates.md"

    hard_negative_candidates_cli.main(
        [
            "--labels-dir",
            str(labels_dir),
            "--outputs-dir",
            str(tmp_path / "outputs"),
            "--verification-dir",
            str(tmp_path / "outputs" / "verification"),
            "--interim-dir",
            str(tmp_path / "interim"),
            "--ready-dates-file",
            str(tmp_path / "ready.txt"),
            "--target-count",
            "50",
            "--output",
            str(output),
            "--report-csv",
            str(report_csv),
            "--report-md",
            str(report_md),
        ]
    )

    assert output.read_text(encoding="utf-8").splitlines() == ["2024-08-17"]
    assert pd.read_csv(report_csv).loc[0, "selection_reason"] == "zero_tornado_hail_wind_heavy"
    assert "total_candidates_found: 1" in report_md.read_text(encoding="utf-8")


def test_delta_comparison_report_identifies_fixed_and_worsened_cases(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    challenger_csv = tmp_path / "challenger.csv"
    output_csv = tmp_path / "delta.csv"
    output_md = tmp_path / "delta.md"
    pd.DataFrame(
        [
            {"init_date": "2024-05-19", "day_rank_within_init": 1, "top_valid_date": "2024-05-22", "top_observed_category": "hail_outbreak_day", "top_minus_best_tornado_score": 0.40, "hail_outranks_tornado_failure": True, "top_day_category_mismatch": True, "non_outbreak_outranks_outbreak_failure": False},
            {"init_date": "2024-08-08", "day_rank_within_init": 1, "top_valid_date": "2024-08-08", "top_observed_category": "tornado_outbreak_day", "top_minus_best_tornado_score": 0.00, "hail_outranks_tornado_failure": False, "top_day_category_mismatch": False, "non_outbreak_outranks_outbreak_failure": False},
        ]
    ).to_csv(baseline_csv, index=False)
    pd.DataFrame(
        [
            {"init_date": "2024-05-19", "day_rank_within_init": 1, "top_valid_date": "2024-05-21", "top_observed_category": "tornado_outbreak_day", "top_minus_best_tornado_score": 0.00, "hail_outranks_tornado_failure": False, "top_day_category_mismatch": False, "non_outbreak_outranks_outbreak_failure": False},
            {"init_date": "2024-08-08", "day_rank_within_init": 1, "top_valid_date": "2024-08-09", "top_observed_category": "non_outbreak_severe_day", "top_minus_best_tornado_score": 0.10, "hail_outranks_tornado_failure": False, "top_day_category_mismatch": True, "non_outbreak_outranks_outbreak_failure": True},
        ]
    ).to_csv(challenger_csv, index=False)

    delta_cli.main(
        [
            "--baseline-csv",
            str(baseline_csv),
            "--challenger-csv",
            str(challenger_csv),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ]
    )

    rows = pd.read_csv(output_csv)
    fixed = rows.loc[rows["init_date"].eq("2024-05-19")].iloc[0]
    worsened = rows.loc[rows["init_date"].eq("2024-08-08")].iloc[0]
    assert bool(fixed["hail_over_tornado_fixed"])
    assert bool(fixed["top_day_mismatch_fixed"])
    assert bool(worsened["non_outbreak_over_outbreak_remained_or_worsened"])
    assert "checkpoint_acceptance_satisfied" in output_md.read_text(encoding="utf-8")


def test_prepare_public_prototype_writes_review_showcase_and_blocked_queue(tmp_path: Path) -> None:
    products_dir = tmp_path / "products"
    products_dir.mkdir(parents=True)
    for date in ["2024-05-07", "2025-04-29"]:
        stem = f"tornado_concern_{date}_00z"
        for suffix in ["png", "md", "json"]:
            (products_dir / f"{stem}.{suffix}").write_text(f"{date} {suffix}", encoding="utf-8")

    candidates_csv = tmp_path / "public_candidates.csv"
    pd.DataFrame(
        [
            {"date": "2024-05-07", "public_ready": True, "status": "ok", "failure_reasons": "", "observed_significant_tornado_support": True, "observed_tornado_relevant": True},
            {"date": "2025-04-29", "public_ready": True, "status": "ok", "failure_reasons": "", "observed_significant_tornado_support": True, "observed_tornado_relevant": True},
            {"date": "2024-05-26", "public_ready": False, "status": "flagged", "failure_reasons": "overbroad_low_risk_footprint;large_contiguous_low_risk_area"},
            {"date": "2024-08-08", "public_ready": False, "status": "flagged", "failure_reasons": "guardrail_heavy_map"},
        ]
    ).to_csv(candidates_csv, index=False)

    outdir = tmp_path / "prototype"
    prototype_cli.main(
        [
            "--public-candidates-csv",
            str(candidates_csv),
            "--products-dir",
            str(products_dir),
            "--output-dir",
            str(outdir),
            "--showcase-count",
            "1",
        ]
    )

    review = pd.read_csv(outdir / "visual_review.csv")
    assert list(review["visual_review_status"]) == ["needs_human_review", "needs_human_review"]
    showcase = pd.read_csv(outdir / "showcase_manifest.csv")
    assert list(showcase["date"]) == ["2024-05-07"]
    assert (outdir / "showcase" / "tornado_concern_2024-05-07_00z.png").exists()
    blocked = pd.read_csv(outdir / "blocked_work_queue.csv")
    assert set(blocked["queue"]) == {"guardrail_heavy", "overbroad_footprint"}
    assert "not official warning guidance" in (outdir / "README.md").read_text(encoding="utf-8")
    gallery = (outdir / "visual_review_gallery.md").read_text(encoding="utf-8")
    assert "Tornado Concern Visual Review Gallery" in gallery
    assert "2024-05-07" in gallery
    assert f"![" in gallery
    assert str(products_dir).replace("\\", "/") in gallery


def test_prepare_public_prototype_overwrite_cleans_stale_generated_files(tmp_path: Path) -> None:
    products_dir = tmp_path / "products"
    products_dir.mkdir(parents=True)
    for suffix in ["png", "md", "json"]:
        (products_dir / f"tornado_concern_2024-05-07_00z.{suffix}").write_text(f"fresh {suffix}", encoding="utf-8")

    candidates_csv = tmp_path / "public_candidates.csv"
    pd.DataFrame(
        [
            {"date": "2024-05-07", "public_ready": True, "status": "ok", "failure_reasons": ""},
        ]
    ).to_csv(candidates_csv, index=False)

    outdir = tmp_path / "prototype"
    outdir.mkdir(parents=True)
    (outdir / "stale_old_format_product.png").write_text("stale", encoding="utf-8")

    prototype_cli.main(
        [
            "--public-candidates-csv",
            str(candidates_csv),
            "--products-dir",
            str(products_dir),
            "--output-dir",
            str(outdir),
            "--overwrite",
        ]
    )

    assert not (outdir / "stale_old_format_product.png").exists()
    assert (outdir / "visual_review.csv").exists()
    assert (outdir / "visual_review_gallery.md").exists()


def test_build_spc_comparison_scaffold_merges_eval_and_public_candidates(tmp_path: Path) -> None:
    eval_csv = tmp_path / "eval.csv"
    public_csv = tmp_path / "public.csv"
    output_csv = tmp_path / "spc.csv"
    output_md = tmp_path / "spc.md"
    pd.DataFrame(
        [
            {"init_date": "2024-05-07", "day_rank_within_init": 1, "top_valid_date": "2024-05-08", "top_observed_category": "tornado_outbreak_day", "ranking_tornado_concern_score": 0.8},
            {"init_date": "2024-05-07", "day_rank_within_init": 2, "top_valid_date": "2024-05-09", "top_observed_category": "hail_outbreak_day", "ranking_tornado_concern_score": 0.2},
            {"init_date": "2024-05-26", "day_rank_within_init": 1, "top_valid_date": "2024-05-27", "top_observed_category": "hail_outbreak_day", "ranking_tornado_concern_score": 0.7},
        ]
    ).to_csv(eval_csv, index=False)
    pd.DataFrame(
        [
            {"date": "2024-05-07", "public_ready": True, "failure_reasons": "", "observed_categories": "tornado_outbreak_day"},
            {"date": "2024-05-26", "public_ready": False, "failure_reasons": "overbroad_low_risk_footprint", "observed_categories": "hail_outbreak_day"},
        ]
    ).to_csv(public_csv, index=False)

    spc_comparison_cli.main(
        [
            "--eval-csv",
            str(eval_csv),
            "--public-candidates-csv",
            str(public_csv),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ]
    )

    rows = pd.read_csv(output_csv)
    assert list(rows["date"]) == ["2024-05-07", "2024-05-26"]
    assert set(rows["spc_label_status"]) == {"needs_spc_label"}
    assert bool(rows.loc[0, "public_ready"])
    assert "needs_spc_label: 2" in output_md.read_text(encoding="utf-8")


def test_checkpoint_orchestrator_runs_baseline_only_subcommands(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str]):
        calls.append(command)
        if "tornado_concern_coverage" in command:
            Path(command[command.index("--output-csv") + 1]).write_text("date\n", encoding="utf-8")
            Path(command[command.index("--output-md") + 1]).write_text("# coverage\n", encoding="utf-8")
        elif "build_tornado_concern_real_dates.py" in " ".join(command):
            Path(command[command.index("--output") + 1]).write_text("2024-04-26\n", encoding="utf-8")
            Path(command[command.index("--ready-output") + 1]).write_text("2024-04-26\n", encoding="utf-8")
            Path(command[command.index("--audit-csv") + 1]).write_text("date\n", encoding="utf-8")
            Path(command[command.index("--ranked-candidates-csv") + 1]).write_text("date\n", encoding="utf-8")
        elif any("tornado_concern_eval" in part for part in command):
            Path(command[command.index("--output-csv") + 1]).write_text("init_date,day_rank_within_init\n2024-04-26,1\n", encoding="utf-8")
            Path(command[command.index("--output-md") + 1]).write_text("# eval\n", encoding="utf-8")
        elif "tornado_concern_failure_review" in command:
            outdir = Path(command[command.index("--outdir") + 1])
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "tornado_concern_failure_review.md").write_text("# review\n", encoding="utf-8")
            (outdir / "tornado_concern_failure_review_cases.csv").write_text("init_date\n2024-04-26\n", encoding="utf-8")
        elif "build_tornado_concern_product" in command:
            outdir = Path(command[command.index("--outdir") + 1])
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "tornado_concern_init_2024-04-26_00z_valid_2024-04-26.png").write_text("png", encoding="utf-8")

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(checkpoint_cli, "_run_command", fake_run)

    outdir = tmp_path / "checkpoint"
    checkpoint_cli.main(
        [
            "--start",
            "2024-01-01",
            "--end",
            "2024-12-31",
            "--output-dir",
            str(outdir),
            "--product-date",
            "2024-04-26",
            "--product-cycle",
            "00",
        ]
    )

    joined = [" ".join(command) for command in calls]
    assert len(calls) == 5
    assert any("severewx.cli.tornado_concern_eval" in command for command in joined)
    assert all("masked_core_steeper" not in command for command in joined)
    product_command = next(command for command in calls if "severewx.cli.build_tornado_concern_product" in command)
    assert product_command[product_command.index("--field") + 1] == "tornado_environment_outlook_hybrid"
    assert product_command[product_command.index("--map-style") + 1] == "outlook"
    assert product_command[product_command.index("--map-domain") + 1] == "regional"
    assert product_command[product_command.index("--display-preset") + 1] == "auto"
    assert (outdir / "ready" / "tornado_concern_ready_dates.txt").exists()
    assert (outdir / "eval" / "tornado_concern_ready_baseline.csv").exists()
