from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from severewx.models.forecast_consensus import (
    CONSENSUS_FIELD,
    ConsensusSource,
    build_forecast_consensus,
    consensus_metadata_path,
    consensus_product_path,
    consensus_weights_for_lead,
    source_metadata_path,
    source_product_path,
)


def _write_source_product(path: Path, values: float, *, times: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "tornado_environment_outlook_hybrid": (
                ("time", "lat", "lon"),
                np.full((len(times or ["2026-05-03T06:00:00"]), 2, 2), values, dtype=np.float32),
            )
        },
        coords={"time": pd.to_datetime(times or ["2026-05-03T06:00:00"]), "lat": [35.0, 36.0], "lon": [-98.0, -97.0]},
    ).to_netcdf(path)


def _write_source_metadata(path: Path, source: str, *, source_mode: str = "real") -> None:
    path.write_text(json.dumps({"ingest_summary": {"source": source, "source_mode": source_mode, "source_model": source}}), encoding="utf-8")


def test_consensus_weight_table_uses_lead_hour_bands() -> None:
    assert consensus_weights_for_lead(6) == {"hrrr_recent": 0.45, "rap_recent": 0.25, "nam_recent": 0.15, "aws_recent": 0.15}
    assert consensus_weights_for_lead(36) == {"hrrr_recent": 0.30, "nam_recent": 0.30, "aws_recent": 0.25, "gefs_mean_recent": 0.15}
    assert consensus_weights_for_lead(72) == {"aws_recent": 0.45, "gefs_mean_recent": 0.40, "nam_recent": 0.15}
    assert consensus_weights_for_lead(120) == {"gefs_mean_recent": 0.65, "aws_recent": 0.35}


def test_consensus_paths_are_deterministic(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    assert source_product_path(outputs, "2026-05-03", "00", "hrrr_recent").name == "forecast_products_2026-05-03_00_hrrr_recent.nc"
    assert source_metadata_path(outputs, "2026-05-03", "00", "hrrr_recent").name == "forecast_metadata_2026-05-03_00_hrrr_recent.json"
    assert consensus_product_path(outputs, "2026-05-03", "00").name == "forecast_consensus_2026-05-03_00.nc"
    assert consensus_metadata_path(outputs, "2026-05-03", "00").name == "forecast_consensus_metadata_2026-05-03_00.json"


def test_consensus_reweights_missing_sources_and_excludes_synthetic(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    rap_product = source_product_path(outputs, "2026-05-03", "00", "rap_recent")
    gfs_product = source_product_path(outputs, "2026-05-03", "00", "aws_recent")
    hrrr_product = source_product_path(outputs, "2026-05-03", "00", "hrrr_recent")
    rap_meta = source_metadata_path(outputs, "2026-05-03", "00", "rap_recent")
    gfs_meta = source_metadata_path(outputs, "2026-05-03", "00", "aws_recent")
    hrrr_meta = source_metadata_path(outputs, "2026-05-03", "00", "hrrr_recent")
    _write_source_product(rap_product, 0.20)
    _write_source_product(gfs_product, 0.60)
    _write_source_product(hrrr_product, 0.90)
    _write_source_metadata(rap_meta, "rap_recent")
    _write_source_metadata(gfs_meta, "aws_recent")
    _write_source_metadata(hrrr_meta, "synthetic_fallback", source_mode="synthetic")

    product_path, metadata_path = build_forecast_consensus(
        date="2026-05-03",
        cycle="00",
        sources=[
            ConsensusSource("rap_recent", rap_product, rap_meta),
            ConsensusSource("aws_recent", gfs_product, gfs_meta),
            ConsensusSource("hrrr_recent", hrrr_product, hrrr_meta),
        ],
        output_path=consensus_product_path(outputs, "2026-05-03", "00"),
        metadata_path=consensus_metadata_path(outputs, "2026-05-03", "00"),
        unavailable_sources=[{"source": "nam_recent", "reason": "source_run_failed_or_unavailable"}],
    )

    with xr.open_dataset(product_path) as dataset:
        # Lead 6 weights re-normalize from RAP 0.25 and GFS 0.15 to 0.625/0.375.
        assert np.allclose(dataset[CONSENSUS_FIELD].values, 0.35)
        assert np.allclose(dataset["model_mean"].values, 0.40)
        assert np.allclose(dataset["model_agreement_count"].values, 2.0)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["included_sources"] == ["aws_recent", "rap_recent"]
    assert metadata["excluded_sources"] == [
        {"source": "nam_recent", "reason": "source_run_failed_or_unavailable"},
        {"source": "hrrr_recent", "reason": "synthetic_source"},
    ]
    assert metadata["time_weights"][0]["applied_weights"] == {"rap_recent": 0.625, "aws_recent": 0.37499999999999994}
