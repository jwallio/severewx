from pathlib import Path

import yaml


def _workflow() -> dict:
    return yaml.safe_load(Path(".github/workflows/manual-model-run.yml").read_text(encoding="utf-8"))


def test_manual_model_run_dispatch_input_count_stays_within_github_limit() -> None:
    payload = _workflow()
    inputs = payload[True]["workflow_dispatch"]["inputs"]
    assert len(inputs) <= 25
    assert inputs["task"]["default"] == "forecast_consensus"
    assert inputs["map_domain"]["default"] == "conus"


def test_manual_model_run_disables_synthetic_forecast_fallback() -> None:
    text = Path(".github/workflows/manual-model-run.yml").read_text(encoding="utf-8")
    assert "allow_synthetic_fallback: false" in text
    assert "allow_synthetic_fallback: true" not in text
    assert "failover_sources:\n              - aws_recent" in text


def test_manual_model_run_restores_released_model_bundle_before_training() -> None:
    text = Path(".github/workflows/manual-model-run.yml").read_text(encoding="utf-8")
    restore_index = text.index("name: Restore released model bundle")
    train_index = text.index("name: Train missing forecast models")
    assert restore_index < train_index
    assert "MODEL_BUNDLE_TAG: models-2026-05-06" in text
    assert "gh release download" in text
    assert "released forecast model bundle restored" in text
    assert "tornado_concern_model.joblib" in text


def test_manual_model_run_uses_consensus_conus_pages_products() -> None:
    text = Path(".github/workflows/manual-model-run.yml").read_text(encoding="utf-8")
    assert "pip install -e .[dev,grib,maps,ml]" in text
    assert "cartopy: true" in text
    assert "build_tornado_preview={str(task in {'forecast', 'forecast_consensus'}" in text
    assert "preview_field={'tornado_environment_outlook_hybrid_consensus' if preview_is_consensus" in text
    assert "preview_artifact_source={'consensus' if preview_is_consensus" in text
    assert "preview_map_domain=conus" in text
    preview_step = text.split("name: Build tornado-concern Day 1-3 Pages maps", maxsplit=1)[1]
    assert "python -m severewx.cli.build_tornado_concern_spc_window_products" in preview_step
    assert "--day1-valid-start auto" in preview_step
    assert "--valid-date" not in preview_step
    assert "--map-domain regional" not in preview_step
    assert "--artifact-source prediction" not in preview_step
