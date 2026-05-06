from pathlib import Path

import yaml


def _workflow() -> dict:
    return yaml.safe_load(Path(".github/workflows/manual-model-run.yml").read_text(encoding="utf-8"))


def test_manual_model_run_dispatch_input_count_stays_within_github_limit() -> None:
    payload = _workflow()
    inputs = payload[True]["workflow_dispatch"]["inputs"]
    assert len(inputs) <= 25


def test_manual_model_run_disables_synthetic_forecast_fallback() -> None:
    text = Path(".github/workflows/manual-model-run.yml").read_text(encoding="utf-8")
    assert "allow_synthetic_fallback: false" in text
    assert "allow_synthetic_fallback: true" not in text
    assert "failover_sources:\n              - aws_recent" in text
