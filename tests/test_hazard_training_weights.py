import numpy as np
import pandas as pd

from severewx.config import load_settings
from severewx.models import train as train_module


class _PassthroughCalibrator:
    def apply(self, frame: pd.DataFrame, score_column: str) -> np.ndarray:
        return frame[score_column].to_numpy(dtype=float)


class _FakeModel:
    def __init__(self) -> None:
        self.sample_weight = None

    def fit(self, x, y, sample_weight=None) -> None:
        self.sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    def predict_proba(self, x) -> np.ndarray:
        positive = np.full(len(x), 0.25, dtype=float)
        return np.column_stack([1.0 - positive, positive])


def test_train_hazard_model_uses_balanced_binary_sample_weight(monkeypatch) -> None:
    settings = load_settings()
    settings.raw["models"]["min_training_rows"] = 1
    settings.raw["models"]["validation_fraction"] = 0.2
    settings.raw["models"]["hazard_sample_weighting"] = {
        "enabled": True,
        "mode": "balanced_binary",
        "min_positive_weight": 1.0,
        "max_positive_weight": 10.0,
        "negative_weight": 1.0,
    }
    fake_model = _FakeModel()
    monkeypatch.setattr(train_module, "_make_backend", lambda settings: (fake_model, "fake_backend"))
    monkeypatch.setattr(train_module, "fit_group_calibrator", lambda frame, score_column, target_column, settings: _PassthroughCalibrator())
    frame = pd.DataFrame(
        {
            "date": ["2024-04-20", "2024-04-21", "2024-04-22", "2024-04-23", "2024-04-24"],
            "lead_day": [1, 1, 1, 1, 1],
            "lat": [35.0, 35.0, 35.0, 35.0, 35.0],
            "lon": [-97.0, -97.0, -97.0, -97.0, -97.0],
            "cape": [2500.0, 900.0, 1000.0, 1100.0, 1200.0],
            "tornado": [1, 0, 0, 0, 0],
        }
    )
    trained = train_module.train_hazard_model("tornado", frame, settings)
    assert fake_model.sample_weight is not None
    assert fake_model.sample_weight.tolist() == [3.0, 1.0, 1.0, 1.0]
    assert trained.artifact["sample_weighting"]["enabled"] is True
    assert trained.artifact["sample_weighting"]["mode"] == "balanced_binary"
    assert trained.artifact["sample_weighting"]["positive_weight"] == 3.0


def test_train_hazard_model_uses_per_hazard_max_positive_weight(monkeypatch) -> None:
    settings = load_settings()
    settings.raw["models"]["min_training_rows"] = 1
    settings.raw["models"]["validation_fraction"] = 0.2
    settings.raw["models"]["hazard_sample_weighting"] = {
        "enabled": True,
        "mode": "balanced_binary",
        "min_positive_weight": 1.0,
        "max_positive_weight": 10.0,
        "per_hazard_max_positive_weight": {
            "wind": 3.0,
        },
        "negative_weight": 1.0,
    }
    fake_model = _FakeModel()
    monkeypatch.setattr(train_module, "_make_backend", lambda settings: (fake_model, "fake_backend"))
    monkeypatch.setattr(train_module, "fit_group_calibrator", lambda frame, score_column, target_column, settings: _PassthroughCalibrator())
    frame = pd.DataFrame(
        {
            "date": ["2024-04-20", "2024-04-21", "2024-04-22", "2024-04-23", "2024-04-24"],
            "lead_day": [1, 1, 1, 1, 1],
            "lat": [35.0, 35.0, 35.0, 35.0, 35.0],
            "lon": [-97.0, -97.0, -97.0, -97.0, -97.0],
            "cape": [2500.0, 900.0, 1000.0, 1100.0, 1200.0],
            "wind": [1, 0, 0, 0, 0],
        }
    )
    trained = train_module.train_hazard_model("wind", frame, settings)
    assert fake_model.sample_weight is not None
    assert fake_model.sample_weight.tolist() == [3.0, 1.0, 1.0, 1.0]
    assert trained.artifact["sample_weighting"]["hazard"] == "wind"
    assert trained.artifact["sample_weighting"]["max_positive_weight"] == 3.0
    assert trained.artifact["sample_weighting"]["positive_weight"] == 3.0
