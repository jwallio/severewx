import numpy as np
import pandas as pd

from severewx.config import AppSettings
from severewx.models import tornado_concern as concern_module


class _FakeModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.clip(frame["sig_tor_support"].to_numpy(dtype=float) / 3.0, 0.0, 1.0)
        return np.column_stack([1.0 - values, values])


class _PassthroughCalibrator:
    def apply(self, frame: pd.DataFrame, score_column: str) -> np.ndarray:
        return frame[score_column].to_numpy(dtype=float)


class _TrainingFakeModel:
    def fit(self, x, y, sample_weight=None) -> None:
        self.sample_weight = sample_weight

    def predict_proba(self, x) -> np.ndarray:
        positive = np.clip(x["sig_tor_support"].to_numpy(dtype=float) / 3.0, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])


def _base_probability_stub(frame: pd.DataFrame, settings, paths) -> pd.DataFrame:
    return frame.assign(tornado_prob=0.08, any_prob=0.1, wind_prob=0.05, hail_prob=0.04, outbreak_risk=0.2)


def test_tornado_environment_outlook_grid_ignores_learned_hotspot_artifact() -> None:
    sig_tor_support = np.zeros((9, 9), dtype=float)
    tornado_favored_overlap = np.zeros((9, 9), dtype=float)
    scp_proxy = np.zeros((9, 9), dtype=float)
    sig_tor_support[1:4, 1:4] = 1.25
    tornado_favored_overlap[1:4, 1:4] = 2.50
    scp_proxy[1:4, 1:4] = 0.75

    result = concern_module.tornado_environment_outlook_grid(
        sig_tor_support=sig_tor_support,
        tornado_favored_overlap=tornado_favored_overlap,
        scp_proxy=scp_proxy,
    )

    assert float(result[4, 4]) == 0.0
    assert float(np.nanmax(result[1:4, 1:4])) >= 0.30
    assert np.unravel_index(int(np.nanargmax(result)), result.shape) != (4, 4)


def test_tornado_environment_outlook_v2_grid_compacts_broad_low_end_field() -> None:
    sig_tor_support = np.zeros((24, 24), dtype=float)
    tornado_favored_overlap = np.zeros((24, 24), dtype=float)
    scp_proxy = np.zeros((24, 24), dtype=float)
    sig_tor_support[3:21, 3:21] = 1.25
    tornado_favored_overlap[3:21, 3:21] = 2.50
    scp_proxy[3:21, 3:21] = 0.75
    sig_tor_support[9:15, 9:15] = 1.9
    tornado_favored_overlap[9:15, 9:15] = 3.4
    scp_proxy[9:15, 9:15] = 1.2

    baseline = concern_module.tornado_environment_outlook_grid(
        sig_tor_support=sig_tor_support,
        tornado_favored_overlap=tornado_favored_overlap,
        scp_proxy=scp_proxy,
    )
    challenger = concern_module.tornado_environment_outlook_v2_grid(
        sig_tor_support=sig_tor_support,
        tornado_favored_overlap=tornado_favored_overlap,
        scp_proxy=scp_proxy,
    )

    assert int(np.count_nonzero(challenger >= 0.02)) < int(np.count_nonzero(baseline >= 0.02))
    assert float(np.nanmax(challenger)) >= 0.30


def test_tornado_environment_outlook_hybrid_only_compacts_broad_fields() -> None:
    broad_sig = np.zeros((24, 24), dtype=float)
    broad_overlap = np.zeros((24, 24), dtype=float)
    broad_scp = np.zeros((24, 24), dtype=float)
    broad_sig[3:21, 3:21] = 1.25
    broad_overlap[3:21, 3:21] = 2.50
    broad_scp[3:21, 3:21] = 0.75
    broad_sig[9:15, 9:15] = 1.9
    broad_overlap[9:15, 9:15] = 3.4
    broad_scp[9:15, 9:15] = 1.2
    compact_sig = np.zeros((24, 24), dtype=float)
    compact_overlap = np.zeros((24, 24), dtype=float)
    compact_scp = np.zeros((24, 24), dtype=float)
    compact_sig[8:14, 8:14] = 1.25
    compact_overlap[8:14, 8:14] = 2.50
    compact_scp[8:14, 8:14] = 0.75

    broad_baseline = concern_module.tornado_environment_outlook_grid(
        sig_tor_support=broad_sig,
        tornado_favored_overlap=broad_overlap,
        scp_proxy=broad_scp,
    )
    broad_hybrid = concern_module.tornado_environment_outlook_hybrid_grid(
        sig_tor_support=broad_sig,
        tornado_favored_overlap=broad_overlap,
        scp_proxy=broad_scp,
    )
    compact_baseline = concern_module.tornado_environment_outlook_grid(
        sig_tor_support=compact_sig,
        tornado_favored_overlap=compact_overlap,
        scp_proxy=compact_scp,
    )
    compact_hybrid = concern_module.tornado_environment_outlook_hybrid_grid(
        sig_tor_support=compact_sig,
        tornado_favored_overlap=compact_overlap,
        scp_proxy=compact_scp,
    )

    assert int(np.count_nonzero(broad_hybrid >= 0.02)) < int(np.count_nonzero(broad_baseline >= 0.02))
    np.testing.assert_array_equal(compact_hybrid, compact_baseline)


def test_tornado_concern_training_frame_uses_spatial_tornado_target_and_significant_weights(monkeypatch) -> None:
    monkeypatch.setattr(concern_module, "_ensure_base_probability_columns", _base_probability_stub)
    frame = pd.DataFrame(
        {
            "date": ["2024-04-26", "2024-04-27", "2024-04-28"],
            "lead_day": [1, 2, 3],
            "lat": [35.0, 35.0, 35.0],
            "lon": [-97.0, -97.0, -97.0],
            "sig_tor_support": [0.2, 0.2, 0.02],
            "tornado_favored_overlap": [0.3, 0.3, 0.01],
            "scp_proxy": [0.2, 0.2, 0.01],
            "low_lcl_support": [0.9, 0.9, 0.2],
            "synoptic_support": [70.0, 70.0, 20.0],
            "qg_support_proxy": [40.0, 32.0, 5.0],
            "analog_outbreak_support": [0.6, 0.4, 0.1],
            "analog_sigtor_support": [0.5, 0.3, 0.1],
            "tornado": [1, 0, 1],
            "tornado_outbreak": [0, 1, 1],
            "significant_tornado_support": [0, 0, 0],
        }
    )
    outbreak_table = pd.DataFrame(
        {
            "date": ["2024-04-26", "2024-04-27", "2024-04-28"],
            "significant_tornado_support": [1, 0, 0],
        }
    )

    result = concern_module.tornado_concern_training_frame(frame, AppSettings(raw={}), paths=None, outbreak_table=outbreak_table)

    assert result["tornado_concern_target"].tolist() == [1, 0, 1]
    weights = dict(zip(result["date"], result["tornado_concern_sample_weight"], strict=False))
    assert weights["2024-04-26"] == 2.0
    assert weights["2024-04-28"] == 1.0


def test_tornado_concern_training_frame_filters_to_supportive_candidate_pixels_and_downsamples_negatives(monkeypatch) -> None:
    monkeypatch.setattr(concern_module, "_ensure_base_probability_columns", _base_probability_stub)
    settings = AppSettings(raw={"models": {"tornado_concern": {"max_negative_rows": 2}, "random_state": 42}})
    frame = pd.DataFrame(
        {
            "date": ["2024-04-26"] * 5,
            "lead_day": [1] * 5,
            "lat": [35.0, 35.5, 36.0, 36.5, 37.0],
            "lon": [-97.0, -97.0, -97.0, -97.0, -97.0],
            "sig_tor_support": [0.02, 0.4, 0.4, 0.4, 0.01],
            "tornado_favored_overlap": [0.01, 0.3, 0.3, 0.3, 0.01],
            "scp_proxy": [0.01, 0.2, 0.2, 0.2, 0.01],
            "low_lcl_support": [0.2, 0.9, 0.9, 0.9, 0.2],
            "synoptic_support": [20.0, 70.0, 70.0, 70.0, 20.0],
            "qg_support_proxy": [5.0] * 5,
            "analog_outbreak_support": [0.1] * 5,
            "analog_sigtor_support": [0.1] * 5,
            "tornado": [1, 0, 0, 0, 0],
            "hail_favored_overlap": [0.0, 0.8, 0.0, 0.0, 0.0],
            "wind_favored_overlap": [0.0, 0.0, 0.0, 0.0, 0.0],
            "hail": [0, 1, 0, 0, 0],
            "wind": [0, 0, 0, 0, 0],
            "any": [1, 1, 0, 0, 0],
        }
    )

    result = concern_module.tornado_concern_training_frame(frame, settings, paths=None)

    assert int(result["tornado_concern_target"].sum()) == 1
    assert len(result) == 3
    assert 37.0 not in result["lat"].tolist()
    assert 35.5 in result["lat"].tolist()


def test_tornado_concern_training_frame_adds_mode_separation_features(monkeypatch) -> None:
    monkeypatch.setattr(concern_module, "_ensure_base_probability_columns", _base_probability_stub)
    frame = pd.DataFrame(
        {
            "date": ["2024-04-26", "2024-04-26"],
            "lead_day": [1, 1],
            "lat": [35.0, 35.5],
            "lon": [-97.0, -97.0],
            "sig_tor_support": [0.4, 0.2],
            "tornado_favored_overlap": [0.6, 0.1],
            "hail_favored_overlap": [0.2, 0.7],
            "wind_favored_overlap": [0.1, 0.3],
            "scp_proxy": [0.4, 0.1],
            "low_lcl_support": [0.9, 0.9],
            "synoptic_support": [70.0, 70.0],
            "qg_support_proxy": [20.0, 20.0],
            "analog_outbreak_support": [0.3, 0.3],
            "analog_sigtor_support": [0.2, 0.2],
            "tornado": [1, 0],
        }
    )

    result = concern_module.tornado_concern_training_frame(frame, AppSettings(raw={}), paths=None)

    assert {
        "hail_favored_overlap",
        "wind_favored_overlap",
        "tornado_hail_mode_delta",
        "tornado_wind_mode_delta",
        "competing_mode_overlap",
        "tornado_mode_advantage",
        "competing_hazard_prob",
        "tornado_prob_advantage",
        "tornado_any_fraction",
        "tornado_scp_interaction",
        "sigtor_scp_interaction",
        "tornado_support_interaction",
    }.issubset(result.columns)
    positive_row = result.loc[result["tornado_concern_target"].eq(1)].iloc[0]
    assert positive_row["tornado_hail_mode_delta"] == 0.39999999999999997
    assert positive_row["tornado_wind_mode_delta"] == 0.5
    assert positive_row["competing_mode_overlap"] == 0.2
    assert positive_row["tornado_mode_advantage"] == 0.39999999999999997
    assert positive_row["competing_hazard_prob"] == 0.05
    assert positive_row["tornado_prob_advantage"] == 0.03
    assert positive_row["tornado_any_fraction"] == 0.7999999999999999
    assert positive_row["tornado_scp_interaction"] == 0.24
    assert positive_row["sigtor_scp_interaction"] == 0.16000000000000003
    assert positive_row["tornado_support_interaction"] == 0.24


def test_tornado_concern_training_frame_upweights_hard_negative_contamination(monkeypatch) -> None:
    monkeypatch.setattr(concern_module, "_ensure_base_probability_columns", _base_probability_stub)
    settings = AppSettings(raw={"models": {"tornado_concern": {"hard_negative_weight": 1.75}}})
    frame = pd.DataFrame(
        {
            "date": ["2024-05-26", "2024-05-26", "2024-05-26"],
            "lead_day": [1, 1, 1],
            "lat": [35.0, 35.5, 36.0],
            "lon": [-97.0, -97.0, -97.0],
            "sig_tor_support": [0.4, 0.2, 0.2],
            "tornado_favored_overlap": [0.6, 0.1, 0.1],
            "hail_favored_overlap": [0.1, 0.8, 0.1],
            "wind_favored_overlap": [0.1, 0.1, 0.8],
            "scp_proxy": [0.4, 0.2, 0.2],
            "low_lcl_support": [0.9, 0.9, 0.9],
            "synoptic_support": [70.0, 70.0, 70.0],
            "qg_support_proxy": [20.0, 20.0, 20.0],
            "analog_outbreak_support": [0.3, 0.3, 0.3],
            "analog_sigtor_support": [0.2, 0.2, 0.2],
            "tornado": [1, 0, 0],
            "hail": [0, 1, 0],
            "wind": [0, 0, 1],
            "any": [1, 1, 1],
        }
    )

    result = concern_module.tornado_concern_training_frame(frame, settings, paths=None)

    hard_negative_weights = result.loc[result["tornado_concern_target"].eq(0), "tornado_concern_sample_weight"].tolist()
    assert hard_negative_weights == [1.75, 1.75]


def test_apply_tornado_concern_model_adds_probability(monkeypatch) -> None:
    artifact = {
        "columns": ["lead_day", "sig_tor_support", "tornado_hail_mode_delta"],
        "model": _FakeModel(),
        "calibrator": _PassthroughCalibrator(),
    }
    monkeypatch.setattr(concern_module, "load_model_artifact", lambda paths, hazard: artifact)
    frame = pd.DataFrame(
        {
            "lead_day": [1, 1],
            "lat": [35.0, 36.0],
            "lon": [-97.0, -96.0],
            "sig_tor_support": [1.5, 2.4],
            "tornado_favored_overlap": [0.8, 0.9],
            "hail_favored_overlap": [0.2, 0.4],
        }
    )

    result = concern_module.apply_tornado_concern_model(frame, paths=object())

    assert "tornado_concern_prob" in result.columns
    assert "tornado_concern_environment_envelope" in result.columns
    assert "tornado_environment_outlook" in result.columns
    assert "tornado_environment_outlook_v2" in result.columns
    assert "tornado_environment_outlook_hybrid" in result.columns
    assert result["tornado_concern_prob"].tolist() == [0.5, 0.7999999999999999]


def test_apply_tornado_concern_model_caps_weak_scattered_public_activation(monkeypatch) -> None:
    artifact = {
        "columns": ["lead_day", "sig_tor_support"],
        "model": _FakeModel(),
        "calibrator": _PassthroughCalibrator(),
    }
    monkeypatch.setattr(concern_module, "load_model_artifact", lambda paths, hazard: artifact)
    frame = pd.DataFrame(
        {
            "lead_day": [1],
            "sig_tor_support": [1.5],
            "tornado_favored_overlap": [0.0],
            "hail_favored_overlap": [0.1],
            "wind_favored_overlap": [0.1],
            "scp_proxy": [0.0],
            "tornado_corridor_index": [0.0],
        }
    )

    result = concern_module.apply_tornado_concern_model(frame, paths=object(), settings=AppSettings(raw={}))

    assert float(result.loc[0, "tornado_concern_prob"]) == 0.049
    assert bool(result.loc[0, "tornado_concern_guardrail_applied"])


def test_apply_tornado_concern_model_preserves_physically_supported_activation(monkeypatch) -> None:
    artifact = {
        "columns": ["lead_day", "sig_tor_support"],
        "model": _FakeModel(),
        "calibrator": _PassthroughCalibrator(),
    }
    monkeypatch.setattr(concern_module, "load_model_artifact", lambda paths, hazard: artifact)
    frame = pd.DataFrame(
        {
            "lead_day": [1],
            "sig_tor_support": [1.5],
            "tornado_favored_overlap": [0.5],
            "hail_favored_overlap": [0.1],
            "wind_favored_overlap": [0.1],
            "scp_proxy": [0.2],
            "tornado_corridor_index": [0.2],
        }
    )

    result = concern_module.apply_tornado_concern_model(frame, paths=object(), settings=AppSettings(raw={}))

    assert float(result.loc[0, "tornado_concern_prob"]) == 0.5
    assert not bool(result.loc[0, "tornado_concern_guardrail_applied"])


def test_apply_tornado_concern_model_preserves_analog_supported_outbreak_activation(monkeypatch) -> None:
    artifact = {
        "columns": ["lead_day", "sig_tor_support"],
        "model": _FakeModel(),
        "calibrator": _PassthroughCalibrator(),
    }
    monkeypatch.setattr(concern_module, "load_model_artifact", lambda paths, hazard: artifact)
    frame = pd.DataFrame(
        {
            "lead_day": [1],
            "sig_tor_support": [1.5],
            "tornado_favored_overlap": [0.0],
            "hail_favored_overlap": [0.1],
            "wind_favored_overlap": [0.1],
            "scp_proxy": [0.0],
            "tornado_corridor_index": [0.0],
            "analog_outbreak_support": [0.65],
        }
    )

    result = concern_module.apply_tornado_concern_model(frame, paths=object(), settings=AppSettings(raw={}))

    assert float(result.loc[0, "tornado_concern_prob"]) == 0.5
    assert not bool(result.loc[0, "tornado_concern_guardrail_applied"])


def test_train_tornado_concern_model_uses_identity_calibrator_for_single_class_validation(monkeypatch) -> None:
    fake_model = _TrainingFakeModel()
    monkeypatch.setattr(concern_module, "_make_backend", lambda settings: (fake_model, "fake_backend"))
    monkeypatch.setattr(concern_module, "fit_group_calibrator", lambda *args, **kwargs: _PassthroughCalibrator())
    monkeypatch.setattr(concern_module, "_ensure_base_probability_columns", _base_probability_stub)
    frame = pd.DataFrame(
        {
            "date": ["2024-04-20", "2024-04-21", "2024-04-22", "2024-04-23", "2024-04-24"],
            "lead_day": [1, 1, 1, 1, 1],
            "lat": [35.0] * 5,
            "lon": [-97.0] * 5,
            "sig_tor_support": [2.2, 2.0, 1.8, 1.6, 0.2],
            "tornado_favored_overlap": [2.4, 2.2, 2.0, 1.8, 0.1],
            "scp_proxy": [0.8, 0.7, 0.6, 0.5, 0.1],
            "low_lcl_support": [0.9] * 5,
            "synoptic_support": [70.0] * 5,
            "qg_support_proxy": [40.0] * 5,
            "analog_outbreak_support": [0.6] * 5,
            "analog_sigtor_support": [0.5] * 5,
            "tornado": [1, 1, 1, 1, 0],
        }
    )

    trained = concern_module.train_tornado_concern_model(frame, AppSettings(raw={"models": {"validation_fraction": 0.2, "random_state": 42}}), paths=None)

    assert isinstance(trained.artifact["calibrator"], concern_module.IdentityCalibrator)
    assert fake_model.sample_weight is not None
    assert "lat" not in trained.artifact["columns"]
    assert "lon" not in trained.artifact["columns"]
    assert trained.artifact["columns"][0] == "lead_day"
