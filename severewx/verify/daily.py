"""Forecast-run verification pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from severewx.calibration.regional import assign_regions
from severewx.calibration.reliability import reliability_table
from severewx.verify.metrics import brier_score, frequency_bias, safe_roc_auc, threshold_summary
from severewx.verify.outbreaks import case_review_table, outbreak_metrics


def _daily_prediction_slices(prediction_dataset: xr.Dataset) -> list[tuple[str, int, xr.Dataset]]:
    times = pd.to_datetime(prediction_dataset["time"].values)
    date_strings = [value.isoformat() for value in times.date]
    days = pd.Series(date_strings, index=times)
    grouped: list[tuple[str, int, xr.Dataset]] = []
    for lead_day, (valid_date, group_times) in enumerate(days.groupby(days.values), start=1):
        _ = group_times
        mask = np.asarray(date_strings) == valid_date
        grouped.append((valid_date, lead_day, prediction_dataset.isel(time=mask).max("time")))
    return grouped


def _empty_label_slice(prediction_daily: xr.Dataset) -> xr.Dataset:
    return xr.Dataset(
        {hazard: xr.zeros_like(prediction_daily["any_prob"]).astype("int8") for hazard in ("tornado", "hail", "wind", "any")},
        coords={"lat": prediction_daily["lat"], "lon": prediction_daily["lon"]},
    )


def _flatten_region_frame(prediction_daily: xr.Dataset, label_slice: xr.Dataset) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "lat": np.repeat(prediction_daily["lat"].values, prediction_daily.sizes["lon"]),
            "lon": np.tile(prediction_daily["lon"].values, prediction_daily.sizes["lat"]),
        }
    )
    for hazard in ("tornado", "hail", "wind", "any"):
        frame[f"{hazard}_prob"] = prediction_daily[f"{hazard}_prob"].values.ravel()
        frame[f"{hazard}_label"] = label_slice[hazard].values.ravel()
    frame["region"] = assign_regions(frame)
    return frame


def _hazard_metric_row(hazard: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "hazard": hazard,
        "brier_score": brier_score(y_true, y_prob),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "frequency_bias": frequency_bias(y_true, y_prob),
    }
    row.update(threshold_summary(y_true, y_prob))
    return row


def verify_daily_probabilities(
    prediction_dataset: xr.Dataset,
    label_dataset: xr.Dataset,
    output_path: Path,
    target_date: str | None = None,
    outbreak_table: pd.DataFrame | None = None,
    training_data_summary: dict[str, object] | None = None,
    evaluation_metadata: dict[str, object] | None = None,
) -> Path:
    per_day_rows: list[dict[str, object]] = []
    lead_day_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    reliability_outputs: dict[str, list[dict[str, object]]] = {}

    outbreak_lookup = outbreak_table.set_index("date") if outbreak_table is not None and not outbreak_table.empty else pd.DataFrame()
    run_date = target_date or pd.to_datetime(prediction_dataset["time"].values[0]).date().isoformat()
    training_quality_tier = str((training_data_summary or {}).get("training_quality_tier", "unknown"))

    for valid_date, lead_day, prediction_daily in _daily_prediction_slices(prediction_dataset):
        if "date" in label_dataset.coords and np.datetime64(valid_date) in label_dataset["date"].values:
            label_slice = label_dataset.sel(date=np.datetime64(valid_date))
        else:
            label_slice = _empty_label_slice(prediction_daily)

        day_row: dict[str, object] = {
            "init_date": run_date,
            "valid_date": valid_date,
            "lead_day": lead_day,
            "forecast_outbreak_prob": float(prediction_daily["outbreak_risk"].max().item()),
            "forecast_tornado_prob": float(prediction_daily["tornado_prob"].max().item()),
            "forecast_hail_prob": float(prediction_daily["hail_prob"].max().item()),
            "forecast_wind_prob": float(prediction_daily["wind_prob"].max().item()),
            "forecast_any_prob": float(prediction_daily["any_prob"].max().item()),
            "forecast_sig_tor_support": float(prediction_daily["sig_tor_support"].max().item()),
            "forecast_confidence": float(prediction_daily["confidence_score"].mean().item()),
            "forecast_signal_quality": float(prediction_daily["signal_quality_score"].mean().item()),
            "forecast_bust_risk": float(prediction_daily["bust_risk_score"].mean().item()),
        }

        if not outbreak_lookup.empty and valid_date in outbreak_lookup.index:
            for column, value in outbreak_lookup.loc[valid_date].to_dict().items():
                day_row[f"observed_{column}"] = value
        else:
            day_row["observed_any_outbreak"] = int(label_slice["any"].sum().item() > 0)
            day_row["observed_tornado_outbreak"] = 0
            day_row["observed_significant_tornado_support"] = 0
            day_row["observed_category"] = "unknown"

        for hazard in ("tornado", "hail", "wind", "any"):
            y_true = label_slice[hazard].values.ravel()
            y_prob = prediction_daily[f"{hazard}_prob"].values.ravel()
            metric_row = _hazard_metric_row(hazard, y_true, y_prob)
            lead_day_rows.append({"valid_date": valid_date, "lead_day": lead_day, **metric_row})
            reliability_outputs[f"{valid_date}_{hazard}"] = reliability_table(y_true, y_prob).to_dict(orient="records")
            day_row[f"{hazard}_brier"] = metric_row["brier_score"]
            day_row[f"{hazard}_roc_auc"] = metric_row["roc_auc"]

        region_frame = _flatten_region_frame(prediction_daily, label_slice)
        for region, subset in region_frame.groupby("region"):
            for hazard in ("tornado", "hail", "wind", "any"):
                region_rows.append(
                    {
                        "valid_date": valid_date,
                        "lead_day": lead_day,
                        "region": region,
                        **_hazard_metric_row(hazard, subset[f"{hazard}_label"].to_numpy(), subset[f"{hazard}_prob"].to_numpy()),
                    }
                )
        per_day_rows.append(day_row)

    per_day = pd.DataFrame(per_day_rows)
    lead_day_summary = pd.DataFrame(lead_day_rows).groupby(["lead_day", "hazard"], as_index=False).agg(
        brier_score=("brier_score", "mean"),
        roc_auc=("roc_auc", "mean"),
        frequency_bias=("frequency_bias", "mean"),
        hit_rate=("hit_rate", "mean"),
        false_alarm_rate=("false_alarm_rate", "mean"),
    )
    regional_summary = pd.DataFrame(region_rows).groupby(["region", "hazard"], as_index=False).agg(
        brier_score=("brier_score", "mean"),
        roc_auc=("roc_auc", "mean"),
        frequency_bias=("frequency_bias", "mean"),
        hit_rate=("hit_rate", "mean"),
        false_alarm_rate=("false_alarm_rate", "mean"),
    )

    outbreak_summary = outbreak_metrics(
        per_day.get("observed_any_outbreak", pd.Series(dtype=int)),
        per_day["forecast_outbreak_prob"],
    )
    tornado_outbreak_summary = outbreak_metrics(
        per_day.get("observed_tornado_outbreak", pd.Series(dtype=int)),
        per_day["forecast_tornado_prob"],
    )
    significant_tornado_summary = outbreak_metrics(
        per_day.get("observed_significant_tornado_support", pd.Series(dtype=int)),
        per_day["forecast_sig_tor_support"],
        threshold=0.4,
    )

    rankings = {
        "top_tornado_outbreak_risk_days": per_day.sort_values("forecast_tornado_prob", ascending=False).head(5).to_dict(orient="records"),
        "top_outbreak_risk_days": per_day.sort_values("forecast_outbreak_prob", ascending=False).head(5).to_dict(orient="records"),
        "worst_bust_risk_days": per_day.sort_values("forecast_bust_risk", ascending=False).head(5).to_dict(orient="records"),
    }
    payload = {
        "run_summary": {
            "init_date": run_date,
            "n_valid_days": int(len(per_day)),
            "training_data_source": (training_data_summary or {}).get("source_preference", "unknown"),
            "training_quality_tier": training_quality_tier,
            "training_guardrails_passed": (training_data_summary or {}).get("guardrails_passed"),
            "training_degraded_mode_reason": (training_data_summary or {}).get("degraded_mode_reason", ""),
            "evaluation_source": ((evaluation_metadata or {}).get("ingest_summary", {}) or {}).get("source", "unknown"),
            **{f"any_{key}": value for key, value in outbreak_summary.items()},
            **{f"tornado_{key}": value for key, value in tornado_outbreak_summary.items()},
            **{f"significant_tornado_{key}": value for key, value in significant_tornado_summary.items()},
        },
        "training_data_summary": training_data_summary or {},
        "evaluation_metadata": evaluation_metadata or {},
        "per_day": per_day.to_dict(orient="records"),
        "lead_day_summary": lead_day_summary.assign(training_quality_tier=training_quality_tier).to_dict(orient="records"),
        "regional_summary": regional_summary.assign(training_quality_tier=training_quality_tier).to_dict(orient="records"),
        "verification_slices": {
            "training_quality_tier": training_quality_tier,
            "guardrails_passed": (training_data_summary or {}).get("guardrails_passed"),
        },
        "reliability": reliability_outputs,
        "rankings": rankings,
        "tornado_outbreak_case_review": case_review_table(per_day).to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
