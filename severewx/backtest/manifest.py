"""Backtest manifest loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BacktestCase:
    case_id: str
    date: str
    cycle: str
    tags: tuple[str, ...]
    expected_signal: str
    fold: str = "eval"
    region: str = "unknown"
    season: str = "unknown"
    regime: str = "unknown"


@dataclass(frozen=True, slots=True)
class BacktestManifest:
    manifest_id: str
    description: str
    target_product: dict[str, Any]
    cases: tuple[BacktestCase, ...]
    path: Path


def _validate_date(value: str, *, field_name: str) -> str:
    try:
        Date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD: {value}") from exc
    return value


def _validate_cycle(value: str) -> str:
    normalized = str(value).zfill(2)
    if normalized not in {"00", "06", "12", "18"}:
        raise ValueError(f"cycle must be one of 00, 06, 12, 18: {value}")
    return normalized


def load_backtest_manifest(path: Path | str) -> BacktestManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_id = str(payload.get("manifest_id", "")).strip()
    if not manifest_id:
        raise ValueError("manifest_id is required")
    target_product = payload.get("target_product", {})
    if not isinstance(target_product, dict):
        raise ValueError("target_product must be an object")
    if target_product.get("synthetic_fallback_allowed") is not False:
        raise ValueError("backtest manifests must set target_product.synthetic_fallback_allowed=false")
    lead_days = target_product.get("lead_days", [])
    if lead_days != [1, 2, 3]:
        raise ValueError("initial backtest target must use lead_days [1, 2, 3]")
    source_values = target_product.get("default_sources", [])
    if not isinstance(source_values, list) or not all(str(value).strip() for value in source_values):
        raise ValueError("target_product.default_sources must be a non-empty list")

    required_case_metadata = bool(payload.get("require_case_metadata", False))
    seen_case_ids: set[str] = set()
    seen_case_keys: dict[tuple[str, str], str] = {}
    cases: list[BacktestCase] = []
    for raw_case in payload.get("cases", []):
        if not isinstance(raw_case, dict):
            raise ValueError("each case must be an object")
        case_id = str(raw_case.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("case_id is required")
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        case_date = _validate_date(str(raw_case.get("date", "")), field_name=f"{case_id}.date")
        case_cycle = _validate_cycle(str(raw_case.get("cycle", "00")))
        case_key = (case_date, case_cycle)
        if case_key in seen_case_keys:
            raise ValueError(f"duplicate date/cycle across cases: {case_date} {case_cycle}Z")
        seen_case_keys[case_key] = case_id
        tags = raw_case.get("tags", [])
        if not isinstance(tags, list):
            raise ValueError(f"tags must be a list for case {case_id}")
        fold = str(raw_case.get("fold", "eval")).strip().lower()
        if fold not in {"train", "tune", "test", "eval"}:
            raise ValueError(f"fold must be train, tune, test, or eval for case {case_id}")
        region = str(raw_case.get("region", "unknown")).strip().lower()
        season = str(raw_case.get("season", "unknown")).strip().lower()
        regime = str(raw_case.get("regime", "unknown")).strip().lower()
        if required_case_metadata and (
            fold == "eval" or region == "unknown" or season == "unknown" or regime == "unknown"
        ):
            raise ValueError(f"case metadata fold/region/season/regime is required for case {case_id}")
        cases.append(
            BacktestCase(
                case_id=case_id,
                date=case_date,
                cycle=case_cycle,
                tags=tuple(str(tag) for tag in tags),
                expected_signal=str(raw_case.get("expected_signal", "")).strip(),
                fold=fold,
                region=region,
                season=season,
                regime=regime,
            )
        )
    if not cases:
        raise ValueError("at least one backtest case is required")
    minimum_counts = payload.get("minimum_case_counts", {})
    if isinstance(minimum_counts, dict):
        tag_counts = {
            "tornado_relevant": sum(1 for case in cases if "tornado_relevant" in case.tags),
            "hard_negative": sum(1 for case in cases if "hard_negative" in case.tags),
            "null": sum(1 for case in cases if "null" in case.tags),
        }
        for key, minimum in minimum_counts.items():
            if int(tag_counts.get(str(key), 0)) < int(minimum):
                raise ValueError(f"manifest requires at least {minimum} {key} cases")
    required_folds = payload.get("required_folds", [])
    if isinstance(required_folds, list):
        present_folds = {case.fold for case in cases}
        missing_folds = [str(fold) for fold in required_folds if str(fold) not in present_folds]
        if missing_folds:
            raise ValueError(f"manifest missing required folds: {', '.join(missing_folds)}")
    return BacktestManifest(
        manifest_id=manifest_id,
        description=str(payload.get("description", "")),
        target_product=target_product,
        cases=tuple(cases),
        path=manifest_path,
    )


def manifest_to_jsonable(manifest: BacktestManifest) -> dict[str, Any]:
    return {
        "manifest_id": manifest.manifest_id,
        "description": manifest.description,
        "target_product": manifest.target_product,
        "cases": [
            {
                "case_id": case.case_id,
                "date": case.date,
                "cycle": case.cycle,
                "tags": list(case.tags),
                "expected_signal": case.expected_signal,
                "fold": case.fold,
                "region": case.region,
                "season": case.season,
                "regime": case.regime,
            }
            for case in manifest.cases
        ],
    }
