import sys

import pandas as pd

from severewx.cli.build_tornado_pilot_dates import main as build_tornado_pilot_dates_main
from severewx.labels.tornado_catalog import build_tornado_pilot_dates


def _summary_frame() -> pd.DataFrame:
    rows = []
    for outbreak_class, rank, base_year in [
        ("significant_tornado_outbreak", 4, 2010),
        ("tornado_outbreak", 3, 2000),
        ("moderate_tornado_day", 2, 1990),
        ("low_tornado_day", 1, 1980),
    ]:
        for idx in range(6):
            rows.append(
                {
                    "date": f"{base_year + idx}-04-{idx + 1:02d}",
                    "tornado_count": 20 - idx if rank == 4 else 12 - idx if rank == 3 else 6 - idx if rank == 2 else 2 - min(idx, 1),
                    "significant_tornado_count": 5 - min(idx, 4) if rank == 4 else 2 - min(idx, 1) if rank == 3 else 1 if idx < 3 else 0,
                    "max_f_scale": 5 if idx == 0 else 4 if rank >= 3 else 2 if rank == 2 else 1,
                    "states_affected": "AL;MS",
                    "states_affected_count": 2 + (idx % 2),
                    "total_path_length_mi": 300.0 - 10 * idx,
                    "outbreak_class": outbreak_class,
                    "outbreak_rank": rank,
                }
            )
    return pd.DataFrame(rows)


def test_build_tornado_pilot_dates_balances_classes() -> None:
    summary = _summary_frame()
    candidates = summary.loc[summary["outbreak_rank"] >= 2].copy()
    pilot = build_tornado_pilot_dates(summary, candidates)

    assert len(pilot) == 20
    assert pilot["outbreak_class"].value_counts().to_dict() == {
        "significant_tornado_outbreak": 5,
        "tornado_outbreak": 5,
        "moderate_tornado_day": 5,
        "low_tornado_day": 5,
    }


def test_build_tornado_pilot_dates_respects_decade_and_spacing_controls() -> None:
    summary = pd.DataFrame(
        [
            {"date": "2011-04-27", "tornado_count": 20, "significant_tornado_count": 5, "max_f_scale": 5, "states_affected": "AL;MS", "states_affected_count": 2, "total_path_length_mi": 300.0, "outbreak_class": "significant_tornado_outbreak", "outbreak_rank": 4},
            {"date": "2011-04-28", "tornado_count": 18, "significant_tornado_count": 4, "max_f_scale": 4, "states_affected": "AL", "states_affected_count": 1, "total_path_length_mi": 260.0, "outbreak_class": "significant_tornado_outbreak", "outbreak_rank": 4},
            {"date": "2021-12-10", "tornado_count": 17, "significant_tornado_count": 4, "max_f_scale": 4, "states_affected": "AR;KY", "states_affected_count": 2, "total_path_length_mi": 240.0, "outbreak_class": "significant_tornado_outbreak", "outbreak_rank": 4},
            {"date": "2022-03-05", "tornado_count": 16, "significant_tornado_count": 3, "max_f_scale": 4, "states_affected": "IA", "states_affected_count": 1, "total_path_length_mi": 220.0, "outbreak_class": "significant_tornado_outbreak", "outbreak_rank": 4},
        ]
    )
    pilot = build_tornado_pilot_dates(
        summary,
        summary,
        class_targets={"significant_tornado_outbreak": 3, "tornado_outbreak": 0, "moderate_tornado_day": 0, "low_tornado_day": 0},
        max_per_decade=1,
        min_spacing_days=2,
    )

    assert pilot["date"].tolist() == ["2011-04-27", "2021-12-10"]


def test_build_tornado_pilot_dates_cli_writes_output(tmp_path, monkeypatch) -> None:
    root = tmp_path
    interim = root / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    summary = _summary_frame()
    candidates = summary.loc[summary["outbreak_rank"] >= 2].copy()
    summary.to_csv(interim / "tornado_daily_summary.csv", index=False)
    candidates.to_csv(interim / "tornado_candidate_dates.csv", index=False)
    config_path = root / "config.yaml"
    config_path.write_text("paths:\n  root: .\n", encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.setenv("SEVEREWX_CONFIG", str(config_path))
    monkeypatch.setattr(sys, "argv", ["build_tornado_pilot_dates"])

    build_tornado_pilot_dates_main()

    pilot = pd.read_csv(interim / "tornado_pilot_dates.csv")
    assert len(pilot) == 20
    assert set(pilot.columns) == {
        "date",
        "outbreak_class",
        "tornado_count",
        "significant_tornado_count",
        "max_f_scale",
        "total_path_length_mi",
        "states_affected_count",
    }
