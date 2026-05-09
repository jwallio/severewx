"""Build deterministic training and skill artifacts from backtest outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from severewx.backtest.training_table import write_training_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training table, calibration candidate, and skill summaries from backtest outputs")
    parser.add_argument("--manifest", required=True, help="Backtest manifest JSON")
    parser.add_argument("--run-dir", action="append", required=True, help="Backtest output directory containing source_availability.json; repeatable")
    parser.add_argument("--output-dir", required=True, help="Directory for training artifacts")
    parser.add_argument("--include-unverified", action="store_true", help="Include rows without verification labels")
    parser.add_argument("--build-spatial", action="store_true", help="Also write bounded gridpoint-level spatial training rows")
    parser.add_argument("--label-path", help="Label NetCDF containing 25-mile tornado labels; required with --build-spatial")
    parser.add_argument("--max-negative-per-day", type=int, default=500, help="Maximum negative gridpoints sampled for each case/day/source set")
    parser.add_argument("--random-seed", type=int, default=20260509, help="Deterministic spatial negative-sampling seed")
    args = parser.parse_args()

    result = write_training_artifacts(
        manifest_path=Path(args.manifest),
        run_dirs=[Path(value) for value in args.run_dir],
        output_dir=Path(args.output_dir),
        include_unverified=bool(args.include_unverified),
        build_spatial=bool(args.build_spatial),
        label_path=Path(args.label_path) if args.label_path else None,
        max_negative_per_day=int(args.max_negative_per_day),
        random_seed=int(args.random_seed),
    )
    print(f"training_table_csv={result.table_csv}")
    print(f"training_table_json={result.table_json}")
    print(f"skill_summary_json={result.summary_json}")
    print(f"skill_summary_csv={result.summary_csv}")
    print(f"calibration_json={result.calibration_json}")
    print(f"archive_coverage_json={result.archive_coverage_json}")
    if result.spatial_table_csv:
        print(f"spatial_training_table_csv={result.spatial_table_csv}")
        print(f"spatial_training_table_json={result.spatial_table_json}")
        print(f"spatial_skill_summary_json={result.spatial_summary_json}")


if __name__ == "__main__":
    main()
