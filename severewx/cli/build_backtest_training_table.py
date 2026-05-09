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
    args = parser.parse_args()

    result = write_training_artifacts(
        manifest_path=Path(args.manifest),
        run_dirs=[Path(value) for value in args.run_dir],
        output_dir=Path(args.output_dir),
        include_unverified=bool(args.include_unverified),
    )
    print(f"training_table_csv={result.table_csv}")
    print(f"training_table_json={result.table_json}")
    print(f"skill_summary_json={result.summary_json}")
    print(f"skill_summary_csv={result.summary_csv}")
    print(f"calibration_json={result.calibration_json}")


if __name__ == "__main__":
    main()
