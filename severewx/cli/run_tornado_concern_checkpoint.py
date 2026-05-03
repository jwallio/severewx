from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("running=" + " ".join(command))
    result = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "command failed: "
            + " ".join(command)
            + "\n"
            + (result.stdout or "")
            + "\n"
            + (result.stderr or "")
        )
    if result.stdout.strip():
        print(result.stdout.strip())
    return result


def _require_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the baseline-only tornado-concern checkpoint workflow")
    parser.add_argument("--start", required=True, help="Inclusive start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Inclusive end date YYYY-MM-DD")
    parser.add_argument("--output-dir", required=True, help="Root output directory for the checkpoint artifacts")
    parser.add_argument("--skip-product", action="store_true", help="Skip the optional product prototype step")
    parser.add_argument("--skip-failure-review", action="store_true", help="Skip the failure-review packet step")
    parser.add_argument("--skip-coverage-refresh", action="store_true", help="Skip the coverage refresh step")
    parser.add_argument("--skip-ready-refresh", action="store_true", help="Skip the candidate/ready refresh step")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the baseline eval step")
    parser.add_argument("--product-date", help="Optional init date to build a baseline product prototype for")
    parser.add_argument("--product-cycle", default="00", help="Cycle for the optional product prototype")
    parser.add_argument("--score-variant", default="baseline", help="Tornado-concern eval score variant")
    parser.add_argument("--tornado-preference-mode", default="off", help="Tornado-concern eval top-day preference mode")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwrite for new product and failure-review outputs")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    coverage_dir = output_dir / "coverage"
    ready_dir = output_dir / "ready"
    eval_dir = output_dir / "eval"
    failure_review_dir = output_dir / "failure_review"
    product_dir = output_dir / "product"
    for path in [coverage_dir, ready_dir, eval_dir, failure_review_dir, product_dir]:
        path.mkdir(parents=True, exist_ok=True)

    coverage_csv = coverage_dir / "tornado_concern_coverage_audit.csv"
    coverage_md = coverage_dir / "tornado_concern_coverage_summary.md"
    real_dates_file = ready_dir / "tornado_concern_real_case_dates.txt"
    ready_dates_file = ready_dir / "tornado_concern_ready_dates.txt"
    ready_audit_csv = ready_dir / "tornado_concern_coverage_audit_refresh.csv"
    ready_ranked_candidates_csv = ready_dir / "tornado_concern_candidates_union_ranked_refresh.csv"
    eval_csv = eval_dir / "tornado_concern_ready_baseline.csv"
    eval_md = eval_dir / "tornado_concern_ready_baseline.md"

    if not args.skip_coverage_refresh:
        _run_command(
            [
                "python",
                "-m",
                "severewx.cli.tornado_concern_coverage",
                "--start",
                args.start,
                "--end",
                args.end,
                "--priority-tier",
                "all",
                "--output-csv",
                str(coverage_csv),
                "--output-md",
                str(coverage_md),
            ]
        )

    if not args.skip_ready_refresh:
        _run_command(
            [
                "python",
                "scripts/build_tornado_concern_real_dates.py",
                "--output",
                str(real_dates_file),
                "--ready-output",
                str(ready_dates_file),
                "--audit-csv",
                str(ready_audit_csv),
                "--ranked-candidates-csv",
                str(ready_ranked_candidates_csv),
                "--start",
                args.start,
                "--end",
                args.end,
                "--priority-tier",
                "all",
                "--backfill",
            ]
        )

    if not args.skip_eval:
        _require_exists(ready_dates_file, "ready dates file")
        _run_command(
            [
                "python",
                "-m",
                "severewx.cli.tornado_concern_eval",
                "--dates-file",
                str(ready_dates_file),
                "--skip-missing",
                "--raw-core-variant",
                "baseline",
                "--core-variant",
                "baseline",
                "--source-variant",
                "baseline",
                "--component-variant",
                "baseline",
                "--score-variant",
                args.score_variant,
                "--tornado-preference-mode",
                args.tornado_preference_mode,
                "--output-csv",
                str(eval_csv),
                "--output-md",
                str(eval_md),
            ]
        )

    if not args.skip_failure_review:
        _require_exists(eval_csv, "baseline eval CSV")
        ready_arg = [str(ready_dates_file)] if ready_dates_file.exists() else []
        command = [
            "python",
            "-m",
            "severewx.cli.tornado_concern_failure_review",
            "--eval-csv",
            str(eval_csv),
            "--outdir",
            str(failure_review_dir),
            "--max-cases",
            "20",
            "--include-hail-outranks",
            "--include-category-mismatches",
            "--include-missed-tornado-first",
        ]
        if ready_arg:
            command.extend(["--ready-dates-file", ready_arg[0]])
        if args.overwrite:
            command.append("--overwrite")
        _run_command(command)

    if not args.skip_product and args.product_date:
        product_target_dir = product_dir / f"{args.product_date}_{str(args.product_cycle).zfill(2)}z"
        command = [
            "python",
            "-m",
            "severewx.cli.build_tornado_concern_product",
            "--date",
            args.product_date,
            "--cycle",
            str(args.product_cycle).zfill(2),
            "--outdir",
            str(product_target_dir),
            "--field",
            "tornado_environment_outlook_hybrid",
            "--map-style",
            "outlook",
            "--map-domain",
            "regional",
            "--display-preset",
            "auto",
        ]
        if args.overwrite:
            command.append("--overwrite")
        _run_command(command)

    print(f"coverage_csv={coverage_csv}")
    print(f"ready_dates_file={ready_dates_file}")
    print(f"eval_csv={eval_csv}")
    print(f"failure_review_dir={failure_review_dir}")
    if args.product_date and not args.skip_product:
        print(f"product_dir={product_dir / f'{args.product_date}_{str(args.product_cycle).zfill(2)}z'}")


if __name__ == "__main__":
    main()
