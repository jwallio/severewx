# SPC-Comparable Day 1-3 Tornado Accuracy Track

`severewx` tornado-concern outlooks should be evaluated as probabilistic guidance, not as deterministic warning polygons. The benchmark definition is SPC-style tornado probability within 25 statute miles of a point for Day 1 and Day 2.

## Product Meaning

- Day 1 and Day 2 tornado probabilities are scored against tornado report neighborhoods using SPC-like bins: 2%, 5%, 10%, 15%, 30%, 45%, and 60%.
- Day 3 tornado guidance remains experimental because SPC Day 3 public guidance is primarily higher-uncertainty categorical or total-severe context.
- Conditional Intensity Groups are a separate intensity layer. They should not be blended into the base tornado probability.

## Accuracy Workflow

- Use `backtests/tornado_environment_consensus_v2.json` for the locked evaluation structure.
- The promoted candidate operational source set is HRRR, RAP, NAM, GFS/AWS, and ECMWF. OpenMeteo GFS remains an optional availability/comparison source until a larger locked backtest shows an independent accuracy gain.
- Keep `synthetic_fallback_allowed=false` for every accuracy, calibration, and public-readiness run.
- Run the backtest CLI with `--build-consensus --build-products --verify --score` once archived public NOAA/SPC data for the fixed cases is available locally.
- Review `backtest_summary.json` for fold coverage, source ablation, SPC threshold metrics, reliability bins, candidate calibration readiness, and public-readiness status.
- Use `docs/forecast_source_mix_decision.md` as the current source-mix decision record for ECMWF/OpenMeteo promotion status.

## Public-Readiness Rules

- A product should remain `internal_review_only` when real-source inputs are missing, Cartopy CONUS rendering fails, verification is missing, or calibration is stale.
- During the ECMWF candidate period, missing ECMWF also keeps the product `internal_review_only`; the artifact is still generated so the run can be inspected.
- A product can be considered a public candidate only after locked out-of-sample scores improve over the previous consensus baseline.
- Public copy must not claim parity with or superiority to official SPC guidance unless the locked test summary supports that claim.
