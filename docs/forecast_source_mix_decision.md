# Forecast Source Mix Decision

This note records the current production source decision for the Day 1-3 CONUS tornado-concern consensus product.

## Current Decision

- Keep `ecmwf_recent` in `sources=auto` with HRRR, RAP, NAM, and AWS/GFS.
- Keep `open_meteo_recent` optional for availability and comparison runs, not as a default production source.
- Keep `synthetic_fallback_allowed=false` for operational, backtest, calibration, and public-readiness workflows.

## Pilot15 Evidence

The Pilot15 comparison used `backtests/tornado_environment_consensus_v2_pilot.json` with trimmed representative leads `0,24,48`.

| source group | cases | public-ready days | 2-15% CSI | 30% CSI | note |
| --- | ---: | ---: | ---: | ---: | --- |
| NOAA/AWS only | 15 | 0/45 | 0.867 | 0.600 | Fails current source-readiness gate because ECMWF is absent. |
| NOAA/AWS + ECMWF | 15 | 31/45 | 0.867 | 0.533 | Production candidate source set. |
| NOAA/AWS + ECMWF + OpenMeteo | 15 | 36/45 | 0.867 | 0.571 | Better readiness count, but not enough skill evidence to promote by default. |

Artifacts:

- `data/outputs/verification/pilot15_source_group_comparison_trimmed.json`
- `data/outputs/verification/pilot15_source_group_comparison_trimmed.csv`
- `data/outputs/verification/pilot15_hydration_status_final/hydration_status.json`

## Public-Readiness Gate

For the current candidate period, a public-ready consensus product must use real ingest, include HRRR/RAP/NAM/AWS, include ECMWF, pass consensus agreement, and render with the production CONUS basemap. Missing ECMWF now marks the product `internal_review_only` but still allows the map artifact to be generated for inspection.

## Next Promotion Criteria

- Expand from Pilot15 to the locked v2 manifest before tuning weights or claiming skill gains.
- Fit calibration only on tune folds and evaluate only on locked test folds.
- Promote OpenMeteo only if it improves locked-test skill, not just product availability.
