# severewx

`severewx` is a production-oriented Python project for Day 1-4 severe-weather forecasting. It is optimized for tornado outbreak detection, significant tornado support, hail and wind outbreak discrimination, confidence quality, bust-risk identification, and increasingly realistic training on archived forecast states.

The system combines:

- hazard-specific tree-based models for tornado, hail, wind, and any severe
- neighborhood-based report labels and configurable outbreak/event labels
- environmental severe proxies and synoptic support features
- light analog/context features inspired by pattern-matching workflows
- lead-aware calibration, confidence scoring, bust-risk heuristics, and verification
- resumable historical forecast-feature archiving for more realistic training and evaluation

The project intentionally excludes Day 5+ analog pages, short-fuse nowcasting, heavy interactive frontend work, teleconnections, GraphCast/Pangu ingest, and storm-scale ensemble infrastructure.

## Install

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
```

Optional extras:

```bash
pip install -e .[dev,grib,maps,ml]
```

`lightgbm` is the preferred model backend. If it is unavailable, the training pipeline falls back to scikit-learn histogram gradient boosting so the project remains runnable on a small local sample dataset.

## Directory Layout

```text
data/
  raw/
  interim/
  processed/
    feature_archive/
    archive_metadata/
  labels/
  models/
  outputs/maps/
  outputs/verification/
  archive/
```

## Data Flow

1. Ingest Day 1-4 forecast grids from NOMADS, a staged local GFS archive, or a configured local-file source, with synthetic fallback reserved for tests and local recovery.
2. Normalize forecast fields into a consistent internal xarray format.
3. Load SPC severe reports and build neighborhood-based hazard labels.
4. Build outbreak/event labels using configurable, broad-region-aware thresholds.
5. Derive environmental, composite, analog, and stability features.
6. Optionally build a resumable historical forecast-feature archive aligned to hazard and outbreak labels.
7. Train or load hazard and outbreak models, preferring archived real forecast-feature history when available.
8. Produce calibrated Day 1-4 hazard probabilities plus outbreak/confidence/bust-risk layers.
9. Render public maps and write gridded outputs.
10. Verify completed days and update a static archive site.

## Configuration

Defaults live in [severewx/config/defaults.yaml](/d:/severewx/severewx/config/defaults.yaml). The settings loader merges optional user overrides from:

- `SEVEREWX_CONFIG`
- `config.yaml` in the working directory

Key configurable areas:

- path roots and archive file locations
- ingest source mode, retry behavior, and lead hours
- label neighborhood radius and outbreak thresholds
- model backend and feature columns
- calibration grouping by hazard, lead day, and region
- confidence and bust-risk weights

## CLI Usage

Ingest forecast:

```bash
python -m severewx.cli.ingest_forecast --date 2026-04-09 --cycle 00
```

Build labels:

```bash
python -m severewx.cli.build_labels --start 2024-03-01 --end 2025-06-30
```

Build a daily tornado event catalog from `data/tor.json`:

```bash
python -m severewx.cli.build_tornado_catalog
```

Build a balanced pilot historical staging list from the tornado catalog outputs:

```bash
python -m severewx.cli.build_tornado_pilot_dates
```

Build historical archive:

```bash
python -m severewx.cli.build_historical_archive --start 2024-03-01 --end 2024-03-31 --cycles 00 12
```

Build the thin cached training-feature archive from those historical archive chunks:

```bash
python -m severewx.cli.build_feature_archive --start 2024-03-01 --end 2024-03-31 --cycles 00 12
```

Optional storage-efficient raw handling can be requested explicitly:

```bash
python -m severewx.cli.build_feature_archive --start 2024-03-01 --end 2024-03-31 --cycles 00 12 --raw-mode move_raw_to_archive
python -m severewx.cli.build_feature_archive --start 2024-03-01 --end 2024-03-31 --cycles 00 12 --raw-mode delete_raw_after_verified_cache
```

Validate staged local GFS files before archive build:

```bash
python -m severewx.cli.validate_staged_gfs --start 2024-03-01 --end 2024-03-31 --cycles 00 12
```

Stage a small pilot historical GFS set directly into the repo's `staged_gfs` layout:

```bash
python -m severewx.cli.stage_historical_gfs --start 2026-04-09 --end 2026-04-09 --cycles 00 --leads 0 6
```

Build a staged-local historical archive by pointing `ingest.source` at `local_staged_gfs` and configuring `ingest.local_staged_gfs.file_patterns` to your on-disk NCEI/AWS staging layout:

```yaml
ingest:
  source: local_staged_gfs
  allow_synthetic_fallback: false
  allow_partial_cycle: true
  local_staged_gfs:
    file_patterns:
      - "{root}/data/raw/staged_gfs/{date}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.grib2"
      - "{root}/data/raw/staged_gfs/{date}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.nc"
```

Summarize archive coverage:

```bash
python -m severewx.cli.archive_report
```

Train models:

```bash
python -m severewx.cli.train_models --hazards tornado hail wind any outbreak
```

If archive-preferred guardrails fail and you intentionally want a degraded training run:

```bash
python -m severewx.cli.train_models --hazards tornado hail wind any outbreak --allow-degraded
```

Run forecast:

```bash
python -m severewx.cli.run_forecast --date 2026-04-09 --cycle 00
```

Verify day:

```bash
python -m severewx.cli.verify_day --date 2026-04-09
```

Build archive site:

```bash
python -m severewx.cli.build_archive_site
```

## GitHub Actions Manual Runs

The repo now includes a manual GitHub Actions workflow at [.github/workflows/manual-model-run.yml](/d:/severewx/.github/workflows/manual-model-run.yml).

Open the `Actions` tab, choose `Manual Model Run`, then click `Run workflow`. The menu exposes these task choices:

- `forecast`: runs `python -m severewx.cli.run_forecast`
- `forecast_consensus`: runs `python -m severewx.cli.run_forecast_consensus`
- `tornado_concern_product`: runs `python -m severewx.cli.build_tornado_concern_product`
- `tornado_concern_run_bundle`: runs `python -m severewx.cli.build_tornado_concern_run_bundle`
- `tornado_concern_checkpoint`: runs `python -m severewx.cli.run_tornado_concern_checkpoint`
- `verify_day`: runs `python -m severewx.cli.verify_day`

Common inputs are passed through the workflow menu:

- `date` and `cycle` for forecast/product runs
- `start_date` and `end_date` for checkpoint runs
- `valid_start` and `valid_end` for custom tornado-concern windows
- `sources` for consensus source selection
- `output_dir` for artifact placement
- `extra_args` for any additional CLI flags not represented directly in the menu

Each run uploads the selected output directory as a GitHub Actions artifact and writes the exact command to the job summary.

## GitHub Pages Run Browser

The repo also includes a Pages deployment workflow at [.github/workflows/github-pages.yml](/d:/severewx/.github/workflows/github-pages.yml).

It builds the static archive browser with:

```bash
python -m severewx.cli.build_archive_site
```

and publishes `data/archive/` to GitHub Pages.

The generated site is self-contained for Pages hosting:

- forecast maps are copied into the publish tree
- verification case-review boards are copied into the publish tree
- tornado-concern run bundles are indexed from `data/outputs/verification/**/manifest.json`
- run-bundle images, summaries, and metadata files are linked directly from the site

To enable it in GitHub, allow Actions-based Pages deployment in the repository Pages settings, then run the `Deploy GitHub Pages` workflow or push the site-related changes to the default branch.

## Staged Local GFS Workflow

Recommended staged folder structure:

```text
data/raw/staged_gfs/
  2024-03-01/
    00/
      gfs.t00z.pgrb2.0p25.f000.grib2
      gfs.t00z.pgrb2.0p25.f006.grib2
      ...
    12/
      gfs.t12z.pgrb2.0p25.f000.grib2
      ...
```

The staged validator inspects each requested `date` / `cycle` / `lead` combination and reports:

- detected dates
- detected cycles
- detected lead hours
- usable files
- missing files
- per-cycle status: `complete`, `partial`, `degraded`, or `unusable`

Sample override config:

```yaml
ingest:
  source: local_staged_gfs
  allow_synthetic_fallback: false
  allow_partial_cycle: true
  local_staged_gfs:
    file_patterns:
      - "{root}/data/raw/staged_gfs/{date}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.grib2"
      - "{root}/data/raw/staged_gfs/{date}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.nc"
```

A ready-to-copy example lives at [local_staged_gfs.example.yaml](d:/severewx/severewx/config/local_staged_gfs.example.yaml).

For small pilot backfills, `stage_historical_gfs` writes directly into the same folder structure using a lightweight URL-template downloader. By default it tries the rolling AWS Open Data GFS archive at `noaa-gfs-bdp-pds`, saving each object as:

```text
data/raw/staged_gfs/{date}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.grib2
```

The downloader:

- accepts `--start`, `--end`, `--cycles`, `--leads`, and `--output-root`
- accepts `--source-strategy auto|ncei_historical|aws_recent`
- skips files already present unless `--force` is used
- supports repeated `--url-template` overrides if you want a different archive source
- writes a deterministic staging report to `data/interim/staged_gfs_download_{start}_{end}_{cycles}.json`

Source selection behavior is:

- `ncei_historical`: try NCEI THREDDS historical GFS paths first
- `aws_recent`: use the rolling AWS Open Data path directly
- `open_meteo_recent`: stage recent historical forecasts into NetCDF lead files using Open-Meteo's archived GFS-backed forecast API
- `auto`: prefer `ncei_historical` for older dates and `aws_recent` for recent dates, with `open_meteo_recent` as an additional recent-case fallback when applicable

Examples:

```bash
python -m severewx.cli.stage_historical_gfs --start 2011-04-27 --cycles 00 12 --source-strategy ncei_historical --timeout 180
python -m severewx.cli.stage_historical_gfs --start 2026-04-09 --cycles 00 --source-strategy aws_recent
python -m severewx.cli.stage_historical_gfs --start 2026-04-09 --cycles 00 --source-strategy open_meteo_recent
```

A practical staged-data workflow is:

```bash
python -m severewx.cli.stage_historical_gfs --start 2024-03-01 --end 2024-03-03 --cycles 00 12 --leads 0 6 12 18 24
python -m severewx.cli.validate_staged_gfs --start 2024-03-01 --end 2024-03-31 --cycles 00 12
python -m severewx.cli.build_historical_archive --start 2024-03-01 --end 2024-03-31 --cycles 00 12
python -m severewx.cli.build_feature_archive --start 2024-03-01 --end 2024-03-31 --cycles 00 12
python -m severewx.cli.archive_report
```

The validator writes a deterministic JSON report to `data/interim/` so you can check coverage before archive build.

### Raw File Handling Modes

The raw staged GFS lifecycle is conservative by default. Supported modes are:

- `keep_raw`
- `move_raw_to_archive`
- `delete_raw_after_verified_cache`

Default behavior is `keep_raw`.

Move/delete is allowed only when all of these are true:

- source is `local_staged_gfs`
- staged source validation status is `complete`
- historical archive chunk status is `local_real`
- cached feature archive metadata is written and marked ready
- cached feature archive row count is greater than zero

If any of those checks fail, raw files are retained.

## Rendering

`run_forecast` now writes a daily graphics suite to `data/outputs/maps/` for each valid Day 1-4 date in the run.

Individual map products:

- `any_severe`
- `tornado`
- `hail`
- `wind`
- `outbreak_risk`
- `confidence`
- `bust_risk`

Each valid day also gets a combined `daily_board` image that assembles those seven panels into a single overview graphic.

Filename format is deterministic:

- `{init_date}_{cycle}_day{lead_day}_{valid_date}_{product}.png`
- `{init_date}_{cycle}_day{lead_day}_{valid_date}_daily_board.png`

Hazard and outbreak products use daily maxima from the forecast run. Confidence and bust-risk maps use daily means so the board reflects signal quality rather than a single noisy hour. If a product field is missing, the renderer writes a placeholder panel instead of failing.

The renderer now uses a fixed CONUS template for both forecast products and case-review boards. When `cartopy` is available and `render.cartopy: true`, panels use a lower-48 `PlateCarree` view with a standard severe-weather national extent, thin state boundaries, and a restrained national outline. If Cartopy is unavailable or disabled, the system falls back to the same fixed CONUS extent in plain Matplotlib without failing the workflow.

### Tornado-Concern Run Bundles

Use the run-bundle workflow for a custom tornado-concern valid window. It builds direct regional, direct CONUS, and consensus CONUS products, then writes machine-readable status and environment artifacts.

```bash
.venv/bin/python -m severewx.cli.build_tornado_concern_run_bundle \
  --date 2026-05-03 \
  --cycle 00 \
  --valid-start 2026-05-05T12:00Z \
  --valid-end 2026-05-06T12:00Z \
  --outdir data/outputs/verification/run_bundle_2026-05-05_12z_to_2026-05-06_12z \
  --overwrite
```

Bundle outputs:

- `direct_regional/`: prediction-sourced regional outlook.
- `direct_conus/`: prediction-sourced CONUS outlook.
- `consensus_conus/`: consensus-sourced CONUS outlook.
- `status.md` and `status.csv`: readiness table across products.
- `run_summary.md`: direct regional quality, consensus source availability, and recommended next actions.
- `manifest.json`: deterministic paths to generated product artifacts.
- `environment.json`: render/basemap readiness, including Cartopy availability.

By default the bundle requires a production basemap. If Cartopy is unavailable, products with otherwise visible signal are marked `needs_render_review`; products with forecast or consensus failures remain `internal_review_only`. Pass `--allow-fallback-publication` only for internal smoke tests.

For a single product, use `build_tornado_concern_product` directly. `--artifact-source prediction` forces `forecast_products_*`; `--artifact-source consensus` forces `forecast_consensus_*`; `--artifact-source auto` preserves the default consensus preference for the hybrid field when consensus exists.

To check generated metadata without rerendering:

```bash
.venv/bin/python -m severewx.cli.tornado_concern_product_status \
  --products-dir data/outputs/verification/run_bundle_2026-05-05_12z_to_2026-05-06_12z \
  --recursive \
  --output-md data/outputs/verification/run_bundle_2026-05-05_12z_to_2026-05-06_12z/status.md \
  --fail-on-blocked \
  --fail-on-fallback-render
```

## Verification Case Review Graphics

`verify_day` now also renders compact case-review boards under `data/outputs/verification/`.

Filename format:

- `{init_date}_{cycle}_day{lead_day}_{valid_date}_case_review.png`

Each board includes:

- forecast tornado map
- forecast any severe map
- forecast outbreak-risk map
- confidence map
- bust-risk map
- observed outcome footprint panel
- concise summary box with outcome, quick metrics, confidence/bust tiers, and training-quality tier

Observed outcomes are shown as a simple footprint panel built from the existing gridded labels, with tornado, hail, wind, and null areas collapsed into a compact categorical view. If observed labels or one of the forecast fields is missing, the board still renders with placeholders instead of failing.

Case-review boards use the same fixed CONUS template as the forecast maps, so the forecast and observed panels share a consistent national framing.

## Outbreak Labels

Outbreak labels are intentionally configurable in [defaults.yaml](/d:/severewx/severewx/config/defaults.yaml), but the default scheme explicitly separates:

- `significant_tornado_outbreak_day`
- `tornado_outbreak_day`
- `significant_tornado_support_day`
- `hail_outbreak_day`
- `wind_mcs_outbreak_day`
- `active_non_outbreak_severe_day`
- `non_outbreak_severe_day`
- `null_day`

The logic combines total report counts, significant-report counts, and broad regional clustering because the target is operational usefulness for tornado outbreaks and corridor-scale severe episodes, not just point-hit report totals.

## Confidence And Signal Quality

`severewx` exposes both `confidence_score` and `signal_quality_score`.

- `signal_quality_score` emphasizes agreement, analog support, run-to-run stability, calibration quality, and coherence between tornado/outbreak support fields.
- `confidence_score` blends signal quality with probability magnitude so a high raw probability can still be downgraded when the signal is noisy or internally inconsistent.

For outbreak forecasting, a lower-probability but highly coherent tornado signal can be more operationally meaningful than a louder but unstable all-severe field.

## Bust Risk

Bust risk is rule-based in V1 and designed to be inspectable. The current score penalizes:

- strong cap / cap-hold scenarios
- low-level moisture failure
- forcing / instability mismatch
- contamination proxies from high PWAT plus weak lapse rates and broad coverage
- hazard disagreement
- weak analog support
- weak outbreak-support coherence

The output remains simple, but the rules are tuned toward practical Day 1-4 failure modes instead of generic "low confidence" labeling.

## Historical Archive And Training Data Preference

The project supports a historical Day 1-4 forecast-feature archive. Each archived init cycle stores:

- init date and cycle
- valid times and lead days
- normalized base fields
- derived features
- region metadata
- linked gridded hazard labels
- linked outbreak labels
- compact ingest and label-coverage diagnostics

The project also supports a thinner cached training-feature archive. This is the preferred long-term training dataset and keeps only:

- training metadata
- model feature columns
- hazard labels
- outbreak labels
- provenance and quality metadata needed for training decisions

Archive outputs are deterministic and resumable under `data/processed/feature_archive/` and `data/processed/archive_metadata/`.

Backfill runs now track:

- fully built local-real chunks
- fully built remote-real chunks
- partially built real chunks
- synthetic/degraded chunks
- skipped chunks
- hard failures

and emit corpus-level coverage summaries after each backfill/update.

Historical reruns preserve healthy real chunks by default, but they retry degraded/synthetic archive chunks when a real ingest source is configured so the corpus can improve instead of freezing synthetic fallback in place.

Training now prefers this archived real forecast-feature history when it is available. The fallback order is:

1. historical feature archive
2. processed forecast files
3. synthetic fallback

`train_models` writes a `training_data_summary.json` manifest so verification and archive views can report whether models were trained from real archived history or fallback paths.

Training now prefers the thin cached training-feature archive when present. The effective order is:

1. cached feature archive
2. historical feature archive
3. processed forecast files
4. synthetic fallback

The training summary now explicitly reports:

- number of real archived rows used
- number of synthetic-augmented rows used
- real rows by source kind (`local_staged_gfs`, `nomads`, etc.)
- real fraction overall and by lead day
- real positive-row coverage by hazard
- coverage by lead day and region
- archive row counts by status (`local_real`, `remote_real`, `partial_real`, `synthetic_degraded`)
- hazard positive counts by lead day
- whether archive-preferred mode was fully satisfied or only partially satisfied through augmentation
- archive-quality tier (`real-heavy`, `mixed`, or `synthetic-heavy`)
- exact guardrail failures and degraded-mode reason when training is not healthy

Archive-preferred training is now guarded by configurable thresholds in [defaults.yaml](/d:/severewx/severewx/config/defaults.yaml):

- minimum real archive rows overall
- minimum real positive rows by hazard
- minimum real rows by lead day
- maximum synthetic fraction for archive-preferred training

`train_models` enforces these guardrails by default. If the archive is too sparse, the CLI now fails instead of silently presenting a degraded archive-backed training run as healthy. Use `--allow-degraded` only when you intentionally want synthetic augmentation or fallback behavior.

## Ingest Resilience

The ingest path remains intentionally simple, but it is more durable now:

- retry/backoff for NOMADS downloads
- cache-first reuse of per-lead raw GRIB artifacts when cache metadata marks them complete
- per-lead raw-cache metadata that distinguishes complete, untracked, failed, and unusable artifacts
- partial-failure diagnostics by lead hour
- explicit missing-field and degraded-feature metadata
- optional `local_file` source mode for historical backfill workflows
- `local_staged_gfs` source mode for per-lead staged NCEI/AWS historical GFS files on disk
- ordered failover hooks via `ingest.source` plus `ingest.failover_sources`
- synthetic fallback retained mainly for tests and local recovery

## Verification

Verification is forecast-run oriented rather than just single-map oriented. `verify_day` writes a richer JSON payload with:

- per-day outbreak outcome summaries across the Day 1-4 valid dates in the run
- per-hazard summaries by lead day
- broad regional verification summaries
- tornado outbreak and significant-tornado-support case review tables
- top-risk and worst-bust daily rankings
- training-data provenance and coverage summaries
- training-quality context for the run, including the archive-quality tier and any degraded-mode reason

These outputs are also surfaced in the static archive.

## Archive Coverage Reporting

`archive_report` and archive builds write machine-readable coverage summaries under `data/processed/archive_metadata/`, including:

- archive chunks by date and cycle
- archive status counts (`local_real`, `remote_real`, `partial_real`, `synthetic_degraded`, `failed`, `skipped`)
- cached feature archive row counts and real/synthetic proportions
- raw-file lifecycle mode counts and raw files retained/moved/deleted
- cached feature archive coverage by lead day and region
- coverage by lead day
- coverage by region
- coverage by outbreak class
- hazard label availability
- real-row fractions and threshold-relevant aggregates
- fraction of current training rows coming from real archive history versus synthetic augmentation

## Current Simplifications

- NOMADS ingest is still focused on a manageable GFS field set rather than a multi-model blend.
- Historical archive building still depends on the configured ingest source rather than a full multi-source backfill framework.
- Advanced low-level helicity and forcing diagnostics use robust proxies when full fields are missing.
- Analogs use lightweight synoptic-vector similarity rather than a full analog page workflow.
- Rendering defaults to plain matplotlib if `cartopy` is unavailable.
- Regional calibration uses broad configurable regions rather than bespoke local climatologies.

## Extension Points

- replace or augment GFS ingest with ECMWF/CMC or curated blended sources
- expand historical backfill sources beyond NOMADS/local-file workflows
- add richer analog archives and regime clustering
- add CAM/WoFS-style short-fuse products
- add storm-mode diagnostics and convective contamination features
- expand archive HTML into an interactive frontend without changing the data products
