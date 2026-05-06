# Model Artifacts

Trained model files are local runtime artifacts, not source files. The repo ignores `data/`, including `data/models/*.joblib`, so model binaries are not promoted by a normal `git push`.

## Current Policy

- Keep model source code, feature engineering, calibration logic, and tests in Git.
- Keep trained binaries and generated data under ignored `data/` folders.
- Package model binaries into an explicit bundle when they need to be shared or restored.
- Do not force-add `data/models/*.joblib` into the repository.

## Bundle Contents

A model bundle should include:

- `data/models/*_model.joblib`
- `data/models/analog_reference.parquet` when present
- `data/models/training_data_summary.json`
- `model_bundle_manifest.json` with file sizes and SHA-256 hashes

Create a local bundle with:

```bash
python scripts/manage_runtime_artifacts.py --inventory --bundle-models
```

The bundle is written to `data/model_bundles/`, which remains ignored. Publish that zip as a GitHub Release asset when a trained model set should be shared outside the local workspace.

## Runtime Cleanup

To inventory ignored runtime folders and clean transient Python caches:

```bash
python scripts/manage_runtime_artifacts.py --inventory --clean-caches
```

This does not delete forecast outputs, raw/staged data, processed archives, labels, model files, or Pages artifacts. Review the generated inventory before removing any `data/` subfolders.
