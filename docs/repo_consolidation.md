# Repository Consolidation

`tornado-concern-wip-snapshot` is the canonical branch for current development and GitHub Pages publication. GitHub `HEAD` and the repository default branch point to it.

The local `master` branch is an older baseline. It should not receive new work unless the branch naming policy changes. If the project later wants `master` or `main` as the canonical name, update that branch from `tornado-concern-wip-snapshot` after a clean test run rather than cherry-picking individual commits.

## Promotion Checklist

Before considering work promoted:

1. Confirm `git status --short --branch` is clean.
2. Confirm `git branch -vv` shows the active branch even with its upstream.
3. Run `pytest -q`.
4. Push `tornado-concern-wip-snapshot`.
5. Confirm GitHub default branch still points to `tornado-concern-wip-snapshot`.

Generated forecast data and trained model binaries under `data/` are not branch content. Promote those through model bundles or release assets, not normal Git commits.
