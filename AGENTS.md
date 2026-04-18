# AGENTS

## Repo Purpose

`severewx` is a Day 1-4 severe-weather forecasting project focused on tornado, hail, wind, any severe, outbreak risk, confidence, bust risk, verification, and archive-driven training quality.

## Working Agreements

- Prefer efficient but accurate changes.
- Preserve deterministic filenames and paths.
- Preserve the current CONUS render template unless explicitly asked to change it.
- Prefer minimal, file-scoped edits over broad rewrites.
- Preserve good existing behavior and inspect the current implementation before editing.

## Do-Not-Touch Rules Unless Explicitly Requested

- Do not change render/board styling during data/archive tasks.
- Do not change model architecture during ingest/archive tasks.
- Do not weaken strict archive/training guardrails.
- Do not add new forecast products unless explicitly asked.

## Build And Test Expectations

- Run targeted tests first, then full `pytest -q` if practical.
- Keep tests network-free when possible.
- Prefer fixture-based validation for archive/backfill workflows.

## Definition Of Done

- Requested feature works end-to-end.
- Deterministic outputs are preserved.
- Tests pass.
- Summary lists changed files, exact behavior added, validation run, and what remains simplified.

## Prompting Preference

- The user prefers efficient but accurate prompts, especially for Codex/build tasks.
