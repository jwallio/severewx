"""Aggregate verification summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def collect_verification_summaries(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for file_path in sorted(directory.glob("*_verification.json")):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        rows.append(payload["run_summary"])
    return pd.DataFrame(rows)
