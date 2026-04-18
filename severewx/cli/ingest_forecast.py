"""CLI for forecast ingest."""

from __future__ import annotations

import argparse

from severewx.config import load_settings
from severewx.ingest.nomads import ingest_forecast_cycle
from severewx.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a forecast cycle")
    parser.add_argument("--date", required=True)
    parser.add_argument("--cycle", required=True)
    args = parser.parse_args()
    logger = configure_logging()
    settings = load_settings()
    output = ingest_forecast_cycle(args.date, args.cycle, settings=settings)
    logger.info("saved forecast dataset to %s", output)


if __name__ == "__main__":
    main()
