"""CLI for cached training-feature archive building."""

from __future__ import annotations

import argparse

from severewx.archive.feature_cache import build_cached_feature_archive
from severewx.config import load_settings
from severewx.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached training-feature archive from historical archive chunks")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--cycles", nargs="+", default=["00", "12"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--raw-mode", choices=["keep_raw", "move_raw_to_archive", "delete_raw_after_verified_cache"])
    parser.add_argument("--raw-archive-root")
    args = parser.parse_args()

    logger = configure_logging()
    settings = load_settings()
    if args.raw_mode:
        settings.raw.setdefault("archive", {}).setdefault("raw_file_handling", {})["mode"] = args.raw_mode
    if args.raw_archive_root:
        settings.raw.setdefault("archive", {}).setdefault("raw_file_handling", {})["archive_root"] = args.raw_archive_root
    results = build_cached_feature_archive(args.start, args.end, args.cycles, settings=settings, force=args.force)
    logger.info(
        "cached feature archive complete built=%d skipped=%d missing_source_archive=%d raw_mode=%s",
        sum(result.status == "built" for result in results),
        sum(result.status == "skipped" for result in results),
        sum(result.status == "missing_source_archive" for result in results),
        settings.get("archive.raw_file_handling.mode", "keep_raw"),
    )


if __name__ == "__main__":
    main()
