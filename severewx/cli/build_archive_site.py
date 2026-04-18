"""CLI for static archive generation."""

from __future__ import annotations

import argparse

from severewx.archive.site import build_archive_site
from severewx.config import load_settings
from severewx.utils.logging import configure_logging
from severewx.utils.paths import build_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build archive site")
    parser.parse_args()
    logger = configure_logging()
    settings = load_settings()
    paths = build_paths(settings)
    output = build_archive_site(paths)
    logger.info("built archive site at %s", output)


if __name__ == "__main__":
    main()
