"""CLI entrypoint for cam-sim."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .supervisor import Supervisor


def setup_logging(level_name: str = "INFO") -> None:
    """Configure structured console logging."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_default_config() -> str:
    """Locate the default config file in container or local workspace."""
    candidates = [
        Path("/app/config.yaml"),
        Path("cam-sim/config.yaml"),
        Path("config.yaml"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="cam-sim: Underwater Rugby Multi-Camera RTSP Stream Simulator"
    )
    parser.add_argument(
        "-c", "--config",
        dest="config_path",
        default=find_default_config(),
        help="Path to YAML configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (default: %(default)s)",
    )
    parser.add_argument(
        "--mediamtx-bin",
        dest="mediamtx_bin",
        default="mediamtx",
        help="Path to mediamtx executable (default: %(default)s)",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        dest="ffmpeg_bin",
        default="ffmpeg",
        help="Path to ffmpeg executable (default: %(default)s)",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("cam-sim")

    logger.info("Initializing cam-sim Camera Stream Simulator...")
    try:
        config = load_config(args.config_path)
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    supervisor = Supervisor(
        config=config,
        mediamtx_bin=args.mediamtx_bin,
        ffmpeg_bin=args.ffmpeg_bin,
    )

    try:
        supervisor.run()
    except Exception as e:
        logger.error(f"Fatal error in supervisor: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
