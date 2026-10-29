#!/usr/bin/env python3
"""Dataset Frame Extraction Utility for Underwater Rugby YOLO Model Training.

Extracts sampled frames from MP4 videos for annotation in labeling tools
(e.g., Roboflow, CVAT, LabelImg).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("extract_frames")


def extract_frames_from_video(
    video_path: Path, output_dir: Path, target_fps: float = 1.0
) -> int:
    """Extract frames from a video file at target_fps."""
    if not video_path.is_file():
        logger.warning(f"Video file not found: {video_path}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_interval = max(1, int(round(video_fps / target_fps)))

    prefix = video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    frame_idx = 0

    logger.info(
        f"Processing {video_path.name} ({total_frames} frames, {video_fps:.1f} fps) -> "
        f"sampling every {frame_interval} frames (~{target_fps} fps)..."
    )

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % frame_interval == 0:
            out_filename = output_dir / f"{prefix}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted_count += 1

        frame_idx += 1

    cap.release()
    logger.info(f"Saved {extracted_count} frames from {video_path.name}")
    return extracted_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract training frames from match MP4 videos for YOLO annotation"
    )
    parser.add_argument(
        "--input-dir",
        default="cam-sim/videos",
        help="Directory containing source MP4 video files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/raw",
        help="Directory to save extracted JPEG frames (default: %(default)s)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Sampling rate in frames per second (default: %(default)s)",
    )

    args = parser.parse_args()
    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_dir).resolve()

    mp4_files = sorted(list(input_path.glob("*.mp4")))
    if not mp4_files:
        logger.warning(f"No MP4 files found in {input_path}")
        return

    logger.info(f"Found {len(mp4_files)} MP4 file(s) in {input_path}")
    total_extracted = 0
    for mp4 in mp4_files:
        total_extracted += extract_frames_from_video(mp4, output_path, args.fps)

    logger.info(
        f"Extraction complete! Total {total_extracted} frames saved to {output_path}"
    )


if __name__ == "__main__":
    main()
