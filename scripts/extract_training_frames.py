#!/usr/bin/env python3
"""Dataset Frame Extraction Utility for YOLO Custom Model Training.

Extracts an evenly distributed number of frames across all .mp4 videos in a source
directory, resizes them to the requested resolution, and saves them for annotation.
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
logger = logging.getLogger("extract_training_frames")


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    target_count: int,
    target_width: int,
    target_height: int,
) -> int:
    """Extract an evenly distributed target_count of frames from a video and resize them."""
    if not video_path.is_file():
        logger.warning(f"Video file not found: {video_path}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        logger.warning(f"Could not determine frame count for: {video_path}")
        cap.release()
        return 0

    # Ensure we don't ask for more frames than exist
    actual_count = min(target_count, total_frames)
    # Determine the step interval between frames to spread them across the whole video
    step = total_frames / actual_count

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = video_path.stem
    extracted_count = 0

    logger.info(
        f"Extracting {actual_count} frames from {video_path.name} "
        f"(Total frames: {total_frames}, Resolution: {target_width}x{target_height})..."
    )

    for i in range(actual_count):
        frame_idx = int(i * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning(f"Failed to read frame {frame_idx} from {video_path.name}")
            continue

        # Resize to specified dimensions
        resized_frame = cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_AREA
        )

        out_filename = output_dir / f"{prefix}_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_filename), resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        extracted_count += 1

    cap.release()
    logger.info(f"Saved {extracted_count} frames from {video_path.name}")
    return extracted_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and resize evenly distributed frames across video files."
    )
    parser.add_argument(
        "--input-dir",
        default="cam-sim/videos",
        help="Directory containing source MP4 video files",
    )
    parser.add_argument(
        "--output-dir",
        default="model-training/frames",
        help="Directory to save extracted JPEG frames",
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=100,
        help="Total number of frames to extract in total across all videos",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Width of output frames",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="Height of output frames",
    )

    args = parser.parse_args()
    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_dir).resolve()

    mp4_files = sorted(list(input_path.glob("*.mp4")))
    if not mp4_files:
        logger.warning(f"No MP4 files found in {input_path}")
        return

    num_videos = len(mp4_files)
    frames_per_video = args.total_frames // num_videos
    remainder = args.total_frames % num_videos

    logger.info(
        f"Found {num_videos} MP4 file(s). Target total frames: {args.total_frames} "
        f"(~{frames_per_video} frames per video)."
    )

    total_extracted = 0
    for i, mp4 in enumerate(mp4_files):
        # Distribute the remainder across the first few videos
        count_for_this_video = frames_per_video + (1 if i < remainder else 0)
        total_extracted += extract_frames_from_video(
            mp4, output_path, count_for_this_video, args.width, args.height
        )

    logger.info(
        f"Extraction complete! Total {total_extracted} frames saved to {output_path}"
    )


if __name__ == "__main__":
    main()
