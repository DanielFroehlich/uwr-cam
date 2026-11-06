#!/usr/bin/env python3
"""Evaluate YOLO models on match footage using Apple Silicon MPS or CUDA."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO, YOLOWorld


def evaluate_video(
    video_path: Path,
    model,
    model_name: str,
    device: str = "mps",
    classes=None,
    sample_count: int = 25,
    conf: float = 0.15,
) -> None:
    print(f"\n=======================================================")
    print(f"Evaluating {model_name} on {video_path.name}")
    print(f"Device: {device} | Testing {sample_count} sampled frames | Conf threshold: {conf}")
    print(f"=======================================================")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100)
    step = max(1, total_frames // sample_count)

    detected_frames = 0
    total_latency_ms = 0.0
    detections_summary = []

    for i in range(sample_count):
        frame_id = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        t0 = time.perf_counter()
        if classes is not None:
            results = model(frame, device=device, classes=classes, verbose=False, conf=conf)
        else:
            results = model(frame, device=device, verbose=False, conf=conf)
        latency = (time.perf_counter() - t0) * 1000.0
        total_latency_ms += latency

        boxes = results[0].boxes
        if len(boxes) > 0:
            detected_frames += 1
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                confidence = float(box.conf[0].item())
                xyxy = [int(x) for x in box.xyxy[0].tolist()]
                detections_summary.append(
                    f"Frame {frame_id:06d} (+{latency:.1f}ms): {cls_name} ({confidence:.2f}) at {xyxy}"
                )

    cap.release()

    avg_latency = total_latency_ms / max(1, sample_count)
    print(f"Average Inference Latency: {avg_latency:.1f} ms per frame")
    print(f"Frames with ball detections: {detected_frames}/{sample_count} ({detected_frames/sample_count*100:.1f}%)")
    if detections_summary:
        print("Detections found:")
        for det in detections_summary[:10]:
            print(f"  - {det}")
        if len(detections_summary) > 10:
            print(f"  ... and {len(detections_summary)-10} more detections")
    else:
        print(f"  (No detections above threshold {conf})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO models on match footage."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO weights (.pt) or name (e.g. yolov8n.pt, yolov8s-worldv2.pt, or weights/custom_ball/.../best.pt)",
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default="cam-sim/videos",
        help="Directory containing video files (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=25,
        help="Number of frames to sample per video (default: %(default)s)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to use for inference (mps, cuda, cpu)",
    )

    args = parser.parse_args()
    videos_dir = Path(args.videos_dir)
    video_files = sorted(list(videos_dir.glob("*.mp4")))
    if not video_files:
        print(f"No MP4 video files found in {videos_dir}")
        return

    print(f"Using compute acceleration: {args.device}")
    print(f"Found {len(video_files)} video file(s) in {videos_dir}")

    if args.model:
        model_path = args.model
        print(f"\n--- Loading Specified Model: {model_path} ---")
        if "world" in model_path.lower():
            model = YOLOWorld(model_path)
            model.set_classes(["ball", "sports ball", "underwater ball"])
        else:
            model = YOLO(model_path)

        for vid in video_files:
            evaluate_video(
                vid,
                model,
                model_name=Path(model_path).name,
                device=args.device,
                sample_count=args.sample_count,
                conf=args.conf,
            )
    else:
        # Default behavior: run comparisons
        print("\n--- Evaluating Pre-Trained YOLOv8n (COCO Class 32: sports ball) ---")
        yolo_coco = YOLO("yolov8n.pt")
        for vid in video_files:
            evaluate_video(
                vid,
                yolo_coco,
                "YOLOv8n (COCO Class 32: sports ball)",
                device=args.device,
                classes=[32],
                sample_count=args.sample_count,
                conf=args.conf,
            )

        print("\n--- Evaluating Pre-Trained YOLO-World (Open-Vocabulary) ---")
        try:
            yolo_world = YOLOWorld("yolov8s-worldv2.pt")
            yolo_world.set_classes(["white ball", "sports ball", "underwater ball"])
            for vid in video_files:
                evaluate_video(
                    vid,
                    yolo_world,
                    "YOLO-World (Prompts: white ball, sports ball, underwater ball)",
                    device=args.device,
                    sample_count=args.sample_count,
                    conf=args.conf,
                )
        except Exception as e:
            print(f"YOLO-World evaluation skipped: {e}")


if __name__ == "__main__":
    main()
