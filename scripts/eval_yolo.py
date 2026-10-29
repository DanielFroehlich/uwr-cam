#!/usr/bin/env python3
"""Evaluate pre-trained YOLO models on match footage using Apple Silicon MPS."""

from __future__ import annotations

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
    sample_count: int = 15,
) -> None:
    print(f"\n=======================================================")
    print(f"Evaluating {model_name} on {video_path.name}")
    print(f"Device: {device} | Testing {sample_count} sampled frames")
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
            results = model(frame, device=device, classes=classes, verbose=False, conf=0.15)
        else:
            results = model(frame, device=device, verbose=False, conf=0.15)
        latency = (time.perf_counter() - t0) * 1000.0
        total_latency_ms += latency

        boxes = results[0].boxes
        if len(boxes) > 0:
            detected_frames += 1
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
                conf = float(box.conf[0].item())
                xyxy = [int(x) for x in box.xyxy[0].tolist()]
                detections_summary.append(
                    f"Frame {frame_id:04d} (+{latency:.1f}ms): {cls_name} ({conf:.2f}) at {xyxy}"
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
        print("  (No detections above threshold 0.15)")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using compute acceleration: {device}")

    videos = [
        Path("cam-sim/videos/goal_left.mp4"),
        Path("cam-sim/videos/center_overview.mp4"),
    ]
    existing_videos = [v for v in videos if v.is_file()]
    if not existing_videos:
        print("No video files found in cam-sim/videos/")
        return

    # 1. Test Standard YOLOv8n (COCO Class 32: sports ball)
    print("\n--- Loading Pre-Trained YOLOv8n (COCO) ---")
    yolo_coco = YOLO("yolov8n.pt")
    # COCO Class 32 is 'sports ball'
    for vid in existing_videos:
        evaluate_video(vid, yolo_coco, "YOLOv8n (COCO Class 32: sports ball)", device=device, classes=[32])

    # 2. Test YOLO-World (Open-Vocabulary Prompts)
    print("\n--- Loading Pre-Trained YOLO-World ---")
    try:
        yolo_world = YOLOWorld("yolov8s-worldv2.pt")
        yolo_world.set_classes(["white ball", "sports ball", "underwater ball"])
        for vid in existing_videos:
            evaluate_video(vid, yolo_world, "YOLO-World (Prompts: white ball, sports ball, underwater ball)", device=device)
    except Exception as e:
        print(f"YOLO-World evaluation skipped: {e}")


if __name__ == "__main__":
    main()
