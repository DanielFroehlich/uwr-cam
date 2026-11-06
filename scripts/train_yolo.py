#!/usr/bin/env python3
"""Custom YOLO Training Script for Underwater Rugby Ball Detection.

Trains a YOLO model (YOLOv8 / YOLO11) using Ultralytics with Apple Silicon
(MPS) or CUDA acceleration on the annotated Roboflow dataset.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_yolo")


def get_default_device() -> str:
    """Detect the most efficient available hardware acceleration device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_model(
    data_yaml: Path,
    base_model: str = "yolov8n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "mps",
    project: str = "weights/custom_ball",
    name: str = "yolov8n_uwr_ball",
    patience: int = 30,
    export_format: str | None = None,
) -> Path:
    """Train the YOLO model and evaluate on validation and test sets."""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset configuration file not found: {data_yaml}")

    logger.info("=" * 60)
    logger.info("Starting Custom YOLO Training for UWR Ball Detection")
    logger.info(f"Dataset YAML: {data_yaml}")
    logger.info(f"Base Architecture: {base_model}")
    logger.info(f"Epochs: {epochs} | Batch: {batch_size} | Image Size: {imgsz}")
    logger.info(f"Device: {device}")
    logger.info(f"Save Location: {project}/{name}")
    logger.info("=" * 60)

    # Initialize model with pre-trained weights for transfer learning
    logger.info(f"Loading base model: {base_model}...")
    model = YOLO(base_model)

    # Train
    train_results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save=True,
        plots=True,
        verbose=True,
        exist_ok=True,
    )

    save_dir = Path(train_results.save_dir) if hasattr(train_results, "save_dir") else Path(project) / name
    best_weights = save_dir / "weights" / "best.pt"
    logger.info(f"Training completed! Best weights saved to: {best_weights}")

    # Evaluate on the test split
    logger.info("\nEvaluating best model on Test split...")
    try:
        best_model = YOLO(str(best_weights))
        metrics = best_model.val(data=str(data_yaml), split="test", device=device)
        logger.info("--- Test Set Performance ---")
        logger.info(f"mAP50:     {metrics.box.map50:.4f}")
        logger.info(f"mAP50-95:  {metrics.box.map:.4f}")
        logger.info(f"Precision: {metrics.box.mp:.4f}")
        logger.info(f"Recall:    {metrics.box.mr:.4f}")
    except Exception as e:
        logger.warning(f"Could not complete test evaluation: {e}")

    # Optional export
    if export_format:
        logger.info(f"\nExporting model to format '{export_format}'...")
        try:
            exported_path = best_model.export(format=export_format)
            logger.info(f"Exported model: {exported_path}")
        except Exception as e:
            logger.error(f"Export failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Next Steps to Use this Model in uwr-cam-engine:")
    logger.info(f"Update uwr-cam-engine/config.yaml under 'ball_detector':")
    logger.info("  ball_detector:")
    logger.info("    enabled: true")
    logger.info(f"    model_name: \"{best_weights.resolve()}\"")
    logger.info("    confidence_threshold: 0.25")
    logger.info("=" * 60)

    return best_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train custom YOLO model for Underwater Rugby ball detection."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="model-training/dataset/data.yaml",
        help="Path to dataset data.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt"],
        help="Base YOLO model for transfer learning (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        help="Compute device (mps, cuda, cpu) (default: %(default)s)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="weights/custom_ball",
        help="Directory to save training runs (default: %(default)s)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov8n_uwr_ball",
        help="Name of the training run (default: %(default)s)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience in epochs without improvement (default: %(default)s)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        choices=["onnx", "coreml", "torchscript", "tflite"],
        help="Optional format to export the best model after training",
    )

    args = parser.parse_args()
    data_yaml_path = Path(args.data).resolve()

    try:
        train_model(
            data_yaml=data_yaml_path,
            base_model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            export_format=args.export,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
