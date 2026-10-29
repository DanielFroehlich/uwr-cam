"""Computer vision motion detection pipeline for Underwater Rugby."""

from __future__ import annotations

import logging
from typing import Protocol, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MotionDetectorProtocol(Protocol):
    """Protocol for interchangeable motion detection algorithms."""

    def process_frame(self, frame: np.ndarray) -> float:
        """Processes a single video frame and returns a normalized motion density score in [0.0, 1.0]."""
        ...


class BallDetectorProtocol(Protocol):
    """Protocol for interchangeable ball detection algorithms (e.g. YOLO, CoreML, Color)."""

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        """Processes a frame to detect the underwater rugby ball.

        Returns:
            Tuple[float, Optional[Tuple[int, int, int, int]]]:
                - Confidence score in range [0.0, 1.0]
                - Bounding box (x, y, w, h) in pixels, or None if not found
        """
        ...


class BallDetectorYOLO:
    """YOLO-based deep learning detector for Underwater Rugby ball tracking.

    Supports pre-trained YOLO-World open-vocabulary models (e.g. yolov8s-worldv2.pt)
    as well as custom trained YOLO models (.pt) with native Apple Silicon Metal (MPS)
    hardware acceleration.
    """

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2.pt",
        confidence_threshold: float = 0.25,
        classes: Optional[list[str]] = None,
        device: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.classes = classes or ["underwater ball", "white ball"]
        self.enabled = enabled

        # Auto-detect Apple Silicon MPS hardware acceleration
        if device is None:
            import torch
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        if self.enabled:
            self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info(
                f"Loading YOLO ball detection model '{self.model_name}' on device: {self.device}..."
            )
            if "world" in self.model_name.lower():
                from ultralytics import YOLOWorld
                self._model = YOLOWorld(self.model_name)
                self._model.set_classes(self.classes)
            else:
                from ultralytics import YOLO
                self._model = YOLO(self.model_name)
            logger.info(
                f"YOLO ball detection model '{self.model_name}' initialized successfully on {self.device}."
            )
        except Exception as e:
            logger.error(
                f"Failed to load YOLO model '{self.model_name}': {e}. "
                "Falling back to disabled ball detection.",
                exc_info=True,
            )
            self._model = None

    def update_config(
        self,
        enabled: bool,
        confidence_threshold: float = 0.25,
        model_name: Optional[str] = None,
        classes: Optional[list[str]] = None,
    ) -> None:
        """Update detector configuration dynamically at runtime."""
        target_model = model_name or self.model_name
        target_classes = classes or self.classes
        model_changed = (self.model_name != target_model)
        classes_changed = (self.classes != target_classes)
        was_disabled = not self.enabled

        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.model_name = target_model
        self.classes = target_classes

        if not self.enabled:
            logger.info("Ball detector disabled dynamically.")
            return

        logger.info(f"Ball detector enabled dynamically (threshold={self.confidence_threshold:.2f}).")
        # Load model if enabling for the first time or if model/classes changed
        if self._model is None or model_changed or classes_changed or was_disabled:
            self._load_model()

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        if not self.enabled or self._model is None or frame is None or frame.size == 0:
            return 0.0, None

        try:
            results = self._model(
                frame,
                device=self.device,
                verbose=False,
                conf=self.confidence_threshold,
            )
            boxes = results[0].boxes
            if len(boxes) == 0:
                return 0.0, None

            # Find the detection with the highest confidence
            best_conf = 0.0
            best_bbox = None
            for box in boxes:
                conf = float(box.conf[0].item())
                if conf > best_conf:
                    best_conf = conf
                    xywh = [int(v) for v in box.xywh[0].tolist()]
                    best_bbox = (xywh[0], xywh[1], xywh[2], xywh[3])

            return best_conf, best_bbox

        except Exception as e:
            logger.debug(f"YOLO ball inference error: {e}")
            return 0.0, None


class MotionDetectorMOG2:
    """Adaptive Background Subtraction motion detector using OpenCV's MOG2.

    Learns dynamic backgrounds (such as water ripples) over time and extracts
    foreground masks corresponding to high-activity player motion.
    """

    def __init__(
        self,
        history: int = 150,
        var_threshold: float = 25.0,
        detect_shadows: bool = False,
        min_contour_area: int = 300,
    ) -> None:
        """
        Args:
            history: Length of the history for background modeling.
            var_threshold: Threshold on the squared Mahalanobis distance to decide whether it is well described by the background model.
            detect_shadows: Whether to detect and mark shadows (disabled for performance).
            min_contour_area: Minimum pixel area to consider a detected blob as valid motion rather than pixel noise.
        """
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.min_contour_area = min_contour_area

        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

        # Reusable morphological kernel for noise filtering
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def process_frame(self, frame: np.ndarray) -> float:
        """Processes a frame to calculate the motion density score.

        Args:
            frame: Input BGR frame.

        Returns:
            float: Motion density score in range [0.0, 1.0].
        """
        if frame is None or frame.size == 0:
            return 0.0

        h, w = frame.shape[:2]
        total_pixels = h * w
        if total_pixels == 0:
            return 0.0

        # Apply MOG2 background subtraction to get foreground mask
        fg_mask = self._subtractor.apply(frame)

        # Filter out minor salt-and-pepper noise using opening
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)

        # Threshold to binary (0 or 255)
        _, fg_binary = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours to filter out small isolated ripple artifacts
        contours, _ = cv2.findContours(fg_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_pixel_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_contour_area:
                motion_pixel_count += area

        # Normalize score between 0.0 and 1.0 (capped at 1.0)
        # Note: In UWR, players taking up ~10-15% of the frame is already high active motion.
        # We normalize such that 15% frame occupancy yields a high motion score near 1.0.
        density = motion_pixel_count / float(total_pixels)
        normalized_score = min(1.0, density / 0.15)

        logger.debug(f"Motion score: {normalized_score:.4f} (raw density: {density:.4f})")
        return float(normalized_score)
