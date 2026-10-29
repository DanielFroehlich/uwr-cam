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

    Prepared for PyTorch/Ultralytics / CoreML integration on Apple Silicon.
    Currently acts as an extensible inference stub.
    """

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.4) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._model = None
        # When actual model weight is provided:
        # if self.model_path and Path(self.model_path).exists():
        #     from ultralytics import YOLO
        #     self._model = YOLO(self.model_path)

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        if frame is None or frame.size == 0:
            return 0.0, None

        # When YOLO model is loaded:
        # results = self._model(frame, device="mps") ...
        # For now, return default baseline confidence
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
