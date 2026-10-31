"""Pipeline manager coordinating multi-camera frame grabbing and computer vision analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import BallDetectorConfig, CameraConfig, CameraWeights
from .cv_pipeline import (
    BallDetectorProtocol,
    BallDetectorYOLO,
    MotionDetectorMOG2,
    MotionDetectorProtocol,
)
from .ingestion import FrameGrabber

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Structured result of computer vision processing for a camera."""
    camera_id: str
    motion_score: float = 0.0          # Phase 3: Motion density [0.0, 1.0]
    ball_confidence: float = 0.0       # Phase 4: Ball detection confidence [0.0, 1.0]
    ball_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    score: float = 0.0                 # Phase 4: Composite score = (w_ball * B) + (w_motion * M)
    frame_width: int = 640
    frame_height: int = 360


class CameraPipeline:
    """Combines a single camera frame grabber with its detection models."""

    def __init__(
        self,
        config: CameraConfig,
        ball_detector: Optional[BallDetectorProtocol] = None,
    ) -> None:
        self.config = config
        self.grabber = FrameGrabber(camera_id=config.id, rtsp_url=config.rtsp_url)
        self.motion_detector: MotionDetectorProtocol = MotionDetectorMOG2()
        self.ball_detector: BallDetectorProtocol = ball_detector or BallDetectorYOLO()

    def start(self) -> None:
        self.grabber.start()

    def stop(self) -> None:
        self.grabber.stop()

    def process_latest(self) -> CVResult:
        """Pulls the latest frame and executes the CV pipeline."""
        ret, frame = self.grabber.read_latest()
        if not ret or frame is None:
            # Stream not ready or offline
            return CVResult(camera_id=self.config.id, motion_score=0.0, score=0.0)

        # 1. Compute motion density
        motion_score = self.motion_detector.process_frame(frame)

        # 2. Check if ball detection is enabled on detector
        is_ball_enabled = getattr(self.ball_detector, "enabled", True)
        if is_ball_enabled:
            ball_conf, ball_bbox = self.ball_detector.process_frame(frame)
            # Calculate composite camera score: S = (w_ball * B) + (w_motion * M)
            weights = self.config.weights or CameraWeights()
            w_ball = weights.ball
            w_motion = weights.motion
            composite_score = (w_ball * ball_conf) + (w_motion * motion_score)
        else:
            # Ball detector disabled: motion receives 100% weight
            ball_conf = 0.0
            ball_bbox = None
            composite_score = motion_score

        fh, fw = frame.shape[:2]
        return CVResult(
            camera_id=self.config.id,
            motion_score=motion_score,
            ball_confidence=ball_conf,
            ball_bbox=ball_bbox,
            score=round(float(composite_score), 4),
            frame_width=fw,
            frame_height=fh,
        )


class PipelineManager:
    """Manages all active camera pipelines, sharing a single GPU ball detector instance."""

    def __init__(
        self,
        camera_configs: List[CameraConfig],
        ball_detector_config: Optional[BallDetectorConfig] = None,
        ball_detector: Optional[BallDetectorProtocol] = None,
    ) -> None:
        if ball_detector is None:
            if ball_detector_config is not None:
                self.shared_ball_detector: BallDetectorProtocol = BallDetectorYOLO(
                    model_name=ball_detector_config.model_name,
                    confidence_threshold=ball_detector_config.confidence_threshold,
                    classes=ball_detector_config.classes,
                    enabled=ball_detector_config.enabled,
                )
            else:
                self.shared_ball_detector = BallDetectorYOLO()
        else:
            self.shared_ball_detector = ball_detector

        self.pipelines: Dict[str, CameraPipeline] = {
            cam.id: CameraPipeline(cam, ball_detector=self.shared_ball_detector)
            for cam in camera_configs
        }

    def update_ball_detector(self, ball_detector_config: BallDetectorConfig) -> None:
        """Update ball detector settings dynamically without restarting video streams."""
        if hasattr(self.shared_ball_detector, "update_config"):
            self.shared_ball_detector.update_config(
                enabled=ball_detector_config.enabled,
                confidence_threshold=ball_detector_config.confidence_threshold,
                model_name=ball_detector_config.model_name,
                classes=ball_detector_config.classes,
            )
        else:
            self.shared_ball_detector = BallDetectorYOLO(
                model_name=ball_detector_config.model_name,
                confidence_threshold=ball_detector_config.confidence_threshold,
                classes=ball_detector_config.classes,
                enabled=ball_detector_config.enabled,
            )
            for pipe in self.pipelines.values():
                pipe.ball_detector = self.shared_ball_detector
        logger.info(
            f"Ball detector dynamically updated: enabled={ball_detector_config.enabled}, "
            f"model='{ball_detector_config.model_name}', conf={ball_detector_config.confidence_threshold:.2f}"
        )

    def update_camera_weights(self, camera_configs: List[CameraConfig]) -> None:
        """Update camera weights dynamically without restarting video streams."""
        for cfg in camera_configs:
            if cfg.id in self.pipelines:
                self.pipelines[cfg.id].config.weights = cfg.weights
        logger.info("Camera weights dynamically updated across all pipelines.")

    def start_all(self) -> None:
        """Starts all background grabbers."""
        logger.info(f"Starting {len(self.pipelines)} camera ingestion pipelines...")
        for pipeline in self.pipelines.values():
            pipeline.start()

    def stop_all(self) -> None:
        """Stops all background grabbers."""
        logger.info("Stopping all camera pipelines...")
        for pipeline in self.pipelines.values():
            pipeline.stop()

    def process_tick(self) -> Dict[str, CVResult]:
        """Pulls latest frames and runs CV inference across all cameras.

        Returns:
            Dict[str, CVResult]: Mapping of camera_id to CVResult.
        """
        results: Dict[str, CVResult] = {}
        for cam_id, pipeline in self.pipelines.items():
            results[cam_id] = pipeline.process_latest()
        return results
