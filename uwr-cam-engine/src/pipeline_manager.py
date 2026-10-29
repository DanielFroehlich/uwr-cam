"""Pipeline manager coordinating multi-camera frame grabbing and computer vision analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import CameraConfig
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


class CameraPipeline:
    """Combines a single camera frame grabber with its detection models."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.grabber = FrameGrabber(camera_id=config.id, rtsp_url=config.rtsp_url)
        self.motion_detector: MotionDetectorProtocol = MotionDetectorMOG2()
        self.ball_detector: BallDetectorProtocol = BallDetectorYOLO()

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

        # 2. Compute ball detection confidence
        ball_conf, ball_bbox = self.ball_detector.process_frame(frame)

        # 3. Calculate composite camera score: S = (w_ball * B) + (w_motion * M)
        w_ball = self.config.weights.ball
        w_motion = self.config.weights.motion
        composite_score = (w_ball * ball_conf) + (w_motion * motion_score)

        return CVResult(
            camera_id=self.config.id,
            motion_score=motion_score,
            ball_confidence=ball_conf,
            ball_bbox=ball_bbox,
            score=round(float(composite_score), 4),
        )


class PipelineManager:
    """Manages all active camera pipelines."""

    def __init__(self, camera_configs: List[CameraConfig]) -> None:
        self.pipelines: Dict[str, CameraPipeline] = {
            cam.id: CameraPipeline(cam) for cam in camera_configs
        }

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
